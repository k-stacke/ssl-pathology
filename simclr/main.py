import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda import amp

from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from simclr.utils import distribute_over_GPUs, reload_weights, validate_arguments
from simclr.get_dataloader import get_dataloader
from simclr.model import Model
from simclr.optimisers import LARS
from simclr.augmentations import Denormalize

torch.backends.cudnn.benchmark=True

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, scaler, opt, epoch):
    denom_transform = Denormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    neg_cosine_sim = {0: [], 1:[]}
    pos_cosine_sim = {0: [], 1:[]}


    for step, data in enumerate(train_bar):
        pos_1, pos_2 = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
        B, C, W, H = pos_1.shape

        with amp.autocast():
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)

            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            cosine_sim_neg = torch.mm(out, out.t().contiguous())
            sim_matrix = torch.exp(cosine_sim_neg / temperature)

            mask_negatives = (torch.ones_like(sim_matrix) - torch.eye(opt.batch_size_multiGPU, device=sim_matrix.device).repeat(2,2)).bool() # for logging
            # [2*B, 2*B-2]
            sim_matrix_neg = sim_matrix.masked_select(mask_negatives).view(2 * opt.batch_size_multiGPU, -1)

            # log negative cosine dist
            cosine_mask = cosine_sim_neg.masked_select(torch.triu(mask_negatives))
            negatives = cosine_mask[torch.randint(2 * B, (B,))].detach().cpu().numpy().ravel()
            neg_cosine_sim[0].extend(negatives) # randomly take B samples

            # compute loss
            cosine_sim = torch.sum(out_1 * out_2, dim=-1)
            pos_sim = torch.exp(cosine_sim / temperature)
            # log positive cosine dist
            pos_cosine_sim[0].extend(cosine_sim.detach().cpu().numpy())

            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            # loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            loss = (- torch.log(pos_sim / (pos_sim + sim_matrix_neg.sum(dim=-1)))).mean()

        train_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(train_optimizer)
        scaler.update()

        total_num += opt.batch_size_multiGPU
        total_loss += loss.item() * opt.batch_size_multiGPU
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def test(net, test_data_loader, opt, epoch):
    denom_transform = Denormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    neg_cosine_sim = {0: [], 1:[]}
    pos_cosine_sim = {0: [], 1:[]}

    net.eval()
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)

        for step, data in enumerate(test_bar):
            pos_1 = data[0].cuda(non_blocking=True)
            pos_2 = data[1].cuda(non_blocking=True)

            B, C, W, H = pos_1.shape

            feat_1, out_1 = net(pos_1)
            feat_2, out_2 = net(pos_2)

            # log negative cosine dist
            out = torch.cat([out_1, out_2], dim=0)
            cosine_sim_neg = torch.mm(out, out.t().contiguous())
            mask_negatives = (torch.ones_like(cosine_sim_neg) - torch.eye(B, device=cosine_sim_neg.device).repeat(2,2)).bool() # for logging

            cosine_mask = cosine_sim_neg.masked_select(torch.triu(mask_negatives))
            negatives = cosine_mask[torch.randint(2 * B, (B,))].detach().cpu().numpy().ravel()
            neg_cosine_sim[0].extend(negatives) # randomly take B samples

            # log positive cosine dist
            cosine_sim = torch.sum(out_1 * out_2, dim=-1)
            pos_sim = torch.exp(cosine_sim / opt.temperature)
            pos_cosine_sim[0].extend(cosine_sim.detach().cpu().numpy())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')

    parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
    parser.add_argument('--start_epoch', default=0, type=int,
                    help='Epoch to start from when cont. training (affects optimizer)')

    parser.add_argument('--training_data_csv', required=True, type=str, help='Path to file to use to read training data')
    parser.add_argument('--test_data_csv', required=True, type=str, help='Path to file to use to read test data')
    # For validation set, need to specify either csv or train/val split ratio
    group_validationset = parser.add_mutually_exclusive_group(required=True)
    group_validationset.add_argument('--validation_data_csv', type=str, help='Path to file to use to read validation data')
    group_validationset.add_argument('--trainingset_split', type=float, help='If not none, training csv with be split in train/val. Value between 0-1')

    parser.add_argument('--dataset', choices=['cam', 'multidata', 'skin'], default='cam', type=str, help='Dataset')
    parser.add_argument('--data_input_dir', type=str, help='Base folder for images')
    parser.add_argument('--data_input_dir_test', type=str, required=False, help='Base folder for images')
    parser.add_argument('--save_dir', type=str, help='Path to save log')
    parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
    parser.add_argument("--balanced_validation_set", action="store_true", default=False, help="Equal size of classes in validation AND test set",)

    parser.add_argument("--optimizer", choices=['adam', 'lars'], default='adam', help="Optimizer to use",)

    parser.add_argument("--use_album", action="store_true", default=False, help="use Albumentations as augmentation lib",)
    parser.add_argument("--balanced_training_set", action="store_true", default=False, help="Equal size of classes in train - SUPERVISED!")
    parser.add_argument("--pretrained", action="store_true", default=False, help="If true, use Imagenet pretrained resnet backbone")

    # Common augmentations
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--scale",  nargs=2, type=float, default=[0.2, 1.0])

    # RGB augmentations
    parser.add_argument("--rgb_gaussian_blur_p", type=float, default=0, help="probability of using gaussian blur (only on rgb)" )
    parser.add_argument("--rgb_jitter_d", type=float, default=1, help="color jitter 0.8*d, val 0.2*d (only on rgb)" )
    parser.add_argument("--rgb_jitter_p", type=float, default=0.8, help="probability of using color jitter(only on rgb)" )
    parser.add_argument("--rgb_contrast", type=float, default=0.2, help="value of contrast (rgb only)")
    parser.add_argument("--rgb_contrast_p", type=float, default=0, help="prob of using contrast (rgb only)")
    parser.add_argument("--rgb_grid_distort_p", type=float, default=0, help="probability of using grid distort (only on rgb)" )
    parser.add_argument("--rgb_grid_shuffle_p", type=float, default=0, help="probability of using grid shuffle (only on rgb)" )

    # args parse
    opt = validate_arguments(parser.parse_args())

    feature_dim, temperature = opt.feature_dim, opt.temperature
    batch_size, epochs = opt.batch_size, opt.epochs - opt.start_epoch

    is_windows = True if os.name == 'nt' else False
    opt.num_workers = 0 if is_windows else 16

    opt.train_supervised = False
    opt.grayscale = False

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = opt.save_dir

    # Write the parameters used to run experiment to file
    with open(f'{opt.log_path}/metadata_train.txt', 'w') as metadata_file:
        metadata_file.write(json.dumps(vars(opt)))

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', opt.device)

    # model setup and optimizer config
    scaler = amp.GradScaler()

    model = Model(feature_dim, pretrained=opt.pretrained)
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    elif opt.optimizer == 'lars':
        params_models = []
        reduced_params = []
        removed_params = []

        skip_lists = ['bn', 'bias']

        m_skip = []
        m_noskip = []
        params_models += list(model.parameters())

        for name, param in model.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
        print("reduced_params len: {}".format(len(reduced_params)))
        print("removed_params len: {}".format(len(removed_params)))
        optimizer = LARS(reduced_params+removed_params, lr=opt.lr,
                         weight_decay=1e-6, eta=0.001, use_nesterov=False, len_reduced=len(reduced_params))

    model.to(opt.device)

    if opt.load_checkpoint_dir:
        print('Loading model from: ', opt.load_checkpoint_dir)
        model, optimizer = reload_weights(
            opt, model, optimizer
        )

    model, num_GPU = distribute_over_GPUs(opt, model)

    train_loader, train_dataset, val_loader, val_dataset, _, _ = get_dataloader(opt)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    if opt.start_epoch > 0:
        print('Moving scheduler ahead')
        for _ in range(opt.start_epoch):
            scheduler.step()

    # training loop
    results = {'train_loss': []}
    save_name_pre = f'{feature_dim}_{temperature}_{batch_size}_{epochs}'
    best_acc = 0.0
    for epoch in range(1, epochs + 1):

        train_loss = train(model, train_loader, optimizer, scaler, opt, epoch)
        scheduler.step()
        results['train_loss'].append(train_loss)

        if val_loader is not None:
            test(model, val_loader, opt, epoch)
            # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'{opt.save_dir}/{save_name_pre}_statistics.csv', index_label='epoch')

        ## Save model
        if epoch % opt.save_after == 0:
            state = {
                #'args': args,
                'model': model.module.state_dict(),
                'optimiser': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, f'{opt.log_path}/{save_name_pre}_model_{epoch}.pth')
        # Delete old ones, save latest, keep every 10th
        if (epoch - 1) % 10 != 0:
            try:
                os.remove(f'{opt.log_path}/{save_name_pre}_model_{epoch - 1}.pth')
            except:
                print("not enough models there yet, nothing to delete")
