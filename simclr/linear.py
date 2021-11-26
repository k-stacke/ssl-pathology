import argparse
import os
import json
import random
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from simclr.utils import distribute_over_GPUs, validate_arguments
from simclr.model import Model, Identity
from simclr.get_dataloader import get_dataloader

torch.backends.cudnn.benchmark=True
class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        # Load pre-trained model
        base_model = Model(pretrained=opt.pretrained)
        if (not opt.random) and (not opt.pretrained):
            print('Loading model from ', opt.model_path)
            base_model.load_state_dict(torch.load(opt.model_path, map_location=opt.device.type)['model'], strict=True)

        self.f = base_model.f

        # classifier
        self.fc = nn.Linear(opt.output_dims, opt.num_classes, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.eval() # train only the last layers.
    #net.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)

    all_preds, all_labels, all_slides, all_outputs0, all_outputs1, all_patches  = [], [], [], [], [], []

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target, patch_id, slide_id in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            _, preds = torch.max(out.data, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().data.numpy())
            all_patches.extend(patch_id)
            all_slides.extend(slide_id)

            probs = torch.nn.functional.softmax(out.data, dim=1).cpu().numpy()
            all_outputs0.extend(probs[:, 0])
            all_outputs1.extend(probs[:, 1])

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description(f'{"Train" if is_train else "Test"} Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f} ACC: {total_correct / total_num * 100:.2f}% ')


    df =  pd.DataFrame({
                'label': all_labels,
                'prediction': all_preds,
                'slide_id': all_slides,
                'patch_id': all_patches,
                'probabilities_0': all_outputs0,
                'probabilities_1': all_outputs1,
            })

    return total_loss / total_num, total_correct / total_num * 100, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    group_modelset = parser.add_mutually_exclusive_group(required=True)
    group_modelset.add_argument('--model_path', type=str, default='results/128_0.5_512_500_model.pth',
                        help='The pretrained model path')
    group_modelset.add_argument("--random", action="store_true", default=False, help="No pre-training, use random weights")
    group_modelset.add_argument("--pretrained", action="store_true", default=False, help="Use Imagenet pretrained Resnet")

    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    parser.add_argument('--training_data_csv', required=True, type=str, help='Path to file to use to read training data')
    parser.add_argument('--test_data_csv', required=True, type=str, help='Path to file to use to read test data')
    # For validation set, need to specify either csv or train/val split ratio
    group_validationset = parser.add_mutually_exclusive_group(required=True)
    group_validationset.add_argument('--validation_data_csv', type=str, help='Path to file to use to read validation data')
    group_validationset.add_argument('--trainingset_split', type=float, help='If not none, training csv with be split in train/val. Value between 0-1')
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

    parser.add_argument('--dataset', choices=['cam', 'patchcam', 'cam_rgb_hed', 'ovary', 'skin'], default='cam', type=str, help='Dataset')
    parser.add_argument('--data_input_dir', type=str, help='Base folder for images')
    parser.add_argument('--data_input_dir_test', type=str, required=False, help='Base folder for images')
    parser.add_argument('--save_dir', type=str, help='Path to save log')
    parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
    parser.add_argument("--balanced_validation_set", action="store_true", default=False, help="Equal size of classes in validation AND test set",)

    parser.add_argument("--finetune", action="store_true", default=False, help="If true, pre-trained model weights will not be frozen.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (l2 reg)')
    parser.add_argument("--model_to_save", choices=['best', 'latest'], default='latest', type=str, help='Save latest or best (based on val acc)')
    parser.add_argument('--seed', type=int, default=44, help='seed')


    parser.add_argument("--use_album", action="store_true", default=False, help="use Albumentations as augmentation lib",)
    parser.add_argument("--balanced_training_set", action="store_true", default=False, help="Equal size of classes in train - SUPERVISED!")


    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="Choice of optimizer")

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


    opt = validate_arguments(parser.parse_args())


    opt.output_dims = 2048


    is_windows = True if os.name == 'nt' else False
    opt.num_workers = 0 if is_windows else 16

    opt.train_supervised = True
    opt.grayscale = False

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = opt.save_dir

    # Write the parameters used to run experiment to file
    with open(f'{opt.log_path}/metadata_train.txt', 'w') as metadata_file:
        metadata_file.write(json.dumps(vars(opt)))

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', opt.device)

    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_path, batch_size, epochs = opt.model_path, opt.batch_size, opt.epochs

    model = Net(opt)
    model, num_GPU = distribute_over_GPUs(opt, model)

    train_loader, train_data, val_loader, val_data, test_loader, test_data = get_dataloader(opt)

    if not opt.finetune:
        for param in model.module.f.parameters():
            param.requires_grad = False

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                              momentum=0.9, nesterov=True)

    scheduler = CosineAnnealingLR(optimizer, opt.epochs)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        val_loss, val_acc, _ = train_val(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'{opt.log_path}/linear_statistics.csv', index_label='epoch')

        if opt.model_to_save == 'best' and val_acc > best_acc:
            # Save only the if the accuracy exceeds previous accuracy
            best_acc = val_acc
            torch.save(model.state_dict(), f'{opt.log_path}/linear_model.pth')
        elif opt.model_to_save == 'latest':
            # Save latest model
            best_acc = val_acc
            torch.save(model.state_dict(), f'{opt.log_path}/linear_model.pth')

    # trainig finished, run test
    print('Training finished, testing started...')
    # Load saved model
    model.load_state_dict(torch.load(f'{opt.log_path}/linear_model.pth'))
    model.eval()
    test_loss, test_acc, df = train_val(model, test_loader, None)

    df.to_csv(
       f"{opt.log_path}/inference_result_model.csv")





