import os
import random
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.transforms import transforms

from torch.utils.data import WeightedRandomSampler

from augmentations import get_transforms, torchvision_transforms, album_transforms, AlbumentationsTransform, get_rgb_transforms
from datasets import ImagePatchesDataset, LmdbDataset
import albumentations as A


def get_dataloader(opt):
    if opt.dataset == 'cam' or opt.dataset == 'skin':
        train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_camelyon_dataloader(
            opt
        )
    elif opt.dataset == 'multidata':
        train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_multidata_dataloader(opt)
        raise Exception("Invalid option")

    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )


def get_weighted_sampler(dataset, num_samples):
    df = dataset.dataframe
    # Get number of sampler per label. Weight = 1/num sampels
    class_weights = { row.label: 1/row[0] for _, row in df.groupby(['label']).size().reset_index().iterrows()}
    print(class_weights)
    # Set weights per sample in dataset
    weights = [class_weights[row.label] for _, row in df.iterrows()]
    return WeightedRandomSampler(weights=weights, num_samples=num_samples)

def clean_data(img_dir, dataframe):
    """ Clean the data """
    for idx, row in dataframe.iterrows():
        if not os.path.isfile(f'{img_dir}/{row.filename}') or (os.stat(f'{img_dir}/{row.filename}').st_size == 0):
            print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
            dataframe = dataframe.drop(idx)
    return dataframe


def get_dataframes(opt):
    if os.path.isfile(opt.training_data_csv):
        print("reading csv file: ", opt.training_data_csv)
        train_df = pd.read_csv(opt.training_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.training_data_csv}')

    if os.path.isfile(opt.test_data_csv):
        print("reading csv file: ", opt.test_data_csv)
        test_df = pd.read_csv(opt.test_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.test_data_csv}')

    if opt.trainingset_split:
        # Split train_df into train and val
        slide_ids = train_df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*opt.trainingset_split)
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        val_df = train_df[train_df.slide_id.isin(valid_req_ids)] # First, take the slides for validation
        train_df = train_df[train_df.slide_id.isin(train_req_ids)] # Update train_df

    else:
        if os.path.isfile(opt.validation_data_csv):
            print("reading csv file: ", opt.validation_data_csv)
            val_df = pd.read_csv(opt.validation_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')

    if opt.balanced_training_set:
        print('Use uniform training set')
        samples_to_take = train_df.groupby('label').size().min()
        train_df = pd.concat([train_df[train_df.label == label].sample(samples_to_take) for label in train_df.label.unique()])

    if opt.balanced_validation_set:
        print('Use uniform validation set')
        samples_to_take = val_df.groupby('label').size().min()
        val_df = pd.concat([val_df[val_df.label == label].sample(samples_to_take) for label in val_df.label.unique()])

        print('Use uniform test set')
        samples_to_take = test_df.groupby('label').size().min()
        test_df = pd.concat([test_df[test_df.label == label].sample(samples_to_take) for label in test_df.label.unique()])

    if not opt.train_supervised:
        val_df = val_df.sample(1000)
        test_df = test_df.sample(1000)

    if not opt.dataset == 'patchcam':
        train_df = clean_data(opt.data_input_dir, train_df)
        val_df = clean_data(opt.data_input_dir, val_df)
        test_df = clean_data(opt.data_input_dir, test_df)

    return train_df, val_df, test_df



def get_camelyon_dataloader(opt):
    base_folder = opt.data_input_dir
    print('opt.data_input_dir: ', opt.data_input_dir)
    print('opt.data_input_dir_test: ', opt.data_input_dir_test)

    train_df, val_df, test_df = get_dataframes(opt)

    print("training patches: ", train_df.groupby('label').size())
    print("Validation patches: ", val_df.groupby('label').size())
    print("Test patches: ", test_df.groupby('label').size())

    print("Saving training/val set to file")
    train_df.to_csv(f'{opt.log_path}/training_patches.csv', index=False)
    val_df.to_csv(f'{opt.log_path}/val_patches.csv', index=False)

    transform_train = get_transforms(opt, eval=False)
    transform_valid = get_transforms(opt, eval=True if opt.train_supervised else False) # we want augm in SSL training
    transform_test = get_transforms(opt, eval=True)

    if opt.dataset == 'cam':
        train_dataset = ImagePatchesDataset(opt, train_df, image_dir=base_folder, transform=transform_train)
        val_dataset = ImagePatchesDataset(opt, val_df, image_dir=base_folder, transform=transform_valid)
        test_dataset = ImagePatchesDataset(opt, test_df, image_dir=opt.data_input_dir_test, transform=transform_test)
    elif opt.dataset == 'skin':
        label_enum = {
                      'normal_dermis': 0,
                      'normal_epidermis': 1,
                      'normal_skinapp': 2,
                      'normal_subcut': 3,
                      'abnormal': 4,
                      }
        train_dataset = ImagePatchesDataset(opt, train_df, image_dir=base_folder, transform=transform_train, label_enum=label_enum)
        val_dataset = ImagePatchesDataset(opt, val_df, image_dir=base_folder, transform=transform_valid, label_enum=label_enum)
        test_dataset = ImagePatchesDataset(opt, test_df, image_dir=opt.data_input_dir_test, transform=transform_test, label_enum=label_enum)

    # Weighted sampler to handle class imbalance
    print('Weighted validation sampler')
    val_sampler = get_weighted_sampler(val_dataset, num_samples=len(val_dataset))

    if opt.train_supervised:
        print('Weighted training sampler')
        train_sampler = get_weighted_sampler(train_dataset, num_samples=len(train_dataset))

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        sampler=None,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    if opt.train_supervised:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=train_sampler,
            shuffle=True if train_sampler is None else False,
            num_workers=opt.num_workers,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size_multiGPU//2,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size_multiGPU//2,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )

def get_multidata_dataloader(opt):
    '''
    Loads pathologydataset from lmdb files, performing augmentaions.
    Only supporting pretraining, as no labels are available
    args:
        opt (dict): Program/commandline arguments.
    Returns:
        dataloaders (): pretraindataloaders.
    '''

    # Base train and test augmentaions
    aug = {
        "multidata": {
            "resize": None,
            "randcrop": None,
            "scale": opt.scale,
            "flip": True,
            "jitter_d": opt.rgb_jitter_d,
            "jitter_p": opt.rgb_jitter_p,
            "grayscale": opt.grayscale,
            "gaussian_blur": opt.rgb_gaussian_blur_p,
            "contrast": opt.rgb_contrast,
            "contrast_p": opt.rgb_contrast_p,
            "grid_distort": opt.rgb_grid_distort_p,
            "grid_shuffle": opt.rgb_grid_shuffle_p,
            "rotation": True,
            "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
            "std": [0.2023, 0.1994, 0.2010],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        },
    }

    #transform_train = transforms.Compose(torchvision_transforms(eval=False, aug=aug['multidata']))
    transform_train = get_transforms(opt, eval=False)

    train_dataset = LmdbDataset(lmdb_path=opt.data_input_dir,
                        transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.num_workers,
                                        pin_memory=True, drop_last=True,
                                        shuffle=True,
                                        batch_size=opt.batch_size_multiGPU)

    return (
        train_loader,
        train_dataset,
        None,
        None,
        None,
        None,
    )
