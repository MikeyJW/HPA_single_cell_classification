'''Transfer learn from efficientnetb2 model'''

import os
from argparse import ArgumentParser

import pandas as pd
import albumentations as A
from tez.callbacks import EarlyStopping

from model import EfficientNetB2
import config


def fetch_augmentations():
    'Return training and validation image augmentation pipelines'
    train_aug = A.Compose([
        A.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1], p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=config.MEAN_CHANNEL_VALUES,
            std=config.CHANNEL_STD_DEV,
            max_pixel_value=1.0,
            p=1.0
        )
    ])
    valid_aug = A.Compose([
        A.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1], p=1.0),
        A.Normalize(
            mean=config.MEAN_CHANNEL_VALUES,
            std=config.CHANNEL_STD_DEV,
            max_pixel_value=1.0,
            p=1.0
        )
    ])
    return train_aug, valid_aug


def main(train_csv, batch_size, num_epochs, fold, save_dir):
    'Train model on cell image data for specified number of epochs'
    # Select fold from training data csv
    dfx = pd.read_csv(train_csv, index_col=0)
    df_train = dfx[dfx['fold'] != fold].reset_index(drop=True)
    df_valid = dfx[dfx['fold'] == fold].reset_index(drop=True)
    # Init augmentations
    train_aug, valid_aug = fetch_augmentations()
    # Train model
    model = EfficientNetB2(
         df_train,
         df_valid,
         batch_size=batch_size,
         train_aug=train_aug,
         valid_aug=valid_aug,
         pretrained=True
    )
    callback = EarlyStopping(
        monitor='valid_loss',
        model_path=os.path.join(save_dir, 'effnetb2_model_checkpoint.bin'),
        patience=3,
        mode='min',
    )
    model.fit(device='cuda', callbacks=[callback], epochs=num_epochs)
    # Save model (with optimizer and scheduler for future usage)
    model.save(os.path.join(save_dir, 'effnetb2_trained_model.bin'))
    print('Success!!!')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    main(
        args.train_csv,
        args.batch_size,
        args.num_epochs,
        args.fold,
        args.save_dir
    )
