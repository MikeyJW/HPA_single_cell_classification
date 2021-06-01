'''EfficientNetB2 Pytorch model class to be trained using the Tez library'''

import tez
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

import config
from custom_metrics import skl_mAP
from dataset import CellDataset

class EfficientNetB2(tez.Model):
    '''Pytorch model class to facilitate transfer learning
    from an EfficientNetB2 model.
    '''
    def __init__(
        self,
        train_df=None,
        valid_df=None,
        batch_size=None,
        train_aug=None,
        valid_aug=None,
        pretrained=True
    ):
        super().__init__()
        # Initialise pretrained net and sub-in final layers for cell classification
        self.effnet = self._load_effnet(pretrained)
        self.effnet._fc = nn.Linear(1408, config.NUM_CLASSES)
        self.out = nn.Sigmoid()
        self.step_scheduler_after = "epoch"
        self.loss_fn = nn.BCELoss()
        # Create torch dataloaders
        if train_df and valid_df and batch_size:
            self.train_loader = self.gen_dataloader(
                train_df,
                batch_size,
                shuffle=True,
                aug=train_aug
            )
            self.valid_loader = self.gen_dataloader(
                valid_df,
                batch_size,
                shuffle=False,
                aug=valid_aug
            )

    def forward(self, image, target=None):
        'Forward prop effnet model w/ sigmoid final layer'
        x = self.effnet(image)
        output = self.out(x)

        if target is not None:
            loss = self.loss_fn(output, target.to(torch.float32))
            metrics = self.monitor_metrics(output, target)
            return output, loss, metrics
        return output, None, None

    def monitor_metrics(self, outputs, targets):
        'Tez compatible metric monitoring'
        if targets is None:
            return {}
        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        # Calculate batch metrics
        mAP = skl_mAP(outputs, targets)
        return {'mAP': mAP}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def gen_dataloader(self, df, bs, shuffle, aug=None):
        'Return pytorch dataloader generated from cell image dataframe'
        # Extract images and targets as numpy arrays from dataframe
        def extract_as_array(str_):
            list_ = str_.strip('][').split(', ')
            return np.array([int(i) for i in list_])
        images = df['cell_id'].values
        targets = df['Label'].apply(extract_as_array).values
        # Init custom dataset class and pass to pytorch loader
        dataset = CellDataset(images, targets, self.IMG_DIR, aug)
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle)

    def _load_effnet(self, pretrained):
        if pretrained is True:
            effnet = EfficientNet.from_pretrained("efficientnet-b2")
        else:
            effnet = EfficientNet.from_name("efficientnet-b2")
        return effnet
