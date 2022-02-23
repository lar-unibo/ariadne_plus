import torch
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        if len(tensor.shape) == 4:
            for b in tensor:
                for t, m, s in zip(b, self.mean, self.std):
                    t.sub_(m).div_(s)
        elif len(tensor.shape) == 3:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class CrossNet(pl.LightningModule):
    """ cross prediction network
    """

    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder.fc = torch.nn.Linear(512, 1)

        self.normalize = Normalize()

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        embedding = self.encoder(self.normalize(x))
        return embedding

    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward
        image = batch['image']
        label = batch['label']

        output = self(image)
        output = output.squeeze()
        label = label.float()

        return {'loss': self.criterion(output, label)}

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']
        
        output = self(image)
        output = output.squeeze()
        label = label.float()

        return {'val_loss': self.criterion(output, label)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer
