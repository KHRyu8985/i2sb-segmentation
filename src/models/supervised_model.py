import autorootcwd
from .base_model import BaseModel
import torch

class SupervisedModel(BaseModel):
    """ Inheriting from BaseModel """

    def __init__(self, arch='FRNet', criterion='MonaiDiceCELoss', mode='train'):
        super(SupervisedModel, self).__init__(arch=arch, criterion=criterion, mode=mode)
        self.optimizer = torch.optim.Adam(self.arch.parameters(), lr=2e-3)

    def feed_data(self, batch):
        img, label = batch['image'], batch['label']
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def train_step(self, batch):
        self.optimizer.zero_grad()
        img, label = self.feed_data(batch)
        output = self.arch(img)
        output = torch.sigmoid(output) # Sigmoid activation for 0 ~ 1
        
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def valid_step(self, batch):
        img, label = self.feed_data(batch)
        output = self.arch(img)
        output = torch.sigmoid(output)
        return output, label
    
    