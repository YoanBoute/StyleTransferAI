import torch
import lightning as L
from torch.utils.data import DataLoader

class EmptyDataModule(L.LightningDataModule) :
    """Custom data module with not content to be used with Lightning Trainer on Style Transfer"""
    def __init__(self):
        super().__init__()
    
    def train_dataloader(self):
        return DataLoader([0], batch_size=1)