import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from copy import copy, deepcopy
import os

from image import Image
from hooked_vgg import Hooked_VGG
from custom_trainer import CustomTrainer

class StyleTransferModel(L.LightningModule) :
    def __init__(self, content_img : Image, style_img : Image, content_feat_layers : list, style_feat_layers : list, vgg_model : Hooked_VGG = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg_model = vgg_model if vgg_model is not None else Hooked_VGG()

        self.content_feat_layers = content_feat_layers
        self.style_feat_layers   = style_feat_layers
        
        self.content_img  = content_img
        self.style_img    = style_img

        train_img = Image(white_noise_shape = self.content_img.shape)
        self._train_img_history = []
        self._train_tensor = torch.nn.Parameter(train_img.to_tensor())

        self._trained_img = None

    
    def ST_loss(self, content_train_features, style_train_features) :
        """Temporary function"""
        loss = torch.nn.MSELoss()
        return loss(list(self.content_features.values())[0], list(content_train_features.values())[0])

    def forward(self, x : torch.Tensor) :
        return self.vgg_model.get_features(x)
    
    def training_step(self, batch, batch_idx) :            
        if not self.trainer :
            print("No trainer")
            
        content_train_feat = self._compute_features(self._train_tensor, 'content')
        style_train_feat   = self._compute_features(self._train_tensor, 'style')

        loss = self.ST_loss(content_train_feat, style_train_feat)

        self.log('General loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def configure_optimizers(self) :
        return torch.optim.Adam([self._train_tensor], lr=0.1)
    
    def on_train_end(self):
        self._trained_img = Image(self.train_tensor)
        return super().on_train_end()
    
    def on_train_epoch_end(self):
        self._train_img_history.append(Image(self._train_tensor.clone().detach()))
        return super().on_train_epoch_end()
    
    def _compute_features(self, x : torch.Tensor, feat_type : str) :
        if feat_type not in ['content', 'style'] :
            raise ValueError("feat_type should be either 'content' or 'style'")
        
        self.vgg_model.remove_all_hooks()
        hooks = self.content_feat_layers if feat_type == 'content' else self.style_feat_layers
        self.vgg_model.add_hooks(hooks)
        
        return self.forward(x)
    
    def train(self, *, trainer : CustomTrainer = None, from_checkpoint = False, **kwargs) :
        if kwargs.get('max_epochs') is None :
            raise ValueError("Please provide a maximum number of epochs to avoid an endless training process")

        if not from_checkpoint :
            # Reset the train Tensor and the training history
            train_img = Image(white_noise_shape = self.content_img.shape)
            self._train_img_history = []
            self._train_tensor = torch.nn.Parameter(train_img.to_tensor())
            ckpt_path = None
            logger = None
            total_epochs = None
        else :
            if kwargs['max_epochs'] - self.current_epoch <= 0 :
                print("The provided max epoch number was already reached in a previous checkpoint. No training is necessary")
                return
            
            logger = TensorBoardLogger(save_dir=Path(self.trainer.default_root_dir), version=self.logger.version)
            ckpt_dir = Path(self.trainer.log_dir) / 'checkpoints'
            if not ckpt_dir.exists() :
                print("Warning : Unable to find the checkpoint path of the model, a new version will be created")
            else :
                ckpt_files_list = [f for f in ckpt_dir.glob('*.ckpt')]

                # If many ckpt files are found in the folder, keep the latest as basis
                if len(ckpt_files_list) > 1 :
                    ckpt_path = max(ckpt_files_list, key=os.path.getctime) 
                else :
                    ckpt_path = ckpt_files_list[0]
            total_epochs = kwargs['max_epochs'] - self.current_epoch # Compute the number of actual epochs of the training to have a coherent progress bar

        if self.__dict__.get('trainer') is None :
            self.trainer = CustomTrainer(logger=logger, total_epochs=total_epochs, **kwargs)

        self.trainer.fit(self, ckpt_path=ckpt_path)

        self.trained_img.show()
    
    
    @property
    def content_img(self) :
        return self._content_img
    
    @content_img.setter
    def content_img(self, new_value) :
        if not isinstance(new_value, Image) :
            raise ValueError("content_img should be an Image")
        self._content_img = new_value
        self._content_features = self._compute_features(self.content_img.to_tensor(), 'content')
    
    @content_img.deleter
    def content_img(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def style_img(self) :
        return self._style_img
    
    @style_img.setter
    def style_img(self, new_value) :
        if not isinstance(new_value, Image) :
            raise ValueError("style_img should be an Image")
        self._style_img = new_value
        self._style_features = self._compute_features(self.style_img.to_tensor(), 'content')
    
    @style_img.deleter
    def style_img(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def vgg_model(self) :
        return self._vgg_model
    
    @vgg_model.setter
    def vgg_model(self, new_value) :
        if not isinstance(new_value, Hooked_VGG) :
            raise ValueError("vgg_model should be a Hooked_VGG")
        self._vgg_model = new_value
    
    @vgg_model.deleter
    def vgg_model(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def content_feat_layers(self) :
        return self._content_feat_layers
    
    @content_feat_layers.setter
    def content_feat_layers(self, new_value) :
        if not isinstance(new_value, list) and not isinstance(new_value, np.ndarray) :
            raise ValueError("Please provide a list of layer indices")
        self._content_feat_layers = new_value
    
    @content_feat_layers.deleter
    def content_feat_layers(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def style_feat_layers(self) :
        return self._style_feat_layers
    
    @style_feat_layers.setter
    def style_feat_layers(self, new_value) :
        if not isinstance(new_value, list) and not isinstance(new_value, np.ndarray) :
            raise ValueError("Please provide a list of layer indices")
        self._style_feat_layers = new_value
    
    @content_feat_layers.deleter
    def content_feat_layers(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def content_features(self) :
        return self._content_features
    
    @content_features.setter
    def content_features(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @content_features.deleter
    def content_features(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def style_features(self) :
        return self._style_features
    
    @style_features.setter
    def style_features(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @style_features.deleter
    def style_features(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def train_tensor(self) :
        return copy(self._train_tensor)
    
    @train_tensor.setter
    def train_tensor(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @train_tensor.deleter
    def train_tensor(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def train_img_history(self) :
        return self._train_img_history.copy()
    
    @train_img_history.setter
    def train_img_history(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @train_img_history.deleter
    def train_img_history(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def trained_img(self) :
        return self._trained_img
    
    @trained_img.setter
    def trained_img(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @trained_img.deleter
    def trained_img(self) :
        raise AttributeError("Deletion of this attribute is not authorized")