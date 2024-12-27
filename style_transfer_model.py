import torch
import lightning as L
# from torchvision.image import TotalVariation 
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import copy, deepcopy
import os

from image import Image
from hooked_vgg import Hooked_VGG
from custom_trainer import CustomTrainer


def gram_matrix(features : torch.Tensor) :
    """Compute the Gram matrix of a layer's features

    Args:
        features (torch.Tensor): Features to compute the Gram matrix on

    Returns:
        Tensor: The Gram matrix of the features
    """
    b, num_features, h, w = features.size()
    # Gram matrix can be computed as the features multiplied by themselves transposed, under the condition that the features are a matrix of size num_features x num_elements_per_feature
    matrix = features.reshape(b, num_features, w*h)
    matrix_t = matrix.transpose(1,2)
    gram = matrix.bmm(matrix_t)
    gram /= num_features * h * w
    return gram


class StyleTransferModel(L.LightningModule) :
    """Lightning Module to encapsulate the training process of creating an image with the content from one image and the style of another"""
    def __init__(self, content_img : Image, style_img : Image, *, start_with : str = 'noise', content_loss_weight : float = 1, style_loss_weight : float = 1e4, tv_loss_weight = 1e1, optimizer : str = 'LBFGS', lr : float = 1, optimizer_kwargs : dict = None, content_feat_layers : list = [22], style_feat_layers : list = [1,6,11,20,29], vgg_model : Hooked_VGG = None, **kwargs):
        """Initialize the model

        Args:
            content_img (Image): Image used for the content
            style_img (Image): Image from which to extract the general style
            content_feat_layers (list): List of indices of the convolutional layers from which to extract features for the content
            style_feat_layers (list): List of indices of the convolutional layers from which to extract features for the style
            vgg_model (Hooked_VGG, optional): Model to use for feature extraction. Defaults to None.
        """
        super().__init__(**kwargs)
        self.vgg_model = vgg_model if vgg_model is not None else Hooked_VGG()

        self.content_feat_layers = content_feat_layers
        self.style_feat_layers   = style_feat_layers
        
        self.content_img  = content_img
        self.style_img    = style_img

        self.start_with = start_with

        self.content_loss_weight = content_loss_weight
        self.style_loss_weight   = style_loss_weight
        self.tv_loss_weight      = tv_loss_weight
        
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        # train_img = Image(white_noise_shape = self.content_img.shape).to(self.used_device)
        self._train_img_history = []
        # self._train_tensor = torch.nn.Parameter(self.style_img.to_tensor().clone())

        self._trained_img = None

    
    def _content_loss(self, content_train_features) :
        content_loss = 0.0
        for key in self.content_features.keys() :
            content_loss += torch.nn.MSELoss(reduction='mean')(self.content_features[key], content_train_features[key])
        content_loss /= len(self.content_features)
        # content_loss = torch.nn.MSELoss(reduction='mean')(list(self.content_features.values())[0], list(content_train_features.values())[0])

        self.log('Content loss', content_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        
        return content_loss

    def _style_loss(self, style_train_features) :
        if len(self.style_features) != len(style_train_features) :
            raise ValueError("The train features do not correspond to the image features")
        
        img_gram_matrices = {}
        train_gram_matrices = {}
        num_features = {}
        num_elems = {}
        for key in self.style_features.keys() :
            if style_train_features.get(key) is None or style_train_features[key].shape != self.style_features[key].shape :
                print(style_train_features[key].shape)
                print(self.style_features[key].shape)
                raise ValueError("The train features do not correspond to the image features")
            
            num_features[key] = self.style_features[key].shape[1]
            num_elems[key] = self.style_features[key].shape[2] * self.style_features[key].shape[3]
            img_gram_matrices[key] = gram_matrix(self.style_features[key])
            train_gram_matrices[key] = gram_matrix(style_train_features[key])

        style_loss = 0.0
        for key in img_gram_matrices.keys() :
            # norm_coef = 1/(4 * num_features[key]**2 * num_elems[key]**2)
            norm_coef = 1
            style_loss += norm_coef * torch.nn.MSELoss(reduction='sum')(img_gram_matrices[key], train_gram_matrices[key])
            # normalized_losses[i] = torch.nn.MSELoss(reduction='sum')(img_gram_matrices[key][0], train_gram_matrices[key][0])
        style_loss /= len(img_gram_matrices)

        self.log('Style loss', style_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return style_loss
    
    def _total_variation_loss(self, img):
        tv_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                  torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

        self.log('TV Loss', tv_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        
        return tv_loss


    def _total_loss(self, content_train_features, style_train_features) :
        content_loss = self._content_loss(content_train_features)
        style_loss = self._style_loss(style_train_features)
        tv_loss = self._total_variation_loss(self._train_tensor)

        return self.content_loss_weight * content_loss + self.style_loss_weight * style_loss + self.tv_loss_weight * tv_loss
    

    def forward(self, x : torch.Tensor) :
        return self.vgg_model.get_features(x)
    
    def training_step(self, batch, batch_idx) :             
        content_train_feat = self._compute_features(self._train_tensor, 'content')
        style_train_feat   = self._compute_features(self._train_tensor, 'style')

        # loss = self._total_loss(content_train_feat, style_train_feat)
        loss = self._total_loss(content_train_feat, style_train_feat)
        self.log('General loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def configure_optimizers(self) :
        return self._optimizer([self._train_tensor], lr=self.lr, **self.optimizer_kwargs)
    
    def on_train_end(self):
        # When the training is complete, retrieve the Image corresponding to the trained tensor
        self._trained_img = Image(self.train_tensor)
        return super().on_train_end()
    
    def on_train_epoch_end(self):
        # After each 5 epoch, save the state of the trained image in history
        if self.current_epoch % 5 == 0 :
            self._train_img_history.append(Image(self._train_tensor.clone().detach()))
        return super().on_train_epoch_end()
    
    
    def _compute_features(self, x : torch.Tensor, feat_type : str) :
        """Compute the features of an image, using specific layers for content extraction and for style extraction.

        Args:
            x (torch.Tensor): Image tensor
            feat_type (str): Either 'content' or 'style'

        Raises:
            ValueError: If the feat_type is invalid

        Returns:
            dict: The specific features for content or style
        """
        if feat_type not in ['content', 'style'] :
            raise ValueError("feat_type should be either 'content' or 'style'")
        
        self.vgg_model.remove_all_hooks()
        hooks = self.content_feat_layers if feat_type == 'content' else self.style_feat_layers
        self.vgg_model.add_hooks(hooks)
        
        return self.forward(x)
    
    def train(self, *, trainer : CustomTrainer = None, from_checkpoint = False, **kwargs) :
        """Train the model using the specified content and style image. 
        If no Trainer is provided, a custom trainer parameterized to display correctly the progres bar will be used.
        Use keyword arguments to change specific parameters of the Trainer, in particular max_epochs

        Args:
            trainer (CustomTrainer, optional): Trainer to use for the training. Defaults to None.
            from_checkpoint (bool, optional): Whether to restart from the current trained state, or to erase all progress before training. Defaults to False.

        Raises:
            ValueError: If no limit for the Trainer is provided
        """
        if kwargs.get('max_epochs') is None :
            raise ValueError("Please provide a maximum number of epochs to avoid an endless training process")

        if not from_checkpoint :
            # Reset the train Tensor and the training history
            self._train_img_history = []
            if self.start_with == 'noise' :
                train_img = Image(white_noise_shape = self.content_img.shape).to(self.used_device)
            elif self.start_with == 'content' :
                train_img = self.content_img
            else :
                train_img = self.style_img
            self._train_tensor = torch.nn.Parameter(train_img.to_tensor().clone())
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
            self._train_tensor = torch.nn.Parameter(self._train_tensor.data.to(self.used_device))

        self.trainer = CustomTrainer(logger=logger, total_epochs=total_epochs, monitor_losses=['General loss'], device=self.used_device, **kwargs)
        self.trainer.fit(self, ckpt_path=ckpt_path)

        self.trained_img.show()
        
    def to(self, device) :
        """Switch the model to the indicated computation device ("cpu" or "cuda", for example). 
        If switching is not possible, the model will remain as before.

        Args:
            device (str): Name of the device to switch the model on

        Returns:
            StyleTransferModel: The updated model
        """

        device = str(device)
        old_device = self.used_device

        if str(device) == old_device :
            return self

        # If any of the components fail to be transferred to the device, all transfers are cancelled (There is no need for an error message here as it is already printed in Image and Hooked_VGG methods)
        self.content_img.to(device)
        if str(self.content_img.device) == old_device :
            return self
        
        self.style_img.to(device)
        if str(self.style_img.device) == old_device :
            self.content_img.to(old_device)
            return self
        
        self.vgg_model.to(device)
        if str(self.vgg_model.device) == old_device :
            self.content_img.to(old_device)
            self.style_img.to(old_device)
            return self
        
        self._train_tensor = torch.nn.Parameter(self._train_tensor.to(device=device))
        for key in self.content_features.keys() :
            self._content_features[key] = self._content_features[key].to(device)
        for key in self.style_features.keys() :
            self._style_features[key] = self._style_features[key].to(device)

        self._used_device = device
        self._device = self.used_device

    
    @property
    def content_img(self) :
        return self._content_img
    
    @content_img.setter
    def content_img(self, new_value) :
        if not isinstance(new_value, Image) :
            raise ValueError("content_img should be an Image")
        self._content_img = new_value
        # self.content_img.resize((224,224))
        if self.content_img.device != self.used_device :
            self.content_img.to(self.used_device)
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
        self.style_img.resize(self.content_img.shape) # Style and content image must have the same size for the trained image to have same feature size as both content and style features
        # self.style_img.resize((224,224))
        if self.style_img.device != self.used_device :
            self.style_img.to(self.used_device)
        self._style_features = self._compute_features(self.style_img.to_tensor(), 'style')
    
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
        
        # If no device is defined (during initialization), the device of the model is used as reference
        if self.__dict__.get('used_device') is None :
            self._used_device = str(self.vgg_model.device)
            self._device = self.used_device
        elif self.vgg_model.device != self.used_device :
            self.vgg_model.to(self.used_device)

        if self.__dict__.get('content_img') is not None :
            self._content_features = self._compute_features(self.content_img.to_tensor(), 'content')
        if self.__dict__.get('style_img') is not None :
            self._style_features = self._compute_features(self.style_img.to_tensor(), 'style')
    
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
        
        if self.__dict__.get('content_img') is not None :
            self._content_features = self._compute_features(self.content_img.to_tensor(), 'content')
    
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

        if self.__dict__.get('style_img') is not None :
            self._style_features = self._compute_features(self.style_img.to_tensor(), 'style')
    
    @style_feat_layers.deleter
    def style_feat_layers(self) :
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
    

    @property
    def used_device(self) :
        return self._used_device
    
    @used_device.setter
    def used_device(self, new_value) :
        raise AttributeError("Direct modification of the device is not authorized. Please use the to(device) method.")
    
    @used_device.deleter
    def used_device(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def start_with(self) :
        return self._start_with
    
    @start_with.setter
    def start_with(self, new_value) :
        if new_value not in ['noise', 'content', 'style'] :
            raise ValueError("Invalid value for start_with. Please give a value in ['noise', 'content', 'style']")
        self._start_with = new_value
    
    @start_with.deleter
    def start_with(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def content_loss_weight(self) :
        return self._content_loss_weight
    
    @content_loss_weight.setter
    def content_loss_weight(self, new_value) :
        if not isinstance(new_value, float) and not isinstance(new_value, int) :
            raise TypeError("content_loss_weight should be a Float")
        self._content_loss_weight = new_value
    
    @content_loss_weight.deleter
    def content_loss_weight(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    
    
    @property
    def style_loss_weight(self) :
        return self._style_loss_weight
    
    @style_loss_weight.setter
    def style_loss_weight(self, new_value) :
        if not isinstance(new_value, float) and not isinstance(new_value, int) :
            raise TypeError("style_loss_weight should be a Float")
        self._style_loss_weight = new_value
    
    @style_loss_weight.deleter
    def style_loss_weight(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def tv_loss_weight(self) :
        return self._tv_loss_weight
    
    @tv_loss_weight.setter
    def tv_loss_weight(self, new_value) :
        if not isinstance(new_value, float) and not isinstance(new_value, int) :
            raise TypeError("tv_loss_weight should be a Float")
        self._tv_loss_weight = new_value
    
    @tv_loss_weight.deleter
    def tv_loss_weight(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def optimizer(self) :
        return self._optimizer_str
    
    @optimizer.setter
    def optimizer(self, new_value) :
        match new_value : 
            case 'LBFGS' :
               self._optimizer = torch.optim.LBFGS
            case 'Adam' :
                self._optimizer = torch.optim.Adam
            case _ :
                try : 
                    self._optimizer = eval(f'torch.optim.{new_value}')
                except :
                    raise ValueError("The specified optimizer was not found in Torch. It is recommended to set it to 'LBFGS' or 'Adam' for style transfer") from None       
        self._optimizer_str = new_value
    
    @optimizer.deleter
    def optimizer(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def lr(self) :
        return self._lr
    
    @lr.setter
    def lr(self, new_value) :
        if not isinstance(new_value, float) and not isinstance(new_value, int) :
            raise TypeError("Learning rate should be a Float value")
        self._lr = new_value
    
    @lr.deleter
    def lr(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def optimizer_kwargs(self) :
        return self._optimizer_kwargs
    
    @optimizer_kwargs.setter
    def optimizer_kwargs(self, new_value) :
        if not isinstance(new_value, dict) :
            raise TypeError("Optimizer's kwargs should be a dict")
        # Check validity of the kwargs
        try :
            blank_tensor = torch.zeros_like(self.content_img.to_tensor(), requires_grad=True)
            o = self._optimizer([blank_tensor], **new_value)
        except :
            raise ValueError("Invalid keyword arguments provided to the optimizer") from None
        
        self._optimizer_kwargs = new_value
    
    @optimizer_kwargs.deleter
    def optimizer_kwargs(self) :
        raise AttributeError("Deletion of this attribute is not authorized")