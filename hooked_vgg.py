import torch
import numpy as np
import torchvision.models as models
from torchvision.models import VGG19_Weights
from copy import copy

from image import Image

class Hooked_VGG :
    """Retrieves a pretrained VGG19 model without classifier, and with facilitated hook registering to collect features of specific layers"""
    def __init__(self, hooked_layers_ixs = []) :
        """Initialize the custom Hooked VGG with hooks on the indicated layers

        Args:
            hooked_layers_ixs (list|int, optional): Index or indices of the layers on which to collect features. Defaults to [].
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Retrieve only the features extraction part of the model, as the classifier won't be used
        self._vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        # self._device = self.vgg[0].device
        self.trainable = False
        self._conv_features = {}
        self._hooks = {}

        self.add_hooks(hooked_layers_ixs)


    @property
    def vgg(self) :
        return self._vgg
    
    @vgg.setter
    def vgg(self, new_value) :
        raise AttributeError("Direct modification of the neural network is not authorized")
    
    @vgg.deleter
    def vgg(self) :
        raise AttributeError("Deletion of the neural network is not authorized")
    

    @property
    def conv_features(self) :
        return self._conv_features
    
    @conv_features.setter
    def conv_features(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @conv_features.deleter
    def conv_features(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    

    @property
    def hooks(self) :
        return self._hooks.copy() # Return a copy to prevent direct modification of the list
    
    @hooks.setter
    def hooks(self, new_value) :
        raise AttributeError("Direct modification of this attribute is not authorized")
    
    @hooks.deleter
    def hooks(self) :
        raise AttributeError("Deletion of this attribute is not authorized")
    
    
    @property
    def device(self) :
        return self._device

    @device.setter
    def device(self, new_value) :
        raise AttributeError("Direct modification of the device is not authorized, please use the method to(device).")
    
    @device.deleter
    def device(self) :
        raise AttributeError("Deletion of the device attribute is not authorized.")
    

    @property
    def trainable(self) :
        return self._trainable
    
    @trainable.setter
    def trainable(self, new_value) :
        if new_value not in (True, False, 0, 1) :
            raise AttributeError("Please provide a boolean value")
        
        self._trainable = bool(new_value)
        for param in self.vgg.parameters() :
            param.requires_grad = self.trainable
    
    @trainable.deleter
    def trainable(self) :
        raise AttributeError("Deletion of the 'trainable' attribute is not authorized.")


    def to(self, device) :
        """Switch the model to the indicated computation device ("cpu" or "cuda", for example). 
        If switching is not possible, the model will remain as before.

        Args:
            device (str): Name of the device to switch the model on

        Returns:
            Hooked_VGG: The updated model
        """
        if self.device == device :
            return self
        
        try :
            self.vgg.to(device)  
        except Exception as e :
            print(f"Unable to switch the model to this device ({e})")
            return self

        self._device = next(self.vgg.parameters()).device  

        return self


    def _define_hook_fn(self, layer_ix) :
        """Utility function for defining a hook on a specific layer

        Args:
            layer_ix (int): id of the layer to put a hook on
        """
        def hook_function(module, input, output) :
            self._conv_features[layer_ix] = output
        return hook_function
    

    def add_hooks(self, layer_ixs) :
        """Add hooks on the indicated layers. The network will collect features from these layers.

        Args:
            layer_ixs (list|int): Index or indices of the layer(s) on which to put a hook
        """
        if not isinstance(layer_ixs, list) and not isinstance(layer_ixs, np.ndarray) :
            layer_ixs = [layer_ixs]

        for layer_ix in layer_ixs:
            if self.hooks.get(layer_ix) is not None :
                continue

            self._hooks[layer_ix] = self.vgg[layer_ix].register_forward_hook(self._define_hook_fn(layer_ix))


    def remove_hooks(self, layer_ixs) :
        """Remove the hooks on the indicated layers. The network will stop collecting features from these layers.

        Args:
            layer_ixs (list|int): Index or indices of the layer(s) from which to remove the hooks
        """
        if not isinstance(layer_ixs, list) and not isinstance(layer_ixs, np.ndarray) :
            layer_ixs = [layer_ixs]
        
        for layer_ix in layer_ixs :
            if self.hooks.get(layer_ix) is None :
                continue

            self.hooks[layer_ix].remove()
            self._hooks.pop(layer_ix)

    
    def remove_all_hooks(self) :
        """Clear the model from all its active hooks."""
        for key, hook in self.hooks.items() :
            hook.remove()
            self._hooks.pop(key)


    def get_features(self, img_tensor : torch.Tensor) :
        """Initiate a forward pass on the Hooked VGG with an image to collect this image's specific features.

        Args:
            img_tensor (torch.Tensor): 4D Tensor of the image, with the shape (1, 3, height, width)

        Returns:
            dict: The features of the layers on which there is a hook, indexed by layer index
        """
        self._conv_features = {}
        self.vgg.to(self.device).forward(img_tensor)
        out_features = self.conv_features
        self._conv_features = {}

        return out_features