import torch
import numpy as np
import torchvision.models as models
from torchvision.models import VGG19_Weights

class Hooked_VGG :
    """Retrieves a pretrained VGG19 model without classifier, and with facilitated hook registering to collect features of specific layers"""
    def __init__(self, hooked_layers_ixs = []) :
        """Initialize the custom Hooked VGG with hooks on the indicated layers

        Args:
            hooked_layers_ixs (list|int, optional): Index or indices of the layers on which to collect features. Defaults to [].
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Retrieve only the features extraction part of the model, as the classifier won't be used
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        self.conv_features = {}
        self.hooks = {}

        self.add_hooks(hooked_layers_ixs)


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
            self.vgg.to(self.device)  
        except Exception as e :
            print(f"Unable to switch the model to this device ({e})")
            return self

        self.device = device  

        return self


    def _define_hook_fn(self, layer_ix) :
        """Utility function for defining a hook on a specific layer

        Args:
            layer_ix (int): id of the layer to put a hook on
        """
        def hook_function(module, input, output) :
            self.conv_features[layer_ix] = output
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

            self.hooks[layer_ix] = self.vgg[layer_ix].register_forward_hook(self._define_hook_fn(layer_ix))


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
            self.hooks.pop(layer_ix)


    def get_features(self, img) :
        """Initiate a forward pass on the Hooked VGG with an image to collect this image's specific features.

        Args:
            img (torch.Tensor): 4D Tensor of the image, with the shape (1, 3, height, width)

        Returns:
            dict: The features of the layers on which there is a hook, indexed by layer index
        """
        self.conv_features = {}
        self.vgg.to(self.device).forward(img)
        out_features = self.conv_features
        self.conv_features = {}

        return out_features