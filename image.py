import torch
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

class Image :
    """Class to encapsulate generic behavior of an image."""
    def __init__(self, img_path : Path = None, shape = None) :
        """Initialize the Image by collecting it from the indicated path

        Args:
            img_path (Path): Path to the image

        Raises:
            RuntimeError: If the image could not be loaded
            ValueError: If the provided shape for a white noise image is invalid, or if no arguments are provided
        """
        if img_path is None and shape is None :
            raise ValueError("Please provide a file path or the shape of the image to generate")

        if img_path is not None :
            self._original_img = cv.imread(img_path)
            if self.original_img is None :
                raise RuntimeError(f'Unable to read image {img_path}')
            
            # Convert single channel image to 3 channels if necessary
            if len(self.original_img.shape) == 2 : 
                self._rgb_img = np.zeros((*self.original_img.shape, 3))
                self._rgb_img[:,:,0] = self.original_img      
                self._rgb_img[:,:,1] = self.original_img      
                self._rgb_img[:,:,2] = self.original_img      
            else : 
                self._rgb_img = self.original_img
            
            self.trainable = False # An image coming from a file usually has no reason to be modified by a training
        else :
            try :
                l = len(shape)
            except Exception as e :
                raise ValueError("The shape must be a 2D or 3D list or tuple") from None
            
            match l :
                case 2 :
                    shape = (*shape, 3) # Add the 3 channels to the image
                case 3 :
                    if shape[-1] != 3 :
                        raise ValueError("Invalid shape, the provided shape must have 3 channels (in the last dimension)")
                case _ :
                    raise ValueError("Please provide a valid 2D or 3D shape for the image")
            
            self._original_img = (np.random.random(shape) * 255).astype(int)
            self._rgb_img = self.original_img
            self.trainable = True # A white noise image is considered trainable by default

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._shape = self.rgb_img.shape
    

    @property
    def original_img(self) :
        return self._original_img
    
    @original_img.setter
    def original_img(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @original_img.deleter
    def original_img(self) :
        raise AttributeError("Deletion of image data is not authorized")
    

    @property
    def rgb_img(self) :
        return self._rgb_img
    
    @rgb_img.setter
    def rgb_img(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @rgb_img.deleter
    def rgb_img(self) :
        raise AttributeError("Deletion of image data is not authorized")
    

    @property
    def shape(self) :
        return self._shape
    
    @shape.setter
    def shape(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @shape.deleter
    def shape(self) :
        raise AttributeError("Deletion of image data is not authorized")
    

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
    
    @trainable.deleter
    def trainable(self) :
        raise AttributeError("Deletion of the 'trainable' attribute is not authorized.")
    

    def to(self, device) :
        """Switch the image to the specified device (For example, "cpu" or "cuda"). Useful mainly when working on Tensors.
        If switching to this device is not possible, the Image will remain as before

        Args:
            device (str): Name of the device to switch the image on

        Returns:
            Image: the updated image
        """
        if self.device == device :
            return self
        
        old_device = self.device
        self._device = device
        # Try to convert the image to a tensor on the specified device to make sure this device is available
        try :
            self.to_tensor()
        except Exception as e :
            print(f"Unable to switch to this device ({e})")
            self._device = old_device

        return self
        

    def to_tensor(self) :
        """Converts the numpy array of the image to a 4D-Tensor compatible with the requirements of convolutional neural networks.

        Returns:
            Tensor: The Tensor corresponding to the image, with shape (1, 3, height, width)
        """
        return torch.Tensor(self.rgb_img).reshape((1,3,self.rgb_img.shape[0], self.rgb_img.shape[1])).to(self.device).requires_grad_(self.trainable)
    
    
    def show(self) :
        """Plot the image to visualize it"""
        plt.imshow(self.rgb_img)
        plt.show()
    

    def save_to(self, save_path : Path) :
        """Save the RGB image to the specified location (If the extension is not 'jpg', 'png', or 'jpeg', it will be set by default to 'png').

        Args:
            save_path (Path): Path to the save the image in
        
        Raises:
            RuntimeError: If the image could not be saved
        """
        if save_path.suffix not in ['.jpg', '.png', '.jpeg'] :
            save_path = save_path.with_suffix('.png')
        
        if save_path.exists() :
            answer = input(f"The file {save_path.absolute()} already exists. Continuing will overwrite this file. Proceed anyway ? [Y/n]")
            if answer not in ['y', 'Y', 'o', 'O', 'yes', 'Yes', 'oui', 'Oui'] :
                print("Saving cancelled")
                return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv.imwrite(save_path, self.rgb_img)
        if not saved :
            raise RuntimeError("Unable to save the image. This can be due to an invalid image or to an impossibility to write in the target folder")