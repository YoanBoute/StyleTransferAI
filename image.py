import torch
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

class Image :
    """Class to encapsulate generic behavior of an image."""
    def __init__(self, img_path : Path) :
        """Initialize the Image by collecting it from the indicated path

        Args:
            img_path (Path): Path to the image

        Raises:
            RuntimeError: If the image could not be loaded
        """
        self.original_img = cv.imread(img_path)
        if self.original_img is None :
            raise RuntimeError(f'Unable to read image {img_path}')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert single channel image to 3 channels if necessary
        if len(self.original_img.shape) == 2 : 
            self.rgb_img = np.zeros((*self.original_img.shape, 3))
            self.rgb_img[:,:,0] = self.original_img      
            self.rgb_img[:,:,1] = self.original_img      
            self.rgb_img[:,:,2] = self.original_img      
        else : 
            self.rgb_img = self.original_img
        
        self.shape = self.rgb_img.shape


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
        self.device = device
        # Try to convert the image to a tensor on the specified device to make sure this device is available
        try :
            self.to_tensor()
        except Exception as e :
            print(f"Unable to switch to this device ({e})")
            self.device = old_device

        return self
        

    def to_tensor(self) :
        """Converts the numpy array of the image to a 4D-Tensor compatible with the requirements of convolutional neural networks.

        Returns:
            Tensor: The Tensor corresponding to the image, with shape (1, 3, height, width)
        """
        return torch.Tensor(self.rgb_img).reshape((1,3,self.rgb_img.shape[0], self.rgb_img.shape[1])).to(self.device)
    
    
    def show(self) :
        """Plot the image to visualize it"""
        plt.imshow(self.rgb_img)
        plt.show()