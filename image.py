import torch
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

class Image :
    """Class to encapsulate generic behavior of an image."""
    def __init__(self, img : Path | str | torch.Tensor = None, *, white_noise_shape : tuple = None, normalize_means : list = [0.485, 0.456, 0.406], normalize_stds : list = [0.229, 0.224, 0.225], is_normalized = None) :
        """Initialize the Image by collecting it from the indicated path

        Args:
            img_path (Path): Path to the image

        Raises:
            RuntimeError: If the image could not be loaded
            ValueError: If the provided shape for a white noise image is invalid, or if no arguments are provided
        """
        if img is None and white_noise_shape is None :
            raise ValueError("Please provide a file path or the shape of the image to generate")

        if img is not None :
            if isinstance(img, str) :
                img = Path(img) # Transform the string to a Path to ensure compatibility with all OS

            if isinstance(img, Path) : 
                self._original_img = cv.imread(img)
                if self.original_img is None :
                    raise RuntimeError(f'Unable to read image {img}')
                # OpenCV uses the BGR convention, so it is converted to have RGB images
                self._original_img = cv.cvtColor(self.original_img, cv.COLOR_BGR2RGB)
                
                # Convert single channel image to 3 channels if necessary
                if len(self.original_img.shape) == 2 : 
                    self._rgb_img = np.zeros((*self.original_img.shape, 3))
                    self._rgb_img[:,:,0] = self.original_img      
                    self._rgb_img[:,:,1] = self.original_img      
                    self._rgb_img[:,:,2] = self.original_img      
                else : 
                    self._rgb_img = self.original_img
                
                self.trainable = False # An image coming from a file usually has no reason to be modified by a training
            elif isinstance(img, torch.Tensor) :
                if len(img.shape) != 4 or img.shape[1] != 3 :
                    raise ValueError("Please provide a 4D Tensor in the format (batch, channels, height, width)")
                
                img_height = img.shape[2]
                img_width = img.shape[3]

                self._original_img = img.cpu().clone().detach().reshape((img_height, img_width, 3)).numpy()
                self._rgb_img = self.original_img

                self.trainable = False # An image coming from a Tensor is usually already trained, and has no reason to be modified
            else :
                raise ValueError("The provided image should be a path to a file or a 4D Tensor. If you wish to create a random image, please use the keyword white_noise_shape.")
        
        else : # white_noise_shape provided
            try :
                l = len(white_noise_shape)
            except Exception as e :
                raise ValueError("The shape must be a 2D or 3D list or tuple") from None
            
            match l :
                case 2 :
                    white_noise_shape = (*white_noise_shape, 3) # Add the 3 channels to the image
                case 3 :
                    if white_noise_shape[-1] != 3 :
                        raise ValueError("Invalid shape, the provided shape must have 3 channels (in the last dimension)")
                case _ :
                    raise ValueError("Please provide a valid 2D or 3D shape for the image")
            
            self._original_img = (np.random.random(white_noise_shape))
            self._rgb_img = self.original_img
            self.trainable = True # A white noise image is considered trainable by default

        # Make sure the 3 channels image is always in Float format (values between 0 and 1)
        if np.issubdtype(self.original_img.dtype, np.integer) :
            self._rgb_img = self.original_img / 255.0

        if not (isinstance(normalize_means, list) or isinstance(normalize_means, np.ndarray) or isinstance(normalize_means, torch.Tensor)) or len(normalize_means) != 3 :
            raise ValueError("normalize_means should be a 3 elements list providing the means with which to normalize respectivley red, green and blue channels in the image.")

        if not (isinstance(normalize_stds, list) or isinstance(normalize_stds, np.ndarray) or isinstance(normalize_stds, torch.Tensor)) or len(normalize_stds) != 3 :
            raise ValueError("normalize_stds should be a 3 elements list providing the means with which to normalize respectivley red, green and blue channels in the image.")  

        self._normalize_means = np.array(normalize_means).astype(float).reshape(1, 1, 3)
        self._normalize_stds = np.array(normalize_stds).astype(float).reshape(1, 1, 3)
        
        if is_normalized is None:
            if isinstance(img, Path) :
                self._is_normalized = False # An image taken from a file is considered unnormalized by default
            else : # if img is a Tensor
                self._is_normalized = True # An image coming from a Tensor is considered normalized by default
        else :
            self._is_normalized = is_normalized

        self._compute_normalized_rgb()
        
        self._normalize = True
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._shape = self.rgb_img.shape
        self._height = self.shape[0]
        self._width = self.shape[1]
    

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
    def normalized_rgb(self) :
        return self._normalized_rgb
    
    @normalized_rgb.setter
    def normalized_rgb(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @normalized_rgb.deleter
    def normalized_rgb(self) :
        raise AttributeError("Deletion of image data is not authorized")


    @property
    def normalize_means(self) :
        return self._normalize_means
    
    @normalize_means.setter
    def normalize_means(self, new_value) :
        if not (isinstance(new_value, list) or isinstance(new_value, np.ndarray) or isinstance(new_value, torch.Tensor)) or len(new_value) != 3 :
            raise ValueError("normalize_means should be a 3 elements list providing the means with which to normalize respectivley red, green and blue channels in the image.")
        self._normalize_means = torch.Tensor(new_value).view(1,3,1,1)
        self._compute_normalized_rgb()
    
    @normalize_means.deleter
    def normalize_means(self) :
        raise AttributeError("Deletion of image data is not authorized")
    

    @property
    def normalize_stds(self) :
        return self._normalize_stds
    
    @normalize_stds.setter
    def normalize_stds(self, new_value) :
        if not (isinstance(new_value, list) or isinstance(new_value, np.ndarray) or isinstance(new_value, torch.Tensor)) or len(new_value) != 3 :
            raise ValueError("normalize_stds should be a 3 elements list providing the means with which to normalize respectivley red, green and blue channels in the image.")
        self._normalize_stds = torch.Tensor(new_value).view(1,3,1,1)
        self._compute_normalized_rgb()
    
    @normalize_stds.deleter
    def normalize_stds(self) :
        raise AttributeError("Deletion of image data is not authorized")


    @property
    def normalize(self) :
        return self._normalize
    
    @normalize.setter
    def normalize(self, new_value) :
        if new_value not in (True, False, 0, 1) :
            raise AttributeError("Please provide a boolean value")
        self._normalize = bool(new_value)

    @normalize.deleter
    def normalize(self) :
        raise AttributeError("Deletion of the 'normalize' error is not authorized")


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
    def height(self) :
        return self._height
    
    @height.setter
    def height(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @height.deleter
    def height(self) :
        raise AttributeError("Deletion of image data is not authorized")
    

    @property
    def width(self) :
        return self._width
    
    @width.setter
    def width(self, new_value) :
        raise AttributeError("Direct modification of the image data is not authorized")
    
    @width.deleter
    def width(self) :
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
    

    def _compute_normalized_rgb(self) :
        """Compute the normalized image using the provided normalization means and stds.
        If the rgb image is already normalized, consider it as normalized and compute the corresponding unnormalized image.
        """
        if not self._is_normalized :
            self._normalized_rgb = (self.rgb_img - self.normalize_means) / self.normalize_stds
        else :
            self._normalized_rgb = self._rgb_img
            self._rgb_img = self.normalized_rgb * self.normalize_stds + self.normalize_means
            self._original_img = self.rgb_img


    def resize(self, new_shape) :
        """Resize rgb and normalized image with the provided shape in the format (height, width, [3]). Original image remains unchanged

        Args:
            new_shape (list): Shape of the new image

        Raises:
            ValueError: If the provided shape is in an invalid format
        """
        try :
            l = len(new_shape)
        except Exception as e :
            raise ValueError("The shape must be a 2D or 3D list or tuple") from None
        
        match l :
            case 2 :
               pass
            case 3 :
                if new_shape[-1] != 3 :
                    raise ValueError("Invalid shape, the provided shape must have 3 channels (in the last dimension)")
                new_shape = new_shape[0:2] # The channel dimension should not be used for resizing the image
            case _ :
                raise ValueError("Please provide a valid 2D or 3D shape for the image")

        new_shape = [new_shape[1], new_shape[0]] # Height and width have to be reversed for the resize function of OpenCV    
        self._rgb_img = cv.resize(self.rgb_img, new_shape)
        self._normalized_rgb = cv.resize(self.normalized_rgb, new_shape)
        self._shape = self.rgb_img.shape
        self._height = self.shape[0]
        self._width = self.shape[1]

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
        if self.normalize :
            return torch.Tensor(self.normalized_rgb).reshape((1,3,self.height, self.width)).to(self.device).requires_grad_(self.trainable)
        else :
            return torch.Tensor(self.rgb_img).reshape((1,3,self.height, self.width)).to(self.device).requires_grad_(self.trainable)
        

    def show(self) :
        """Plot the image to visualize it"""
        plt.imshow(self.original_img)
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