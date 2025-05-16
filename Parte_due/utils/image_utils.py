"""
Utility functions for image handling.
"""

import cv2
import numpy as np

class ImageUtils:
    """
    Class providing utility methods for image manipulation.
    """
    
    def __init__(self):
        """Initialize ImageUtils."""
        pass
    
    def load_grayscale_image(self, file_path):
        """
        Load an image and convert it to grayscale.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Grayscale image as a 2D array
        """
        # Load image
        image = cv2.imread(file_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image
            
        return grayscale
    
    def save_grayscale_image(self, image, file_path):
        """
        Save a grayscale image to file.
        
        Args:
            image (numpy.ndarray): Grayscale image as a 2D array
            file_path (str): Path where the image will be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        return cv2.imwrite(file_path, image)