"""
Image processor for handling DCT2-based image compression.
"""

import numpy as np
from transformer.DCT2Transformer import DCT2Transformer
from utils.image_utils import ImageUtils

class ImageProcessor:
    """
    Class for processing images using DCT2 transformation for compression.
    """
    
    def __init__(self):
        """Initialize the ImageProcessor."""
        self.transformer = DCT2Transformer()
        self.image_utils = ImageUtils()
        
    def load_image(self, file_path):
        """
        Load an image from the file system.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded grayscale image as a 2D array
        """
        return self.image_utils.load_grayscale_image(file_path)
    
    def compress_image(self, image, block_size, threshold):
        """
        Compress an image using DCT2 transformation.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            block_size (int): Size of the blocks for DCT2 (F)
            threshold (int): Threshold for frequency cutoff (d)
            
        Returns:
            tuple: (compressed_image, compression_ratio)
                compressed_image (numpy.ndarray): Compressed grayscale image
                compression_ratio (float): Ratio of frequencies kept after compression
        """
        # Get image dimensions
        height, width = image.shape
        
        # Calculate dimensions for blocks
        # We'll truncate the image to be a multiple of block_size
        blocks_height = height // block_size
        blocks_width = width // block_size
        
        # Create output image (might be smaller due to truncation)
        output_height = blocks_height * block_size
        output_width = blocks_width * block_size
        output_image = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Calculate theoretical compression ratio based on threshold
        total_coefficients = block_size * block_size
        kept_coefficients = sum(i + j < threshold for i in range(block_size) for j in range(block_size))
        compression_ratio = total_coefficients / kept_coefficients if kept_coefficients > 0 else float('inf')
        
        # Process each block
        for block_y in range(blocks_height):
            for block_x in range(blocks_width):
                # Extract block
                y_start = block_y * block_size
                y_end = y_start + block_size
                x_start = block_x * block_size
                x_end = x_start + block_size
                
                block = image[y_start:y_end, x_start:x_end]
                
                # Apply DCT2
                dct_block = self.transformer.dct2_fast(block)
                
                # Apply frequency threshold (filter)
                for k in range(block_size):
                    for l in range(block_size):
                        if k + l >= threshold:
                            dct_block[k, l] = 0
                
                # Apply inverse DCT2
                reconstructed_block = self.transformer.idct2_fast(dct_block)
                
                # Clip values to valid range and round to nearest integer
                reconstructed_block = np.clip(np.round(reconstructed_block), 0, 255).astype(np.uint8)
                
                # Place block back in the output image
                output_image[y_start:y_end, x_start:x_end] = reconstructed_block
        
        return output_image, compression_ratio