"""
Processore di immagini per la compressione basata sulla trasformata DCT2.
"""

import numpy as np
from transformer.DCT2Transformer import DCT2Transformer
from utils.image_utils import ImageUtils

class ImageProcessor:
    """
    Classe per la gestione della compressione di immagini tramite la trasformata DCT2.
    """
    
    def __init__(self):
        """Inizializza il processore caricando il trasformatore DCT2 e gli strumenti per l'immagine."""
        self.transformer = DCT2Transformer()
        self.image_utils = ImageUtils()
        
    def load_image(self, file_path):
        """
        Carica un'immagine in scala di grigi dal file system.

        Args:
            file_path (str): percorso del file immagine
        
        Returns:
            numpy.ndarray: immagine caricata in scala di grigi come array 2D
        """
        return self.image_utils.load_grayscale_image(file_path)
    
    def compress_image(self, image, block_size, threshold):
        """
        Comprimi un'immagine usando la trasformata DCT2 a blocchi.

        Args:
            image (numpy.ndarray): immagine di input in scala di grigi
            block_size (int): dimensione del blocco (F)
            threshold (int): soglia per il taglio delle frequenze (d)

        Returns:
            tuple: (immagine_compressa, rapporto_compressione)
                immagine_compressa (numpy.ndarray): immagine compressa in scala di grigi
                rapporto_compressione (float): rapporto tra coefficenti totali e quelli mantenuti
        """
        # Dimensioni dell'immagine originale
        height, width = image.shape
        
        # Calcolo del numero di blocchi interi che possiamo estrarre (troncamento)
        blocks_height = height // block_size
        blocks_width = width // block_size
        
        # Dimensioni dell'immagine output (potrebbe essere più piccola a causa del troncamento)
        output_height = blocks_height * block_size
        output_width = blocks_width * block_size
        
        # Inizializza immagine di output con zeri (nero)
        output_image = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Calcolo teorico del rapporto di compressione basato sulla soglia
        total_coefficients = block_size * block_size
        
        # Numero di coefficienti DCT mantenuti: somma degli indici i+j < threshold
        kept_coefficients = sum(
            (i + j) < threshold
            for i in range(block_size)
            for j in range(block_size)
        )
        
        # Evita divisione per zero
        compression_ratio = total_coefficients / kept_coefficients if kept_coefficients > 0 else float('inf')
        
        # Elaborazione blocco per blocco
        for block_y in range(blocks_height):
            for block_x in range(blocks_width):
                # Estrazione del blocco corrente dall'immagine originale
                y_start = block_y * block_size
                y_end = y_start + block_size
                x_start = block_x * block_size
                x_end = x_start + block_size
                
                block = image[y_start:y_end, x_start:x_end]
                
                # Applicazione della trasformata DCT2 veloce al blocco
                dct_block = self.transformer.dct2_fast(block)
                
                # Applicazione della soglia: azzera le frequenze con somma indici >= threshold
                for k in range(block_size):
                    for l in range(block_size):
                        if k + l >= threshold:
                            dct_block[k, l] = 0
                
                # Applicazione della trasformata inversa IDCT2 per ricostruire il blocco
                reconstructed_block = self.transformer.idct2_fast(dct_block)
                
                # Clipping dei valori nel range [0, 255] e conversione in interi uint8
                reconstructed_block = np.clip(np.round(reconstructed_block), 0, 255).astype(np.uint8)
                
                # Inserimento del blocco ricostruito nell'immagine di output
                output_image[y_start:y_end, x_start:x_end] = reconstructed_block
        
        return output_image, compression_ratio
