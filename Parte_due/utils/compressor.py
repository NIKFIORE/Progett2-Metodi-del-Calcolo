import numpy as np
from transformer.DCT import DCT2Transformer

class DCTCompressor:
    """
    Classe per la compressione di immagini utilizzando la trasformata DCT2.
    Implementa un algoritmo simile a JPEG ma senza matrice di quantizzazione.
    """
    
    def __init__(self):
        """Inizializza il compressore DCT."""
        self.transformer = DCT2Transformer()
        
    def compress_image(self, image, block_size, threshold):
        """
        Comprime un'immagine utilizzando la DCT2 e un taglio delle frequenze.
        
        Argomenti:
            image (numpy.ndarray): immagine input in formato numpy (toni di grigio)
            block_size (int): dimensione dei blocchi quadrati (F)
            threshold (int): soglia per il taglio delle frequenze (d)
            
        Restituisce:
            numpy.ndarray: immagine compressa
        """
        # Ottiene dimensioni dell'immagine
        height, width = image.shape
        
        # Calcola quanti blocchi completi ci stanno
        blocks_y = height // block_size
        blocks_x = width // block_size
        
        # Crea una nuova immagine con le dimensioni esatte dei blocchi
        new_height = blocks_y * block_size
        new_width = blocks_x * block_size
        
        # Taglia l'immagine per avere dimensioni multiple di block_size
        trimmed_image = image[:new_height, :new_width].copy()
        
        # Crea l'immagine di output
        compressed_image = np.zeros_like(trimmed_image)
        
        # Processa ogni blocco
        for i in range(blocks_y):
            for j in range(blocks_x):
                # Estrae il blocco corrente
                y_start = i * block_size
                y_end = (i + 1) * block_size
                x_start = j * block_size
                x_end = (j + 1) * block_size
                
                block = trimmed_image[y_start:y_end, x_start:x_end]
                
                # Applica DCT2 al blocco
                dct_block = self.transformer.dct2_fast(block)
                
                # Elimina le frequenze in base alla soglia (k + l >= d)
                for k in range(block_size):
                    for l in range(block_size):
                        if k + l >= threshold:
                            dct_block[k, l] = 0
                
                # Applica DCT2 inversa
                reconstructed_block = self.transformer.idct2_fast(dct_block)
                
                # Arrotonda all'intero più vicino e limita valori a [0, 255]
                reconstructed_block = np.round(reconstructed_block)
                reconstructed_block = np.clip(reconstructed_block, 0, 255)
                
                # Inserisce il blocco ricostruito nell'immagine compressa
                compressed_image[y_start:y_end, x_start:x_end] = reconstructed_block
        
        return compressed_image.astype(np.uint8)
    
    def calculate_compression_ratio(self, original, threshold, block_size):
        """
        Calcola il rapporto di compressione teorico in base alla soglia e dimensione blocco.
        
        Argomenti:
            original (numpy.ndarray): immagine originale
            threshold (int): soglia di taglio delle frequenze
            block_size (int): dimensione del blocco
            
        Restituisce:
            float: rapporto di compressione stimato
        """
        # Calcola quanti coefficienti sono conservati in ogni blocco
        kept_coefficients = 0
        for k in range(block_size):
            for l in range(block_size):
                if k + l < threshold:
                    kept_coefficients += 1
        
        # Rapporto tra coefficienti conservati e totali
        compression_ratio = kept_coefficients / (block_size * block_size)
        
        return compression_ratio
    
    def calculate_psnr(self, original, compressed):
        """
        Calcola il Peak Signal-to-Noise Ratio tra l'immagine originale e quella compressa.
        
        Argomenti:
            original (numpy.ndarray): immagine originale
            compressed (numpy.ndarray): immagine compressa
            
        Restituisce:
            float: PSNR in dB
        """
        # Trimma le dimensioni se necessario
        height = min(original.shape[0], compressed.shape[0])
        width = min(original.shape[1], compressed.shape[1])
        
        original_trimmed = original[:height, :width]
        compressed_trimmed = compressed[:height, :width]
        
        # Calcola MSE (Mean Squared Error)
        mse = np.mean((original_trimmed - compressed_trimmed) ** 2)
        if mse == 0:  # Immagini identiche
            return float('inf')
        
        # Calcola PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr