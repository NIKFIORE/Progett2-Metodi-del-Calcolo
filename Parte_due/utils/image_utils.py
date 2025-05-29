"""
Funzioni di utilità per la gestione delle immagini. Classe che fornisce metodi utili per la manipolazione delle immagini.
"""

import cv2
import numpy as np

class ImageUtils:
    def __init__(self):
        """Inizializza ImageUtils."""
        pass
    
    def load_grayscale_image(self, file_path):
        """
        Carica un'immagine e la converte in scala di grigi.

        Args: file_path (str): percorso del file immagine

        Returns: numpy.ndarray: immagine in scala di grigi come array 2D
        """
        # Carica immagine dal file
        image = cv2.imread(file_path)
        
        # Controlla che il caricamento sia riuscito
        if image is None:
            raise ValueError(f"Impossibile caricare l'immagine da {file_path}")
        
        # Se l'immagine è a colori (3 canali), la converte in scala di grigi
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Se è già in scala di grigi, la restituisce così com'è
            grayscale = image
            
        return grayscale
    
    def save_grayscale_image(self, image, file_path):
        """
        Salva un'immagine in scala di grigi su file.

        Args: image (numpy.ndarray): immagine in scala di grigi come array 2D
              file_path (str): percorso dove salvare l'immagine

        Returns:bool: True se il salvataggio ha successo, False altrimenti
        """
        # Scrive l'immagine sul disco e ritorna True/False in base al risultato
        return cv2.imwrite(file_path, image)
