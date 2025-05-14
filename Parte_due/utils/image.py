import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageHandler:
    """
    Classe per la gestione delle immagini: caricamento, conversione e visualizzazione.
    """
    
    def __init__(self):
        """Inizializza l'handler per le immagini."""
        self.original_image = None
        self.processed_image = None
        self.grayscale_array = None
        self.processed_array = None
        self.current_file = None
    
    def load_image(self, filepath):
        """
        Carica un'immagine da file e la converte in scala di grigi.
        
        Argomenti:
            filepath (str): percorso del file immagine
            
        Restituisce:
            numpy.ndarray: array dell'immagine in scala di grigi
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        try:
            # Carica l'immagine
            self.original_image = Image.open(filepath)
            self.current_file = filepath
            
            # Converti in scala di grigi
            grayscale_image = self.original_image.convert('L')
            
            # Converti in array numpy
            self.grayscale_array = np.array(grayscale_image)
            
            return self.grayscale_array
        
        except Exception as e:
            print(f"Errore durante il caricamento dell'immagine: {e}")
            return None
    
    def update_processed_image(self, processed_array):
        """
        Aggiorna l'immagine elaborata con un nuovo array.
        
        Argomenti:
            processed_array (numpy.ndarray): array dell'immagine elaborata
        """
        self.processed_array = processed_array
        self.processed_image = Image.fromarray(processed_array)
    
    def get_image_for_display(self, image, size=None):
        """
        Prepara un'immagine per la visualizzazione in Tkinter.
        
        Argomenti:
            image (PIL.Image): immagine da preparare
            size (tuple, optional): dimensione di ridimensionamento (larghezza, altezza)
            
        Restituisce:
            ImageTk.PhotoImage: immagine pronta per la visualizzazione in Tkinter
        """
        if image is None:
            return None
        
        # Crea una copia per evitare modifiche all'originale
        display_image = image.copy()
        
        # Ridimensiona se necessario
        if size is not None:
            display_image = display_image.resize(size, Image.Resampling.LANCZOS)
        
        # Converti in formato per Tkinter
        return ImageTk.PhotoImage(display_image)
    
    def get_original_and_processed_for_display(self, size=None):
        """
        Ottiene entrambe le immagini (originale e processata) per la visualizzazione.
        
        Argomenti:
            size (tuple, optional): dimensione di ridimensionamento (larghezza, altezza)
            
        Restituisce:
            tuple: (ImageTk.PhotoImage originale, ImageTk.PhotoImage processata)
        """
        if self.original_image is None:
            return None, None
        
        original_tk = self.get_image_for_display(self.original_image.convert('L'), size)
        
        if self.processed_image is not None:
            processed_tk = self.get_image_for_display(self.processed_image, size)
        else:
            processed_tk = None
            
        return original_tk, processed_tk
    
    def save_processed_image(self, filepath):
        """
        Salva l'immagine elaborata su file.
        
        Argomenti:
            filepath (str): percorso del file di destinazione
            
        Restituisce:
            bool: True se il salvataggio è riuscito, False altrimenti
        """
        if self.processed_image is None:
            print("Nessuna immagine elaborata da salvare.")
            return False
        
        try:
            self.processed_image.save(filepath)
            return True
        except Exception as e:
            print(f"Errore durante il salvataggio dell'immagine: {e}")
            return False
    
    def plot_original_vs_processed(self):
        """
        Visualizza un confronto tra l'immagine originale e quella elaborata.
        """
        if self.grayscale_array is None or self.processed_array is None:
            print("Mancano immagini da confrontare.")
            return
        
        # Crea figura con subplot
        plt.figure(figsize=(12, 6))
        
        # Mostra immagine originale
        plt.subplot(1, 2, 1)
        plt.imshow(self.grayscale_array, cmap='gray', vmin=0, vmax=255)
        plt.title('Immagine Originale')
        plt.axis('off')
        
        # Mostra immagine elaborata
        plt.subplot(1, 2, 2)
        plt.imshow(self.processed_array, cmap='gray', vmin=0, vmax=255)
        plt.title('Immagine Elaborata')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_image_dimensions(self):
        """
        Ottiene le dimensioni dell'immagine originale.
        
        Restituisce:
            tuple: (larghezza, altezza) dell'immagine
        """
        if self.original_image is None:
            return (0, 0)
        return self.original_image.size
    
    def get_file_info(self):
        """
        Ottiene le informazioni sul file immagine corrente.
        
        Restituisce:
            dict: dizionario con le informazioni sul file
        """
        if self.current_file is None:
            return None
        
        file_size = os.path.getsize(self.current_file)
        file_name = os.path.basename(self.current_file)
        dimensions = self.get_image_dimensions()
        
        return {
            'name': file_name,
            'size': file_size,
            'dimensions': dimensions,
            'path': self.current_file
        }