"""
Classe che implementa la Trasformata Discreta del Coseno bidimensionale (DCT2),
sia con un'implementazione manuale che con una veloce basata su libreria (SciPy).
"""

import numpy as np
from scipy.fft import dct, idct

class DCT2Transformer:
    def __init__(self):
        pass

    def dct2_fast(self, f_mat):
        """
        Implementazione veloce della DCT2 utilizzando la funzione DCT di scipy.

        Argomenti: f_mat (numpy.ndarray): matrice di input

        Restituisce: numpy.ndarray: matrice trasformata con DCT2
        """
        # Applica DCT lungo l'asse 0 (righe), poi lungo l'asse 1 (colonne), con normalizzazione ortogonale
        return dct(dct(f_mat, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def idct2_fast(self, c_mat):
        """
        Implementazione veloce della IDCT2 (DCT2 inversa) utilizzando la funzione IDCT di scipy.

        Argomenti:c_mat (numpy.ndarray): matrice trasformata con DCT2

        Restituisce:numpy.ndarray: matrice originale ricostruita
        """
        # Applica IDCT lungo l'asse 1 (colonne), poi lungo l'asse 0 (righe), con normalizzazione ortogonale
        return idct(idct(c_mat, axis=1, norm='ortho'), axis=0, norm='ortho')