import numpy as np
from scipy.fft import dct, idct

class DCT2Transformer:
    """
    Classe che implementa la Trasformata Discreta del Coseno bidimensionale (DCT2),
    sia con un'implementazione manuale che con una veloce basata su libreria (SciPy).
    """

    def __init__(self):
        pass

    def compute_D(self, N):
        """
        Calcola la matrice D per la trasformata DCT.

        Argomenti:
            N (int): dimensione della matrice quadrata

        Restituisce:
            numpy.ndarray: matrice D della DCT (NxN)
        """
        D = np.zeros((N, N))  # inizializza matrice D a zeri
        for k in range(N):
            if k == 0:
                coef = np.sqrt(1/N)  # coefficiente di normalizzazione per k = 0
            else:
                coef = np.sqrt(2/N)  # coefficiente di normalizzazione per k > 0

            for j in range(N):
                # Calcolo dell'elemento (k, j) della matrice D secondo la formula della DCT
                D[k, j] = coef * np.cos((np.pi * k * (2*j + 1)) / (2 * N))
                
        return D

    def dct2_manual(self, f_mat):
        """
        Implementazione manuale della trasformata DCT2 (2D).

        Argomenti:
            f_mat (numpy.ndarray): matrice di input

        Restituisce:
            numpy.ndarray: matrice trasformata con DCT2
        """
        N = f_mat.shape[0]
        D = self.compute_D(N)  # calcola la matrice D

        # Copia della matrice di input
        c_mat = f_mat.copy()

        # Applica la DCT alle colonne: c = D @ f
        for j in range(N):
            c_mat[:, j] = D @ c_mat[:, j]  # moltiplicazione matrice-vettore colonna per ogni colonna

        # Applica la DCT alle righe: c = c @ D^T
        for i in range(N):
            c_mat[i, :] = (D @ c_mat[i, :].T).T  # moltiplicazione matrice-vettore riga

        return c_mat

    def dct2_fast(self, f_mat):
        """
        Implementazione veloce della DCT2 utilizzando la funzione DCT di scipy.

        Argomenti:
            f_mat (numpy.ndarray): matrice di input

        Restituisce:
            numpy.ndarray: matrice trasformata con DCT2
        """
        # Applica DCT lungo l'asse 0 (righe), poi lungo l'asse 1 (colonne), con normalizzazione ortogonale
        return dct(dct(f_mat, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def idct2_fast(self, c_mat):
        """
        Implementazione veloce della IDCT2 (DCT2 inversa) utilizzando la funzione IDCT di scipy.

        Argomenti:
            c_mat (numpy.ndarray): matrice trasformata con DCT2

        Restituisce:
            numpy.ndarray: matrice originale ricostruita
        """
        # Applica IDCT lungo l'asse 1 (colonne), poi lungo l'asse 0 (righe), con normalizzazione ortogonale
        return idct(idct(c_mat, axis=1, norm='ortho'), axis=0, norm='ortho')