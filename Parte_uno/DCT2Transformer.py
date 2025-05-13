import numpy as np
import matplotlib.pyplot as plt
import time
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
        Equivalente alla funzione compute_D del codice MATLAB.

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
        Equivalente alla funzione dct_2D del codice MATLAB.

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
        # Applica DCT lungo l’asse 0 (righe), poi lungo l’asse 1 (colonne), con normalizzazione ortogonale
        return dct(dct(f_mat, axis=0, norm='ortho'), axis=1, norm='ortho')

    def validate_implementation(self):
        """
        Valida l'implementazione della DCT utilizzando i dati di test presenti nel testo del progetto.
        """
        # Blocco di test 8x8 fornito nel progetto
        test_block = np.array([
            [231, 32, 233, 161, 24, 71, 140, 245],
            [247, 40, 248, 245, 124, 204, 36, 107],
            [234, 202, 245, 167, 9, 217, 239, 173],
            [193, 190, 100, 167, 43, 180, 8, 70],
            [11, 24, 210, 177, 81, 243, 8, 112],
            [97, 195, 203, 47, 125, 114, 165, 181],
            [193, 70, 174, 167, 41, 30, 127, 245],
            [87, 149, 57, 192, 65, 129, 178, 228]
        ])

        # Prima riga per il test 1D
        first_row = test_block[0, :]

        # Applica la DCT1D alla prima riga con la matrice D
        D = self.compute_D(8)
        dct_row = D @ first_row  # moltiplicazione matrice D per vettore riga

        # Risultati attesi dal testo
        expected_dct_row = np.array([
            4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02,
            6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01
        ])

        # Esegue DCT2 manuale sul blocco
        dct2_result = self.dct2_manual(test_block)

        # Output del confronto tra risultato ottenuto e atteso
        print("Validazione DCT 1D:")
        print("Risultato ottenuto:")
        print(dct_row)
        print("\nRisultato atteso:")
        print(expected_dct_row)
        print("\nDifferenza assoluta:")
        print(np.abs(dct_row - expected_dct_row))

        # Stampa i primi coefficienti della DCT2 manuale
        print("\nDCT2 manuale (prime due righe):")
        print(dct2_result[0, :])
        print(dct2_result[1, :])

        # Verifica consistenza con la DCT2 veloce di SciPy
        dct2_scipy = self.dct2_fast(test_block)

        print("\nDCT2 con SciPy (prime due righe):")
        print(dct2_scipy[0, :])
        print(dct2_scipy[1, :])

        # Nota: Le differenze nei risultati possono dipendere dalle normalizzazioni diverse
