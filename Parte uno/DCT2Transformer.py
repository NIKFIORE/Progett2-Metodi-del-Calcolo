import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import dct, idct

class DCT2Transformer:
    """
    Class implementing the Discrete Cosine Transform (DCT) 2D functionality
    both using manual implementation and fast library implementation.
    """

    def __init__(self):
        pass

    def compute_D(self, N):
        """
        Compute the D matrix for DCT transformation.
        This is the equivalent of the compute_D function from the MATLAB code.
        
        Args:
            N (int): Size of the matrix
            
        Returns:
            numpy.ndarray: The D matrix for DCT
        """
        D = np.zeros((N, N))
        for k in range(N):
            if k == 0:
                coef = np.sqrt(1/N)
            else:
                coef = np.sqrt(2/N)
            
            for j in range(N):
                D[k, j] = coef * np.cos((np.pi * k * (2*j + 1)) / (2 * N))
                
        return D

    def dct2_manual(self, f_mat):
        """
        Manual implementation of 2D DCT transformation.
        This is the equivalent of the dct_2D function from the MATLAB code.
        
        Args:
            f_mat (numpy.ndarray): Input matrix
            
        Returns:
            numpy.ndarray: 2D DCT transformed matrix
        """
        N = f_mat.shape[0]
        D = self.compute_D(N)
        
        # Make a copy of the input matrix
        c_mat = f_mat.copy()
        
        # Apply DCT by columns
        for j in range(N):
            c_mat[:, j] = D @ c_mat[:, j]
        
        # Apply DCT by rows
        for i in range(N):
            c_mat[i, :] = (D @ c_mat[i, :].T).T
            
        return c_mat
    
    def dct2_fast(self, f_mat):
        """
        Fast implementation of 2D DCT using scipy's DCT function.
        
        Args:
            f_mat (numpy.ndarray): Input matrix
            
        Returns:
            numpy.ndarray: 2D DCT transformed matrix
        """
        return dct(dct(f_mat, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def validate_implementation(self):
        """
        Validate the DCT implementation with test data from the problem statement.
        """
        # Test block from the problem statement
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
        
        # First row for 1D DCT test
        first_row = test_block[0, :]
        
        # Apply 1D DCT to the first row
        D = self.compute_D(8)
        dct_row = D @ first_row
        
        # Expected results from the problem statement
        expected_dct_row = np.array([
            4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01
        ])
        
        # Test 2D DCT using manual implementation
        dct2_result = self.dct2_manual(test_block)
        
        # Print validation results
        print("1D DCT Validation:")
        print("Our result:")
        print(dct_row)
        print("\nExpected result:")
        print(expected_dct_row)
        print("\nDifference:")
        print(np.abs(dct_row - expected_dct_row))
        
        # Print first few elements of 2D DCT
        print("\n2D DCT (first few elements):")
        print(dct2_result[0, :])
        print(dct2_result[1, :])
        
        # Test with scipy's DCT to check consistency
        dct2_scipy = self.dct2_fast(test_block)
        
        print("\nSciPy's DCT2 (first few elements):")
        print(dct2_scipy[0, :])
        print(dct2_scipy[1, :])
        
        # Note: The output might differ from the expected result due to
        # different normalizations in different implementations
        

