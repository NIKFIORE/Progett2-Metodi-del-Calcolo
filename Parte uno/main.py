import numpy as np
import matplotlib.pyplot as plt
import time
from DCT2Transformer import DCT2Transformer

class Main:
    """
    Main class for benchmarking the DCT2 implementations and visualizing the results.
    """
    
    def __init__(self):
        self.transformer = DCT2Transformer()
        
    def benchmark(self, sizes):
        """
        Benchmark the manual and fast DCT2 implementations for various matrix sizes.
        
        Args:
            sizes (list): List of matrix sizes to benchmark
            
        Returns:
            dict: Dictionary with benchmark results
        """
        results = {
            'sizes': sizes,
            'manual_times': [],
            'fast_times': []
        }
        
        for N in sizes:
            print(f"Benchmarking for size {N}x{N}...")
            
            # Create random matrix
            f_mat = np.random.rand(N, N)
            
            # Benchmark manual implementation
            start_time = time.time()
            self.transformer.dct2_manual(f_mat)
            manual_time = time.time() - start_time
            results['manual_times'].append(manual_time)
            
            # Benchmark fast implementation - run multiple times for small matrices
            # to get a more accurate measurement
            num_runs = 10 if N < 32 else 1
            fast_times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.transformer.dct2_fast(f_mat)
                fast_times.append(time.time() - start_time)
            
            # Take average time
            fast_time = sum(fast_times) / num_runs
            results['fast_times'].append(fast_time)
            
            print(f"  Manual: {manual_time:.6f}s, Fast: {fast_time:.6f}s")
            
        return results
    
    def plot_results(self, results):
        """
        Plot benchmark results in a semilogarithmic scale.
        
        Args:
            results (dict): Benchmark results from the benchmark method
        """
        plt.figure(figsize=(10, 6))
        
        # Set style if seaborn is available
        # sns.set_style("whitegrid")
        
        # Plot times vs sizes
        plt.semilogy(results['sizes'], results['manual_times'], 'o-', label='Manual DCT2 (O(N³))')
        plt.semilogy(results['sizes'], results['fast_times'], 's-', label='Fast DCT2 (O(N²log(N)))')
        
        # Add theoretical complexity curves for reference
        sizes_np = np.array(results['sizes'])
        
        # Scale the curves to match the data
        # Find first non-zero value to avoid division by zero
        for i, val in enumerate(results['manual_times']):
            if val > 0:
                manual_scale = val / (sizes_np[i]**3)
                break
        else:
            manual_scale = 1e-10  # Default value if all are zero
            
        for i, val in enumerate(results['fast_times']):
            if val > 0:
                fast_scale = val / (sizes_np[i]**2 * np.log(sizes_np[i]))
                break
        else:
            fast_scale = 1e-10  # Default value if all are zero
        
        plt.semilogy(sizes_np, manual_scale * sizes_np**3, '--', label='O(N³) reference')
        plt.semilogy(sizes_np, fast_scale * sizes_np**2 * np.log(sizes_np), '--', label='O(N²log(N)) reference')
        
        plt.xlabel('Matrix Size (N)')
        plt.ylabel('Execution Time (s)')
        plt.title('DCT2 Execution Time Comparison')
        plt.legend()
        plt.grid(True)
        
        # Add annotations with time ratio (with check for zero division)
        for i, N in enumerate(results['sizes']):
            manual_time = results['manual_times'][i]
            fast_time = results['fast_times'][i]
            
            # Avoid division by zero
            if fast_time > 0:
                ratio = manual_time / fast_time
                plt.annotate(f'{ratio:.1f}x', 
                            xy=(N, np.sqrt(manual_time * fast_time) if manual_time * fast_time > 0 else manual_time),
                            xytext=(5, 0),
                            textcoords='offset points',
                            fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Additional Plot: Matrix size vs. speedup ratio
        plt.figure(figsize=(10, 6))
        
        # Calculate speedup ratios, handling zero division
        speedup_ratios = []
        for m, f in zip(results['manual_times'], results['fast_times']):
            if f > 0:
                speedup_ratios.append(m/f)
            else:
                # If fast time is zero, use a placeholder value or skip
                # Here we'll use a high value to indicate "much faster"
                speedup_ratios.append(1000)  # Arbitrary high value
        
        # Filter out matrix sizes where we couldn't compute a proper ratio
        valid_sizes = [s for i, s in enumerate(results['sizes']) if results['fast_times'][i] > 0]
        valid_ratios = [r for i, r in enumerate(speedup_ratios) if results['fast_times'][i] > 0]
        
        if valid_sizes and valid_ratios:  # Only plot if we have valid data
            plt.plot(valid_sizes, valid_ratios, 'o-')
            plt.xlabel('Matrix Size (N)')
            plt.ylabel('Speedup Ratio (Manual/Fast)')
            plt.title('DCT2 Implementation Speedup Ratio')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("Could not generate speedup ratio plot - insufficient valid data")


if __name__ == "__main__":
    # Create the transformer and main objects
    transformer = DCT2Transformer()
    main = Main()
    
    # Validate the implementation
    print("Validating DCT2 implementation...")
    transformer.validate_implementation()
    
    # Define matrix sizes for benchmarking
    # Vary this based on your computer's capabilities
    sizes = [8, 16, 32, 64, 128, 256]
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = main.benchmark(sizes)
    
    # Plot results
    print("\nPlotting results...")
    main.plot_results(results)