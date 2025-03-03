import numpy as np
import matplotlib.pyplot as plt

class FourierSeries:
    def __init__(self, x, y, n=501):
        # Convert input data to numpy arrays
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_array = x.astype(np.float32)
            y_array = y.astype(np.float32)
        elif hasattr(x, 'x') and hasattr(x, 'y'):  # Handle DataFrame input
            x_array = x.x.values.astype(np.float32)
            y_array = x.y.values.astype(np.float32)
        else:
            x_array = np.array(x, dtype=np.float32)
            y_array = np.array(y, dtype=np.float32)
            
        self.data = x_array + 1j * y_array
        self.n = n
        self.deg_start, self.deg_end = self.compute_deg_start_end()
        self.c_n = np.zeros(self.n, dtype=np.complex64)

        self.total_points = len(self.data)
        self.t = np.linspace(0, 1, self.total_points)

        self.series_constant = np.arange(self.deg_start, self.deg_end + 1)
        self.series_constant = np.exp(2j * np.pi * self.series_constant)

    def compute_deg_start_end(self):
        tmp = (self.n - 1) / 2
        start, end = -np.floor(tmp), np.ceil(tmp)
        return int(start), int(end)
    
    def compute_c_n(self, n):
        exp_term = np.exp(-2j * np.pi * n * self.t)
        return np.sum(self.data * exp_term) / self.total_points
        
    def compute_series(self):
        degrees = np.arange(self.deg_start, self.deg_end + 1)
        for i, deg in enumerate(degrees):
            self.c_n[i] = self.compute_c_n(deg)

    def compute_series_value(self, t):
        if not isinstance(t, np.ndarray):
            t = np.array(t, dtype=np.float32)
        
        degrees = np.arange(self.deg_start, self.deg_end + 1)
        result = np.zeros(len(t), dtype=np.complex64)
        
        for i, deg in enumerate(degrees):
            result += self.c_n[i] * np.exp(2j * np.pi * deg * t)
        
        return [result.real, result.imag]

    def return_series(self):
        return self.c_n
    
    def export_series(self):
        return {
            "c_n": self.c_n,
            "deg_start": self.deg_start,
            "deg_end": self.deg_end,
            "n": self.n,
            "total_points": self.total_points,
            "t": self.t,
            "series_constant": self.series_constant
        }
    
    def sample_n(self, n=10000):
        t_smooth = np.linspace(0, 1, n)
        reconstructed_points = self.compute_series_value(t_smooth)
        reconstructed_x, reconstructed_y = reconstructed_points
        return reconstructed_x, reconstructed_y

    def plot_series(self, n=10000, plot_original=True):
        reconstructed_x, reconstructed_y = self.sample_n(n)
        if plot_original:
            plt.scatter(self.data.real, self.data.imag, color='red', label='Original Points')
        plt.plot(reconstructed_x, reconstructed_y, color='blue', label='Fourier Reconstruction')
        plt.legend()
        plt.grid(True)
        plt.savefig("fourier_series.png")

    def prepare_for_manim(self):
        """
        Prepare Fourier coefficients for Manim visualization
        Converts numpy arrays to Python native types for JSON serialization
        
        Returns:
            List of tuples (frequency, complex_coefficient) sorted by magnitude
            with coefficients normalized to fit within the Manim screen
        """
        coeffs = []
        for i, idx in enumerate(range(self.n)):
            freq = self.deg_start + i
            if freq == 0:
                continue
            # Convert numpy complex to Python complex number
            coeff_value = complex(self.c_n[i].real, self.c_n[i].imag)
            coeffs.append((freq, coeff_value))
        
        # Sort by magnitude (largest first)
        coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Normalize coefficients to ensure they fit within the Manim screen
        # Find the maximum magnitude
        max_magnitude = max(abs(coeff) for _, coeff in coeffs) if coeffs else 1.0
        
        # Scale factor - adjust this value based on desired arrow size in Manim
        # A value of 3-4 typically works well for Manim's default camera frame
        scale_factor = 1
        
        # Normalize all coefficients
        normalized_coeffs = [(freq, coeff * scale_factor / max_magnitude) 
                             for freq, coeff in coeffs]
        
        return normalized_coeffs
 
    # f(t) = sum_{n = -infty}^{infty} c_n * e^{2pi*i*n*t}
    # integral from 0 to 1 of f(t) * e^(-2pi*i*n*t) dt = c_n


if __name__ == "__main__":
    x = np.linspace(-4, 4 - 1e-8, 2000)
    y = np.floor(x)%2

    fourier = FourierSeries(x, y)
    fourier.compute_series()
    fourier.plot_series(n=10000)
    
    # Plot the histogram of the magnitude of c_n values
    plt.figure(figsize=(10, 6))
    c_n = fourier.return_series()
    c_n_magnitude = np.abs(c_n)
    degrees = np.arange(fourier.deg_start, fourier.deg_end + 1)
    
    plt.bar(degrees, c_n_magnitude)
    plt.grid(True)
    plt.title("Magnitude of Fourier Coefficients")
    plt.xlabel("Frequency (n)")
    plt.ylabel("|c_n|")
    plt.savefig("fourier_coefficients.png")