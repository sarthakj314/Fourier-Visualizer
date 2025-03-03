import numpy as np
import matplotlib.pyplot as plt
import torch

class FourierSeries:
    def __init__(self, x, y, n=501):
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert input data to torch tensors
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        elif hasattr(x, 'x') and hasattr(x, 'y'):  # Handle DataFrame input
            x_tensor = torch.tensor(x.x.values, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(x.y.values, dtype=torch.float32, device=self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            
        self.data = x_tensor + 1j * y_tensor
        self.n = n
        self.deg_start, self.deg_end = self.compute_deg_start_end()
        self.c_n = torch.zeros(self.n, dtype=torch.complex64, device=self.device)

        self.total_points = len(self.data)
        self.t = torch.linspace(0, 1, self.total_points, device=self.device)

        self.series_constant = torch.linspace(self.deg_start, self.deg_end, self.n, device=self.device)
        self.series_constant = torch.exp(2j * torch.pi * self.series_constant)

    def compute_deg_start_end(self):
        tmp = (self.n - 1) / 2
        start, end = -np.floor(tmp), np.ceil(tmp)
        return int(start), int(end)
    
    def compute_c_n(self, n):
        exp_term = torch.exp(-2j * torch.pi * n * self.t)
        return torch.sum(self.data * exp_term) / self.total_points
        
    def compute_series(self):
        degrees = torch.arange(self.deg_start, self.deg_end + 1, device=self.device)
        for i, deg in enumerate(degrees):
            self.c_n[i] = self.compute_c_n(deg)

    def compute_series_value(self, t):
        if isinstance(t, np.ndarray):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        degrees = torch.arange(self.deg_start, self.deg_end + 1, device=self.device)
        result = torch.zeros(len(t), dtype=torch.complex64, device=self.device)
        
        for i, deg in enumerate(degrees):
            result += self.c_n[i] * torch.exp(2j * torch.pi * deg * t)
        
        # Convert to CPU and numpy for compatibility with the rest of the code
        return [result.real.cpu().numpy(), result.imag.cpu().numpy()]

    def return_series(self):
        return self.c_n.cpu().numpy()
 
    # f(t) = sum_{n = -infty}^{infty} c_n * e^{2pi*i*n*t}
    # integral from 0 to 1 of f(t) * e^(-2pi*i*n*t) dt = c_n


if __name__ == "__main__":
    x = np.linspace(-2, 2 - 1e-8, 1000)
    y = np.floor(x)%2

    fourier = FourierSeries(x, y)
    fourier.compute_series()
    
    t_smooth = np.linspace(0, 1, 1000) 
    reconstructed_points = fourier.compute_series_value(t_smooth)
    reconstructed_x, reconstructed_y = reconstructed_points
 
    # Plot the original points and Fourier reconstruction
    plt.figure(figsize=(10, 6))    
    plt.scatter(x, y, color='red', label='Original Points')
    plt.plot(reconstructed_x, reconstructed_y, color='blue', label='Fourier Reconstruction')
    
    plt.legend()
    plt.grid(True)
    plt.title("Original Points vs Fourier Series Reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("fourier_series.png")
    
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