# Fourier Series Visualizer

A web application that allows you to draw any shape and visualize its Fourier Series representation through animated vector diagrams.

![Fourier Series Visualizer](https://i.imgur.com/example.gif)

## Overview

This application demonstrates the power of Fourier Series by transforming hand-drawn shapes into a sum of rotating vectors (epicycles). The app provides:

- Interactive drawing canvas
- Real-time visualization of the drawing process
- Animated Fourier vector representation
- Drawing statistics and analysis

## Features

### Drawing Interface
- Free-form drawing canvas with responsive design
- Clear canvas functionality with one-click reset
- Automatic processing of drawing data including interpolation of points
- Support for various drawing commands (move, line, quadratic bezier curves)

### Visualizations
- Original drawing animation showing the drawing process
- Fourier vector animation showing epicycles with configurable number of terms
- Scatter plot of drawing points for data visualization
- Comprehensive drawing statistics (total points, path length, coordinate ranges)
- Dark mode UI for better viewing experience

### Technical Details
- Fourier Series computation using PyTorch for performance and GPU acceleration
- Manim-based animations for smooth vector visualization
- Streamlit web interface for interactive experience
- Base64 encoding for efficient video embedding
- Responsive design that works on various screen sizes

## How It Works

1. **Drawing Capture**: The app captures your drawing as a series of path commands (move, line, quadratic bezier).
2. **Point Extraction**: These commands are converted to a series of (x,y) coordinates with interpolation for smoother curves.
3. **Fourier Transform**: The coordinates are transformed into the frequency domain using Fourier analysis with configurable number of terms (default: 50).
4. **Vector Animation**: The Fourier coefficients are visualized as a series of rotating vectors that recreate your drawing.

## Mathematical Background

A Fourier series decomposes any periodic function into a sum of simple sine and cosine waves. For a drawing, we can represent the path as a complex function where:

- The x-coordinate is the real part
- The y-coordinate is the imaginary part

The Fourier coefficients are calculated using:

$$c_n = \frac{1}{T} \int_{0}^{T} f(t) e^{-i n \omega t} dt$$

Where:
- $c_n$ is the nth Fourier coefficient
- $T$ is the period of the function
- $f(t)$ is our complex function representing the drawing
- $\omega = \frac{2\pi}{T}$ is the angular frequency

## Installation and Usage

### Prerequisites
- Python 3.7+
- PyTorch
- Streamlit
- Manim (for animations)
- NumPy and Pandas

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

### Using the Application
1. Draw any shape on the canvas using your mouse or touchscreen
2. The application automatically processes your drawing
3. View the original drawing animation and Fourier representation side by side
4. Experiment with different shapes to see how they're represented in the frequency domain

## Performance Considerations
- Complex drawings with many points may take longer to process
- The number of Fourier terms (default: 50) affects both accuracy and performance
- For optimal performance, use a device with GPU support for PyTorch acceleration

## Implementation Details
- Drawing data is stored in session state for persistence between interactions
- Canvas is made responsive using custom CSS
- Animations are generated as MP4 files and embedded using base64 encoding
- Loading animations provide feedback during computation-intensive operations