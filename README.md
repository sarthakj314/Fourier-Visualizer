# ✨ Fourier Series Visualizer ✨

Transform your doodles into mesmerizing mathematical animations with this interactive Fourier Series web app! Draw any shape and watch in awe as it's recreated through an elegant dance of rotating vectors, revealing how even the most complex patterns can emerge from simple circular motions.

<div align="center">
  <table>
    <tr>
      <td width="50%">
        <img src="https://raw.githubusercontent.com/sarthakj314/fourier-visualizer/main/examples/fourier_series.png" alt="Fourier Series Visualization" width="100%">
        <p align="center"><em>✨ Mathematical Magic in Motion ✨</em></p>
      </td>
      <td width="50%">
        <img src="https://raw.githubusercontent.com/sarthakj314/fourier-visualizer/main/examples/fourier_vectors.gif" alt="Fourier Vectors Animation" width="100%">
        <p align="center"><em>🔄 Epicycles in Action 🔄</em></p>
      </td>
    </tr>
  </table>
</div>

## 🚀 Overview

Dive into the fascinating world of Fourier Series as this application transforms your hand-drawn creations into a symphony of rotating vectors (epicycles).

## 🧠 The Math Behind the Magic

A Fourier series transforms any periodic function into a beautiful symphony of sine and cosine waves. For your drawing, we represent the path as a complex function where:

- The x-coordinate becomes the real component
- The y-coordinate transforms into the imaginary component

The mathematical heart of the process:

$$c_n = \frac{1}{T} \int_{0}^{T} f(t) e^{-i n \omega t} dt$$

Where:
- $c_n$ is the nth Fourier coefficient (the DNA of your drawing)
- $T$ is the period of the function
- $f(t)$ is the complex function representing your artistic creation
- $\omega = \frac{2\pi}{T}$ is the angular frequency

The complete Fourier series reconstruction of your drawing is given by:

$$f(t) = \sum_{n=-\infty}^{\infty} c_n e^{i n \omega t}$$

In practice, we use a finite number of terms (default: 60) to approximate your drawing. Each term corresponds to a rotating vector (epicycle) in our animation:

- The frequency $n$ determines how fast each vector rotates
- The magnitude $|c_n|$ determines the length of each vector
- The phase $\arg(c_n)$ determines the starting angle of each vector

### 🔄 From Drawing to Fourier Coefficients

1. Your drawing is sampled as a series of (x,y) points along a path
2. These points are converted to complex numbers: $z(t) = x(t) + i y(t)$
3. We normalize the path parameter $t$ to range from 0 to 1
4. The discrete Fourier transform is applied to calculate coefficients
5. Coefficients are sorted by magnitude to prioritize the most significant vectors

## 🚀 Get Started

### 🧰 Prerequisites
- Python 3.7+ (the foundation)
- PyTorch (the engine)
- Streamlit (the interface)
- Manim (the animator)
- NumPy and Pandas (the data wizards)

### ⚙️ Setup
1. Clone the repository to your machine
2. Summon the dependencies: `pip install -r requirements.txt`
3. For system dependencies: `apt-get install $(cat packages.txt)`
4. Launch into the experience: `streamlit run app.py`

## ⚡ Performance Tips
- Adjust the number of Fourier terms (default: 60) to balance detail and speed
- Canvas dimensions (default: 1200×500)
- Drawing processing parameters:
  - Interpolation points (default: 15) control smoothness between path commands
  - Point simplification distance (default: 1.0) reduces unnecessary points
- Animation settings:
  - Drawing animation duration (default: 5s)
  - Fourier animation duration (default: 6s)
  - Fourier animation frames (default: 100)
  - Fourier scale factor (default: 1.5)

## 🔧 Behind the Scenes
- Your drawing data persists between interactions thanks to session state magic
- The canvas adapts perfectly to your device with responsive CSS
- Animations are crafted as MP4 files and seamlessly embedded using base64 encoding
- Elegant loading animations keep you informed during intensive calculations