# ✨ Fourier Series Visualizer: Art Meets Mathematics ✨

Transform your doodles into mesmerizing mathematical animations with this interactive Fourier Series web app! Draw any shape and watch in awe as it's recreated through an elegant dance of rotating vectors, revealing how even the most complex patterns can emerge from simple circular motions.

<div align="center">
  <table>
    <tr>
      <td width="50%">
        <img src="https://raw.githubusercontent.com/sarthakj314/fourier-visualizer/fourier_series.png" alt="Fourier Series Visualization" width="100%">
        <p align="center"><em>✨ Mathematical Magic in Motion ✨</em></p>
      </td>
      <td width="50%">
        <video width="100%" autoplay loop muted playsinline>
          <source src="https://raw.githubusercontent.com/sarthakj314/fourier-visualizer/fourier_vectors.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <p align="center"><em>🔄 Epicycles in Action 🔄</em></p>
      </td>
    </tr>
  </table>
</div>

## 🚀 Overview

Dive into the fascinating world of Fourier Series as this application transforms your hand-drawn creations into a symphony of rotating vectors (epicycles). Experience the perfect blend of art, mathematics, and interactive visualization:

- ✏️ Intuitive drawing canvas for your creative expression
- 🎬 Watch your drawing come to life in real-time
- 🔄 Marvel at the hypnotic Fourier vector representation
- 📊 Explore the data behind your artistic creation

## ✨ Features

### 🎨 Drawing Interface
- Fluid, responsive canvas that captures your every stroke
- One-click reset to unleash your creativity again and again
- Smart interpolation that smooths out your drawing
- Support for various drawing techniques (move, line, quadratic bezier curves)

### 🌈 Stunning Visualizations
- See your drawing unfold in a captivating animation
- Witness the mesmerizing dance of epicycles with adjustable complexity
- Explore your creation through interactive data visualization
- Track comprehensive statistics about your masterpiece
- Sleek dark mode for an enhanced viewing experience

### 🔧 Powerful Technology
- Lightning-fast Fourier computations powered by PyTorch with GPU acceleration
- Cinematic-quality animations courtesy of Manim
- Seamless interactive experience built on Streamlit
- Optimized video embedding using Base64 encoding
- Responsive design that looks gorgeous on any device

## 🔍 How It Works

1. **✏️ Capture Your Creation**: Your drawing is captured as a series of precise path commands
2. **🧮 Extract the Essence**: These commands transform into coordinates with smart interpolation for smooth curves
3. **⚡ Fourier Magic**: Your drawing enters the frequency domain through mathematical wizardry
4. **🎭 Vector Performance**: Watch as rotating vectors dance in perfect harmony to recreate your drawing

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
3. Launch into the experience: `streamlit run app.py`

### 🎮 Creating Your Masterpiece
1. Let your creativity flow on the canvas
2. Watch as the app instantly processes your artistic expression
3. Compare your original drawing with its mathematical twin
4. Experiment with different shapes and discover new patterns in the frequency domain

## ⚡ Performance Tips
- Complex masterpieces with intricate details may require more processing time
- Adjust the number of Fourier terms (default: 50) to balance detail and speed
- For the ultimate experience, use a device with GPU support

## 🔧 Behind the Scenes
- Your drawing data persists between interactions thanks to session state magic
- The canvas adapts perfectly to your device with responsive CSS
- Animations are crafted as MP4 files and seamlessly embedded using base64 encoding
- Elegant loading animations keep you informed during intensive calculations