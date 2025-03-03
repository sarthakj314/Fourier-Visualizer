import os
# Disable Streamlit's file watcher completely
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import sys
sys.path.append("drawing_manim")
sys.path.append("fourier")

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from fourier import FourierSeries
from draw_fourier import generate_fourier_drawing_video, generate_fourier_vector_video
import base64

# Set page configuration
st.set_page_config(
    page_title="Fourier Series Visualizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dark-card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #1E1E1E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .dark-card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #64B5F6;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .dark-metric-card {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 1rem;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .dark-metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #64B5F6;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .dark-metric-label {
        font-size: 0.9rem;
        color: #BBBBBB;
    }
    .video-container {
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .instructions-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        padding: 1.5rem;
        background-color: #1E1E1E;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .instructions-container h3 {
        color: #64B5F6;
        margin-bottom: 1rem;
    }
    .instructions-container p {
        margin-bottom: 0.7rem;
    }
    .instructions-container strong {
        color: #64B5F6;
    }
    /* Make the canvas container take full width */
    .canvas-container {
        width: 100% !important;
        margin-top: 1rem;
    }
    
    /* Make the canvas element itself take full width */
    .canvas-container > div {
        width: 100% !important;
    }
    
    /* Target the actual canvas element */
    .canvas-container canvas {
        width: 100% !important;
    }
    
    /* Ensure the streamlit-drawable-canvas wrapper takes full width */
    .streamlit-drawable-canvas {
        width: 100% !important;
    }
    
    /* Make the button full width */
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'stored_coordinates' not in st.session_state:
        st.session_state.stored_coordinates = []
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = "default"
    if 'canvas_width' not in st.session_state:
        # Calculate canvas width based on column width (approximately 66% of screen width)
        st.session_state.canvas_width = 1000
    if 'canvas_height' not in st.session_state:
        # Keep a reasonable aspect ratio
        st.session_state.canvas_height = 500
    if 'drawing_data' not in st.session_state:
        st.session_state.drawing_data = None
    if 'fourier_animation_ready' not in st.session_state:
        st.session_state.fourier_animation_ready = False
    if 'fourier_output_path' not in st.session_state:
        st.session_state.fourier_output_path = None

def calculate_stats(x_coords, y_coords):
    if len(x_coords) == 0 or len(y_coords) == 0:
        return {}
    
    return {
        'x_min': round(np.min(x_coords), 2),
        'x_max': round(np.max(x_coords), 2),
        'y_min': round(np.min(y_coords), 2),
        'y_max': round(np.max(y_coords), 2),
        'total_points': len(x_coords),
        'path_length': round(np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)), 2)
    }

def display_metric_card(label, value, dark_mode=False):
    """Display a styled metric card with custom HTML/CSS"""
    card_class = "dark-metric-card" if dark_mode else "metric-card"
    value_class = "dark-metric-value" if dark_mode else "metric-value"
    label_class = "dark-metric-label" if dark_mode else "metric-label"
    
    metric_html = f"""
    <div class="{card_class}">
        <div class="{value_class}">{value}</div>
        <div class="{label_class}">{label}</div>
    </div>
    """
    return metric_html

def display_looping_video(video_path, width=None, autoplay=True):
    """
    Display a video with autoplay and loop enabled
    
    Args:
        video_path: Path to the video file
        width: Width of the video in pixels (optional)
        autoplay: Whether to autoplay the video (default: True)
    """
    # Get the file name from the path
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    # Create HTML with autoplay and loop attributes
    width_str = f"width=\"{width}\"" if width else "width=\"100%\""
    autoplay_str = "autoplay" if autoplay else ""
    
    video_html = f"""
    <div class="video-container">
        <video {width_str} {autoplay_str} loop muted playsinline controls>
            <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    
    # Display the HTML
    st.markdown(video_html, unsafe_allow_html=True)

def display_loading_animation(main_text="Generating Fourier Vector Animation...", sub_text="This may take a few moments"):
    """Display a loading animation while the Fourier animation is being generated"""
    loading_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 2rem 0;">
        <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        <p style="margin-top: 1rem; font-size: 1.2rem; color: #3498db;">{main_text}</p>
        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">{sub_text}</p>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

def display_card(title, content, dark_mode=False):
    """Display content in a styled card"""
    card_class = "dark-card" if dark_mode else "card"
    title_class = "dark-card-title" if dark_mode else "card-title"
    
    card_html = f"""
    <div class="{card_class}">
        <div class="{title_class}">{title}</div>
        {content}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    initialize_session_state()

    st.title("✏️ Fourier Series Visualizer")
    st.markdown("Draw anything on the canvas below and see its Fourier Series representation!")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Drawing canvas
        st.markdown("### 🎨 Drawing Canvas")
        
        # Create a container for the button
        clear_canvas = st.button("🗑️ Clear Canvas", use_container_width=True)
        
        # Check if clear button was pressed
        if clear_canvas:
            st.session_state.canvas_key = str(pd.Timestamp.now())
            st.session_state.stored_coordinates = []
            st.session_state.drawing_data = None
            st.session_state.fourier_animation_ready = False
            st.session_state.fourier_output_path = None
        
        # Calculate canvas width based on the column width
        # This is a placeholder value that will be overridden by CSS
        canvas_width = 1000
        
        # Add a container div to make canvas responsive
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)

        # Use st.container() to get a full-width container
        with st.container():
            # The canvas will now inherit the container's width
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=3,
                stroke_color="#000000",
                background_color="#FFFFFF",
                update_streamlit=True,
                height=st.session_state.canvas_height,
                # Set a base width that will be overridden by CSS
                width=canvas_width,
                drawing_mode='freedraw',
                key=f"canvas_{st.session_state.canvas_key}",
                initial_drawing=st.session_state.drawing_data
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Vertically centered instructions with dark mode
        st.markdown("""
        <div class="instructions-container">
            <h3>ℹ️ Instructions</h3>
            <p>1. Draw any shape or pattern on the canvas</p>
            <p>2. The app will automatically process your drawing</p>
            <p>3. View the Fourier Series representation below</p>
            <p>4. Experiment with different shapes to see how they're represented</p>
            <p><strong>Tip</strong>: Simple, closed shapes work best!</p>
        </div>
        """, unsafe_allow_html=True)

    if canvas_result.json_data is not None and len(canvas_result.json_data.get("objects", [])) > 0:
        st.session_state.drawing_data = canvas_result.json_data
        
        # Process the drawing data
        x_coords = []
        y_coords = []
        num_interpolated_points = 3
        
        # Display results in a two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎬 Original Drawing")
            
            # Create a placeholder for the original drawing animation
            drawing_animation_placeholder = st.empty()
            
            # Show loading animation for the original drawing
            with drawing_animation_placeholder.container():
                display_loading_animation(main_text="Generating Original Drawing Animation...", sub_text="This may take a few moments")
            
            # Generate a video of the drawing (this may take some time)
            drawing_output_path = generate_fourier_drawing_video(canvas_result.json_data, "drawing.mp4")
            
            # Replace the loading animation with the actual video
            with drawing_animation_placeholder.container():
                display_looping_video(drawing_output_path, width="100%")
        
        # Extract coordinates from the drawing
        for obj in canvas_result.json_data["objects"]:
            for command in obj["path"]:
                if isinstance(command, list):
                    if command[0] == 'M':
                        # Move command - just add the point
                        x, y = command[1], command[2]
                        if 0 <= x <= st.session_state.canvas_width and 0 <= y <= st.session_state.canvas_height:
                            x_coords.append(x)
                            y_coords.append(y)
                    elif command[0] == 'L':
                        # Line command
                        x1, y1 = command[1], command[2]
                        if x_coords and y_coords:
                            x0, y0 = x_coords[-1], y_coords[-1]
                            for i in range(1, num_interpolated_points + 1):
                                t = i / num_interpolated_points
                                x = x0 + t * (x1 - x0)
                                y = y0 + t * (y1 - y0)
                                if 0 <= x <= st.session_state.canvas_width and 0 <= y <= st.session_state.canvas_height:
                                    x_coords.append(x)
                                    y_coords.append(y)
                    elif command[0] == 'Q':
                        # Quadratic bezier curve
                        x0, y0 = x_coords[-1], y_coords[-1] # Start point
                        x1, y1 = command[1], command[2] # Control point
                        x2, y2 = command[3], command[4] # End point
                        
                        for i in range(1, num_interpolated_points + 1):
                            t = i / num_interpolated_points
                            # Quadratic bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                            x = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
                            y = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
                            if 0 <= x <= st.session_state.canvas_width and 0 <= y <= st.session_state.canvas_height:
                                x_coords.append(x)
                                y_coords.append(y)
        
        if x_coords and y_coords:
            st.session_state.stored_coordinates = list(zip(x_coords, y_coords))
            y_coords = [-y for y in y_coords]
            stats = calculate_stats(x_coords, y_coords)

            df = pd.DataFrame({
                'x': x_coords,
                'y': y_coords
            })

            # Continue with the left column (drawing points)
            with col1:
                # Display drawing points
                st.markdown("### 📈 Drawing Points")
                st.scatter_chart(df, x='x', y='y')
            
            # Continue with the right column (Fourier animation and other content)
            with col2:
                # Create a placeholder for the Fourier animation
                fourier_animation_placeholder = st.empty()
                
                # Show loading animation if not already generated
                if not st.session_state.fourier_animation_ready:
                    with fourier_animation_placeholder.container():
                        st.markdown("### 🔄 Fourier Series Representation")
                        display_loading_animation()
                    
                    # Display drawing statistics while Fourier animation is loading
                    metrics_html = '<div class="metric-container">'
                    metrics_html += display_metric_card("Points", stats['total_points'], dark_mode=True)
                    metrics_html += display_metric_card("Path Length", f"{stats['path_length']}", dark_mode=True)
                    metrics_html += display_metric_card("X Range", f"{stats['x_min']} to {stats['x_max']}", dark_mode=True)
                    metrics_html += display_metric_card("Y Range", f"{stats['y_min']} to {stats['y_max']}", dark_mode=True)
                    metrics_html += '</div>'
                    
                    display_card("📊 Drawing Statistics", metrics_html, dark_mode=True)
                    
                    # Add explanation about Fourier Series
                    st.markdown("### 📚 About Fourier Series")
                    st.markdown("""
                    A Fourier series decomposes any periodic function into a sum of simple sine and cosine waves.
                    
                    The animation shows how rotating vectors (epicycles) can recreate your drawing. Each vector:
                    - Rotates at a different frequency
                    - Has a specific length and phase
                    - Contributes to the overall shape
                    
                    The more terms (vectors) used, the more accurate the representation!
                    """)
                    
                    # Generate the Fourier animation (this may take some time)
                    fourier = FourierSeries(df['x'], df['y'], n=50)
                    fourier.compute_series()
                    coeffs = fourier.prepare_for_manim()
                    output_path = generate_fourier_vector_video(coeffs, "fourier_vectors.mp4")
                    
                    # Store the output path and mark as ready
                    st.session_state.fourier_output_path = output_path
                    st.session_state.fourier_animation_ready = True
                    
                    # Replace the loading animation with the actual video
                    with fourier_animation_placeholder.container():
                        st.markdown("### 🔄 Fourier Series Representation")
                        display_looping_video(output_path, width="100%")
                else:
                    # Display the already generated animation
                    with fourier_animation_placeholder.container():
                        st.markdown("### 🔄 Fourier Series Representation")
                        display_looping_video(st.session_state.fourier_output_path, width="100%")
                    
                    # Display drawing statistics
                    metrics_html = '<div class="metric-container">'
                    metrics_html += display_metric_card("Points", stats['total_points'], dark_mode=True)
                    metrics_html += display_metric_card("Path Length", f"{stats['path_length']}", dark_mode=True)
                    metrics_html += display_metric_card("X Range", f"{stats['x_min']} to {stats['x_max']}", dark_mode=True)
                    metrics_html += display_metric_card("Y Range", f"{stats['y_min']} to {stats['y_max']}", dark_mode=True)
                    metrics_html += '</div>'
                    
                    display_card("📊 Drawing Statistics", metrics_html, dark_mode=True)
                    
                    # Add explanation about Fourier Series
                    st.markdown("### 📚 About Fourier Series")
                    st.markdown("""
                    A Fourier series decomposes any periodic function into a sum of simple sine and cosine waves.
                    
                    The animation shows how rotating vectors (epicycles) can recreate your drawing. Each vector:
                    - Rotates at a different frequency
                    - Has a specific length and phase
                    - Contributes to the overall shape
                    
                    The more terms (vectors) used, the more accurate the representation!
                    """)

if __name__ == "__main__":
    main()