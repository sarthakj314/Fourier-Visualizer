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
import time
import hashlib

# ====================== HYPERPARAMETERS ======================
# Canvas settings
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 500

# Drawing processing
INTERPOLATION_POINTS = 15  # Number of points to interpolate between path commands
POINT_SIMPLIFICATION_DISTANCE = 1.0  # Minimum distance between points after simplification

# Fourier series
FOURIER_TERMS = 60  # Number of Fourier terms to use (higher = more detail, slower)

# Animation settings
DRAWING_ANIMATION_DURATION = 5  # Duration of the drawing animation in seconds
FOURIER_ANIMATION_DURATION = 6  # Duration of the Fourier animation in seconds
FOURIER_ANIMATION_FRAMES = 100  # Number of frames for the Fourier animation
FOURIER_SCALE_FACTOR = 1.5  # Scale factor for the Fourier animation
# =============================================================

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
        margin-top: 3.5rem;
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
    
    /* Add styles for the process button */
    .process-button {
        margin-top: 1rem;
    }
    
    /* Center the process button */
    .process-button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 1rem 0;
    }
    
    .process-button-container button {
        width: auto !important;
        min-width: 200px;
        margin: 0 auto;
        display: block;
    }
    
    /* Fix for the stButton class */
    .process-button-container .stButton {
        width: auto;
        display: flex;
        justify-content: center;
    }
    
    /* Style for the instructions container to align with canvas center */
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
        margin-top: 3.5rem;
    }
    
    /* Zoomable video container */
    .zoomable-video-container {
        position: relative;
        overflow: hidden;
        border-radius: 5px;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .zoomable-video-container:hover {
        transform: scale(1.05);
        z-index: 100;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Fullscreen button for videos */
    .video-controls {
        display: flex;
        justify-content: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .video-controls button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background-color 0.3s;
    }
    
    .video-controls button:hover {
        background-color: #1565C0;
    }
    
    /* Center the process button - more direct approach */
    .truly-center-button {
        text-align: center;
        margin: 1rem auto;
        width: 100%;
        padding: 0 1.5rem; /* Match the padding of the instructions container */
    }
    
    .truly-center-button .stButton {
        width: 100% !important;
    }
    
    .truly-center-button .stButton > button {
        width: 100% !important;
        min-width: unset !important;
        display: block !important;
        margin: 0 !important;
        background-color: #1E88E5 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.75rem 1rem !important;
        border-radius: 5px !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Hover effect for the button */
    .truly-center-button .stButton > button:hover {
        background-color: #1565C0 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Override any Streamlit default styles that might interfere */
    .stButton {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'stored_coordinates' not in st.session_state:
        st.session_state.stored_coordinates = []
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = "default"
    if 'canvas_width' not in st.session_state:
        st.session_state.canvas_width = CANVAS_WIDTH
    if 'canvas_height' not in st.session_state:
        st.session_state.canvas_height = CANVAS_HEIGHT
    if 'drawing_data' not in st.session_state:
        st.session_state.drawing_data = None
    if 'fourier_output_path' not in st.session_state:
        st.session_state.fourier_output_path = None
    if 'drawing_output_path' not in st.session_state:
        st.session_state.drawing_output_path = None
    if 'drawing_count' not in st.session_state:
        st.session_state.drawing_count = 0
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'simplified_points_df' not in st.session_state:
        st.session_state.simplified_points_df = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

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

def display_looping_video(video_path, width=None, autoplay=True, zoomable=True):
    """
    Display a video with autoplay and loop enabled, with optional zoom capability
    
    Args:
        video_path: Path to the video file
        width: Width of the video in pixels (optional)
        autoplay: Whether to autoplay the video (default: True)
        zoomable: Whether to make the video zoomable (default: True)
    """
    # Check if video_path is None
    if video_path is None:
        st.error("Video file not found or could not be generated")
        return
        
    try:
        # Get the file name from the path
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        video_file.close()
        
        # Create a unique ID for this video
        video_id = f"video_{hash(video_path)}"
        
        # Create HTML with autoplay and loop attributes
        width_str = f"width=\"{width}\"" if width else "width=\"100%\""
        autoplay_str = "autoplay" if autoplay else ""
        
        container_class = "zoomable-video-container" if zoomable else "video-container"
        
        # Create the HTML for the video container and video element
        video_html = f"""
        <div class="{container_class}" id="{video_id}_container">
            <video id="{video_id}" {width_str} {autoplay_str} loop muted playsinline controls>
                <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """
        
        # Display the video HTML
        st.markdown(video_html, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error displaying video: {str(e)}")

def display_loading_animation(main_text="Generating Fourier Vector Animation...", sub_text="This may take up to a minute depending on the complexity"):
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

def simplify_points(x_coords, y_coords, min_distance=1.0):
    """
    Simplify a list of points by removing points that are too close together
    
    Args:
        x_coords: List of x coordinates
        y_coords: List of y coordinates
        min_distance: Minimum distance between points
        
    Returns:
        Simplified lists of x and y coordinates
    """
    if len(x_coords) <= 2:
        return x_coords, y_coords
    
    # Create new lists for the simplified points
    simplified_x = [x_coords[0]]  # Always keep the first point
    simplified_y = [y_coords[0]]
    
    # Keep track of the last point we added
    last_x, last_y = x_coords[0], y_coords[0]
    
    # Go through all points and only add those that are far enough from the last added point
    for i in range(1, len(x_coords)):
        current_x, current_y = x_coords[i], y_coords[i]
        
        # Calculate distance to the last added point
        distance = ((current_x - last_x) ** 2 + (current_y - last_y) ** 2) ** 0.5
        
        # If the distance is greater than the minimum, add the point
        if distance >= min_distance:
            simplified_x.append(current_x)
            simplified_y.append(current_y)
            last_x, last_y = current_x, current_y
    
    # Always add the last point to ensure the shape is complete
    if len(x_coords) > 1 and (simplified_x[-1] != x_coords[-1] or simplified_y[-1] != y_coords[-1]):
        simplified_x.append(x_coords[-1])
        simplified_y.append(y_coords[-1])

    print(f"Compressed from {len(x_coords)} to {len(simplified_x)} points")

    return simplified_x, simplified_y

def main():
    initialize_session_state()

    st.title("✏️ Fourier Series Visualizer")
    st.markdown("Draw anything on the canvas below and see its Fourier Series representation!")

    # Drawing section
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
            st.session_state.fourier_output_path = None
            st.session_state.drawing_output_path = None
            st.session_state.drawing_count = 0
            st.session_state.stats = None
            st.session_state.simplified_points_df = None
            st.session_state.processing = False
        
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
                width=st.session_state.canvas_width,
                drawing_mode='freedraw',
                key=f"canvas_{st.session_state.canvas_key}",
                initial_drawing=st.session_state.drawing_data
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Instructions section
        st.markdown("### ℹ️ Instructions")
        st.markdown("""
        1. Draw any shape or pattern on the canvas
        2. Click the "Process Drawing" button when finished
        3. View the Fourier Series representation below
        4. Experiment with different shapes to see how they're represented
        
        **Tip**: Simple, closed shapes work best!
        """)
        
        # About Fourier Series section
        st.markdown("### 📚 About Fourier Series")
        st.markdown("""
        A Fourier series decomposes any periodic function into a sum of simple sine and cosine waves.
        
        The animation shows how rotating vectors (epicycles) can recreate your drawing. Each vector:
        - Rotates at a different frequency
        - Has a specific length and phase
        - Contributes to the overall shape
        
        The more terms (vectors) used, the more accurate the representation!
        """)
        
        # Add the Process Drawing button under the instructions
        process_drawing = st.button("🔄 Process Drawing", 
                                  help="Click this button when you've finished your drawing to process it",
                                  use_container_width=True)

    # Create placeholders for the results
    st.markdown("---")
    
    # Create a 3-column layout for the top section
    top_col1, top_col2, top_col3 = st.columns(3)
    
    # Create placeholders for the top section
    with top_col1:
        drawing_animation_placeholder = st.empty()
    
    with top_col2:
        drawing_points_placeholder = st.empty()
    
    with top_col3:
        stats_placeholder = st.empty()
    
    # Create a full-width section for the Fourier animation
    st.markdown("---")
    fourier_animation_placeholder = st.empty()

    # Only process the drawing when the Process Drawing button is clicked
    # and there's a valid drawing
    should_process = (process_drawing and 
                     canvas_result.json_data is not None and 
                     len(canvas_result.json_data.get("objects", [])) > 0)

    if should_process:
        # Mark that we're processing
        st.session_state.processing = True
        
        # Increment the drawing count to force reprocessing
        st.session_state.drawing_count += 1
        st.session_state.drawing_data = canvas_result.json_data
        
        # Reset all placeholders to show loading animations
        with drawing_animation_placeholder.container():
            st.markdown("### 🎬 Original Drawing")
            display_loading_animation(main_text="Generating Original Drawing Animation...", sub_text="This may take a few moments")
        
        with drawing_points_placeholder.container():
            st.markdown("### 📈 Drawing Points")
            display_loading_animation(main_text="Processing Drawing Points...", sub_text="Extracting and simplifying points")
        
        with stats_placeholder.container():
            st.markdown("### 📊 Drawing Statistics")
            display_loading_animation(main_text="Calculating Statistics...", sub_text="Analyzing drawing data")
        
        with fourier_animation_placeholder.container():
            st.markdown("### 🔄 Fourier Series Representation")
            display_loading_animation(main_text="Generating Fourier Vector Animation...", 
                                    sub_text="This may take up to a minute depending on the complexity")
        
        # Process the drawing data
        x_coords = []
        y_coords = []
        num_interpolated_points = INTERPOLATION_POINTS
        
        # Generate a video of the drawing (this may take some time)
        try:
            drawing_output_path = generate_fourier_drawing_video(canvas_result.json_data, "drawing.mp4", 
                                                               drawing_duration=DRAWING_ANIMATION_DURATION)
            
            # Store the new drawing output path
            st.session_state.drawing_output_path = drawing_output_path
            
            # Replace the loading animation with the actual video
            with drawing_animation_placeholder.container():
                st.markdown("### 🎬 Original Drawing")
                if drawing_output_path is not None and os.path.exists(drawing_output_path):
                    display_looping_video(drawing_output_path, width="100%", zoomable=True)
                else:
                    st.error("Failed to generate drawing animation. Showing drawing points instead.")
        except Exception as e:
            with drawing_animation_placeholder.container():
                st.markdown("### 🎬 Original Drawing")
                st.error(f"Error generating drawing animation: {str(e)}")
        
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
            # Store the original points before simplification
            original_x_coords = x_coords.copy()
            original_y_coords = [-y for y in y_coords]  # Flip y-coordinates
            
            # Simplify the points to remove unnecessary ones
            min_distance = POINT_SIMPLIFICATION_DISTANCE
            x_coords_simplified, y_coords_simplified = simplify_points(original_x_coords, original_y_coords, min_distance)
            
            # Store both original and simplified coordinates
            st.session_state.stored_coordinates = list(zip(x_coords_simplified, y_coords_simplified))
            
            # Calculate stats based on simplified points
            stats = calculate_stats(x_coords_simplified, y_coords_simplified)
            st.session_state.stats = stats

            # Create DataFrame for the scatter plot
            simplified_points_df = pd.DataFrame({
                'x': x_coords_simplified,
                'y': y_coords_simplified
            })
            st.session_state.simplified_points_df = simplified_points_df

            # Display drawing points
            with drawing_points_placeholder.container():
                st.markdown("### 📈 Drawing Points")
                if st.session_state.simplified_points_df is not None:
                    st.scatter_chart(simplified_points_df, x='x', y='y')
                else:
                    st.info("Processing drawing points...")

            # Display drawing statistics - make sure this is updated properly
            with stats_placeholder.container():
                st.markdown("### 📊 Drawing Statistics")
                if stats:
                    metrics_html = '<div class="metric-container">'
                    metrics_html += display_metric_card("Points", stats['total_points'], dark_mode=True)
                    metrics_html += display_metric_card("Path Length", f"{stats['path_length']}", dark_mode=True)
                    metrics_html += display_metric_card("X Range", f"{stats['x_min']} to {stats['x_max']}", dark_mode=True)
                    metrics_html += display_metric_card("Y Range", f"{stats['y_min']} to {stats['y_max']}", dark_mode=True)
                    metrics_html += '</div>'
                    
                    display_card("📊 Drawing Statistics", metrics_html, dark_mode=True)
                else:
                    st.info("No statistics available yet.")
            
            # Create DataFrame for visualization using original points
            df = pd.DataFrame({
                'x': original_x_coords,
                'y': original_y_coords
            })

            # Generate the Fourier animation (this may take some time)
            try:
                # Start timing the Fourier vector generation
                fourier_start_time = time.time()
                
                fourier = FourierSeries(df['x'], df['y'], n=FOURIER_TERMS)
                fourier.compute_series()
                coeffs = fourier.prepare_for_manim(scale_factor=FOURIER_SCALE_FACTOR)
                output_path = generate_fourier_vector_video(coeffs, "fourier_vectors.mp4", 
                                                          num_frames=FOURIER_ANIMATION_FRAMES, 
                                                          drawing_duration=FOURIER_ANIMATION_DURATION)
                
                # Calculate and print the time taken
                fourier_end_time = time.time()
                fourier_generation_time = fourier_end_time - fourier_start_time
                print(f"Fourier vector animation generation took {fourier_generation_time:.2f} seconds")
                
                # Store the new Fourier output path
                st.session_state.fourier_output_path = output_path
                
                # Replace the loading animation with the actual video
                with fourier_animation_placeholder.container():
                    st.markdown(f"### 🔄 Fourier Series Representation (Generated in {fourier_generation_time:.2f}s)")
                    display_looping_video(output_path, width="100%", zoomable=True)
            except Exception as e:
                # Show error message if animation generation fails
                with fourier_animation_placeholder.container():
                    st.markdown("### 🔄 Fourier Series Representation")
                    st.error(f"Error generating Fourier animation: {str(e)}")
            
            # Mark that we're done processing
            st.session_state.processing = False
    
    # If we're not processing a new drawing, display the previous results if available
    elif not st.session_state.processing:
        # Display the original drawing animation
        with drawing_animation_placeholder.container():
            st.markdown("### 🎬 Original Drawing")
            if st.session_state.drawing_output_path and os.path.exists(st.session_state.drawing_output_path):
                display_looping_video(st.session_state.drawing_output_path, width="100%", zoomable=True)
            else:
                st.info("Draw something and click 'Process Drawing' to see the animation.")
        
        # Display drawing points
        with drawing_points_placeholder.container():
            st.markdown("### 📈 Drawing Points")
            if st.session_state.simplified_points_df is not None:
                st.scatter_chart(st.session_state.simplified_points_df, x='x', y='y')
            else:
                st.info("No drawing points available yet.")
        
        # Display drawing statistics
        with stats_placeholder.container():
            if st.session_state.stats:
                metrics_html = '<div class="metric-container">'
                metrics_html += display_metric_card("Points", st.session_state.stats['total_points'], dark_mode=True)
                metrics_html += display_metric_card("Path Length", f"{st.session_state.stats['path_length']}", dark_mode=True)
                metrics_html += display_metric_card("X Range", f"{st.session_state.stats['x_min']} to {st.session_state.stats['x_max']}", dark_mode=True)
                metrics_html += display_metric_card("Y Range", f"{st.session_state.stats['y_min']} to {st.session_state.stats['y_max']}", dark_mode=True)
                metrics_html += '</div>'
                
                display_card("📊 Drawing Statistics", metrics_html, dark_mode=True)
        
        # Display the Fourier animation
        with fourier_animation_placeholder.container():
            st.markdown("### 🔄 Fourier Series Representation")
            if st.session_state.fourier_output_path and os.path.exists(st.session_state.fourier_output_path):
                display_looping_video(st.session_state.fourier_output_path, width="100%", zoomable=True)
            else:
                st.info("Process a drawing to see the Fourier representation.")

if __name__ == "__main__":
    main()