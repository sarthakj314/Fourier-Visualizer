import os
# Disable Streamlit's file watcher completely
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from fourier.fourier import FourierSeries

def initialize_session_state():
    if 'stored_coordinates' not in st.session_state:
        st.session_state.stored_coordinates = []
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = "default"
    if 'canvas_width' not in st.session_state:
        st.session_state.canvas_width = 800  # Set larger initial width
    if 'canvas_height' not in st.session_state:
        st.session_state.canvas_height = 600  # Set larger initial height
    if 'drawing_data' not in st.session_state:
        st.session_state.drawing_data = None

def calculate_stats(x_coords, y_coords):
    if len(x_coords) == 0 or len(y_coords) == 0:
        return {}
    
    return {
        'x_min': np.min(x_coords),
        'x_max': np.max(x_coords),
        'y_min': np.min(y_coords),
        'y_max': np.max(y_coords),
        'total_points': len(x_coords),
        'path_length': np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
    }

def display_stats_card(title, value, description=""):
    """Display a styled stats card using Streamlit's metric with controlled formatting."""
    st.metric(label=title, value=value, help=description)

def main():
    initialize_session_state()

    st.title("✏️ Fourier Series")
    st.write("Use the canvas below to draw anything you want! Soon you will be able to see the Fourier Series of your drawing.")

    if st.button("🗑️ Clear Canvas"):
        st.session_state.canvas_key = str(pd.Timestamp.now())
        st.session_state.stored_coordinates = []
        st.session_state.drawing_data = None

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=st.session_state.canvas_height,
        width=st.session_state.canvas_width,
        drawing_mode='freedraw',
        key=f"canvas_{st.session_state.canvas_key}",
        initial_drawing=st.session_state.drawing_data
    )

    if canvas_result.json_data is not None:
        x_coords = []
        y_coords = []
        num_interpolated_points = 3
        
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


            fourier = FourierSeries(df['x'], df['y'])
            fourier.compute_series()
            fourier.plot_series(n=1000)

            # Separate rows for Drawing Points and Drawing Statistics
            st.write("📊 Drawing Points")
            st.scatter_chart(df, x='x', y='y')
            
            st.write("📈 Drawing Statistics")
            if stats:
                st.markdown("### 📌 Overview")
                overview_cols = st.columns(2)
                with overview_cols[0]:
                    display_stats_card("Total Points", f"{stats['total_points']:,}", "Number of points in the drawing")
                with overview_cols[1]:
                    display_stats_card("Path Length", f"{stats['path_length']:.3f}px", "Total length of the drawn path")
                
                # Enhanced Detailed Statistics using Columns
                st.markdown("### 🎯 Detailed Statistics")
                detailed_cols = st.columns(4)
                with detailed_cols[0]:
                    st.metric(label="Min X", value=f"{stats['x_min']:.3f}", help="Minimum X coordinate")
                with detailed_cols[1]:
                    st.metric(label="Max X", value=f"{stats['x_max']:.3f}", help="Maximum X coordinate")
                with detailed_cols[2]:
                    st.metric(label="Min Y", value=f"{stats['y_min']:.3f}", help="Minimum Y coordinate")
                with detailed_cols[3]:
                    st.metric(label="Max Y", value=f"{stats['y_max']:.3f}", help="Maximum Y coordinate")

if __name__ == "__main__":
    main()