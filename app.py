import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from fourier import FourierSeries

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
        'x_mean': np.mean(x_coords),
        'y_mean': np.mean(y_coords),
        'x_std': np.std(x_coords),
        'y_std': np.std(y_coords),
        'total_points': len(x_coords),
        'path_length': np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
    }

def display_stats_card(title, value, description=""):
    """Display a styled stats card using Streamlit's metric with controlled formatting."""
    st.metric(label=title, value=value, help=description)

def main():
    initialize_session_state()

    st.title("✏️ Drawing Analyzer")
    st.write("Use the canvas below to free-draw. Click 'Clear' to erase your drawing.")

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
        
        for obj in canvas_result.json_data["objects"]:
            for command in obj["path"]:
                if isinstance(command, list):
                    if command[0] in ['M', 'L']:
                        x, y = command[1], command[2]
                        # Ignore points outside the canvas
                        if 0 <= x <= st.session_state.canvas_width and 0 <= y <= st.session_state.canvas_height:
                            x_coords.append(x)
                            y_coords.append(y)
                    elif command[0] == 'Q':
                        x, y = command[3], command[4]
                        # Ignore points outside the canvas
                        if 0 <= x <= st.session_state.canvas_width and 0 <= y <= st.session_state.canvas_height:
                            x_coords.append(x)
                            y_coords.append(y)
        
        if x_coords and y_coords:
            st.session_state.stored_coordinates = list(zip(x_coords, y_coords))
            stats = calculate_stats(x_coords, y_coords)

            df = pd.DataFrame({
                'x': x_coords,
                'y': y_coords
            })

            fourier = FourierSeries(df)
            fourier.compute_series()
            times = np.linspace(0, 1, stats['total_points'])            
            series_values = fourier.compute_series_value(times)
            print(series_values)

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
                    st.metric(label="Mean X", value=f"{stats['x_mean']:.3f}", help="Average X coordinate")
                with detailed_cols[1]:
                    st.metric(label="Mean Y", value=f"{stats['y_mean']:.3f}", help="Average Y coordinate")
                with detailed_cols[2]:
                    st.metric(label="Std Dev X", value=f"{stats['x_std']:.3f}", help="Variation in X coordinates")
                with detailed_cols[3]:
                    st.metric(label="Std Dev Y", value=f"{stats['y_std']:.3f}", help="Variation in Y coordinates")

if __name__ == "__main__":
    main()