import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from manim import config

# Disable preview in headless environments
config["preview"] = False

def render_drawing(drawing_data, output_file="drawing_animation.mp4"):
    """
    Render a drawing animation from drawing data in completely headless mode
    
    Args:
        drawing_data: JSON data from Streamlit's canvas
        output_file: Output file path for the animation
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    # Create a temporary directory for the rendering process
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a temporary scene file with the drawing data embedded
        temp_scene_path = os.path.join(temp_dir, "temp_scene.py")
        drawing_data_json = json.dumps(drawing_data)
        
        # Create a custom config file to ensure headless operation
        config_path = os.path.join(temp_dir, "manim.cfg")
        with open(config_path, "w") as f:
            f.write("""
[CLI]
preview = False
show_in_file_browser = False
progress_bar = none
verbosity = ERROR

[renderer]
use_opengl_renderer = False

[window]
enable_gui = False
            """)
        
        with open(temp_scene_path, "w") as f:
            f.write(f"""
from manim import *
import json
import numpy as np
import os
import sys

# Disable GUI and display operations
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = ""

# Disable any interactive mode
import matplotlib
matplotlib.use('Agg')

class DrawingScene(Scene):
    def construct(self):
        self.camera.background_color = "#000000"  # Black background
        
        drawing_data = json.loads('''{drawing_data_json}''')
        
        def create_path_from_commands(commands):
            path = VMobject()
            path.set_stroke(YELLOW, 4)  # More visible color and thicker
            
            current_point = None
            
            for cmd in commands:
                if isinstance(cmd, list):
                    if cmd[0] == 'M':
                        x, y = cmd[1], -cmd[2]
                        current_point = np.array([x, y, 0])
                        path.start_new_path(current_point)
                    
                    elif cmd[0] == 'L' and current_point is not None:
                        x, y = cmd[1], -cmd[2]
                        new_point = np.array([x, y, 0])
                        path.add_line_to(new_point)
                        current_point = new_point
                    
                    elif cmd[0] == 'Q' and current_point is not None:
                        control_x, control_y = cmd[1], -cmd[2]
                        end_x, end_y = cmd[3], -cmd[4]
                        
                        control_point = np.array([control_x, control_y, 0])
                        end_point = np.array([end_x, end_y, 0])
                        
                        path.add_quadratic_bezier(control_point, end_point)
                        current_point = end_point
            
            return path
        
        # Create paths from all objects in the drawing
        all_paths = VGroup()
        for obj in drawing_data["objects"]:
            path = create_path_from_commands(obj["path"])
            all_paths.add(path)
        
        # Center and scale the drawing to fit within the frame
        all_paths.center()
        all_paths.scale_to_fit_width(6)  # Smaller width
        all_paths.scale_to_fit_height(4)  # Smaller height
        
        # Animate the drawing
        self.play(Create(all_paths), run_time=3)
        self.wait(2)
            """)
        
        # Set environment variables to prevent any GUI operations
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["DISPLAY"] = ""
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["XDG_RUNTIME_DIR"] = ""
        
        # Run manim with all GUI and preview operations disabled
        try:
            result = subprocess.run(
                [
                    "python", "-m", "manim",
                    "-qh",
                    "--disable_caching",
                    "--format=mp4",
                    "--write_to_movie",
                    "--renderer=cairo",  # Use cairo renderer which works better headless
                    "--config_file", config_path,
                    temp_scene_path,
                    "DrawingScene",
                    "--media_dir", os.path.join(temp_dir, "media")
                ],
                env=env,
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            
            # For debugging if needed
            if result.returncode != 0:
                print(f"Manim error: {result.stderr}")
        except Exception as e:
            print(f"Error running manim: {e}")
        
        # Find the generated video file
        video_path = None
        for root, dirs, files in os.walk(os.path.join(temp_dir, "media", "videos", "temp_scene", "1080p60")):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    break
        
        # If not found in the expected location, search the entire temp directory
        if not video_path:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        video_path = os.path.join(root, file)
                        break
                if video_path:
                    break
        
        # Copy the video to the working directory
        if video_path and os.path.exists(video_path):
            shutil.copy2(video_path, output_file)
            print(f"Animation saved to: {os.path.abspath(output_file)}")
            return os.path.abspath(output_file)
        else:
            print("Failed to generate animation.")
            return None
            
    finally:
        # Clean up all temporary files
        shutil.rmtree(temp_dir)

def render_fourier_vectors(fourier_coefficients, output_file="fourier_vectors.mp4", num_frames=240, drawing_duration=5, show_progress=True, zoom_factor=1.0):
    """
    Render an animation of Fourier vectors drawing a path
    
    Args:
        fourier_coefficients: List of tuples (frequency, complex_coefficient)
                             ordered by coefficient magnitude (largest first)
        output_file: Output file path for the animation
        num_frames: Number of frames for the animation
        drawing_duration: Duration of the drawing animation in seconds
        show_progress: Whether to show a progress bar in the console
        zoom_factor: Controls how much to zoom out (higher values show more context)
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    # Import tqdm for progress bar if requested
    if show_progress:
        try:
            from tqdm import tqdm
            print("Rendering Fourier animation...")
        except ImportError:
            print("For progress bar, install tqdm: pip install tqdm")
            show_progress = False
    
    # Create a temporary directory for the rendering process
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a temporary scene file with the Fourier coefficients embedded
        temp_scene_path = os.path.join(temp_dir, "temp_scene.py")
        
        # Convert coefficients to a format that can be embedded in the Python script
        if show_progress:
            print("Processing Fourier coefficients...")
            coeffs_data = []
            for freq, coeff in tqdm(fourier_coefficients):
                # Convert tensor to Python complex if needed
                if hasattr(coeff, 'item'):
                    coeff = complex(coeff.item().real, coeff.item().imag)
                coeffs_data.append((freq, coeff.real, coeff.imag))
        else:
            coeffs_data = []
            for freq, coeff in fourier_coefficients:
                # Convert tensor to Python complex if needed
                if hasattr(coeff, 'item'):
                    coeff = complex(coeff.item().real, coeff.item().imag)
                coeffs_data.append((freq, coeff.real, coeff.imag))

        print("Coeffs data: ", coeffs_data)
        coeffs_json = json.dumps(coeffs_data)
        print("Coeffs json: ", coeffs_json)
        
        # Create a custom config file to ensure headless operation
        config_path = os.path.join(temp_dir, "manim.cfg")
        with open(config_path, "w") as f:
            f.write("""
[CLI]
preview = False
show_in_file_browser = False
progress_bar = none
verbosity = ERROR

[renderer]
use_opengl_renderer = False

[window]
enable_gui = False
            """)
        
        if show_progress:
            print("Creating scene file...")
            
        with open(temp_scene_path, "w") as f:
            f.write(f"""
from manim import *
import json
import numpy as np
import os
import sys

# Disable GUI and display operations
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = ""

# Disable any interactive mode
import matplotlib
matplotlib.use('Agg')

class FourierVectorScene(Scene):
    def construct(self):
        # Force a square aspect ratio for the camera
        config.frame_width = config.frame_height
        
        self.camera.background_color = "#000000"  # Black background
        
        # Load Fourier coefficients
        coeffs_data = json.loads('''{coeffs_json}''')
        
        # Convert back to complex numbers
        fourier_coefficients = [(freq, complex(real, imag)) for freq, real, imag in coeffs_data]
        
        # Sort by magnitude (largest first)
        fourier_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create vectors, circles, and path
        vectors = VGroup()
        circles = VGroup()
        
        # Create a dot to trace the path
        trace_dot = Dot(color=YELLOW)
        
        # Create a path to show the drawing
        path = TracedPath(trace_dot.get_center, stroke_width=2, stroke_color=YELLOW)
        
        # Function to calculate the position of each vector
        def get_vector_positions(t):
            positions = []
            current_pos = ORIGIN
            
            for freq, coeff in fourier_coefficients:
                # Calculate the position of this vector
                angle = 2 * np.pi * freq * t
                radius = abs(coeff)
                phase = np.angle(coeff)
                
                # Calculate the end position of this vector
                vector_end = current_pos + np.array([
                    radius * np.cos(angle + phase),
                    radius * np.sin(angle + phase),
                    0
                ])
                
                positions.append((current_pos, vector_end, radius))
                current_pos = vector_end
            
            return positions
        
        # Calculate the path for the entire animation to determine frame size
        num_samples = 100
        path_points = []
        
        for i in range(num_samples):
            t = i / num_samples
            positions = get_vector_positions(t)
            if positions:
                path_points.append(positions[-1][1])  # End position of last vector
        
        # Create a path object to calculate bounds
        preview_path = VMobject()
        preview_path.set_points_as_corners(path_points)
        
        # Create initial vectors and circles
        positions = get_vector_positions(0)
        
        # Color gradient for vectors and circles
        colors = color_gradient([BLUE_D, BLUE_B, BLUE_A, WHITE], len(positions))
        
        for i, (start, end, radius) in enumerate(positions):
            # Create vector
            vector = Arrow(start, end, buff=0, color=colors[i])
            vectors.add(vector)
            
            # Create circle
            circle = Circle(radius=radius, color=colors[i])
            circle.move_to(start)
            circle.set_stroke(opacity=0.5)
            circles.add(circle)
        
        # Position the trace dot at the end of the last vector
        if positions:
            trace_dot.move_to(positions[-1][1])
        
        # Add everything to the scene
        self.add(circles, vectors, trace_dot, path)
        
        # Function to update vectors and circles
        def update_vectors_and_progress(t):
            # Update vectors and circles
            positions = get_vector_positions(t)
            
            for i, (vector, circle) in enumerate(zip(vectors, circles)):
                if i < len(positions):
                    start, end, radius = positions[i]
                    vector.put_start_and_end_on(start, end)
                    circle.move_to(start)
            
            # Update the trace dot position
            if positions:
                trace_dot.move_to(positions[-1][1])
        
        # Animate the drawing
        self.play(
            UpdateFromAlphaFunc(
                VGroup(vectors, circles, trace_dot),
                lambda mob, alpha: update_vectors_and_progress(alpha)
            ),
            run_time={drawing_duration},
            rate_func=linear
        )
        
        self.wait(1)
            """)
        
        # Set environment variables to prevent any GUI operations
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["DISPLAY"] = ""
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["XDG_RUNTIME_DIR"] = ""
        
        # Run manim with all GUI and preview operations disabled
        if show_progress:
            print("Rendering animation with Manim (this may take a while)...")
            
        try:
            result = subprocess.run(
                [
                    "python", "-m", "manim",
                    "-qh",
                    "--disable_caching",
                    "--format=mp4",
                    "--write_to_movie",
                    "--renderer=cairo",  # Use cairo renderer which works better headless
                    "--config_file", config_path,
                    temp_scene_path,
                    "FourierVectorScene",
                    "--media_dir", os.path.join(temp_dir, "media")
                ],
                env=env,
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            
            # For debugging if needed
            if result.returncode != 0:
                print(f"Manim error: {result.stderr}")
        except Exception as e:
            print(f"Error running manim: {e}")
        
        # Find the generated video file
        if show_progress:
            print("Finding and copying output file...")
            
        video_path = None
        for root, dirs, files in os.walk(os.path.join(temp_dir, "media", "videos", "temp_scene", "1080p60")):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    break
        
        # If not found in the expected location, search the entire temp directory
        if not video_path:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        video_path = os.path.join(root, file)
                        break
                if video_path:
                    break
        
        # Copy the video to the working directory
        if video_path and os.path.exists(video_path):
            shutil.copy2(video_path, output_file)
            if show_progress:
                print(f"✅ Fourier vector animation saved to: {os.path.abspath(output_file)}")
            return os.path.abspath(output_file)
        else:
            print("Failed to generate Fourier vector animation.")
            return None
            
    finally:
        # Clean up all temporary files
        if show_progress:
            print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir) 