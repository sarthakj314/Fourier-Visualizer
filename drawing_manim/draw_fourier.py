from manim import *
import numpy as np
import json
from render_utils import render_drawing


class FourierDrawing(Scene):
    def construct(self):
        self.camera.background_color = "#000000"  # Darker background
        
        # Function to parse and render a drawing path
        def create_path_from_commands(commands):
            path = VMobject()
            path.set_stroke(YELLOW, 4)  # More visible color and thicker
            
            current_point = None
            
            for cmd in commands:
                if isinstance(cmd, list):
                    if cmd[0] == 'M':  # Move command
                        x, y = cmd[1], -cmd[2]  # Negate y to match canvas coordinates
                        current_point = np.array([x, y, 0])
                        path.start_new_path(current_point)
                    
                    elif cmd[0] == 'L' and current_point is not None:  # Line command
                        x, y = cmd[1], -cmd[2]  # Negate y to match canvas coordinates
                        new_point = np.array([x, y, 0])
                        path.add_line_to(new_point)
                        current_point = new_point
                    
                    elif cmd[0] == 'Q' and current_point is not None:  # Quadratic bezier
                        control_x, control_y = cmd[1], -cmd[2]  # Control point
                        end_x, end_y = cmd[3], -cmd[4]  # End point
                        
                        control_point = np.array([control_x, control_y, 0])
                        end_point = np.array([end_x, end_y, 0])
                        
                        # Use add_quadratic_bezier_curve_to for quadratic bezier curves
                        path.add_quadratic_bezier_curve_to(control_point, end_point)
                        current_point = end_point
            
            return path
        
        # Create the path from the drawing commands
        for obj in self.drawing_data["objects"]:
            path = create_path_from_commands(obj["path"])
            
            # Center and scale the path to fit the screen
            path.center()
            path.scale(3)
            
            # Animate the drawing of the path
            self.play(Create(path), run_time=3)
            self.wait(1)
        
        self.wait(2)


def load_drawing_from_json(json_file_path):
    """
    Load a drawing from a JSON file exported from Streamlit's canvas
    """
    with open(json_file_path, 'r') as f:
        drawing_data = json.load(f)
    return drawing_data


def render_drawing(drawing_data, output_file="drawing_animation.mp4"):
    """
    Render a drawing animation from drawing data
    
    Args:
        drawing_data: JSON data from Streamlit's canvas
        output_file: Output file path for the animation
    """
    import os
    import shutil
    import tempfile
    
    # Create a temporary directory for manim files
    temp_dir = tempfile.mkdtemp()
    temp_scene_path = os.path.join(temp_dir, "temp_scene.py")
    
    # Convert drawing data to JSON string first
    drawing_data_json = json.dumps(drawing_data)
    
    # Create a temporary scene file
    with open(temp_scene_path, "w") as f:
        f.write(f"""
from manim import *
import json

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
                        
                        path.add_quadratic_bezier_curve_to(control_point, end_point)
                        current_point = end_point
            
            return path
        
        # Create paths from all objects in the drawing
        all_paths = VGroup()
        for obj in drawing_data["objects"]:
            path = create_path_from_commands(obj["path"])
            all_paths.add(path)
        
        # Center and scale the drawing to fit within the frame
        all_paths.center()
        # Scale to fit within the visible area with some margin
        all_paths.scale_to_fit_width(6)  # Smaller width
        all_paths.scale_to_fit_height(4)  # Smaller height
        
        # Animate the drawing
        self.play(Create(all_paths), run_time=3)
        self.wait(2)
        """)
    
    # Run manim to render the scene in the temp directory
    os.system(f"cd {temp_dir} && manim -pqh temp_scene.py DrawingScene --media_dir {temp_dir}/media")
    
    # Find the generated video file
    video_path = None
    for root, dirs, files in os.walk(os.path.join(temp_dir, "media", "videos", "temp_scene", "1080p60")):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                break
    
    # Copy the video to the working directory
    if video_path and os.path.exists(video_path):
        shutil.copy2(video_path, output_file)
        print(f"Animation saved to: {os.path.abspath(output_file)}")
    else:
        print("Failed to generate animation.")
    
    # Clean up all temporary files
    shutil.rmtree(temp_dir)


def generate_fourier_drawing_video(drawing_data, output_file="fourier_drawing.mp4"):
    """
    Generate a video of a drawing using the FourierDrawingScene
    
    Args:
        drawing_data: JSON data from Streamlit's canvas or a dictionary with path commands
        output_file: Output file path for the animation
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    render_drawing(drawing_data, output_file)
    return output_file


def generate_fourier_vector_video(fourier_coefficients, output_file="fourier_vectors.mp4", num_frames=240, drawing_duration=5, show_progress=True, zoom_factor=1.0):
    """
    Generate a video of Fourier vectors drawing a path
    
    Args:
        fourier_coefficients: List of tuples (frequency, complex_coefficient)
                             ordered by coefficient magnitude (largest first)
        output_file: Output file path for the animation
        num_frames: Number of frames for the animation
        drawing_duration: Duration of the drawing animation in seconds
        show_progress: Whether to show a progress bar
        zoom_factor: Controls how much to zoom out (higher values show more context)
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    from render_utils import render_fourier_vectors
    return render_fourier_vectors(
        fourier_coefficients, 
        output_file, 
        num_frames, 
        drawing_duration,
        show_progress,
        zoom_factor
    )


if __name__ == "__main__":
    # Example usage with a properly sized shape
    sample_drawing = {
        "objects": [
            {
                "path": [
                    ['M', -2, -2],
                    ['L', 2, -2],
                    ['L', 2, 2],
                    ['L', -2, 2],
                    ['L', -2, -2]
                ]
            }
        ]
    }
    
    # Generate a video of the drawing
    output_path = generate_fourier_drawing_video(sample_drawing, "square_drawing.mp4")
    print(f"Video saved to: {output_path}")
    
    # Generate a sample Fourier series and create a vector animation
    import numpy as np
    
    # Create sample Fourier coefficients (a simple circle)
    coefficients = []
    # Add the fundamental frequency (n=1)
    coefficients.append((1, complex(0, 1)))  # Coefficient for e^(2πi*t)
    # Add some harmonics with decreasing magnitudes
    for n in range(2, 10):
        magnitude = 1.0 / n  # Decreasing magnitude
        phase = np.random.uniform(0, 2*np.pi)  # Random phase
        coefficients.append((n, complex(magnitude * np.cos(phase), magnitude * np.sin(phase))))
        coefficients.append((-n, complex(magnitude * np.cos(-phase), magnitude * np.sin(-phase))))
    
    # Sort by magnitude (largest first)
    coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Generate the Fourier vector animation
    fourier_output = generate_fourier_vector_video(coefficients, "fourier_vectors_demo.mp4")
    print(f"Fourier vector animation saved to: {fourier_output}")
