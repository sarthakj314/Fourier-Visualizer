from manim import *
import numpy as np
import json

class FourierDrawingScene(Scene):
    """
    A Manim scene that renders a drawing using path commands.
    """
    
    def __init__(self, drawing_data=None, **kwargs):
        super().__init__(**kwargs)
        self.drawing_data = drawing_data or {"objects": []}
    
    def construct(self):
        self.camera.background_color = "#000000"  # Black background
        
        # Create paths from all objects in the drawing
        all_paths = self.create_paths_from_drawing()
        
        # Center and scale the drawing to fit within the frame
        all_paths.center()
        all_paths.scale_to_fit_width(6)  # Adjust width
        all_paths.scale_to_fit_height(4)  # Adjust height
        
        # Animate the drawing of the path
        self.play(Create(all_paths), run_time=3)
        self.wait(2)
    
    def create_paths_from_drawing(self):
        """Create a VGroup containing all paths from the drawing data."""
        all_paths = VGroup()
        
        for obj in self.drawing_data["objects"]:
            path = self.create_path_from_commands(obj["path"])
            all_paths.add(path)
            
        return all_paths
    
    def create_path_from_commands(self, commands):
        """Create a VMobject path from drawing commands."""
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
                    
                    # Add quadratic bezier curve
                    path.add_quadratic_bezier(control_point, end_point)
                    current_point = end_point
        
        return path 