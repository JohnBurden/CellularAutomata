"""
Cellular Automaton Visualizers

This module contains various visualization implementations for cellular automata:
- BaseVisualizer: Abstract base class for all visualizers
- OpenCVVisualizer: Real-time OpenCV window display
- ConsoleVisualizer: Simple console statistics output
- NoVisualizer: No visualization for benchmarking
- OpenCV3DVisualizer: OpenCV-based 3D visualizer showing slices of 3D grid
- OpenGL3DVisualizer: True volumetric 3D visualization using OpenGL
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL dependencies not available. Install with: pip install pygame PyOpenGL")

# Abstract Base Classes
class BaseVisualizer(ABC):
    """Abstract base class for different visualization methods."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    @abstractmethod
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """Update visualization with current state. Returns False to stop simulation."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up visualization resources."""
        pass

# Concrete Visualizer Implementations
class OpenCVVisualizer(BaseVisualizer):
    """OpenCV-based real-time visualizer."""
    
    def __init__(self, width: int, height: int, display_size: Tuple[int, int] = (1024, 1024), 
                 window_name: str = "Game of Life"):
        super().__init__(width, height)
        self.display_size = display_size
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """Update OpenCV display."""
        # Convert to 8-bit grayscale (0 or 255)
        display_state = (state * 255).astype(np.uint8)
        
        # Resize for display if needed
        if self.display_size != (self.height, self.width):
            display_state = cv2.resize(display_state, self.display_size, interpolation=cv2.INTER_NEAREST)
        
        # Add FPS and iteration info
        display_state = cv2.cvtColor(display_state, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_state, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_state, f"Iteration: {iteration}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the image
        cv2.imshow(self.window_name, display_state)
        
        # Check for 'q' key press to quit
        return cv2.waitKey(1) & 0xFF != ord('q')
    
    def cleanup(self) -> None:
        """Clean up OpenCV resources."""
        cv2.destroyAllWindows()

class ConsoleVisualizer(BaseVisualizer):
    """Simple console-based visualizer."""
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
    
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """Print basic stats to console."""
        alive_count = np.sum(state)
        total_cells = state.size
        density = alive_count / total_cells * 100
        
        print(f"Iteration {iteration:6d} | FPS: {fps:6.1f} | "
              f"Alive: {alive_count:6d}/{total_cells} ({density:5.1f}%)")
        
        return True  # Never stop from console
    
    def cleanup(self) -> None:
        """No cleanup needed for console."""
        pass

class NoVisualizer(BaseVisualizer):
    """No visualization - for benchmarking."""
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
    
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """No visualization."""
        return True  # Never stop
    
    def cleanup(self) -> None:
        """No cleanup needed."""
        pass

class OpenCV3DVisualizer(BaseVisualizer):
    """OpenCV-based 3D visualizer showing slices of 3D grid."""
    
    def __init__(self, width: int, height: int, depth: int, 
                 display_size: Tuple[int, int] = (1024, 1024), 
                 window_name: str = "3D Game of Life",
                 slices_per_row: int = 4):
        super().__init__(width, height)
        self.depth = depth
        self.display_size = display_size
        self.window_name = window_name
        self.slices_per_row = slices_per_row
        self.current_slice = depth // 2  # Start at middle slice
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """Update 3D visualization showing multiple slices."""
        # Calculate slice display layout
        rows = (self.depth + self.slices_per_row - 1) // self.slices_per_row
        slice_width = self.display_size[0] // self.slices_per_row
        slice_height = self.display_size[1] // rows
        
        # Create composite image
        composite = np.zeros((self.display_size[1], self.display_size[0]), dtype=np.uint8)
        
        for z in range(self.depth):
            # Get 2D slice at depth z
            slice_2d = state[z, :, :]
            
            # Convert to display format
            display_slice = (slice_2d * 255).astype(np.uint8)
            
            # Resize slice to fit in grid
            if (slice_height, slice_width) != (self.height, self.width):
                display_slice = cv2.resize(display_slice, (slice_width, slice_height), 
                                         interpolation=cv2.INTER_NEAREST)
            
            # Calculate position in composite image
            row = z // self.slices_per_row
            col = z % self.slices_per_row
            y_start = row * slice_height
            y_end = min(y_start + slice_height, self.display_size[1])
            x_start = col * slice_width
            x_end = min(x_start + slice_width, self.display_size[0])
            
            # Place slice in composite
            composite[y_start:y_end, x_start:x_end] = display_slice[:y_end-y_start, :x_end-x_start]
        
        # Convert to color for text overlay
        display_image = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
        
        # Add FPS and iteration info
        cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f"Iteration: {iteration}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f"3D Grid: {self.width}x{self.height}x{self.depth}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f"Showing all {self.depth} slices", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add grid lines to separate slices
        for i in range(1, self.slices_per_row):
            x = i * slice_width
            cv2.line(display_image, (x, 0), (x, self.display_size[1]), (100, 100, 100), 1)
        for i in range(1, rows):
            y = i * slice_height
            cv2.line(display_image, (0, y), (self.display_size[0], y), (100, 100, 100), 1)
        
        # Show the image
        cv2.imshow(self.window_name, display_image)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('n'):  # Next slice (for future single-slice mode)
            self.current_slice = (self.current_slice + 1) % self.depth
        elif key == ord('p'):  # Previous slice (for future single-slice mode)
            self.current_slice = (self.current_slice - 1) % self.depth
        
        return True
    
    def cleanup(self) -> None:
        """Clean up OpenCV resources."""
        cv2.destroyAllWindows()

class OpenGL3DVisualizer(BaseVisualizer):
    """True volumetric 3D visualizer using OpenGL - heavily optimized for performance."""
    
    def __init__(self, width: int, height: int, depth: int, 
                 display_size: Tuple[int, int] = (1024, 768),
                 window_name: str = "3D Game of Life - Volumetric",
                 downsample_factor: int = 4,  # Increased from 4 to 8 for better performance
                 max_points: int = 10000):    # Limit max points to render
        if not OPENGL_AVAILABLE:
            raise ImportError("OpenGL dependencies not available. Install with: pip install pygame PyOpenGL")
        
        super().__init__(width, height)
        self.depth = depth
        self.display_size = display_size
        self.window_name = window_name
        self.downsample_factor = downsample_factor
        self.max_points = max_points
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -max(width, height, depth) * 1.5
        self.auto_rotate = True
        
        # Calculate downsampled dimensions
        self.vis_width = width // downsample_factor
        self.vis_height = height // downsample_factor
        self.vis_depth = depth // downsample_factor
        
        print(f"OpenGL 3D Visualizer Configuration:")
        print(f"  Original grid: {width}×{height}×{depth} = {width*height*depth:,} cells")
        print(f"  Downsampled: {self.vis_width}×{self.vis_height}×{self.vis_depth} = {self.vis_width*self.vis_height*self.vis_depth:,} cells")
        print(f"  Downsample factor: {downsample_factor}x")
        print(f"  Max points to render: {max_points:,}")
        
        self._init_opengl()
    
    def _init_opengl(self):
        """Initialize OpenGL context and settings."""
        pygame.init()
        pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption(self.window_name)
        
        # Enable depth testing and point sprites
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        # Set up perspective projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 0.1, 1000.0)
        
        # Set up model view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        
        # Set clear color to black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Set point size
        glPointSize(4.0)  # Slightly larger points
    
    def _downsample_state_fast(self, state: np.ndarray) -> np.ndarray:
        """Fast downsampling using NumPy operations instead of loops."""
        # Reshape and use max pooling with NumPy
        # This is much faster than nested loops
        
        # Calculate actual dimensions we can downsample
        d_max = (self.depth // self.downsample_factor) * self.downsample_factor
        h_max = (self.height // self.downsample_factor) * self.downsample_factor
        w_max = (self.width // self.downsample_factor) * self.downsample_factor
        
        # Crop to divisible dimensions
        cropped = state[:d_max, :h_max, :w_max]
        
        # Reshape for max pooling
        reshaped = cropped.reshape(
            self.vis_depth, self.downsample_factor,
            self.vis_height, self.downsample_factor,
            self.vis_width, self.downsample_factor
        )
        
        # Max pool across the downsample dimensions
        downsampled = np.max(reshaped, axis=(1, 3, 5))
        
        return downsampled.astype(np.uint8)
    
    def update(self, state: np.ndarray, fps: float, iteration: int) -> bool:
        """Update 3D volumetric visualization with aggressive optimizations."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.auto_rotate = not self.auto_rotate
                elif event.key == pygame.K_w:  # Zoom in
                    glTranslatef(0.0, 0.0, 20.0)
                elif event.key == pygame.K_s:  # Zoom out
                    glTranslatef(0.0, 0.0, -20.0)
                elif event.key == pygame.K_UP:
                    glRotatef(10, 1, 0, 0)
                elif event.key == pygame.K_DOWN:
                    glRotatef(-10, 1, 0, 0)
                elif event.key == pygame.K_LEFT:
                    glRotatef(10, 0, 1, 0)
                elif event.key == pygame.K_RIGHT:
                    glRotatef(-10, 0, 1, 0)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    glPointSize(min(10.0, glGetFloatv(GL_POINT_SIZE)[0] + 1.0))
                elif event.key == pygame.K_MINUS:
                    glPointSize(max(1.0, glGetFloatv(GL_POINT_SIZE)[0] - 1.0))
        
        # Auto-rotation (faster)
        if self.auto_rotate:
            glRotatef(1.0, 1, 1, 0)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Fast downsampling
        vis_state = self._downsample_state_fast(state)
        
        # Calculate scaling to center the grid
        scale_x = 4.0 / self.vis_width
        scale_y = 4.0 / self.vis_height  
        scale_z = 4.0 / self.vis_depth
        
        # Get living cell positions using NumPy (much faster)
        alive_indices = np.where(vis_state == 1)
        
        if len(alive_indices[0]) > 0:
            # Convert to world coordinates using vectorized operations
            world_z = (alive_indices[0] - self.vis_depth/2) * scale_z
            world_y = (alive_indices[1] - self.vis_height/2) * scale_y
            world_x = (alive_indices[2] - self.vis_width/2) * scale_x
            
            # Limit number of points for performance
            num_points = min(len(world_x), self.max_points)
            if num_points < len(world_x):
                # Randomly sample points if too many
                indices = np.random.choice(len(world_x), num_points, replace=False)
                world_x = world_x[indices]
                world_y = world_y[indices]
                world_z = world_z[indices]
            
            # Batch render all points at once
            glColor4f(1.0, 1.0, 1.0, 0.9)
            glBegin(GL_POINTS)
            for i in range(num_points):
                glVertex3f(world_x[i], world_y[i], world_z[i])
            glEnd()
            
            alive_count = num_points
        else:
            alive_count = 0
        
        # Render text overlay less frequently
        if iteration % 60 == 0:  # Print every 60 iterations
            print(f"Iteration: {iteration:6d} | FPS: {fps:6.1f} | "
                  f"Alive: {alive_count:6d}/{self.max_points} (downsampled {self.downsample_factor}x) | "
                  f"Controls: SPACE=rotate, WASD=move, +/-=point size, Q=quit")
        
        pygame.display.flip()
        
        return True
    
    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        pygame.quit()
