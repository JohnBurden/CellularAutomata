"""
Cellular Automaton Visualizers

This module contains various visualization implementations for cellular automata:
- BaseVisualizer: Abstract base class for all visualizers
- OpenCVVisualizer: Real-time OpenCV window display
- ConsoleVisualizer: Simple console statistics output
- NoVisualizer: No visualization for benchmarking
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple

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
