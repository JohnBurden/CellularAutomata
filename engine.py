"""
Simulation Engine

This module contains the SimulationEngine class that coordinates between
simulators and visualizers to run cellular automaton simulations.
"""

import time
from simulators import BaseSimulator
from visualizers import BaseVisualizer

class SimulationEngine:
    """Coordinates between simulator and visualizer."""
    
    def __init__(self, simulator: BaseSimulator, visualizer: BaseVisualizer, 
                 display_interval: int = 5):
        """
        Initialize simulation engine.
        
        Args:
            simulator: The cellular automaton simulator to use
            visualizer: The visualizer to display results
            display_interval: Update display every N iterations (for performance)
        """
        self.simulator = simulator
        self.visualizer = visualizer
        self.display_interval = display_interval
    
    def run(self, max_iterations: int = 10000) -> None:
        """
        Run the simulation.
        
        Args:
            max_iterations: Maximum number of iterations to run
        """
        print(f"Starting simulation with {type(self.simulator).__name__} and {type(self.visualizer).__name__}")
        print(f"Grid size: {self.simulator.width}x{self.simulator.height}")
        print(f"Display interval: {self.display_interval}")
        print("Press 'q' in OpenCV window to quit early")
        
        try:
            start_time = time.time()
            last_display_time = start_time
            frame_count = 0
            
            for iteration in range(max_iterations):
                # Perform simulation step
                self.simulator.step()
                frame_count += 1
                
                # Update visualization at specified intervals
                if iteration % self.display_interval == 0:
                    current_time = time.time()
                    elapsed_time = current_time - last_display_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Get current state and update visualization
                    state = self.simulator.get_state()
                    should_continue = self.visualizer.update(state, fps, iteration)
                    
                    if not should_continue:
                        break
                    
                    # Reset timing for next interval
                    last_display_time = current_time
                    frame_count = 0
        
        finally:
            self.visualizer.cleanup()
            print("Simulation completed")
