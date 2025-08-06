#!/usr/bin/env python3
"""
GUI Interface for Cellular Automaton Framework

This module provides a graphical user interface for:
- Selecting simulation types and parameters
- Drawing initial configurations on a grid
- Launching simulations with custom settings
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import Dict, Any, Optional, Callable
import threading

from simulators import CUDAGameOfLife, CPUGameOfLife, CUDALUTBNN, CUDA3DGameOfLife, make_totalistic_table
from visualizers import OpenCVVisualizer, ConsoleVisualizer, NoVisualizer, OpenCV3DVisualizer, OpenGL3DVisualizer
from engine import SimulationEngine


class CellularAutomatonGUI:
    """Main GUI application for cellular automaton framework."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU-Accelerated Cellular Automaton Framework")
        self.root.geometry("1200x800")
        
        # Configuration state
        self.grid_width = tk.IntVar(value=512)
        self.grid_height = tk.IntVar(value=512)
        self.grid_depth = tk.IntVar(value=128)
        self.max_iterations = tk.IntVar(value=10000)
        self.display_interval = tk.IntVar(value=1)
        
        # Rule configuration
        self.birth_rules = tk.StringVar(value="3")
        self.survival_rules = tk.StringVar(value="2,3")
        
        # Simulation type
        self.sim_type = tk.StringVar(value="Conway 2D")
        self.vis_type = tk.StringVar(value="OpenCV")
        
        # Initial configuration grid
        self.initial_config = None
        self.canvas_scale = 4  # Pixels per cell
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the main GUI layout."""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_controls(control_frame)
        self.setup_canvas(canvas_frame)
    
    def setup_controls(self, parent):
        """Set up the control panel."""
        # Title
        title_label = ttk.Label(parent, text="Cellular Automaton Controls", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Simulation Type Selection
        sim_frame = ttk.LabelFrame(parent, text="Simulation Type", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)
        
        sim_types = [
            "Conway 2D", "HighLife 2D", "Custom 2D", 
            "3D Game of Life", "3D Balanced", "3D Conservative", "3D Moderate"
        ]
        ttk.Label(sim_frame, text="Type:").pack(anchor=tk.W)
        sim_combo = ttk.Combobox(sim_frame, textvariable=self.sim_type, 
                                values=sim_types, state="readonly")
        sim_combo.pack(fill=tk.X, pady=2)
        sim_combo.bind('<<ComboboxSelected>>', self.on_sim_type_changed)
        
        # Visualizer Selection
        vis_frame = ttk.LabelFrame(parent, text="Visualizer", padding=10)
        vis_frame.pack(fill=tk.X, pady=5)
        
        vis_types = ["OpenCV", "Console", "OpenCV 3D", "OpenGL 3D"]
        ttk.Label(vis_frame, text="Visualizer:").pack(anchor=tk.W)
        vis_combo = ttk.Combobox(vis_frame, textvariable=self.vis_type, 
                                values=vis_types, state="readonly")
        vis_combo.pack(fill=tk.X, pady=2)
        
        # Grid Parameters
        grid_frame = ttk.LabelFrame(parent, text="Grid Parameters", padding=10)
        grid_frame.pack(fill=tk.X, pady=5)
        
        # Width
        ttk.Label(grid_frame, text="Width:").pack(anchor=tk.W)
        width_spin = ttk.Spinbox(grid_frame, from_=64, to=2048, increment=64,
                                textvariable=self.grid_width, command=self.update_canvas_size)
        width_spin.pack(fill=tk.X, pady=2)
        
        # Height
        ttk.Label(grid_frame, text="Height:").pack(anchor=tk.W)
        height_spin = ttk.Spinbox(grid_frame, from_=64, to=2048, increment=64,
                                 textvariable=self.grid_height, command=self.update_canvas_size)
        height_spin.pack(fill=tk.X, pady=2)
        
        # Depth (for 3D)
        ttk.Label(grid_frame, text="Depth (3D only):").pack(anchor=tk.W)
        depth_spin = ttk.Spinbox(grid_frame, from_=64, to=512, increment=64,
                                textvariable=self.grid_depth)
        depth_spin.pack(fill=tk.X, pady=2)
        
        # Rule Configuration
        rules_frame = ttk.LabelFrame(parent, text="Rules (Custom 2D)", padding=10)
        rules_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(rules_frame, text="Birth rules (comma-separated):").pack(anchor=tk.W)
        birth_entry = ttk.Entry(rules_frame, textvariable=self.birth_rules)
        birth_entry.pack(fill=tk.X, pady=2)
        
        ttk.Label(rules_frame, text="Survival rules (comma-separated):").pack(anchor=tk.W)
        survival_entry = ttk.Entry(rules_frame, textvariable=self.survival_rules)
        survival_entry.pack(fill=tk.X, pady=2)
        
        # Simulation Parameters
        params_frame = ttk.LabelFrame(parent, text="Simulation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Max Iterations:").pack(anchor=tk.W)
        iter_spin = ttk.Spinbox(params_frame, from_=100, to=100000, increment=1000,
                               textvariable=self.max_iterations)
        iter_spin.pack(fill=tk.X, pady=2)
        
        ttk.Label(params_frame, text="Display Interval:").pack(anchor=tk.W)
        display_spin = ttk.Spinbox(params_frame, from_=1, to=10, increment=1,
                                  textvariable=self.display_interval)
        display_spin.pack(fill=tk.X, pady=2)
        
        # Action Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="Clear Grid", 
                  command=self.clear_grid).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Random Fill", 
                  command=self.random_fill).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Load Pattern", 
                  command=self.load_pattern).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Pattern", 
                  command=self.save_pattern).pack(fill=tk.X, pady=2)
        
        # Launch button
        launch_button = ttk.Button(button_frame, text="Launch Simulation", 
                                  command=self.launch_simulation,
                                  style="Accent.TButton")
        launch_button.pack(fill=tk.X, pady=(20, 2))
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(parent, textvariable=self.status_var, 
                                foreground="blue")
        status_label.pack(pady=10)
    
    def setup_canvas(self, parent):
        """Set up the drawing canvas for initial configuration."""
        canvas_label = ttk.Label(parent, text="Initial Configuration (Click to draw)", 
                                font=("Arial", 12, "bold"))
        canvas_label.pack(pady=(0, 10))
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="white", 
                               scrollregion=(0, 0, 2048, 2048))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, 
                                   command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, 
                                   command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, 
                             xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_cell)
        self.canvas.bind("<Button-3>", self.erase_cell)  # Right click to erase
        self.canvas.bind("<B3-Motion>", self.erase_cell)
        
        # Initialize grid
        self.update_canvas_size()
    
    def update_canvas_size(self):
        """Update canvas size based on grid dimensions."""
        width = self.grid_width.get()
        height = self.grid_height.get()
        
        # Update canvas scroll region
        canvas_width = width * self.canvas_scale
        canvas_height = height * self.canvas_scale
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # Initialize or resize the grid
        self.initial_config = np.zeros((height, width), dtype=np.uint8)
        self.redraw_grid()
    
    def redraw_grid(self):
        """Redraw the entire grid on the canvas."""
        self.canvas.delete("all")
        
        height, width = self.initial_config.shape
        
        # Draw grid lines (light gray)
        for i in range(0, width + 1, 8):  # Every 8th line
            x = i * self.canvas_scale
            self.canvas.create_line(x, 0, x, height * self.canvas_scale, 
                                   fill="lightgray", width=1)
        
        for i in range(0, height + 1, 8):  # Every 8th line
            y = i * self.canvas_scale
            self.canvas.create_line(0, y, width * self.canvas_scale, y, 
                                   fill="lightgray", width=1)
        
        # Draw living cells
        for y in range(height):
            for x in range(width):
                if self.initial_config[y, x] == 1:
                    self.draw_cell_at(x, y, "black")
    
    def draw_cell_at(self, x: int, y: int, color: str = "black"):
        """Draw a cell at grid coordinates (x, y)."""
        x1 = x * self.canvas_scale
        y1 = y * self.canvas_scale
        x2 = x1 + self.canvas_scale
        y2 = y1 + self.canvas_scale
        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color,
                                    tags=f"cell_{x}_{y}")
    
    def start_draw(self, event):
        """Start drawing cells."""
        self.draw_cell(event)
    
    def draw_cell(self, event):
        """Draw a cell at mouse position."""
        x = self.canvas.canvasx(event.x) // self.canvas_scale
        y = self.canvas.canvasy(event.y) // self.canvas_scale
        
        if 0 <= x < self.grid_width.get() and 0 <= y < self.grid_height.get():
            self.initial_config[int(y), int(x)] = 1
            self.draw_cell_at(int(x), int(y), "black")
    
    def erase_cell(self, event):
        """Erase a cell at mouse position."""
        x = self.canvas.canvasx(event.x) // self.canvas_scale
        y = self.canvas.canvasy(event.y) // self.canvas_scale
        
        if 0 <= x < self.grid_width.get() and 0 <= y < self.grid_height.get():
            self.initial_config[int(y), int(x)] = 0
            # Remove the cell from canvas
            self.canvas.delete(f"cell_{int(x)}_{int(y)}")
    
    def clear_grid(self):
        """Clear all cells from the grid."""
        self.initial_config.fill(0)
        self.redraw_grid()
        self.status_var.set("Grid cleared")
    
    def random_fill(self):
        """Randomly fill the grid with living cells."""
        density = 0.1  # 10% density
        self.initial_config = np.random.choice([0, 1], 
                                              size=self.initial_config.shape, 
                                              p=[1-density, density])
        self.redraw_grid()
        self.status_var.set("Grid randomly filled")
    
    def load_pattern(self):
        """Load a pattern from file."""
        filename = filedialog.askopenfilename(
            title="Load Pattern",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            try:
                pattern = np.load(filename)
                if pattern.ndim == 2:
                    # Resize grid if necessary
                    h, w = pattern.shape
                    self.grid_height.set(h)
                    self.grid_width.set(w)
                    self.update_canvas_size()
                    self.initial_config = pattern.astype(np.uint8)
                    self.redraw_grid()
                    self.status_var.set(f"Pattern loaded: {filename}")
                else:
                    messagebox.showerror("Error", "Pattern must be 2D array")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pattern: {e}")
    
    def save_pattern(self):
        """Save current pattern to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Pattern",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            try:
                np.save(filename, self.initial_config)
                self.status_var.set(f"Pattern saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save pattern: {e}")
    
    def on_sim_type_changed(self, event=None):
        """Handle simulation type change."""
        sim_type = self.sim_type.get()
        
        # Update default visualizer based on simulation type
        if "3D" in sim_type:
            self.vis_type.set("OpenCV 3D")
        else:
            self.vis_type.set("OpenCV")
        
        # Update default rules
        if sim_type == "Conway 2D":
            self.birth_rules.set("3")
            self.survival_rules.set("2,3")
        elif sim_type == "HighLife 2D":
            self.birth_rules.set("3,6")
            self.survival_rules.set("2,3")
    
    def parse_rules(self, rule_string: str) -> frozenset:
        """Parse comma-separated rule string into frozenset."""
        try:
            rules = [int(x.strip()) for x in rule_string.split(",") if x.strip()]
            return frozenset(rules)
        except ValueError:
            return frozenset()
    
    def launch_simulation(self):
        """Launch the simulation with current settings."""
        try:
            self.status_var.set("Launching simulation...")
            
            # Create simulator
            sim_type = self.sim_type.get()
            width = self.grid_width.get()
            height = self.grid_height.get()
            depth = self.grid_depth.get()
            
            # Parse rules for custom simulations
            birth_rules = frozenset()
            survival_rules = frozenset()
            if 'Custom' in sim_type:
                try:
                    birth_rules = frozenset(map(int, self.birth_rules.get().split(','))) if self.birth_rules.get().strip() else frozenset({3})
                    survival_rules = frozenset(map(int, self.survival_rules.get().split(','))) if self.survival_rules.get().strip() else frozenset({2, 3})
                except ValueError:
                    birth_rules = frozenset({3})
                    survival_rules = frozenset({2, 3})
            
            simulator = None
            try:
                if sim_type == "Conway 2D":
                    simulator = CUDAGameOfLife(width, height, birth_rules=frozenset({3}), survival_rules=frozenset({2, 3}))
                elif sim_type == "Custom 2D":
                    simulator = CUDAGameOfLife(width, height, birth_rules=birth_rules, survival_rules=survival_rules)
                elif sim_type == "3D Game of Life":
                    simulator = CUDA3DGameOfLife(width, height, depth, birth_rules=frozenset({3}), survival_rules=frozenset({2, 3}))
                elif sim_type == "3D Balanced":
                    simulator = CUDA3DGameOfLife(width, height, depth, birth_rules=frozenset({7}), survival_rules=frozenset({5, 6}))
                elif sim_type == "3D Conservative":
                    simulator = CUDA3DGameOfLife(width, height, depth, birth_rules=frozenset({8}), survival_rules=frozenset({6, 7}))
                elif sim_type == "3D Moderate":
                    simulator = CUDA3DGameOfLife(width, height, depth, birth_rules=frozenset({6, 7}), survival_rules=frozenset({5, 6}))
                else:
                    # Default fallback
                    simulator = CUDAGameOfLife(width, height)
            except Exception as cuda_error:
                # CUDA failed, fall back to CPU
                print(f"CUDA initialization failed: {cuda_error}")
                print("Falling back to CPU simulation...")
                self.status_var.set("CUDA failed, using CPU...")
                
                if '3D' in sim_type:
                    messagebox.showwarning("CUDA Error", 
                                         f"CUDA failed: {cuda_error}\n\n3D simulations require GPU. Please try 2D simulations or fix CUDA setup.")
                    self.status_var.set("Ready")
                    return
                else:
                    # Use CPU fallback for 2D
                    if sim_type == "Conway 2D":
                        simulator = CPUGameOfLife(width, height, birth_rules=frozenset({3}), survival_rules=frozenset({2, 3}))
                    elif sim_type == "Custom 2D":
                        simulator = CPUGameOfLife(width, height, birth_rules=birth_rules, survival_rules=survival_rules)
                    else:
                        simulator = CPUGameOfLife(width, height)
            
            # Create visualizer
            vis_type = self.vis_type.get()
            if vis_type in ("OpenCV", "OpenCV 3D"):
                # Choose correct visualizer
                if vis_type == "OpenCV":
                    visualizer = OpenCVVisualizer(width, height)
                else:  # "OpenCV 3D"
                    visualizer = OpenCV3DVisualizer(width, height, depth)
                # Apply initial configuration to simulator BEFORE running
                if not '3D' in sim_type and self.initial_config is not None:
                    if self.initial_config.shape != (height, width):
                        new_state = np.zeros((height, width), dtype=np.uint8)
                        copy_h = min(height, self.initial_config.shape[0])
                        copy_w = min(width, self.initial_config.shape[1])
                        new_state[:copy_h, :copy_w] = self.initial_config[:copy_h, :copy_w]
                        simulator.reset(new_state)
                    else:
                        simulator.reset(self.initial_config)
                # Run simulation in the MAIN thread because OpenCV windows
                # don't display correctly from background threads.
                # Temporarily hide/minimise the GUI so it doesn't appear frozen.
                self.root.withdraw()
                self.status_var.set("Simulation running (OpenCV window)... Press 'q' in the OpenCV window to quit.")
                try:
                    engine = SimulationEngine(simulator, visualizer, self.display_interval.get())
                    engine.run(max_iterations=self.max_iterations.get())
                finally:
                    self.root.deiconify()
                    self.status_var.set("Ready")
                return  # Nothing else to schedule – we're done
            elif vis_type == "Console":
                visualizer = ConsoleVisualizer(width, height)
            elif vis_type == "OpenGL 3D":
                try:
                    visualizer = OpenGL3DVisualizer(width, height, depth)
                except Exception as gl_error:
                    print(f"OpenGL failed: {gl_error}")
                    messagebox.showwarning("OpenGL Error", 
                                         f"3D visualization failed: {gl_error}\nFalling back to 2D visualization.")
                    visualizer = OpenCVVisualizer(width, height)
            else:
                visualizer = OpenCVVisualizer(width, height)
            
            # Create engine and run simulation in thread
            engine = SimulationEngine(simulator, visualizer, self.display_interval.get())
            
            def run_simulation():
                try:
                    engine.run(max_iterations=self.max_iterations.get())
                    self.status_var.set("Simulation completed")
                except Exception as e:
                    print(f"Simulation error: {e}")
                    # Update status in main thread
                    self.root.after(0, lambda: self.status_var.set(f"Simulation error: {e}"))
                finally:
                    # Reset status in main thread
                    self.root.after(0, lambda: self.status_var.set("Ready"))
            
            sim_thread = threading.Thread(target=run_simulation, daemon=True)
            sim_thread.start()
            
            self.status_var.set("Simulation running...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch simulation: {e}")
            self.status_var.set("Ready")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for GUI application."""
    app = CellularAutomatonGUI()
    app.run()


if __name__ == "__main__":
    main()
