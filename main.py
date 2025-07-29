#!/usr/bin/env python3
"""
GPU-Accelerated Cellular Automaton Main Runner

This script demonstrates various cellular automaton configurations using
the modular simulator and visualizer architecture.

Examples:
- Conway's Game of Life
- HighLife variant
- Custom explosive growth rules
- Different visualization options
"""

from simulators import CUDAGameOfLife, CPUGameOfLife, CUDALUTBNN, make_totalistic_table
from visualizers import OpenCVVisualizer, ConsoleVisualizer, NoVisualizer
from engine import SimulationEngine

def conway_example():
    """Standard Conway's Game of Life (B3/S23)."""
    print("=== Conway's Game of Life (B3/S23) ===")
    WIDTH, HEIGHT = 512, 512
    
    sim = CUDAGameOfLife.conway(WIDTH, HEIGHT)
    vis = OpenCVVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=1)
    engine.run(max_iterations=10_000)

def highlife_example():
    """HighLife variant (B36/S23) - has replicators."""
    print("=== HighLife (B36/S23) ===")
    WIDTH, HEIGHT = 512, 512
    
    sim = CUDAGameOfLife.highlife(WIDTH, HEIGHT)
    vis = OpenCVVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=1)
    engine.run(max_iterations=10_000)

def explosive_growth_example():
    """Custom explosive growth rules using CUDALUTBNN."""
    print("=== Explosive Growth (B123/S01234) ===")
    WIDTH, HEIGHT = 256, 256  # Smaller grid for explosive rules
    
    # Von Neumann neighborhood + center
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    
    # Create lookup table for explosive growth
    tbl = make_totalistic_table(
        neigh,
        birth_set={3, 4},        # Born with 1, 2, or 3 neighbors
        survival_set={0, 1, 2, 3, 4},  # Never die
        include_center=True
    )
    
    sim = CUDALUTBNN(WIDTH, HEIGHT, neigh, tbl)
    vis = OpenCVVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=2)
    engine.run(max_iterations=1_000)  # Shorter run for explosive rules

def console_benchmark_example():
    """Benchmark different simulators with console output."""
    print("=== Performance Benchmark ===")
    WIDTH, HEIGHT = 1024, 1024
    
    # Test CUDA simulator
    print("Testing CUDA simulator...")
    sim = CUDAGameOfLife.conway(WIDTH, HEIGHT)
    vis = ConsoleVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=10)
    engine.run(max_iterations=100)
    
    print("\nTesting CPU simulator...")
    sim = CPUGameOfLife(WIDTH, HEIGHT)
    vis = ConsoleVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=10)
    engine.run(max_iterations=100)

def lut_conway_example():
    """Conway's Life using lookup table approach - built manually."""
    print("=== Conway's Life (Lookup Table Implementation - Manual) ===")
    WIDTH, HEIGHT = 512, 512
    
    # Manually build 8-neighbor Moore neighborhood (including center)
    neigh = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    print(f"Neighborhood: {neigh}")
    print(f"Center cell at index: {neigh.index((0, 0))}")
    
    # Create lookup table for Conway's B3/S23 rules
    tbl = make_totalistic_table(
        neigh,
        birth_set={3},          # Born with exactly 3 neighbors
        survival_set={2, 3},    # Survive with 2 or 3 neighbors
        include_center=True     # Center cell is part of neighborhood
    )
    
    print(f"Lookup table size: {len(tbl)} (2^{len(neigh)} patterns)")
    print(f"Patterns resulting in life: {sum(tbl)}/{len(tbl)}")
    
    sim = CUDALUTBNN(WIDTH, HEIGHT, neigh, tbl)
    vis = OpenCVVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=1)
    engine.run(max_iterations=10_000)

def custom_rules_example():
    """Custom cellular automaton rules."""
    print("=== Custom Rules (B2/S23) - Seeds-like ===")
    WIDTH, HEIGHT = 512, 512
    
    sim = CUDAGameOfLife(WIDTH, HEIGHT, 
                        birth_rules=frozenset({2}),      # Born with 2 neighbors
                        survival_rules=frozenset({2, 3})) # Survive with 2-3 neighbors
    vis = OpenCVVisualizer(WIDTH, HEIGHT)
    engine = SimulationEngine(sim, vis, display_interval=1)
    engine.run(max_iterations=10_000)

def main():
    """Main menu for selecting examples."""
    examples = {
        '1': ('Conway\'s Game of Life', conway_example),
        '2': ('HighLife Variant', highlife_example),
        '3': ('Explosive Growth Rules', explosive_growth_example),
        '4': ('Performance Benchmark', console_benchmark_example),
        '5': ('Conway (Lookup Table)', lut_conway_example),
        '6': ('Custom Rules (B2/S23)', custom_rules_example),
    }
    
    print("GPU-Accelerated Cellular Automaton Examples")
    print("=" * 45)
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect an example (1-6, q to quit): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}")
            try:
                func()
            except KeyboardInterrupt:
                print("\nExample interrupted by user")
            except Exception as e:
                print(f"Error running example: {e}")
            print("\nExample completed. Returning to menu...")
        else:
            print("Invalid choice. Please select 1-6 or q.")

if __name__ == "__main__":
    main()
