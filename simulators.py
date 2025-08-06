"""
GPU-Accelerated Cellular Automaton Simulators

This module contains various cellular automaton simulation implementations:
- BaseSimulator: Abstract base class for all simulators
- CUDAGameOfLife: GPU-accelerated Conway's Game of Life
- CPUGameOfLife: CPU-based NumPy implementation  
- CUDALUTBNN: Generalized GPU cellular automaton with lookup tables
- Utility functions for creating rule tables
"""

import numpy as np
from abc import ABC, abstractmethod
from numba import cuda
from typing import Optional, Tuple, FrozenSet, List

# Abstract Base Classes
class BaseSimulator(ABC):
    """Abstract base class for different simulation implementations."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.current_state = None
    
    @abstractmethod
    def step(self) -> None:
        """Perform one simulation step."""
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Get current simulation state as numpy array."""
        pass
    
    @abstractmethod
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset simulation with optional initial state."""
        pass

# CUDA Kernels
@cuda.jit
def _update_kernel(current, next_state, width, height, birth_rules_array, survival_rules_array):
    """
    CUDA kernel for Conway's Game of Life with configurable rules.
    
    Args:
        current: Current state grid (device array)
        next_state: Next state grid (device array) 
        width, height: Grid dimensions
        birth_rules_array: Array where birth_rules_array[i] = 1 if i neighbors cause birth
        survival_rules_array: Array where survival_rules_array[i] = 1 if i neighbors allow survival
    """
    # Get thread position
    i, j = cuda.grid(2)
    
    if i < height and j < width:
        # Count living neighbors using toroidal boundary conditions
        neighbors = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue  # Skip center cell
                ni = (i + di) % height
                nj = (j + dj) % width
                neighbors += current[ni, nj]
        
        # Apply rules using array lookup
        current_cell = current[i, j]
        if current_cell == 1:
            # Living cell - check survival rules
            next_state[i, j] = 1 if survival_rules_array[neighbors] else 0
        else:
            # Dead cell - check birth rules  
            next_state[i, j] = 1 if birth_rules_array[neighbors] else 0

@cuda.jit
def _update_kernel_3d(current, next_state, width, height, depth, birth_rules_array, survival_rules_array):
    """
    CUDA kernel for 3D Game of Life with configurable rules.
    
    Args:
        current: Current 3D state grid (device array)
        next_state: Next 3D state grid (device array) 
        width, height, depth: Grid dimensions
        birth_rules_array: Array where birth_rules_array[i] = 1 if i neighbors cause birth
        survival_rules_array: Array where survival_rules_array[i] = 1 if i neighbors allow survival
    """
    # Get thread position in 3D
    i, j, k = cuda.grid(3)
    
    if i < depth and j < height and k < width:
        # Count living neighbors in 3D (26 neighbors in Moore neighborhood)
        neighbors = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue  # Skip center cell
                    ni = (i + di) % depth
                    nj = (j + dj) % height
                    nk = (k + dk) % width
                    neighbors += current[ni, nj, nk]
        
        # Apply rules using array lookup
        current_cell = current[i, j, k]
        if current_cell == 1:
            # Living cell - check survival rules
            next_state[i, j, k] = 1 if survival_rules_array[neighbors] else 0
        else:
            # Dead cell - check birth rules  
            next_state[i, j, k] = 1 if birth_rules_array[neighbors] else 0

@cuda.jit
def lut_kernel(curr, nxt, offs_y, offs_x, truth):
    """
    CUDA kernel for lookup-table based cellular automaton.
    
    Args:
        curr: Current state grid
        nxt: Next state grid
        offs_y, offs_x: Neighborhood offset arrays
        truth: Lookup table for all possible patterns
    """
    i, j = cuda.grid(2)
    H, W = curr.shape
    
    if i < H and j < W:
        K = len(offs_y)
        pattern = 0
        
        # Build bit pattern from neighborhood
        for k in range(K):
            ni = (i + offs_y[k]) % H
            nj = (j + offs_x[k]) % W
            if curr[ni, nj]:
                pattern |= (1 << k)
        
        # Look up result in truth table
        nxt[i, j] = truth[pattern]

# Concrete Simulator Implementations
class CUDAGameOfLife(BaseSimulator):
    """
    GPU-accelerated Conway's Game of Life with configurable rules.
    
    Uses CUDA kernels for parallel computation with configurable birth and survival rules.
    Supports toroidal (wraparound) boundary conditions.
    """
    
    def __init__(self, width: int, height: int, 
                 birth_rules: FrozenSet[int] = frozenset({3}),
                 survival_rules: FrozenSet[int] = frozenset({2, 3}),
                 threads_per_block: Tuple[int, int] = (16, 16)):
        """
        Initialize CUDA Game of Life simulator.
        
        Args:
            width, height: Grid dimensions
            birth_rules: Set of neighbor counts that cause birth (default: {3})
            survival_rules: Set of neighbor counts that allow survival (default: {2, 3})
            threads_per_block: CUDA thread block size
        """
        super().__init__(width, height)
        self.birth_rules = birth_rules
        self.survival_rules = survival_rules
        self.threads_per_block = threads_per_block
        
        # Calculate grid dimensions for CUDA launch
        self.blocks_per_grid = (
            (height + threads_per_block[0] - 1) // threads_per_block[0],
            (width + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Pre-compute rule arrays and move to GPU memory (PERFORMANCE OPTIMIZATION)
        max_neighbors = 8
        birth_rules_array = np.zeros(max_neighbors + 1, dtype=np.uint8)
        survival_rules_array = np.zeros(max_neighbors + 1, dtype=np.uint8)
        
        for rule in birth_rules:
            birth_rules_array[rule] = 1
        for rule in survival_rules:
            survival_rules_array[rule] = 1
        
        # Move rule arrays to GPU memory once during initialization
        self.d_birth_rules = cuda.to_device(birth_rules_array)
        self.d_survival_rules = cuda.to_device(survival_rules_array)
        
        # Initialize device arrays
        self.d_current = None
        self.d_next = None
        self.reset()
    
    def step(self) -> None:
        """Perform one simulation step using CUDA kernel."""
        # No more array creation or host-to-device transfers!
        _update_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_current, self.d_next, self.width, self.height,
            self.d_birth_rules, self.d_survival_rules
        )
        # Swap buffers
        self.d_current, self.d_next = self.d_next, self.d_current
    
    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self.d_current.copy_to_host()
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset simulation with random or provided initial state."""
        if initial_state is None:
            # Create random initial state (50% alive)
            initial_state = np.random.choice([0, 1], size=(self.height, self.width), p=[0.5, 0.5])
        
        # Ensure correct data type and shape
        initial_state = initial_state.astype(np.uint8)
        if initial_state.shape != (self.height, self.width):
            raise ValueError(f"Initial state shape {initial_state.shape} doesn't match grid ({self.height}, {self.width})")
        
        # Copy to GPU
        self.d_current = cuda.to_device(initial_state)
        self.d_next = cuda.device_array_like(self.d_current)
    
    @classmethod
    def conway(cls, width: int, height: int, **kwargs):
        """Create standard Conway's Game of Life (B3/S23)."""
        return cls(width, height, frozenset({3}), frozenset({2, 3}), **kwargs)
    
    @classmethod  
    def highlife(cls, width: int, height: int, **kwargs):
        """Create HighLife variant (B36/S23)."""
        return cls(width, height, frozenset({3, 6}), frozenset({2, 3}), **kwargs)
    
    @classmethod
    def seeds(cls, width: int, height: int, **kwargs):
        """Create Seeds variant (B2/S).""" 
        return cls(width, height, frozenset({2}), frozenset(), **kwargs)

class CUDA3DGameOfLife(BaseSimulator):
    """
    GPU-accelerated 3D Game of Life with configurable rules.
    
    Uses CUDA kernels for parallel computation with configurable birth and survival rules.
    Supports toroidal (wraparound) boundary conditions.
    """
    
    def __init__(self, width: int, height: int, depth: int, 
                 birth_rules: FrozenSet[int] = frozenset({3}),
                 survival_rules: FrozenSet[int] = frozenset({2, 3}),
                 threads_per_block: Tuple[int, int, int] = (8, 8, 8)):
        """
        Initialize CUDA 3D Game of Life simulator.
        
        Args:
            width, height, depth: Grid dimensions
            birth_rules: Set of neighbor counts that cause birth (default: {3})
            survival_rules: Set of neighbor counts that allow survival (default: {2, 3})
            threads_per_block: CUDA thread block size
        """
        super().__init__(width, height)
        self.depth = depth
        self.birth_rules = birth_rules
        self.survival_rules = survival_rules
        self.threads_per_block = threads_per_block
        
        # Print GPU information for verification
        self._print_gpu_info()
        
        # Calculate grid dimensions for CUDA launch
        self.blocks_per_grid = (
            (depth + threads_per_block[0] - 1) // threads_per_block[0],
            (height + threads_per_block[1] - 1) // threads_per_block[1],
            (width + threads_per_block[2] - 1) // threads_per_block[2]
        )
        
        print(f"CUDA 3D Grid Configuration:")
        print(f"  Threads per block: {threads_per_block}")
        print(f"  Blocks per grid: {self.blocks_per_grid}")
        print(f"  Total threads: {np.prod(threads_per_block) * np.prod(self.blocks_per_grid)}")
        print(f"  Grid cells: {width * height * depth}")
        
        # Pre-compute rule arrays and move to GPU memory (PERFORMANCE OPTIMIZATION)
        max_neighbors = 26
        birth_rules_array = np.zeros(max_neighbors + 1, dtype=np.uint8)
        survival_rules_array = np.zeros(max_neighbors + 1, dtype=np.uint8)
        
        for rule in birth_rules:
            birth_rules_array[rule] = 1
        for rule in survival_rules:
            survival_rules_array[rule] = 1
        
        # Move rule arrays to GPU memory once during initialization
        self.d_birth_rules = cuda.to_device(birth_rules_array)
        self.d_survival_rules = cuda.to_device(survival_rules_array)
        print(f"  Rule arrays moved to GPU memory")
        
        # Initialize device arrays
        self.d_current = None
        self.d_next = None
        self.reset()
    
    def _print_gpu_info(self):
        """Print GPU information for verification."""
        try:
            from numba import cuda
            print(f"GPU Verification:")
            print(f"  CUDA available: {cuda.is_available()}")
            if cuda.is_available():
                gpu = cuda.get_current_device()
                print(f"  GPU device: {gpu.name.decode()}")
                print(f"  Compute capability: {gpu.compute_capability}")
                print(f"  Total memory: {gpu.total_memory / (1024**3):.1f} GB")
                
                # Get memory info
                meminfo = cuda.current_context().get_memory_info()
                used_mb = (meminfo.total - meminfo.free) / (1024**2)
                total_mb = meminfo.total / (1024**2)
                print(f"  GPU memory used: {used_mb:.1f} MB / {total_mb:.1f} MB")
            else:
                print("  WARNING: CUDA not available - falling back to CPU!")
        except Exception as e:
            print(f"  Error checking GPU info: {e}")
    
    def step(self) -> None:
        """Perform one simulation step using CUDA kernel."""
        # No more array creation or host-to-device transfers!
        _update_kernel_3d[self.blocks_per_grid, self.threads_per_block](
            self.d_current, self.d_next, self.width, self.height, self.depth,
            self.d_birth_rules, self.d_survival_rules
        )
        # Swap buffers
        self.d_current, self.d_next = self.d_next, self.d_current
    
    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self.d_current.copy_to_host()
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset simulation with random or provided initial state."""
        if initial_state is None:
            # Create random initial state (50% alive)
            initial_state = np.random.choice([0, 1], size=(self.depth, self.height, self.width), p=[0.5, 0.5])
        
        # Ensure correct data type and shape
        initial_state = initial_state.astype(np.uint8)
        if initial_state.shape != (self.depth, self.height, self.width):
            raise ValueError(f"Initial state shape {initial_state.shape} doesn't match grid ({self.depth}, {self.height}, {self.width})")
        
        # Copy to GPU
        self.d_current = cuda.to_device(initial_state)
        self.d_next = cuda.device_array_like(self.d_current)

class CPUGameOfLife(BaseSimulator):
    """CPU-based Game of Life using NumPy."""
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.next_state = None
        self.reset()
    
    def step(self) -> None:
        """Perform one CPU simulation step."""
        # Count neighbors using convolution
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = np.zeros_like(self.current_state)
        
        for i in range(3):
            for j in range(3):
                if kernel[i, j] == 1:
                    # Shift and add with wraparound
                    shifted = np.roll(np.roll(self.current_state, i-1, axis=0), j-1, axis=1)
                    neighbors += shifted
        
        # Apply Conway's rules
        self.next_state = ((self.current_state == 1) & ((neighbors == 2) | (neighbors == 3))) | \
                         ((self.current_state == 0) & (neighbors == 3))
        self.next_state = self.next_state.astype(np.uint8)
        
        # Swap states
        self.current_state, self.next_state = self.next_state, self.current_state
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.current_state.copy()
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset simulation with random or provided initial state."""
        if initial_state is None:
            initial_state = np.random.choice([0, 1], size=(self.height, self.width), p=[0.5, 0.5])
        
        self.current_state = initial_state.astype(np.uint8)
        self.next_state = np.zeros_like(self.current_state)

class CUDALUTBNN(BaseSimulator):
    """
    GPU Boolean neural net governed by an arbitrary lookup‑table.
    neighbourhood : list of (dy, dx) tuples
    truthTable    : uint8 array length 2**K giving next bit for every pattern
    """
    
    def __init__(self,
                 width: int,
                 height: int,
                 neighbourhood: List[Tuple[int, int]],
                 truthTable: np.ndarray,
                 threadsPerBlock: Tuple[int, int] = (16, 16)):
        super().__init__(width, height)

        self.neigh = neighbourhood
        self.K     = len(neighbourhood)

        if truthTable.shape != (1 << self.K,):
            raise ValueError(f"truthTable length must be 2**K = {1<<self.K}")
        self.truth_h = truthTable.astype(np.uint8)

        # --- CUDA launch config
        self.threads = threadsPerBlock
        self.blocks  = ((height + self.threads[0] - 1) // self.threads[0],
                        (width  + self.threads[1] - 1) // self.threads[1])

        # copy lookup data to GPU
        offs_y, offs_x = map(np.int16, zip(*self.neigh))
        self.d_offs_y  = cuda.to_device(np.asarray(offs_y, dtype=np.int16))
        self.d_offs_x  = cuda.to_device(np.asarray(offs_x, dtype=np.int16))
        self.d_truth   = cuda.to_device(self.truth_h)

        self.d_curr = None
        self.d_next = None
        self.reset()

    def step(self) -> None:
        """Perform one simulation step using lookup table kernel."""
        lut_kernel[self.blocks, self.threads](
            self.d_curr, self.d_next, self.d_offs_y, self.d_offs_x, self.d_truth
        )
        # swap
        self.d_curr, self.d_next = self.d_next, self.d_curr

    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return self.d_curr.copy_to_host()

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset simulation with random or provided initial state."""
        if initial_state is None:
            initial_state = np.random.choice([0, 1], size=(self.height, self.width), p=[0.5, 0.5])
        
        initial_state = initial_state.astype(np.uint8)
        if initial_state.shape != (self.height, self.width):
            raise ValueError(f"Initial state shape {initial_state.shape} doesn't match grid ({self.height}, {self.width})")
        
        self.d_curr = cuda.to_device(initial_state)
        self.d_next = cuda.device_array_like(self.d_curr)

    @classmethod
    def conway(cls, w: int, h: int, **kw):
        """Create Conway's Game of Life using lookup table approach."""
        neigh = [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
        K = 9
        table = np.zeros(1 << K, dtype=np.uint8)
        for pat in range(1 << K):
            bits = [(pat >> b) & 1 for b in range(K)]
            c    = bits[4]  # center at index 4
            nbs  = sum(bits) - c
            if (c and nbs in (2,3)) or (not c and nbs==3):
                table[pat] = 1
        return cls(w, h, neigh, table, **kw)

# Utility Functions
def make_totalistic_table(neigh, birth_set, survival_set, include_center=True):
    """
    Build a truth‑table (uint8 array) for any totalistic B/S rule.

    neigh : list of (dy,dx) offsets  — must match the simulator neighbourhood
    birth_set, survival_set : sets of neighbour counts (integers)
    include_center : if True, the centre cell is one of the inputs
    """
    K = len(neigh)
    table = np.zeros(1 << K, dtype=np.uint8)
    
    # Find the center cell position in the neighborhood
    center_idx = -1
    if include_center:
        try:
            center_idx = neigh.index((0, 0))
        except ValueError:
            raise ValueError("Center cell (0, 0) not found in neighborhood when include_center=True")
    
    for pattern in range(1 << K):
        bits = [(pattern >> b) & 1 for b in range(K)]
        centre = bits[center_idx] if include_center and center_idx >= 0 else 0
        neighbours_alive = sum(bits) - centre
        
        if (centre and neighbours_alive in survival_set) or \
           (not centre and neighbours_alive in birth_set):
            table[pattern] = 1
    return table
