import numpy as np
from numba import njit, typed, types

@njit
def diffusion_step_food_numba(food, neighbors, D_food, dt, dx):
    n = food.shape[0]
    C_new = food.copy()
    for j in range(n):
        flux = 0.0
        # neighbors[j] is expected to be a list or array of indices
        for k in neighbors[j]:
            D_avg = 0.5 * (D_food[j] + D_food[k])
            flux += D_avg * (food[k] - food[j])
        C_new[j] += dt / (dx**2) * flux
        if C_new[j] < 0:
            C_new[j] = 0.0
    return C_new

@njit
def diffusion_step_antibiotics_numba(antibiotics, neighbors, D_antibiotics, dt, dx):
    n = antibiotics.shape[0]
    C_new = antibiotics.copy()
    for j in range(n):
        flux = 0.0
        for k in neighbors[j]:
            D_avg = 0.5 * (D_antibiotics[j] + D_antibiotics[k])
            flux += D_avg * (antibiotics[k] - antibiotics[j])
        C_new[j] += dt / (dx**2) * flux
        if C_new[j] < 0:
            C_new[j] = 0.0
    return C_new

@njit
def diffusion_steps_antibiotics_FAST(antibiotics, neighbors, D_antibiotics, dt, dx, steps):
    # Perform diffusion for a specified number of steps
    n = antibiotics.shape[0]
    for _ in range(steps):
        C_new = antibiotics.copy()
        for j in range(n):
            flux = 0.0
            for k in neighbors[j]:
                D_avg = 0.5 * (D_antibiotics[j] + D_antibiotics[k])
                flux += D_avg * (antibiotics[k] - antibiotics[j])
            C_new[j] += dt / (dx**2) * flux
            if C_new[j] < 0:
                C_new[j] = 0.0
        antibiotics = C_new
    return antibiotics

@njit
def diffusion_steps_antibiotics_FAST_optimized(antibiotics, buffer, neighbors, D_antibiotics, dt, dx, steps):
    n = antibiotics.shape[0]
    src = antibiotics
    dst = buffer
    
    # Precompute the modified coefficient
    coeff_prime = 0.5 * dt / (dx**2) 
    
    for _ in range(steps):
        for j in range(n):
            # Hoist loop-invariant values for cell j
            D_j = D_antibiotics[j]
            src_j = src[j]
            
            sum_of_neighbor_terms = 0.0
            for k in neighbors[j]:
                # D_k = D_antibiotics[k] # Accessed directly
                # src_k = src[k]         # Accessed directly
                
                # Calculate (D_j + D_k) * (src_k - src_j)
                term = (D_j + D_antibiotics[k]) * (src[k] - src_j)
                sum_of_neighbor_terms += term
                
            dst[j] = src_j + coeff_prime * sum_of_neighbor_terms
            
            if dst[j] < 0:
                dst[j] = 0.0
        # Swap source and destination arrays for the next iteration
        src, dst = dst, src
    
    if src is not antibiotics:
        antibiotics[:] = src[:] # Copy data back if src is the buffer
        return antibiotics
    return src

def get_close_neighbors_3d(index, size, height):
    """
    Get the immediate (face-sharing) neighbors for a cell in a 3D grid with customizable height.
    The grid is size x size x height, and indexing is 1D.
    """
    layer_size = size * size
    z = index // layer_size
    rem = index % layer_size
    row = rem // size
    col = rem % size
    neighbors = []
    # Same layer neighbors
    if row > 0:    # Up in row
        neighbors.append(index - size)
    if row < size - 1:    # Down in row
        neighbors.append(index + size)
    if col > 0:    # Left in column
        neighbors.append(index - 1)
    if col < size - 1:    # Right in column
        neighbors.append(index + 1)
    # Neighbors in layers above and below
    if z > 0:
        neighbors.append(index - layer_size)
    if z < height - 1:
        neighbors.append(index + layer_size)
    return neighbors

class ResourceManager:

    def __init__(self, grid_size, grid_height, resource_steps_per_time_unit, dx,
                 D_antibiotics, D_food, D_antibiotics_multiplyer, D_food_multiplyer):
        self.dt = 1/resource_steps_per_time_unit
        self.resource_steps_per_time_unit = resource_steps_per_time_unit
        self.dx = dx

        self.D_antibiotics = D_antibiotics
        self.D_food = D_food
        self.D_antibiotics_multiplyer = D_antibiotics_multiplyer
        self.D_food_multiplyer = D_food_multiplyer

        self.num_cells = grid_size * grid_size * grid_height

        # For speed we now use numpy arrays instead of dictionaries.
        self.food = np.zeros(self.num_cells, dtype=np.float64)
        self.antibiotics = np.zeros(self.num_cells, dtype=np.float64)
        self.antibiotics_buffer = np.zeros(self.num_cells, dtype=np.float64)
        # Create arrays for D values (dictionary data converted to numpy arrays)
        self.D_food_arr = np.full(self.num_cells, D_food, dtype=np.float64)
        self.D_antibiotics_arr = np.full(self.num_cells, D_antibiotics, dtype=np.float64)

        # Precompute neighbors as a typed list of arrays for numba compatibility.
        self.neighbors = typed.List()
        for index in range(self.num_cells):
            neigh = np.array(get_close_neighbors_3d(index, grid_size, grid_height), 
                             dtype=np.int64)
            self.neighbors.append(neigh)

    def diffusion_step_food(self):
        self.food = diffusion_step_food_numba(self.food, self.neighbors, self.D_food_arr, 
                                              self.dt, self.dx)
    
    def diffusion_step_antibiotics(self):
        self.antibiotics = diffusion_step_antibiotics_numba(self.antibiotics,  self.neighbors, 
                                                            self.D_antibiotics_arr, self.dt, 
                                                            self.dx)
    
    def diffusion_step(self):
        for _ in range(self.resource_steps_per_time_unit):
            self.diffusion_step_food()
            self.diffusion_step_antibiotics()


        # Perform diffusion for a specified number of steps efficiently ignoring food
    # def diffusion_step(self):
    #     self.antibiotics = diffusion_steps_antibiotics_FAST_optimized(self.antibiotics, self.antibiotics_buffer, self.neighbors, 
    #                                                          self.D_antibiotics_arr, self.dt, 
    #                                                          self.dx, self.resource_steps_per_time_unit)
        

    def update_D(self, states):
        for i, s in enumerate(states):
            if s == 0:
                self.D_food_arr[i] = self.D_food
                self.D_antibiotics_arr[i] = self.D_antibiotics
            else:
                self.D_food_arr[i] = self.D_food * self.D_food_multiplyer
                self.D_antibiotics_arr[i] = self.D_antibiotics * self.D_antibiotics_multiplyer