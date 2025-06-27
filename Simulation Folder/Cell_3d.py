import numpy as np
import random
from BookKeepers_3d import Bookkeeper



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

def get_large_neighbors_3d(index, size, height):
    """
    Get "large" neighbors for a cell in a 3D grid with customizable height.
    "Large" neighbors are defined as cells that are exactly two edges away
    in any combination of the three dimensions.
    """
    layer_size = size * size
    z = index // layer_size
    rem = index % layer_size
    row = rem // size
    col = rem % size
    neighbors = []
    
    # Loop over all combinations of dz, dr, dc that sum to 2 steps away
    for dz in [-2, -1, 0, 1, 2]:
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                # Skip the cell itself
                if dz == 0 and dr == 0 and dc == 0:
                    continue
                # Ensure the total distance is exactly 2 edges away
                if abs(dz) + abs(dr) + abs(dc) != 2:
                    continue
                # Calculate the new coordinates
                nz = z + dz
                nr = row + dr
                nc = col + dc
                # Check bounds
                if 0 <= nz < height and 0 <= nr < size and 0 <= nc < size:
                    neighbor_index = nz * layer_size + nr * size + nc
                    neighbors.append(neighbor_index)
    return neighbors



class Cell():
    good = 1
    bad = -1
    empty = 0
    dead_good = 2
    dead_bad = -2

    lambd_map = {1: 0.64,
                -1: 0.87}
    p_mutation = 0.01
    
    def __init__(self, index, init_state, grid_size, grid_height, bookkeeper):
        self.bookkeeper = bookkeeper

        self.index = index
        self.state = init_state
        self.neighbors = get_close_neighbors_3d(index, grid_size, grid_height)
        self.lambd = None
        self.death_date = None
        self.alive_time = None
        self.reproduction_timer = None
        self.reproduction_count = 0  # Number of reproductions performed before death

        self.antibiotics_decay = 0.005
        self.antibiotics_resistance = 0.05  # Resistance to antibiotics for good bacteria
        self.good_dead_due_to_antibiotics = False  # indicate if cell died due to antibiotics
        self.bad_dead_due_to_antibiotics = False  # indicate if cell died due to antibiotics

        self.eat_amount = 0.1

        if init_state == self.good or init_state == self.bad:
            self.lambd = self.lambd_map[init_state]
            # Death timer in simulation steps.
            self.death_date = np.random.exponential(3 / self.lambd)
            self.alive_time = self.death_date
            # Reproduction timer.
            self.reproduction_timer = np.random.exponential(1 / self.lambd)


    def death_of_good(self):
        # A good cell "dies".
        self.state = self.dead_good
        self.bookkeeper.record_death(self.index, 'good', 'age', self.reproduction_count, 
                                     self.alive_time)  # Record the death event

    def death_of_bad(self):
        # A bad cell "dies".
        self.state = self.dead_bad
        self.bookkeeper.record_death(self.index, 'bad', 'age', self.reproduction_count, 
                                     self.alive_time)  # Record the death event

    def death_of_good_due_to_antibiotics(self):
        # A good cell "dies" due to antibiotics.
        self.state = self.dead_good
        self.bookkeeper.record_death(self.index, 'good', 'antibiotics', self.reproduction_count, 
                                     self.alive_time)  # Record the death event
        self.good_dead_due_to_antibiotics = True  # flag to indicate death due to antibiotics

    def death_of_bad_due_to_antibiotics(self):
        # A bad cell "dies" due to antibiotics.
        self.state = self.dead_bad
        self.bookkeeper.record_death(self.index, 'bad', 'antibiotics', self.reproduction_count, 
                                     self.alive_time)  # Record the death event
        self.bad_dead_due_to_antibiotics = True  # flag to indicate death due to antibiotics


    def death_of_good_due_to_food(self):
        # A good cell "dies" due to food depletion.
        self.state = self.dead_good
        self.bookkeeper.record_death(self.index, 'good', 'food', self.reproduction_count, 
                                     self.alive_time)  # Record the death event

    def death_of_bad_due_to_food(self):
        # A bad cell "dies" due to food depletion.
        self.state = self.dead_bad
        self.bookkeeper.record_death(self.index, 'bad', 'food', self.reproduction_count, 
                                     self.alive_time)  # Record the death event



    def reproduction_of_any(self, grid):
        while self.reproduction_timer <= 0:  # next reproduction is due within the same timestep
            empty_neighbors = [grid[i] for i in self.neighbors if grid[i].state == Cell.empty]
            if empty_neighbors:
                child = random.choice(empty_neighbors)
                # Inherit parent's state; with a small chance of mutation.
                new_state = self.state
                if self.state == self.good and random.random() < Cell.p_mutation:
                    # For example, a mutation flips the state.
                    new_state *= -1
                child.state = new_state
                child.lambd = self.lambd_map[new_state]
                # Initialize child's timers.
                child.death_date = self.reproduction_timer + \
                   np.random.exponential(3 / child.lambd)
                child.reproduction_timer = self.reproduction_timer + \
                   np.random.exponential(1 / child.lambd)
                child.alive_time = self.death_date

                self.reproduction_count += 1  # Increment the reproduction count of parent cell
                # Reset the parent's reproduction timer.
                timer_addition = np.random.exponential(1 / self.lambd)
                self.reproduction_timer += timer_addition
            else:
                break  # No empty neighbors available for reproduction, exit the loop

    def step(self, grid, ResourceManager, dt=1):
        # --- Resource Manager --- #
        food_dict = ResourceManager.food
        antibiotics_dict = ResourceManager.antibiotics


        # --- Reproduction and Death Mechanism --- #
        if self.state == self.good:
            # Move steps forward in time.
            self.death_date -= dt  # Decrement timer by dt.
            if self.reproduction_timer >= 0: 
                self.reproduction_timer -= dt  # Decrement timer by dt.



            if self.reproduction_timer <= 0:
                if self.death_date <= 0:
                    # Check if the good cell dies before reproduction.
                    if self.death_date < self.reproduction_timer:
                        # Cell dies first.
                        self.death_of_good()
                    else:
                        # Cell reproduces first, then dies.
                        self.reproduction_of_any(grid)
                        self.death_of_good()
                
                else: # Cell reproduces only.
                    self.reproduction_of_any(grid)


            if self.death_date <= 0:
                # Cell dies.
                self.death_of_good()

        elif self.state == self.bad:
            # Move steps forward in time.
            self.death_date -= dt  # Decrement timer by dt.
            if self.reproduction_timer >= 0:
                self.reproduction_timer -= dt  # Decrement timer by dt.

            if self.reproduction_timer <= 0:
                if self.death_date <= 0:
                    # Check if the bad cell dies before reproduction.
                    if self.death_date < self.reproduction_timer:
                        # Cell dies first.
                        self.death_of_bad()
                    else:
                        # Cell reproduces first, then dies.
                        self.reproduction_of_any(grid)
                        self.death_of_bad()
                
                else: # Cell reproduces only.
                    self.reproduction_of_any(grid)


            if self.death_date <= 0:
                # Cell dies.
                self.death_of_bad()

        
        antibiotics_dict[self.index] -= self.antibiotics_decay * dt
        if antibiotics_dict[self.index] < 0:
            antibiotics_dict[self.index] = 0
        
        # --- Antibiotics Effect on Cells -- #
        
        if self.state == self.bad:  # Effect on bad bacteria
            if random.random() < antibiotics_dict[self.index]:
                # Bad bacteria die due to antibiotics.
                self.alive_time -= self.death_date  # adjust the alive time of the cell
                self.death_of_bad_due_to_antibiotics()
                antibiotics_dict[self.index] -= 0.05
                if antibiotics_dict[self.index] < 0:
                    antibiotics_dict[self.index] = 0
        elif self.state == self.good:  # Effect on good bacteria
            if random.random() < antibiotics_dict[self.index] * self.antibiotics_resistance:
                # Good bacteria die due to antibiotics.
                self.alive_time -= self.death_date  # adjust the alive time of the cell
                self.death_of_good_due_to_antibiotics()
                antibiotics_dict[self.index] -= 0.05
                if antibiotics_dict[self.index] < 0:
                    antibiotics_dict[self.index] = 0


        # --- Food Consumption Mechanism --- #
        if self.state == self.good:
            # Good bacteria consume food.
            food_dict[self.index] -= self.eat_amount * dt
            if food_dict[self.index] < 0:
                food_dict[self.index] = 0
                self.alive_time -= self.death_date # adjust the alive time of the cell
                self.death_of_good_due_to_food()  # If food is depleted, the good bacteria die.
        elif self.state == self.bad:
            # Bad bacteria consume food.
            food_dict[self.index] -= self.eat_amount * dt
            if food_dict[self.index] < 0:
                food_dict[self.index] = 0
                self.alive_time -= self.death_date # adjust the alive time of the cell
                self.death_of_bad_due_to_food()
    
        # --- Update the Resource Manager --- #
        ResourceManager.food = food_dict
        ResourceManager.antibiotics = antibiotics_dict
        
