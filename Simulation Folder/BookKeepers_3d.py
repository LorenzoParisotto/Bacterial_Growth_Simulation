class Bookkeeper:
    def __init__(self):
        # List of dictionaries: one per simulation step.
        self.step_summaries = []
        # Counters for deaths per cell type and per cause.
        self.death_counts = {
            'good': {'age': 0, 'antibiotics': 0, 'food': 0},
            'bad': {'age': 0, 'antibiotics': 0, 'food': 0}
        }
        # Dictionary mapping cell index to the number of reproductions performed before dying.
        self.reproduction_records = {}
        self.alive_times = {'good': [], 'bad': []}

    def record_death(self, cell_index, cell_type, cause, reproduction_count, time):
        """
        Record a death event.
        
        cell_index: unique id or index of the cell.
        cell_type: either 'good' or 'bad'
        cause: one of 'age', 'antibiotics', or 'food'
        reproduction_count: number of reproductions performed before death.
        """
        if cell_type in self.death_counts:
            if cause in self.death_counts[cell_type]:
                self.death_counts[cell_type][cause] += 1
            else:
                self.death_counts[cell_type][cause] = 1
        else:
            self.death_counts[cell_type] = {cause: 1}
        # Record the cell's reproduction count at death.
        self.reproduction_records[cell_index] = reproduction_count
        self.alive_times[cell_type].append(time)

    def record_step_summary(self, step, grid, resource_manager):
        """
        Record counts for the current simulation step.
        """
        from Cell_3d import Cell
        from Resource_Manager_3d import ResourceManager
        summary = {
            'step': step,
            'alive_good': 0,
            'alive_bad': 0,
            'dead_good': 0,
            'dead_bad': 0,
            'good_dead_due_to_antibiotics': 0,
            'bad_dead_due_to_antibiotics': 0,
            'antibiotics_concentration': 0
        }
        for cell in grid:
            if cell.state == Cell.good:
                summary['alive_good'] += 1
            elif cell.state == Cell.bad:
                summary['alive_bad'] += 1
            elif cell.state == Cell.dead_good:
                summary['dead_good'] += 1
            elif cell.state == Cell.dead_bad:
                summary['dead_bad'] += 1
            if cell.good_dead_due_to_antibiotics:
                summary['good_dead_due_to_antibiotics'] += 1
            elif cell.bad_dead_due_to_antibiotics:
                summary['bad_dead_due_to_antibiotics'] += 1

        summary['antibiotics_concentration'] += sum(
            resource_manager.antibiotics[k] for k in range(len(grid))
        ) # Sum the antibiotics concentration across all cells in the grid.
        self.step_summaries.append(summary)

        