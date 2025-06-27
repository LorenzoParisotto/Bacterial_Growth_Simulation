# Bacterial_Growth_Simulation
Bacterial growth simulation model from my Thesis project submitted to Eindhoven University of Technology

This repository contains a 3D bacterial growth simulation model. The model simulates the growth of two types of bacteria ("good" and "bad") in a 3D grid environment, considering factors like nutrient availability and the presence of antibiotics.

## How to Run the Simulation

To run the simulation, you need to use the `3d_grid_model.ipynb` Jupyter Notebook.

1.  **Open `Simulation Folder/3d_grid_model.ipynb`** in a Jupyter environment.
2.  **Run the first cell.** This cell contains the main simulation logic. It will initialize the simulation, run through the specified number of steps, and store the simulation data. You can modify the simulation parameters within this cell to change the conditions of the experiment.
3.  **Run the subsequent cells** to visualize the simulation results. These cells generate various plots, such as the progression of bacteria populations over time, the concentration of antibiotics, and heatmaps of the grid at different time steps. The plots can be saved as image files.

## File Descriptions

*   `README.md`: This file.
*   `Simulation Folder/3d_grid_model.ipynb`: The main Jupyter Notebook for running the simulation and visualizing the results.
*   `Simulation Folder/Cell_3d.py`: Defines the `Cell` class, which represents a single bacterium in the simulation. Here it is possible to change parameters specific to the bacteria type.
*   `Simulation Folder/BookKeepers_3d.py`: Defines the `Bookkeeper` class, which tracks statistics and data from the simulation.
*   `Simulation Folder/Efficient_Resource_Manager_3d.py`: Defines the `ResourceManager` class, which manages the diffusion of resources like food and antibiotics in the 3D grid.
