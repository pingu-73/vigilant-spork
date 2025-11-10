# benchmark.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
import os

# Import your PSO algorithms from main.py
from main import APSO, SPSO, ARPSO 
# Import the CEC2022 benchmark functions
from cec import cec2022_func

# Ensure a 'plots' directory exists
if not os.path.exists('plots_new'):
    os.makedirs('plots_new')


# --- Adapter Classes for Minimization ---
class MinimizationAPSO(APSO):
    def __init__(self, num_particles, max_iter, dim=2):
        super().__init__(num_particles=num_particles, max_iter=max_iter, dim=dim, initial_positions=None)
        self.dim = dim
        self.positions = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.accelerations = np.zeros((self.num_particles, self.dim))

    def run(self, fitness_func, dim, bounds):
        if self.dim != dim:
            raise ValueError(f"Object initialized with dim={self.dim} but run with dim={dim}")
        
        # Initialize positions within the specified bounds
        self.positions = np.random.uniform(bounds[0], bounds[1], (self.num_particles, dim))
        self.velocities.fill(0)
        self.accelerations.fill(0)

        # Evaluate initial fitness
        fitness_values = np.array([fitness_func(p) for p in self.positions])
        
        b_positions = self.positions.copy()
        b_fitness = fitness_values.copy()
        
        g_index = np.argmin(fitness_values) # MIN instead of MAX
        g_position = self.positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        convergence_curve = [g_fitness]
        position_history = [self.positions.copy()]

        for i in range(self.max_iter):
            # Update positions
            self.accelerations = self.update_acceleration(self.accelerations, self.positions, b_positions, g_position)
            self.velocities = self.update_velocity(self.velocities, self.accelerations)
            self.positions = self.update_position(self.positions, self.velocities)
            
            # Clamp positions to the bounds
            self.positions = np.clip(self.positions, bounds[0], bounds[1])

            fitness_values = np.array([fitness_func(p) for p in self.positions])

            # Update personal best (look for smaller values)
            update_indices = fitness_values < b_fitness # LESS THAN instead of GREATER THAN
            b_positions[update_indices] = self.positions[update_indices].copy()
            b_fitness[update_indices] = fitness_values[update_indices]

            # Update global best (look for smaller values)
            current_min_fitness = np.min(fitness_values)
            if current_min_fitness < g_fitness: # LESS THAN instead of GREATER THAN
                g_index = np.argmin(fitness_values) # MIN instead of MAX
                g_position = self.positions[g_index].copy()
                g_fitness = current_min_fitness

            convergence_curve.append(g_fitness)
            position_history.append(self.positions.copy())

        return g_fitness, g_position, convergence_curve, position_history

class MinimizationSPSO(SPSO):
    def __init__(self, num_particles, max_iter, dim=2):
        self.c1 = 1.193   # same as main.py
        self.c2 = 1.193   # same as main.py
        self.w = 0.721    # same as main.py
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = dim
        self.positions = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))

    def run(self, fitness_func, dim, bounds):
        if self.dim != dim:
            raise ValueError(f"Object initialized with dim={self.dim} but run with dim={dim}")
        self.positions = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        self.velocities.fill(0)
        
        fitness_values = np.array([fitness_func(p) for p in self.positions])
        
        b_positions = self.positions.copy()
        b_fitness = fitness_values.copy()
        
        g_index = np.argmin(fitness_values)
        g_position = self.positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        convergence_curve = [g_fitness]
        position_history = [self.positions.copy()]

        for i in range(self.max_iter):
            self.velocities = self.update_velocity(self.velocities, self.positions, b_positions, g_position)
            self.positions = self.update_positions(self.velocities, self.positions)
            self.positions = np.clip(self.positions, bounds[0], bounds[1])

            fitness_values = np.array([fitness_func(p) for p in self.positions])

            update_indices = fitness_values < b_fitness
            b_positions[update_indices] = self.positions[update_indices].copy()
            b_fitness[update_indices] = fitness_values[update_indices]

            current_min_fitness = np.min(fitness_values)
            if current_min_fitness < g_fitness:
                g_index = np.argmin(fitness_values)
                g_position = self.positions[g_index].copy()
                g_fitness = current_min_fitness
            
            convergence_curve.append(g_fitness)
            position_history.append(self.positions.copy())

        return g_fitness, g_position, convergence_curve, position_history


# In benchmark.py

def run_benchmark(cec_functions, dimensions, algorithms, num_runs, max_iter, num_particles):
    """
    Runs the benchmarking experiments and returns the results.
    """
    results_list = []
    bounds = [-100, 100]

    for func_id in cec_functions:
        for dim in dimensions:
            print(f"--- Testing F{func_id} in {dim}D ---")
            
            cec_func_obj = cec2022_func(func_num=func_id)

            # --- THIS IS THE CORRECTED WRAPPER ---
            def fitness_wrapper(x):
                # The cec function expects a shape of (dim, num_particles)
                # Here, we are evaluating one particle at a time, so shape is (dim, 1)
                particle_position = x.reshape(-1, 1)
                
                # Call the values method, which returns the object itself
                cec_func_obj.values(particle_position)
                
                # The result is stored in the ObjFunc attribute
                return cec_func_obj.ObjFunc[0]
            # --- END OF CORRECTION ---

            for alg_name, alg_class in algorithms.items():
                print(f"  -> Algorithm: {alg_name}")
                for run in range(num_runs):
                    print(f"    -> Run: {run + 1}/{num_runs}")
                    
                    # algorithm = alg_class(num_particles=num_particles, max_iter=max_iter)
                    # best_fitness, best_pos, conv_curve, pos_history = algorithm.run(fitness_wrapper, dim, bounds)
                    algorithm = alg_class(num_particles=num_particles, max_iter=max_iter, dim=dim)
                    best_fitness, best_pos, conv_curve, pos_history = algorithm.run(fitness_wrapper, dim, bounds)

                    results_list.append({
                        'Function': f'F{func_id}',
                        'Dimension': dim,
                        'Algorithm': alg_name,
                        'Run': run + 1,
                        'BestFitness': best_fitness,
                        'ConvergenceCurve': conv_curve,
                        'PositionHistory': pos_history
                    })

    return pd.DataFrame(results_list)


# --- Visualization Functions ---

def plot_convergence_curves(df, path='plots_new/'):
    """Plots the average convergence curve for each algorithm on each function."""
    for (func, dim), group in df.groupby(['Function', 'Dimension']):
        plt.figure(figsize=(12, 8))
        
        for alg in group['Algorithm'].unique():
            alg_df = group[group['Algorithm'] == alg]
            # Stack all convergence curves for this alg and compute the mean
            curves = np.vstack(alg_df['ConvergenceCurve'].values)
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            
            iterations = np.arange(len(mean_curve))
            plt.plot(iterations, mean_curve, label=alg)
            plt.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        plt.title(f'Convergence Curve for {func} ({dim}D)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f'{path}convergence_{func}_{dim}D.png', dpi=300)
        plt.close()

def plot_boxplots(df, path='plots_new/'):
    """Creates box plots of the final best fitness for each algorithm and function."""
    for dim in df['Dimension'].unique():
        plt.figure(figsize=(16, 10))
        dim_df = df[df['Dimension'] == dim]
        
        sns.boxplot(x='Function', y='BestFitness', hue='Algorithm', data=dim_df)
        
        plt.title(f'Final Fitness Distribution ({dim}D)')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{path}boxplot_fitness_{dim}D.png', dpi=300)
        plt.close()

# In benchmark.py

def create_animation(df, func_id, dim, alg_name, path='plots_new/'):
    """Creates and saves an animation of the swarm's movement for a 2D function."""
    if dim != 2:
        # This check was already correct, but good to confirm.
        # print(f"Animation only available for 2D functions. Skipping for {dim}D.")
        return

    # Get the position history from the first run
    run_data = df[(df['Function'] == f'F{func_id}') & 
                  (df['Dimension'] == dim) & 
                  (df['Algorithm'] == alg_name)].iloc[0]
    
    position_history = run_data['PositionHistory']

    # Create a contour plot of the fitness function
    cec_func_obj = cec2022_func(func_num=func_id)
    x = np.linspace(-100, 100, 150)
    y = np.linspace(-100, 100, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # --- THIS LOOP IS CORRECTED ---
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j]])
            # Call the values method
            cec_func_obj.values(pos.reshape(-1, 1))
            # Access the result from the ObjFunc attribute
            Z[i, j] = cec_func_obj.ObjFunc[0]
    # --- END OF CORRECTION ---

    fig, ax = plt.subplots(figsize=(10, 8))
    # Using log scale for the contour plot can often reveal more detail
    contour = ax.contourf(X, Y, Z, levels=np.logspace(np.log10(Z.min()+1e-8), np.log10(Z.max()), 50), cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Fitness (Log Scale)')
    
    ax.set_title(f'Swarm Animation for {alg_name} on F{func_id} (2D)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Initialize the scatter plot for particles
    particles_scatter = ax.scatter([], [], c='red', zorder=2, label='Particles')
    ax.legend()

    def update(frame):
        positions = position_history[frame]
        particles_scatter.set_offsets(positions)
        ax.set_title(f'Swarm Animation for {alg_name} on F{func_id} (Iteration {frame})')
        return particles_scatter,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(position_history), blit=True, interval=50)
    
    # Save the animation
    anim_filename = f'{path}animation_{alg_name}_F{func_id}_{dim}D.gif'
    anim.save(anim_filename, writer='pillow', fps=15)
    print(f"Saved animation to {anim_filename}")
    plt.close()


if __name__ == "__main__":
    # --- Experiment Configuration ---
    
    # Select which CEC functions to test (1-12)
    # Let's start with a few to keep it fast
    CEC_FUNCTIONS_TO_TEST = [1, 2, 5] 
    
    # Select dimensions to test (CEC2022 supports 2, 10, 20)
    DIMENSIONS_TO_TEST = [2, 10]
    
    # Define the algorithms to compare
    ALGORITHMS_TO_COMPARE = {
        'APSO': MinimizationAPSO,
        'SPSO': MinimizationSPSO
    }
    
    # Set experiment parameters
    NUM_RUNS = 5       # For real results, use 20-30. For testing, 5 is fine.
    MAX_ITERATIONS = 500
    NUM_PARTICLES = 30

    # --- Run the Benchmark ---
    results_df = run_benchmark(
        cec_functions=CEC_FUNCTIONS_TO_TEST,
        dimensions=DIMENSIONS_TO_TEST,
        algorithms=ALGORITHMS_TO_COMPARE,
        num_runs=NUM_RUNS,
        max_iter=MAX_ITERATIONS,
        num_particles=NUM_PARTICLES
    )
    
    # Save raw results to a CSV file for later analysis
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmark finished. Results saved to benchmark_results.csv")

    # --- Generate Visualizations ---
    print("\nGenerating plots...")
    
    # Create a version of the dataframe without the large history objects for plotting
    plotting_df = results_df.drop(columns=['PositionHistory'])
    
    plot_convergence_curves(plotting_df)
    plot_boxplots(plotting_df)
    
    print("Plots saved to the 'plots_new' directory.")

    # --- Generate Animations (for 2D cases) ---
    print("\nGenerating animations for 2D functions...")
    for func_id in CEC_FUNCTIONS_TO_TEST:
        for alg_name in ALGORITHMS_TO_COMPARE.keys():
            create_animation(results_df, func_id=func_id, dim=2, alg_name=alg_name)

    print("\nAll tasks complete.")