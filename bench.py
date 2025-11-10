import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import os
import json

from main import APSO, SPSO, ARPSO 

from cec import cec2022_func


if not os.path.exists('plots_new2'):
    os.makedirs('plots_new2')


class MinimizationAPSO(APSO):
    def __init__(self, num_particles, max_iter, dim=2, w1=0.675, w2=-0.285, c1=1.193, c2=1.193):
        super().__init__(num_particles=num_particles, max_iter=max_iter, dim=dim, 
                         w1=w1, w2=w2, c1=c1, c2=c2)
        self.dim = dim
        self.positions = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.accelerations = np.zeros((self.num_particles, self.dim))

    def run(self, fitness_func, dim, bounds):
        if self.dim != dim:
            raise ValueError(f"Object initialized with dim={self.dim} but run with dim={dim}")
        
        self.positions = np.random.uniform(bounds[0], bounds[1], (self.num_particles, dim))
        self.velocities.fill(0)
        self.accelerations.fill(0)

        fitness_values = np.array([fitness_func(p) for p in self.positions])
        
        b_positions = self.positions.copy()
        b_fitness = fitness_values.copy()
        
        g_index = np.argmin(fitness_values)
        g_position = self.positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        convergence_curve = [g_fitness]
        position_history = [self.positions.copy()]

        for i in range(self.max_iter):
            self.accelerations = self.update_acceleration(self.accelerations, self.positions, b_positions, g_position)
            self.velocities = self.update_velocity(self.velocities, self.accelerations)
            self.positions = self.update_position(self.positions, self.velocities)
            
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

class MinimizationSPSO(SPSO):
    def __init__(self, num_particles, max_iter, dim=2, w=0.721, c1=1.193, c2=1.193):
        self.w = w
        self.c1 = c1
        self.c2 = c2
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


def run_benchmark(cec_functions, dimensions, algorithms, num_runs, max_iter, num_particles):
    results_list = []
    bounds = [-100, 100]

    invalid_combinations = {
        2: [6, 7, 8]
    }


    for func_id in cec_functions:
        for dim in dimensions:
            if dim in invalid_combinations and func_id in invalid_combinations[dim]:
                print(f"--- SKIPPING F{func_id} in {dim}D (not defined in benchmark) ---")
                continue

            print(f"--- Testing F{func_id} in {dim}D ---")
            
            cec_func_obj = cec2022_func(func_num=func_id)

            def fitness_wrapper(x):
                particle_position = x.reshape(-1, 1)
                cec_func_obj.values(particle_position)
                return cec_func_obj.ObjFunc[0]

            for alg_name, alg_class in algorithms.items():
                print(f"  -> Algorithm: {alg_name}")
                for run in range(num_runs):
                    print(f"    -> Run: {run + 1}/{num_runs}")
                    
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


def plot_convergence_curves(df, path='plots_new2/'):
    for (func, dim), group in df.groupby(['Function', 'Dimension']):
        plt.figure(figsize=(12, 8))
        
        for alg in group['Algorithm'].unique():
            alg_df = group[group['Algorithm'] == alg]
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

def plot_boxplots(df, path='plots_new2/'):
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


def create_animation(df, func_id, dim, alg_name, path='plots_new2/'):
    if dim != 2:
        return

    run_data = df[(df['Function'] == f'F{func_id}') & 
                  (df['Dimension'] == dim) & 
                  (df['Algorithm'] == alg_name)].iloc[0]
    
    position_history = run_data['PositionHistory']

    cec_func_obj = cec2022_func(func_num=func_id)
    x = np.linspace(-100, 100, 150)
    y = np.linspace(-100, 100, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j]])
            
            cec_func_obj.values(pos.reshape(-1, 1))
            Z[i, j] = cec_func_obj.ObjFunc[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=np.logspace(np.log10(Z.min()+1e-8), np.log10(Z.max()), 50), cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Fitness (Log Scale)')
    
    ax.set_title(f'Swarm Animation for {alg_name} on F{func_id} (2D)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    particles_scatter = ax.scatter([], [], c='red', zorder=2, label='Particles')
    ax.legend()

    def update(frame):
        positions = position_history[frame]
        particles_scatter.set_offsets(positions)
        ax.set_title(f'Swarm Animation for {alg_name} on F{func_id} (Iteration {frame})')
        return particles_scatter,

    anim = FuncAnimation(fig, update, frames=len(position_history), blit=True, interval=50)
    
    anim_filename = f'{path}animation_{alg_name}_F{func_id}_{dim}D.gif'
    anim.save(anim_filename, writer='pillow', fps=15)
    print(f"Saved animation to {anim_filename}")
    plt.close()



def plot_radar_chart(df, path='plots_new2/'):
    for dim in df['Dimension'].unique():
        dim_df = df[df['Dimension'] == dim]
        
        mean_fitness = dim_df.groupby(['Function', 'Algorithm'])['BestFitness'].mean().reset_index()
        
        mean_fitness['Rank'] = mean_fitness.groupby('Function')['BestFitness'].rank()
        
        pivot_df = mean_fitness.pivot(index='Function', columns='Algorithm', values='Rank')
        
        functions = pivot_df.index.tolist()
        algorithms = pivot_df.columns.tolist()

        fig = go.Figure()

        for alg in algorithms:
            ranks = pivot_df[alg].values
            fig.add_trace(go.Scatterpolar(
                r=ranks,
                theta=functions,
                fill='toself',
                name=alg
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, len(algorithms) + 0.5],
                    tickvals=list(range(1, len(algorithms) + 1)),
                    ticktext=[f"Rank {i}" for i in range(1, len(algorithms) + 1)]
                )
            ),
            showlegend=True,
            title=f'Algorithm Performance Ranking ({dim}D) - Closer to Center is Better'
        )
        
        fig.write_html(f"{path}radar_chart_{dim}D.html")
        try:
            fig.write_image(f"{path}radar_chart_{dim}D.png", scale=2)
        except ValueError as e:
            print(f"Could not save static radar chart image. Error: {e}")


def tune_hyperparameters(tuning_config):
    print("--- STARTING HYPERPARAMETER TUNING ---")
    best_params = {}

    bounds = [-100, 100]
    
    tuning_functions = tuning_config['functions']
    tuning_dims = tuning_config['dims']
    num_tuning_runs = tuning_config['num_runs']
    
    for alg_name, params_to_tune in tuning_config['algorithms'].items():
        print(f"\n--- Tuning Algorithm: {alg_name} ---")
        
        param_names = list(params_to_tune.keys())
        param_values = list(params_to_tune.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        best_avg_fitness = float('inf')
        current_best_combo = None

        for combo in param_combinations:
            current_params = dict(zip(param_names, combo))
            
            total_fitness = 0
            num_evals = 0

            for func_id in tuning_functions:
                for dim in tuning_dims:
                    cec_func_obj = cec2022_func(func_num=func_id)
                    def fitness_wrapper(x):
                        particle_position = x.reshape(-1, 1)
                        cec_func_obj.values(particle_position)
                        return cec_func_obj.ObjFunc[0]

                    alg_class = MinimizationAPSO if alg_name == 'APSO' else MinimizationSPSO
                    
                    for _ in range(num_tuning_runs):
                        algorithm = alg_class(
                            num_particles=tuning_config['num_particles'],
                            max_iter=tuning_config['max_iter'],
                            dim=dim,
                            **current_params # Unpack the hyperparameter combo
                        )
                        
                        final_fitness, _, _, _ = algorithm.run(fitness_wrapper, dim, bounds)
                        total_fitness += final_fitness
                        num_evals += 1
            
            avg_fitness = total_fitness / num_evals
            print(f"  Tested {current_params}, Avg Fitness: {avg_fitness:.4e}")

            if avg_fitness < best_avg_fitness:
                best_avg_fitness = avg_fitness
                current_best_combo = current_params
        
        best_params[alg_name] = current_best_combo
        print(f"\n  ==> Best parameters for {alg_name}: {current_best_combo} (Avg Fitness: {best_avg_fitness:.4e})")

    print("\n--- HYPERPARAMETER TUNING FINISHED ---")
    return best_params

if __name__ == "__main__":
    hyperparams_cache_file = 'best_hyperparams.json'
    if os.path.exists(hyperparams_cache_file):
        print(f"--- LOADING tuned hyperparameters from {hyperparams_cache_file} ---")
        with open(hyperparams_cache_file, 'r') as f:
            best_hyperparams = json.load(f)
    else:
        tuning_config = {
            'functions': [1, 5, 11],
            'dims': [10],
            'num_runs': 3,
            'num_particles': 30,
            'max_iter': 200,
            'algorithms': {
                'SPSO': {
                    'w': [0.4, 0.6, 0.721, 0.8],
                    'c1': [1.193, 1.5, 2.0],
                    'c2': [1.193, 1.5, 2.0]
                },
                'APSO': {
                    'w1': [0.5, 0.675, 0.8],
                    'w2': [-0.1, -0.285, -0.4],
                    'c1': [1.193, 1.5, 2.0],
                    'c2': [1.193, 1.5, 2.0]
                }
            }
        }
        
        best_hyperparams = tune_hyperparameters(tuning_config)
        print(f"--- SAVING tuned hyperparameters to {hyperparams_cache_file} ---")
        with open(hyperparams_cache_file, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)

    print("\nUsing the following hyperparameters for the final benchmark:")
    print(json.dumps(best_hyperparams, indent=4))

    print("\n--- STARTING FINAL BENCHMARK WITH TUNED PARAMETERS ---")
    TunedMinimizationAPSO = lambda **kwargs: MinimizationAPSO(**kwargs, **best_hyperparams['APSO'])
    TunedMinimizationSPSO = lambda **kwargs: MinimizationSPSO(**kwargs, **best_hyperparams['SPSO'])

    CEC_FUNCTIONS_TO_TEST = list(range(1, 13)) 

    DIMENSIONS_TO_TEST = [2, 10, 20]
    
    ALGORITHMS_TO_COMPARE = {
        # 'APSO': MinimizationAPSO,
        # 'SPSO': MinimizationSPSO
        'APSO_Tuned': TunedMinimizationAPSO,
        'SPSO_Tuned': TunedMinimizationSPSO
    }
    
    NUM_RUNS = 20
    MAX_ITERATIONS = 500
    NUM_PARTICLES = 30

    results_df = run_benchmark(
        cec_functions=CEC_FUNCTIONS_TO_TEST,
        dimensions=DIMENSIONS_TO_TEST,
        algorithms=ALGORITHMS_TO_COMPARE,
        num_runs=NUM_RUNS,
        max_iter=MAX_ITERATIONS,
        num_particles=NUM_PARTICLES
    )
    
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmark finished. Results saved to benchmark_results.csv")


    print("\nGenerating plots...")
    
    plotting_df = results_df.drop(columns=['PositionHistory'])
    
    plot_convergence_curves(plotting_df)
    plot_boxplots(plotting_df)
    plot_radar_chart(plotting_df)
    
    print("Plots saved to the 'plots_new2' directory.")

    print("\nGenerating animations for 2D functions...")
    for func_id in CEC_FUNCTIONS_TO_TEST:
        for alg_name in ALGORITHMS_TO_COMPARE.keys():
            create_animation(results_df, func_id=func_id, dim=2, alg_name=alg_name)

    print("\nAll tasks complete.")