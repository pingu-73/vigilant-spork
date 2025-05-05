import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main import APSO, ARPSO, SPSO
from cec import CEC2022Benchmark


class APSO_CEC(APSO):
    def __init__(self, c1=1.193, c2=1.193, w1=0.675, w2=-0.285, num_particles=5, dim=10, max_iter=1000, T=1, threshold=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w1 = w1
        self.w2 = w2
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.T = T
        self.threshold_dis = threshold

        self.positions = np.random.uniform(-100, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.accelerations = np.zeros((num_particles, dim))
    

    def optimize(self, benchmark, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter

        # initialize within bounds
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        accelerations = np.zeros((self.num_particles, self.dim))

        # calculate initial fitness
        fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmin(fitness_values)  # CEC functions are minimization
        g_position = positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        fitness_history = [g_fitness]
        position_history = [positions.copy()]

        for i in range(max_iter):
            # update positions using APSO
            accelerations = self.update_acceleration(
                accelerations, positions, b_positions, g_position
            )
            velocities = self.update_velocity(velocities, accelerations)
            positions = self.update_position(positions, velocities)

            # boundary handling
            positions = np.clip(positions, -100, 100)

            fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])

            # update personal best (for minimization)
            improved = fitness_values < b_fitness
            b_positions[improved] = positions[improved].copy()
            b_fitness[improved] = fitness_values[improved]

            # update global best
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < g_fitness:
                g_position = positions[min_idx].copy()
                g_fitness = fitness_values[min_idx]

            fitness_history.append(g_fitness)
            position_history.append(positions.copy())

        return g_position, g_fitness, fitness_history, position_history


class SPSO_CEC(SPSO):
    def __init__(self, c1=1.193, c2=1.193, w=0.721, num_particles=5, dim=10, max_iter=1000, threshold=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold

        self.positions = np.random.uniform(-100, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(num_particles, np.inf)  # For minimization


    def optimize(self, benchmark, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter

        # initialize within bounds
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))

        # calculate initial fitness
        fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmin(fitness_values)  # CEC functions are minimization
        g_position = positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        fitness_history = [g_fitness]
        position_history = [positions.copy()]

        for i in range(max_iter):
            velocities = self.update_velocity(
                velocities, positions, b_positions, g_position
            )
            positions = self.update_positions(velocities, positions)

            positions = np.clip(positions, -100, 100)

            fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])

            improved = fitness_values < b_fitness
            b_positions[improved] = positions[improved].copy()
            b_fitness[improved] = fitness_values[improved]

            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < g_fitness:
                g_position = positions[min_idx].copy()
                g_fitness = fitness_values[min_idx]

            fitness_history.append(g_fitness)
            position_history.append(positions.copy())

        return g_position, g_fitness, fitness_history, position_history


class ARPSO_CEC(ARPSO):
    def __init__(self, c1=1.193, c2=1.193, c3=0, num_particles=5, dim=10, max_iter=1000, threshold=0.1, sensing_radius=10):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold
        self.sensing_radius = sensing_radius

        self.positions = np.random.uniform(-100, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(num_particles, np.inf)  # For minimization

    
    def optimize(self, benchmark, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter

        # initialize within bounds
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))

        # initial fitness
        fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmin(fitness_values)  # CEC functions are minimization
        g_position = positions[g_index].copy()
        g_fitness = fitness_values[g_index]

        attractive_position = np.random.uniform(
            -100, 100, (self.num_particles, self.dim)
        )

        obstacles = np.array([[0, 0]])

        fitness_history = [g_fitness]
        position_history = [positions.copy()]

        for i in range(max_iter):
            # calculate adaptive parameters
            w = self.calculate_inertia_weight(fitness_values)
            c3_values = self.calculate_c3(positions, obstacles)

            # update velocities and positions
            velocities = self.update_velocity(
                velocities,
                positions,
                b_positions,
                g_position,
                attractive_position,
                w,
                c3_values,
            )
            positions = self.update_position(positions, velocities)

            # boundary handling
            positions = np.clip(positions, -100, 100)

            # evaluate new positions
            fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])

            # update personal best (for minimization)
            improved = fitness_values < b_fitness
            b_positions[improved] = positions[improved].copy()
            b_fitness[improved] = fitness_values[improved]

            # update global best
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < g_fitness:
                g_position = positions[min_idx].copy()
                g_fitness = fitness_values[min_idx]

            fitness_history.append(g_fitness)
            position_history.append(positions.copy())

        return g_position, g_fitness, fitness_history, position_history


def benchmark_on_cec(
    algorithms, func_ids=[1, 2, 3, 4, 5], dim=10, runs=10, max_iter=500
):
    results = {}

    for func_id in func_ids:
        print(f"Testing function F{func_id}")
        benchmark = CEC2022Benchmark(func_num=func_id, dim=dim)

        func_results = {}
        for alg_name, algorithm in algorithms.items():
            print(f"  Running {alg_name}...")
            alg_results = []
            convergence_data = []

            for run in range(runs):
                # new instance
                if alg_name == "APSO":
                    alg = APSO_CEC(
                        c1=algorithm.c1,
                        c2=algorithm.c2,
                        w1=algorithm.w1,
                        w2=algorithm.w2,
                        num_particles=algorithm.num_particles,
                        dim=dim,
                        max_iter=max_iter,
                        T=algorithm.T,
                    )
                elif alg_name == "SPSO":
                    alg = SPSO_CEC(
                        c1=algorithm.c1,
                        c2=algorithm.c2,
                        w=algorithm.w,
                        num_particles=algorithm.num_particles,
                        dim=dim,
                        max_iter=max_iter,
                    )
                else:  # ARPSO
                    alg = ARPSO_CEC(
                        c1=algorithm.c1,
                        c2=algorithm.c2,
                        c3=algorithm.c3,
                        num_particles=algorithm.num_particles,
                        dim=dim,
                        max_iter=max_iter,
                        sensing_radius=algorithm.sensing_radius,
                    )

                # Run optimization
                # if alg_name == "APSO":
                #     _, best_fitness, history = alg.optimize(benchmark, max_iter)
                # else:
                _, best_fitness, history, _ = alg.optimize(benchmark, max_iter)
                
                alg_results.append(best_fitness)
                convergence_data.append(history)

                print(f"    Run {run + 1}/{runs}: {best_fitness:.2e}")

            func_results[alg_name] = {
                "mean": np.mean(alg_results),
                "std": np.std(alg_results),
                "min": np.min(alg_results),
                "max": np.max(alg_results),
                "median": np.median(alg_results),
                "convergence": np.mean(convergence_data, axis=0),
                "all_results": alg_results,
            }

            print(
                f"  {alg_name} results: Mean={func_results[alg_name]['mean']:.2e}, Std={func_results[alg_name]['std']:.2e}"
            )

        results[func_id] = func_results

    return results


def plot_cec_results(results):
    func_ids = list(results.keys())
    alg_names = list(results[func_ids[0]].keys())

    # Mean performance
    plt.figure(figsize=(12, 6))
    for alg_name in alg_names:
        means = [results[f][alg_name]["mean"] for f in func_ids]
        plt.semilogy(func_ids, means, label=alg_name, marker="o")

    plt.title("Mean Performance on CEC2022 Functions")
    plt.xlabel("Function ID")
    plt.ylabel("Mean Function Value (log scale)")
    plt.xticks(func_ids)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cec_mean_performance.png", dpi=300)
    plt.show()

    for func_id in func_ids:
        plt.figure(figsize=(10, 6))
        
        data = [results[func_id][alg_name]["all_results"] for alg_name in alg_names]
        
        plt.boxplot(data, tick_labels=alg_names)
        plt.title(f"Statistical Comparison on Function F{func_id}")
        plt.ylabel("Function Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"cec_boxplot_f{func_id}.png", dpi=300)
        plt.show()

    for func_id in func_ids:
        plt.figure(figsize=(10, 6))
        for alg_name in alg_names:
            convergence = results[func_id][alg_name]["convergence"]
            plt.semilogy(range(len(convergence)), convergence, label=alg_name)

        plt.title(f"Convergence on Function F{func_id}")
        plt.xlabel("Iterations")
        plt.ylabel("Function Value (log scale)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cec_convergence_f{func_id}.png", dpi=300)
        plt.show()


def visualize_optimization_3d(algorithm, benchmark, max_iter=50, func_id=1):
    # Only works for 2D problems
    if algorithm.dim != 2:
        print("3D visualization only works for 2D problems")
        return

    _, _, _, position_history = algorithm.optimize(benchmark, max_iter)

    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = benchmark.evaluate(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, antialiased=True)

    particles = ax.scatter([], [], [], c="red", s=50, label="Particles")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Function Value")
    ax.set_title(f"Optimization Process on CEC2022 Function F{func_id}")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    def update(frame):
        positions = position_history[frame]
        z_values = np.array([benchmark.evaluate(pos) for pos in positions])
        particles._offsets3d = (positions[:, 0], positions[:, 1], z_values)
        ax.set_title(
            f"Optimization Process on CEC2022 Function F{func_id} - Iteration {frame}"
        )
        return (particles,)

    ani = FuncAnimation(
        fig, update, frames=len(position_history), interval=200, blit=True
    )

    ani.save(f"optimization_3d_f{func_id}.mp4", writer="ffmpeg", fps=5, dpi=300)

    plt.show()


def visualize_optimization_2d(algorithm, benchmark, max_iter=50, func_id=1):
    # Only works for 2D problems
    if algorithm.dim != 2:
        print("2D visualization only works for 2D problems")
        return

    _, _, fitness_history, position_history = algorithm.optimize(benchmark, max_iter)

    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = benchmark.evaluate(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax)

    particles = ax.scatter([], [], c="red", s=50, label="Particles")
    best_particle = ax.scatter(
        [], [], c="white", s=100, marker="*", label="Global Best"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Optimization Process on CEC2022 Function F{func_id}")
    ax.legend()

    fitness_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.5)
    )

    def update(frame):
        positions = position_history[frame]
        particles.set_offsets(positions)

        fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])
        best_idx = np.argmin(fitness_values)
        best_particle.set_offsets([positions[best_idx]])

        fitness_text.set_text(
            f"Iteration: {frame}\nBest Fitness: {fitness_history[frame]:.2e}"
        )

        return particles, best_particle, fitness_text

    ani = FuncAnimation(
        fig, update, frames=len(position_history), interval=200, blit=True
    )

    ani.save(f"optimization_2d_f{func_id}.mp4", writer="ffmpeg", fps=5, dpi=300)

    plt.show()


def compare_trajectories(algorithms, benchmark, max_iter=50, func_id=1):
    # Only works for 2D problems
    if list(algorithms.values())[0].dim != 2:
        print("Trajectory comparison only works for 2D problems")
        return

    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = benchmark.evaluate(np.array([X[i, j], Y[i, j]]))

    fig, axes = plt.subplots(1, len(algorithms), figsize=(15, 5))
    if len(algorithms) == 1:
        axes = [axes]  # Make it iterable for single algorithm case

    # run optimization for each algorithm
    for i, (alg_name, algorithm) in enumerate(algorithms.items()):
        _, _, fitness_history, position_history = algorithm.optimize(
            benchmark, max_iter
        )

        contour = axes[i].contourf(X, Y, Z, 50, cmap="viridis", alpha=0.8)  # noqa: F841

        # trajectories of global best
        global_best_positions = []
        for frame in range(len(position_history)):
            positions = position_history[frame]
            fitness_values = np.array([benchmark.evaluate(pos) for pos in positions])
            best_idx = np.argmin(fitness_values)
            global_best_positions.append(positions[best_idx])

        global_best_positions = np.array(global_best_positions)
        axes[i].plot(
            global_best_positions[:, 0], global_best_positions[:, 1], "r-", linewidth=2
        )
        axes[i].scatter(
            global_best_positions[-1, 0],
            global_best_positions[-1, 1],
            c="white",
            s=100,
            marker="*",
        )

        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_title(f"{alg_name} on F{func_id}")

    plt.tight_layout()
    plt.savefig(f"trajectory_comparison_f{func_id}.png", dpi=300)
    plt.show()


def create_performance_heatmap(results):
    func_ids = list(results.keys())
    alg_names = list(results[func_ids[0]].keys())

    # create data matrix
    data = np.zeros((len(alg_names), len(func_ids)))
    for i, alg_name in enumerate(alg_names):
        for j, func_id in enumerate(func_ids):
            data[i, j] = results[func_id][alg_name]["mean"]

    # normalize data for better visualization
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        if col_max > col_min:
            data_normalized[:, j] = (data[:, j] - col_min) / (col_max - col_min)
        else:
            data_normalized[:, j] = 0.5

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data_normalized,
        annot=True,
        cmap="viridis",
        xticklabels=[f"F{f}" for f in func_ids],
        yticklabels=alg_names,
    )
    plt.title("Relative Performance Across Functions (Lower is Better)")
    plt.tight_layout()
    plt.savefig("performance_heatmap.png", dpi=300)
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=True,
        cmap="viridis",
        fmt=".2e",
        xticklabels=[f"F{f}" for f in func_ids],
        yticklabels=alg_names,
    )
    plt.title("Mean Function Values Across Functions")
    plt.tight_layout()
    plt.savefig("performance_heatmap_values.png", dpi=300)
    plt.show()


def create_radar_chart(results):
    func_ids = list(results.keys())
    alg_names = list(results[func_ids[0]].keys())

    data = np.zeros((len(alg_names), len(func_ids)))
    for i, alg_name in enumerate(alg_names):
        for j, func_id in enumerate(func_ids):
            data[i, j] = results[func_id][alg_name]["mean"]

    # normalize data for radar chart
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        if col_max > col_min:
            # invert normalization so lower values are better
            data_normalized[:, j] = 1 - (data[:, j] - col_min) / (col_max - col_min)
        else:
            data_normalized[:, j] = 0.5

    # radar chart
    angles = np.linspace(0, 2 * np.pi, len(func_ids), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, alg_name in enumerate(alg_names):
        values = data_normalized[i].tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, linewidth=2, label=alg_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"F{f}" for f in func_ids])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(0, 1)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Algorithm Performance Across Functions (Higher is Better)")
    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=300)
    plt.show()


def plot_function_surface(benchmark, func_id, bounds=(-100, 100), resolution=100):
    # Only works for 2D problems
    if benchmark.dim != 2:
        print("Surface plot only works for 2D problems")
        return

    # grid for the function surface
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = benchmark.evaluate(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True
    )

    offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
    contour = ax.contourf(X, Y, Z, zdir="z", offset=offset, cmap="viridis", alpha=0.5)  # noqa: F841

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Function Value")
    ax.set_title(f"CEC2022 Function F{func_id} Surface")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f"function_surface_f{func_id}.png", dpi=300)
    plt.show()

    # 2D contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, 50, cmap="viridis")
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"CEC2022 Function F{func_id} Contour")
    plt.tight_layout()
    plt.savefig(f"function_contour_f{func_id}.png", dpi=300)
    plt.show()


dim = 10
func_ids = [1, 2, 3, 4, 5]

apso_cec = APSO_CEC(
    c1=1.193,
    c2=1.193,
    w1=0.675,
    w2=-0.285,
    num_particles=30,
    dim=dim,
    max_iter=500,
    T=1,
)

spso_cec = SPSO_CEC(
    c1=1.193,
    c2=1.193,
    w=0.721,
    num_particles=30,
    dim=dim,
    max_iter=500,
)

arpso_cec = ARPSO_CEC(
    c1=1.193,
    c2=1.193,
    c3=0,
    num_particles=30,
    dim=dim,
    max_iter=500,
    sensing_radius=10,
)

algorithms = {"APSO": apso_cec, "SPSO": spso_cec, "ARPSO": arpso_cec}

results = benchmark_on_cec(algorithms, func_ids=func_ids, dim=dim, runs=5, max_iter=200)

plot_cec_results(results)
create_performance_heatmap(results)
create_radar_chart(results)

if dim >= 2:
    for func_id in func_ids:
        benchmark_2d = CEC2022Benchmark(func_num=func_id, dim=2)

        plot_function_surface(benchmark_2d, func_id)

        apso_2d = APSO_CEC(
            c1=apso_cec.c1,
            c2=apso_cec.c2,
            w1=apso_cec.w1,
            w2=apso_cec.w2,
            num_particles=20,
            dim=2,
            max_iter=50,
            T=apso_cec.T,
        )
        visualize_optimization_3d(apso_2d, benchmark_2d, max_iter=50, func_id=func_id)

        visualize_optimization_2d(apso_2d, benchmark_2d, max_iter=50, func_id=func_id)

        algorithms_2d = {
            "APSO": APSO_CEC(
                c1=apso_cec.c1,
                c2=apso_cec.c2,
                w1=apso_cec.w1,
                w2=apso_cec.w2,
                num_particles=20,
                dim=2,
                max_iter=50,
                T=apso_cec.T,
            ),
            "SPSO": SPSO_CEC(
                c1=spso_cec.c1, c2=spso_cec.c2, w=spso_cec.w, num_particles=20, dim=2, max_iter=50
            ),
            "ARPSO": ARPSO_CEC(
                c1=arpso_cec.c1,
                c2=arpso_cec.c2,
                c3=arpso_cec.c3,
                num_particles=20,
                dim=2,
                max_iter=50,
                sensing_radius=arpso_cec.sensing_radius,
            ),
        }
        compare_trajectories(algorithms_2d, benchmark_2d, max_iter=50, func_id=func_id)
