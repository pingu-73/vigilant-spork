import matplotlib.pyplot as plt
import numpy as np

def fitness_signal_strength(pos, src_pos, src_pow=100, a=0.1, noise=0.0):
    distance = np.linalg.norm(pos - src_pos)
    signal_strength = src_pow * np.exp(-a * distance**2)
    noise_amount = noise * signal_strength
    rnd_noise = np.random.normal(0, noise_amount)
    return max(0, signal_strength + rnd_noise)

class APSO:
    def __init__(self, c1=1.193, c2=1.193, w1=0.675, w2=-0.285, num_particles=5, dim=2, max_iter=1000, T=1, threshold=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w1 = w1
        self.w2 = w2
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.T = T
        self.threshold_dis = threshold

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.accelerations = np.zeros((num_particles, dim))

    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.accelerations = np.zeros((self.num_particles, self.dim))

    def update_acceleration(self, accelerations, positions, p_best, g_best):
        r1 = np.random.uniform(0, self.c1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, self.c2, (self.num_particles, self.dim))
        accelerations = self.w1 * accelerations + r1 * (p_best - positions) + r2 * (g_best - positions)
        return accelerations

    def update_velocity(self, velocities, accelerations):
        velocities = self.w2 * velocities + accelerations * self.T
        return velocities

    def update_position(self, positions, velocities):
        positions = positions + velocities * self.T
        return positions

    def apso(self, src_position=np.array([80, 34])):
        # Initialize positions and velocities
        positions = self.positions.copy()
        velocities = self.velocities.copy()
        accelerations = self.accelerations.copy()
        
        # Calculate initial fitness
        fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmax(fitness_values)
        g_position = positions[g_index].copy()
        
        total_distance = 0

        for i in range(self.max_iter):
            # Update positions
            accelerations = self.update_acceleration(accelerations, positions, b_positions, g_position)
            velocities = self.update_velocity(velocities, accelerations)
            positions = self.update_position(positions, velocities)

            total_distance += np.sum(np.linalg.norm(velocities, axis=1))

            # Calculate new fitness values
            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])

            # Update personal best
            for j in range(self.num_particles):
                if fitness_values[j] > b_fitness[j]:
                    b_positions[j] = positions[j].copy()
                    b_fitness[j] = fitness_values[j]
            
            # Update global best
            if np.max(fitness_values) > fitness_signal_strength(g_position, src_position):
                g_index = np.argmax(fitness_values)
                g_position = positions[g_index].copy()

            # Check if any UAV is close enough to the source
            if np.min([np.linalg.norm(pos - src_position) for pos in positions]) < self.threshold_dis:
                # Update class state before returning
                self.positions = positions
                self.velocities = velocities
                self.accelerations = accelerations
                return i + 1, total_distance

        # Update class state before returning
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        return self.max_iter, total_distance

class SPSO:
    def __init__(self, c1=1.193, c2=1.193, w=0.721, num_particles=5, dim=2, max_iter=1000, threshold=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.array([fitness_signal_strength(pos, np.array([80, 34])) for pos in self.positions])

    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.array([fitness_signal_strength(pos, np.array([80, 34])) for pos in self.positions])

    def update_velocity(self, velocities, positions, p_best, g_best):
        r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        velocities = self.w * velocities + r1*self.c1*(p_best - positions) + r2*self.c2*(g_best - positions)
        return velocities

    def update_positions(self, velocities, positions):
        positions = positions + velocities
        return positions

    def spso(self, src_position=np.array([80, 34])):
        # Initialize positions and velocities
        positions = self.positions.copy()
        velocities = self.velocities.copy()
        
        # Calculate initial fitness
        fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmax(fitness_values)
        g_position = positions[g_index].copy()
        g_fitness = fitness_values[g_index]
        
        total_distance = 0

        for i in range(self.max_iter):
            # Update positions
            velocities = self.update_velocity(velocities, positions, b_positions, g_position)
            positions = self.update_positions(velocities, positions)

            total_distance += np.sum(np.linalg.norm(velocities, axis=1))

            # Calculate new fitness values
            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])

            # Update personal best
            for j in range(self.num_particles):
                if fitness_values[j] > b_fitness[j]:
                    b_positions[j] = positions[j].copy()
                    b_fitness[j] = fitness_values[j]
            
            # Update global best
            if np.max(fitness_values) > g_fitness:
                g_index = np.argmax(fitness_values)
                g_position = positions[g_index].copy()
                g_fitness = fitness_values[g_index]

            # Check if any UAV is close enough to the source
            if np.min([np.linalg.norm(pos - src_position) for pos in positions]) < self.threshold_dis:
                # Update class state before returning
                self.positions = positions
                self.velocities = velocities
                self.b_positions = b_positions
                self.b_scores = b_fitness
                return i + 1, total_distance

        # Update class state before returning
        self.positions = positions
        self.velocities = velocities
        self.b_positions = b_positions
        self.b_scores = b_fitness
        return self.max_iter, total_distance

class ARPSO:
    def __init__(self, c1=1.193, c2=1.193, c3=0, num_particles=5, dim=2, max_iter=1000, threshold=0.1, sensing_radius=10):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold
        self.sensing_radius = sensing_radius

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.array([fitness_signal_strength(pos, np.array([80, 34])) for pos in self.positions])

    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.array([fitness_signal_strength(pos, np.array([80, 34])) for pos in self.positions])

    def calculate_inertia_weight(self, fitness_values):
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        
        # Avoid division by zero
        if max_fitness == min_fitness:
            return np.ones(self.num_particles)
            
        evolutionary_speed = 1 - (min_fitness / max_fitness)
        aggregation_degree = min_fitness / max_fitness
        
        return 1 * (1 - 0.5 * evolutionary_speed + 0.5 * aggregation_degree) * np.ones(self.num_particles)

    def calculate_c3(self, positions, obstacles):
        c3 = np.zeros(self.num_particles)

        # using only the first 2 dimensions for obstacle avoidance
        for i, pos in enumerate(positions):
            distance_to_obstacles = np.linalg.norm(obstacles - pos[:2], axis=1)
            if np.any(distance_to_obstacles < self.sensing_radius):
                c3[i] = 2 * self.c1 + self.c2
        return c3

    def update_velocity(self, velocities, positions, p_best, g_best, attractive_pos, w, c3_values):
        r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r3 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        
        velocities = (
            w[:, np.newaxis] * velocities +
            self.c1 * r1 * (p_best - positions) +
            self.c2 * r2 * (g_best - positions) +
            c3_values[:, np.newaxis] * r3 * (attractive_pos - positions)
        )
        return velocities

    def update_position(self, positions, velocities):
        positions += velocities
        return positions

    def arpso(self, src_position=np.array([80, 34]), obstacles=np.array([[50, 50]])):
        # Initialize positions and velocities
        positions = self.positions.copy()
        velocities = self.velocities.copy()
        
        # Calculate initial fitness
        fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])
        b_positions = positions.copy()
        b_fitness = fitness_values.copy()
        g_index = np.argmax(fitness_values)
        g_position = positions[g_index].copy()
        g_fitness = fitness_values[g_index]
        
        # Initialize attractive positions (random points for UAVs to explore)
        attractive_position = np.random.uniform(0, 100, (self.num_particles, self.dim))
        
        total_distance = 0

        for i in range(self.max_iter):
            # Calculate adaptive parameters
            w = self.calculate_inertia_weight(fitness_values)
            c3_values = self.calculate_c3(positions, obstacles)
            
            # Update positions
            velocities = self.update_velocity(velocities, positions, b_positions, g_position, attractive_position, w, c3_values)
            positions = self.update_position(positions, velocities)

            total_distance += np.sum(np.linalg.norm(velocities, axis=1))

            # Calculate new fitness values
            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])

            # Update personal best
            for j in range(self.num_particles):
                if fitness_values[j] > b_fitness[j]:
                    b_positions[j] = positions[j].copy()
                    b_fitness[j] = fitness_values[j]
            
            # Update global best
            if np.max(fitness_values) > g_fitness:
                g_index = np.argmax(fitness_values)
                g_position = positions[g_index].copy()
                g_fitness = fitness_values[g_index]

            # Check if any UAV is close enough to the source
            if np.min([np.linalg.norm(pos - src_position) for pos in positions]) < self.threshold_dis:
                # Update class state before returning
                self.positions = positions
                self.velocities = velocities
                self.b_positions = b_positions
                self.b_scores = b_fitness
                return i + 1, total_distance

        # Update class state before returning
        self.positions = positions
        self.velocities = velocities
        self.b_positions = b_positions
        self.b_scores = b_fitness
        return self.max_iter, total_distance

def stability_analysis(apso_instance):
    w1, w2, c1, c2, T = apso_instance.w1, apso_instance.w2, apso_instance.c1, apso_instance.c2, apso_instance.T
    stability_1 = (2 / T) * (1 + w1 + w2 + w1 * w2) > c1 + c2
    stability_2 = abs(w1 * w2) < 1
    stability_3 = abs((1 - w1 * w2) * (w1 + w2) + (w1 * w2) * (c1 * T + c2 * T)) < abs(1 - (w1 * w2)**2)
    stability_4 = abs(w1 + w2) < abs(1 + w1 * w2)

    is_stable = stability_1 and stability_2 and stability_3 and stability_4
    print(f"Stability Analysis: {'Passed' if is_stable else 'Failed'}")
    print(f"Condition 1: {stability_1}")
    print(f"Condition 2: {stability_2}")
    print(f"Condition 3: {stability_3}")
    print(f"Condition 4: {stability_4}")

def convergence_analysis(apso_instance, src_position=np.array([80, 34])):
    # Create a new instance to avoid modifying the original
    apso = APSO(
        c1=apso_instance.c1, 
        c2=apso_instance.c2, 
        w1=apso_instance.w1, 
        w2=apso_instance.w2,
        num_particles=apso_instance.num_particles, 
        dim=apso_instance.dim, 
        max_iter=apso_instance.max_iter,
        T=apso_instance.T, 
        threshold=apso_instance.threshold_dis
    )
    
    # Initialize tracking variables
    global_best_fitness = []
    positions = apso.positions.copy()
    velocities = apso.velocities.copy()
    accelerations = apso.accelerations.copy()
    
    # Calculate initial fitness
    fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])
    b_positions = positions.copy()
    b_fitness = fitness_values.copy()
    g_index = np.argmax(fitness_values)
    g_position = positions[g_index].copy()
    g_fitness = fitness_values[g_index]
    
    # Record initial global best fitness
    global_best_fitness.append(g_fitness)
    
    # Run for 24 more iterations
    for _ in range(24):
        # Update positions using APSO algorithm
        accelerations = apso.update_acceleration(accelerations, positions, b_positions, g_position)
        velocities = apso.update_velocity(velocities, accelerations)
        positions = apso.update_position(positions, velocities)
        
        # Calculate new fitness values
        fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in positions])
        # print(f"Iteration {_}, fitness values: {fitness_values}")
        # print(f"Global best fitness: {g_fitness}")

        # Update personal best
        for j in range(apso.num_particles):
            if fitness_values[j] > b_fitness[j]:
                b_positions[j] = positions[j].copy()
                b_fitness[j] = fitness_values[j]
        
        # Update global best
        current_max = np.max(fitness_values)
        if current_max > g_fitness:
            g_index = np.argmax(fitness_values)
            g_position = positions[g_index].copy()
            g_fitness = current_max
        
        # Record the global best fitness
        global_best_fitness.append(g_fitness)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(25), global_best_fitness, label="Global Best Fitness")
    plt.title("Convergence Analysis of APSO", fontsize=16)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Global Best Fitness", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("./plots/convergence_analysis.png", dpi = 300)
    plt.show()
    plt.close()

class PerformanceMetrics:
    def __init__(self, algorithms, num_simulations=10, uav_counts=None, src_position=np.array([80, 34])):
        self.algorithms = algorithms
        self.num_simulations = num_simulations
        self.uav_counts = uav_counts or [5, 10, 15, 20]
        self.src_position = src_position

    def evaluate(self):
        results = {alg_name: {"time": [], "iterations": [], "distance": []} for alg_name in self.algorithms.keys()}

        for num_uavs in self.uav_counts:
            for alg_name, alg in self.algorithms.items():
                alg_results = {"time": [], "iterations": [], "distance": []}

                for _ in range(self.num_simulations):
                    # Create a new instance with the current number of UAVs
                    if alg_name == "APSO":
                        algorithm = APSO(
                            c1=alg.c1, c2=alg.c2, w1=alg.w1, w2=alg.w2,
                            num_particles=num_uavs, dim=alg.dim, max_iter=alg.max_iter,
                            T=alg.T, threshold=alg.threshold_dis
                        )
                    elif alg_name == "SPSO":
                        algorithm = SPSO(
                            c1=alg.c1, c2=alg.c2, w=alg.w,
                            num_particles=num_uavs, dim=alg.dim, max_iter=alg.max_iter,
                            threshold=alg.threshold_dis
                        )
                    elif alg_name == "ARPSO":
                        algorithm = ARPSO(
                            c1=alg.c1, c2=alg.c2, c3=alg.c3,
                            num_particles=num_uavs, dim=alg.dim, max_iter=alg.max_iter,
                            threshold=alg.threshold_dis, sensing_radius=alg.sensing_radius
                        )

                    # Dynamically call the appropriate method
                    method = getattr(algorithm, alg_name.lower())
                    iterations, distance = method(self.src_position)

                    # Collect metrics
                    alg_results["time"].append(iterations)  # 1 iteration = 1 unit time
                    alg_results["iterations"].append(iterations)
                    alg_results["distance"].append(distance)

                # Compute averages
                results[alg_name]["time"].append(np.mean(alg_results["time"]))
                results[alg_name]["iterations"].append(np.mean(alg_results["iterations"]))
                results[alg_name]["distance"].append(np.mean(alg_results["distance"]))

        return results

    def plot_results(self, results):
        for metric in ["time", "iterations", "distance"]:
            plt.figure(figsize=(10, 6))
            for alg_name in self.algorithms.keys():
                plt.plot(self.uav_counts, results[alg_name][metric], label=alg_name, marker='o')

            plt.title(f"Average {metric.capitalize()} vs Number of UAVs", fontsize=16)
            plt.xlabel("Number of UAVs", fontsize=12)
            plt.ylabel(f"Average {metric.capitalize()}", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            # plt.savefig(f"./plots/{metric}_vs_#UAVs.png", dpi = 300)
            plt.show()
            plt.close()


# Create algorithm instances
apso_sim = APSO()
spso_sim = SPSO()
arpso_sim = ARPSO()

# Run individual simulations and compare
dis_apso_list = []
dis_spso_list = []
dis_arpso_list = []
iter_apso_list = []
iter_spso_list = []
iter_arpso_list = []
    
print("APSO\t\t\t SPSO\t\t\t ARPSO")
print("Iter\t Dist\t\t Iter\t Dist\t\t Iter\t Dist")
print("----------------------------------------------------------------")
    
for _ in range(10):
    # Create new instances for each run to ensure independence
    apso_sim = APSO()
    spso_sim = SPSO()
    arpso_sim = ARPSO()
        
    iterations_apso, distance_apso = apso_sim.apso()
    iterations_spso, distance_spso = spso_sim.spso()
    iterations_arpso, distance_arpso = arpso_sim.arpso()

    dis_apso_list.append(distance_apso)
    dis_spso_list.append(distance_spso)
    dis_arpso_list.append(distance_arpso)
        
    iter_apso_list.append(iterations_apso)
    iter_spso_list.append(iterations_spso)
    iter_arpso_list.append(iterations_arpso)

    print(f"{iterations_apso:<8} {distance_apso:<8.3f}\t {iterations_spso:<8} {distance_spso:<8.3f}\t {iterations_arpso:<8} {distance_arpso:<8.3f}")

# Plot distance comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), dis_apso_list, label='APSO Distance', marker='o')
plt.plot(range(1, 11), dis_spso_list, label='SPSO Distance', marker='s')
plt.plot(range(1, 11), dis_arpso_list, label='ARPSO Distance', marker='*')
plt.title("APSO vs SPSO vs ARPSO Performance", fontsize=16)
plt.xlabel("Simulation Run", fontsize=12)
plt.ylabel("Distance Traveled to Find Source", fontsize=12)
plt.xticks(range(1, 11))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()
    
# Plot iteration comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), iter_apso_list, label='APSO Iterations', marker='o')
plt.plot(range(1, 11), iter_spso_list, label='SPSO Iterations', marker='s')
plt.plot(range(1, 11), iter_arpso_list, label='ARPSO Iterations', marker='*')
plt.title("APSO vs SPSO vs ARPSO Convergence Speed", fontsize=16)
plt.xlabel("Simulation Run", fontsize=12)
plt.ylabel("Iterations to Find Source", fontsize=12)
plt.xticks(range(1, 11))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


metrics = PerformanceMetrics(
    algorithms={"APSO": apso_sim, "SPSO": spso_sim, "ARPSO": arpso_sim},
    num_simulations=5,  # Reduced for faster execution
    uav_counts=[5, 10, 15, 20]
)

results = metrics.evaluate()
metrics.plot_results(results)

stability_analysis(apso_sim)
convergence_analysis(apso_sim)



def noise_analysis(algorithms, noise_levels=[0.0, 0.05, 0.1], num_simulations=10, uav_count=10, src_position=np.array([80, 34])):
    results = {alg_name: [] for alg_name in algorithms.keys()}
    
    # Store the original fitness function
    original_fitness = globals()['fitness_signal_strength']
    
    for noise in noise_levels:
        print(f"Testing noise level: {noise}")
        
        # Create a modified fitness function with the current noise level
        def current_fitness(pos, src_pos, src_pow=100, a=0.1):
            return original_fitness(pos, src_pos, src_pow, a, noise=noise)
        
        # Replace the global fitness function temporarily
        globals()['fitness_signal_strength'] = current_fitness
        
        for alg_name, alg in algorithms.items():
            total_iterations, total_distance = 0, 0
            
            for sim in range(num_simulations):
                # Create a fresh instance for each simulation
                if alg_name == "APSO":
                    algorithm = APSO(
                        c1=alg.c1, c2=alg.c2, w1=alg.w1, w2=alg.w2,
                        num_particles=uav_count, dim=alg.dim, max_iter=alg.max_iter,
                        T=alg.T, threshold=alg.threshold_dis
                    )
                elif alg_name == "SPSO":
                    algorithm = SPSO(
                        c1=alg.c1, c2=alg.c2, w=alg.w,
                        num_particles=uav_count, dim=alg.dim, max_iter=alg.max_iter,
                        threshold=alg.threshold_dis
                    )
                elif alg_name == "ARPSO":
                    algorithm = ARPSO(
                        c1=alg.c1, c2=alg.c2, c3=alg.c3,
                        num_particles=uav_count, dim=alg.dim, max_iter=alg.max_iter,
                        threshold=alg.threshold_dis, sensing_radius=alg.sensing_radius
                    )
                
                # Run the algorithm
                method = getattr(algorithm, alg_name.lower())
                iterations, distance = method(src_position)
                total_iterations += iterations
                total_distance += distance
                
                print(f"  {alg_name} simulation {sim+1}/{num_simulations}: {iterations} iterations, {distance:.2f} distance")
            
            avg_iterations = total_iterations / num_simulations
            avg_distance = total_distance / num_simulations
            print(f"  {alg_name} average: {avg_iterations:.2f} iterations, {avg_distance:.2f} distance")
            
            results[alg_name].append({
                "noise": noise,
                "avg_iterations": avg_iterations,
                "avg_distance": avg_distance,
            })
    
    # Restore the original fitness function
    globals()['fitness_signal_strength'] = original_fitness
    
    return results



def plot_noise_results(results):
    for metric in ["avg_iterations", "avg_distance"]:
        plt.figure(figsize=(10, 6))
        for alg_name, metrics in results.items():
            noise_levels = [m["noise"] for m in metrics]
            values = [m[metric] for m in metrics]
            plt.plot(noise_levels, values, label=alg_name, marker='o')

        plt.title(f"Effect of Noise on {metric.replace('_', ' ').capitalize()}", fontsize=16)
        plt.xlabel("Noise Level", fontsize=12)
        plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(f"./plots/{metric}_vs_noise.png", dpi = 300)
        plt.show()
        plt.close()




noise_results = noise_analysis(
    algorithms={"APSO": apso_sim, "SPSO": spso_sim, "ARPSO": arpso_sim},
    noise_levels=[0.00, 0.05, 0.10, 0.15],
    num_simulations=10,
    uav_count=20,
    src_position=np.array([80, 34])
)

plot_noise_results(noise_results)
stability_analysis(apso_sim)
convergence_analysis(apso_sim)