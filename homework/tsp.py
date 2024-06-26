import numpy as np
import random
import matplotlib.pyplot as plt

# 31个省会城市的平面坐标
coordinates = np.array([
    [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535],
    [3326, 1556], [3238, 1229], [4196, 1004], [4312, 790], [4386, 570],
    [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695],
    [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578],
    [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367], [3394, 2643],
    [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
    [2370, 2975]
])


# 计算两点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# 计算整个路径的总距离
def calculate_total_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path)):
        total_distance += distance_matrix[path[i - 1], path[i]]
    return total_distance


# 生成距离矩阵
num_cities = len(coordinates)
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])


# 创建初始种群
def create_initial_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        population.append(random.sample(range(num_cities), num_cities))
    return population


# 选择父母进行交叉
def selection(population, fitness_scores, num_parents):
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False,
                                        p=fitness_scores / fitness_scores.sum())
    return [population[i] for i in selected_indices]


# 部分映射交叉 (PMX)
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size

    crossover_points = sorted(random.sample(range(size), 2))
    start, end = crossover_points[0], crossover_points[1]

    # Copy the crossover segment from the parents to the children
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    def fill_child(child, parent, start, end):
        for i in range(start, end):
            if parent[i] not in child:
                pos = i
                while start <= pos < end:
                    pos = parent.index(child[pos])
                child[pos] = parent[i]

    fill_child(child1, parent2, start, end)
    fill_child(child2, parent1, start, end)

    # Fill the remaining positions
    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]

    return child1, child2


# 变异
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]


# 遗传算法求解TSP
def genetic_algorithm_tsp(coordinates, population_size, num_generations, mutation_rate):
    num_cities = len(coordinates)
    population = create_initial_population(population_size, num_cities)
    best_path = None
    best_distance = float('inf')

    for generation in range(num_generations):
        fitness_scores = np.zeros(population_size)

        for i in range(population_size):
            fitness_scores[i] = 1 / calculate_total_distance(population[i], distance_matrix)

        new_population = []

        for _ in range(population_size // 2):
            parents = selection(population, fitness_scores, 2)
            child1, child2 = pmx_crossover(parents[0], parents[1])
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

        for individual in population:
            current_distance = calculate_total_distance(individual, distance_matrix)
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = individual

        print(f"Generation {generation}: Best Distance = {best_distance}")

    return best_path, best_distance


# 参数设置
population_size = 100
num_generations = 500
mutation_rate = 0.01

# 运行遗传算法
best_path, best_distance = genetic_algorithm_tsp(coordinates, population_size, num_generations, mutation_rate)

print("Best Path:", best_path)
print("Best Distance:", best_distance)


# 可视化结果
def plot_path(coordinates, path):
    plt.figure(figsize=(10, 6))
    for i in range(len(path)):
        start = coordinates[path[i - 1]]
        end = coordinates[path[i]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'bo-')
    plt.plot([coordinates[path[-1]][0], coordinates[path[0]][0]], [coordinates[path[-1]][1], coordinates[path[0]][1]],
             'ro-')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Best TSP Path')
    plt.show()


plot_path(coordinates, best_path)
