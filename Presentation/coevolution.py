import random
import time

def initialize_population(population_size, chromosome_length):
    return [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(population_size)]

def select_opponents(population, individual, num_opponents=5):
    opponents = population.copy()
    opponents.remove(individual)  # исключаем текущую особь
    return random.sample(opponents, min(num_opponents, len(opponents)))

def evaluate_fitness(individual, opponents):
    fitness = sum(1 for opponent in opponents if sum(individual) > sum(opponent))
    return fitness / len(opponents)  # нормируем фитнесс

#  Mutation
def evolve_population(population, mutation_rate=0.1):
    new_population = []
    for individual in population:
        new_individual = [gene if random.random() > mutation_rate else 1 - gene for gene in individual]
        new_population.append(new_individual)
    return new_population

def coevolution(population_size, chromosome_length, num_generations, num_opponents=5):
    population = initialize_population(population_size, chromosome_length)
    
    for generation in range(num_generations):
        fitness_scores = []
        
        # each indiviudal grade score
        for individual in population:
            opponents = select_opponents(population, individual, num_opponents)
            fitness = evaluate_fitness(individual, opponents)
            fitness_scores.append((individual, fitness))
        
        # pick best fitness scores
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_individual = fitness_scores[0][0]
        # create new generation
        population = evolve_population([individual for individual, _ in fitness_scores])
    
    return best_individual

population_size = 5
chromosome_length = 4
num_generations = 100

s = time.time()
best_solution = coevolution(population_size, chromosome_length, num_generations)
t_s = time.time()-s
print("Лучшее решение:", best_solution)
print(f"Time spend", t_s)