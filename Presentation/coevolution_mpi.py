from mpi4py import MPI
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def initialize_population(population_size, chromosome_length):
    return [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(population_size)]

def select_opponents(population, individual, num_opponents=5):
    opponents = population.copy()
    opponents.remove(individual)  # исключаем текущую особь
    return random.sample(opponents, min(num_opponents, len(opponents)))

def evaluate_fitness(individual, opponents):
    fitness = sum(1 for opponent in opponents if sum(individual) > sum(opponent))
    return fitness / len(opponents)

# Mutation
def evolve_population(population, mutation_rate=0.1):
    new_population = []
    for individual in population:
        new_individual = [gene if random.random() > mutation_rate else 1 - gene for gene in individual]
        new_population.append(new_individual)
    return new_population

def coevolution(population_size, chromosome_length, num_generations, num_opponents=5):
    if rank == 0:
        population = initialize_population(population_size, chromosome_length)
    else:
        population = None
    
    # send popilation to al processors
    population = comm.bcast(population, root=0)
    
    for generation in range(num_generations):
        fitness_scores = []

        # each processor count fitness for his part of generation
        chunk_size = len(population) // size
        start_index = rank * chunk_size
        end_index = start_index + chunk_size if rank != size - 1 else len(population)
        local_population = population[start_index:end_index]

        # each indiviudal grade score for a given generation
        local_fitness_scores = []
        for individual in local_population:
            if generation == num_generations-1:
                print(f"Epoch {generation+1}, Processor {rank} -> Chromosome: {individual}")
            opponents = select_opponents(population, individual, num_opponents)
            fitness = evaluate_fitness(individual, opponents)
            local_fitness_scores.append((individual, fitness))

        # pick best fitness scores from each processor
        all_fitness_scores = comm.gather(local_fitness_scores, root=0)

        if rank == 0:
            # merge scores from all processors
            fitness_scores = [item for sublist in all_fitness_scores for item in sublist]
            # sort by fitness and pick best
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            best_individual = fitness_scores[0][0]
            # create new generation
            population = evolve_population([individual for individual, _ in fitness_scores])
        else:
            best_individual = None

        # send new generaion to processors
        population = comm.bcast(population, root=0)

    # return best fitness scores
    if rank == 0:
        return best_individual
    else:
        return None

population_size = 20
chromosome_length = 10
num_generations = 5
s = time.time()
best_solution = coevolution(population_size, chromosome_length, num_generations)
t_s = time.time()-s
if rank == 0:
    time.sleep(1)
    print("Best count:", best_solution)
    print(f"Time spend", t_s)

#mpiexec -n 4 python coevolution_mpi.py
