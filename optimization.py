from genotype import generate_population, check_valid_genotype
from crossover import uniform_crossover
from fitness import lift_coef_based_fitness_function_multi, lift_coef_based_fitness_function
from mutation import uniform_mutation
from selection import roulette_wheel_selection
from survivor_selection import truncation_selection
from loggers import result_logger

from typing import List
import random

def optimization_strategy_one_parent_pairing(gene_list: List[List[float]], number_of_pairs: int) -> List[List[float]]:
    """
    Given candidate parents this function returns a list of parent pairs.

    Parameters:
    - gene_list (List[List[float]]): A list of candidate parents.
    - number_of_pairs (int): The number of parent pairs to be returned.

    Returns:
    - List[List[float]]: A list of parent pairs.
    """

    selected_parent_pairs = set()
    parent_pairs = []

    # Create pairs of parents
    while len(selected_parent_pairs) < number_of_pairs:  
        index1, index2 = random.sample(range(len(gene_list)), 2)  # Randomly select two indices
        pair = (min(index1, index2), max(index1, index2))  # Order the pair indices
        if pair not in selected_parent_pairs:  # Check if the pair is not already selected
            selected_parent_pairs.add(pair)
            parent_pairs.append((gene_list[pair[0]], gene_list[pair[1]]))  # Append the selected pair

    return parent_pairs

# Test optimization_strategy_one_parent_pairing
# print(optimization_strategy_one_parent_pairing(generate_population(10), 5))


def optimization_strategy_one(current_gen: List[List[float]], population_size: int) -> List[List[float]]:
    """
    Performs one generation of a genetic algorithm using strategy one.

    Parameters:
    - current_gen (List[List[float]]): The current generation of genotypes.
    - population_size (int): The size of the population/generation.

    Returns:
    - List[List[float]]: The next generation of genotypes.
    """

    mutation_rate = 0.1 # Placeholder mutation rate    


    # Evaluate fitness of current generation
    fitness_scores = lift_coef_based_fitness_function_multi(current_gen)

    # Perform selection (Roulette wheel selection)
    next_generation_parent_individuals = roulette_wheel_selection(current_gen, 100, lift_coef_based_fitness_function_multi)

    #  Get Parent pairs for crossover
    parent_pairs = optimization_strategy_one_parent_pairing(next_generation_parent_individuals, 50)

    # Perform crossover (Uniform crossover)
    offspring = []

    for parent1, parent2 in parent_pairs:
        offsprings = uniform_crossover(parent1, parent2)
        offspring.extend(offsprings)

    tuple_offspring = {tuple(genotype) for genotype in offspring}

    # Convert unique tuples back to lists to get unique lists
    crossover_population = [list(unique_tuple) for unique_tuple in tuple_offspring]

    # Perform mutation (Uniform mutation)
    mutated_population = []
    for genotype in crossover_population:
        mutated_genotype = uniform_mutation(genotype, mutation_rate)
        mutated_population.append(mutated_genotype)

    # Check validity of genotypes
    # for genotype in mutated_population:
    #     if not check_valid_genotype(genotype):
    #         # If invalid, generate a new valid genotype and replace it
    #         new_valid_genotype = generate_population(1)[0]
    #         mutated_population[mutated_population.index(genotype)] = new_valid_genotype

    combined_generations = current_gen + mutated_population
    final_genaration = truncation_selection(combined_generations, population_size, lift_coef_based_fitness_function_multi)

    return final_genaration

# # Test optimization_strategy_one
# print(len(optimization_strategy_one(generate_population(10), 20)))
# result = optimization_strategy_one(generate_population(10), 20)
# print(result[0], len(result))

from tqdm import tqdm

def log_genration_results(generation: List[List[float]], generation_number: int):
    """
    Logs the results of a generation to a text file.

    Parameters:
    - generation (List[List[float]]): The generation of genotypes.
    - generation_number (int): The generation number.
    """
    root_folder = './RESULTS'
    experiment_name = 'Experiment1'

    fitness_results = lift_coef_based_fitness_function_multi(generation, return_full_dict=True)
    fitness_values = [x[0] for x in fitness_results]
    results_dicts = [x[1] for x in fitness_results]
    result_logger(root_folder, experiment_name, generation_number, generation, fitness_values, results_dicts)
    print(f"Generation {generation_number} results logged.")


def simulation_strategy_one():
    # Run optimization_strategy_one for 100 Generations

    # Generate initial population
    initial_population = generate_population(10)

    # Run optimization_strategy_one for 100 generations
    current_generation = initial_population
    for i in tqdm(list(range(2))):
        current_generation = optimization_strategy_one(current_generation, 10)
        log_genration_results(current_generation, i+1)

    # Evaluate fitness of final generation
    fitness_scores = lift_coef_based_fitness_function_multi(current_generation)
    return current_generation, fitness_scores

# Test simulation_strategy_one
final_generation, fitness_scores = simulation_strategy_one()
print(fitness_scores)
print(final_generation[fitness_scores.index(max(fitness_scores))][:5])

print(final_generation[:5])

# [[0.022028398172698832, 0.014909972469847377, 0.19131302277026258, -0.03145004697386025, 0.348600346393238, -0.11270809714274087, 0.019478974011888583, -0.09137591307596914, 0.001051826118050081, 358.57716542057204, 243.7584605017981], [0.009184781744351635, 0.014909972469847377, 0.1668981036767055, -0.05174832623838861, 0.2652554774311876, -0.11270809714274087, -0.08082863268240983, 0.046632160153082886, 0.001051826118050081, 358.57716542057204, 160.91629829169645], [0.014071653932302846, 0.014909972469847377, 0.1668981036767055, 0.012927437061695121, 0.19069779518561036, -0.1934743008979389, -0.08082863268240983, 0.06570611638799778, 0.001051826118050081, 268.5672836290555, 160.91629829169645], [0.014071653932302846, 0.014909972469847377, 0.1668981036767055, 0.012927437061695121, 0.19069779518561036, -0.1934743008979389, -0.08082863268240983, -0.09137591307596914, 0.001051826118050081, 268.5672836290555, 282.5659800093517], [0.014071653932302846, 0.014909972469847377, 0.1668981036767055, 0.07129968424638725, 0.4134505039093871, 0.137912988943254, -0.08082863268240983, -0.0527572320351819, 0.001051826118050081, 358.57716542057204, 160.91629829169645]]