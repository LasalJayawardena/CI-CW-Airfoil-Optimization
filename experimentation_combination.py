import os
import random
from tqdm import tqdm
from typing import *

from genotype import generate_population, check_valid_genotype
from fitness import lift_coef_based_fitness_function_multi, lift_coef_based_fitness_function
from crossover import single_point_crossover, multi_point_crossover, uniform_crossover, blend_crossover, arithmetic_crossover
from mutation import random_resetting_mutation, creep_mutation, gaussian_mutation, boundary_mutation, uniform_mutation
from selection import roulette_wheel_selection, tournament_selection, rank_selection, stochastic_universal_sampling, elitism_selection
from loggers import result_logger
import survivor_selection
import itertools
import inspect

# Define a configuration dictionary to hold the function references
GA_Configuration = Dict[str, Callable]

configurations = {
    "crossover": {
        "Single Point Crossover": lambda parent1, parent2: single_point_crossover(parent1=parent1, parent2=parent2),
        "Two Point Crossover": lambda parent1, parent2: multi_point_crossover(parent1=parent1, parent2=parent2,  n_points=2),
        "Uniform Crossover": lambda parent1, parent2: uniform_crossover(parent1=parent1, parent2=parent2),
        "Blend Crossover": lambda parent1, parent2: blend_crossover(parent1=parent1, parent2=parent2),
        "Arithmetic Crossover": lambda parent1, parent2: arithmetic_crossover(parent1=parent1, parent2=parent2, alpha=0.4)
    },
    "mutation_rate":{
        "1 Percent": 0.01,
        "5 Percent": 0.05,
        "10 Percent": 0.1,
    },
    "mutation": {
        # "Random Resetting Mutation": lambda individual, mutation_rate: random_resetting_mutation(individual=individual, mutation_rate=mutation_rate),
        "Creep Mutation": lambda individual, mutation_rate: creep_mutation(individual=individual, mutation_rate=mutation_rate, creep_magnitude=0.05),
        "Gaussian Mutation": lambda individual, mutation_rate: gaussian_mutation(individual=individual, mutation_rate=mutation_rate, std_dev=1.0),
        # "Boundary Mutation": lambda individual, mutation_rate: boundary_mutation(individual=individual, mutation_rate=mutation_rate),
        "Uniform Mutation": lambda individual, mutation_rate: uniform_mutation(individual=individual, mutation_rate=mutation_rate, uniform_range_fraction=0.1)
    },
    "selection": {
        "Roulette Wheel Selection": lambda population, num_selected, fitness_function: roulette_wheel_selection(population=population, num_selected=num_selected, fitness_function= fitness_function),
        "Binary Tournament Selection": lambda population, num_selected, fitness_function: tournament_selection(population=population, num_selected=num_selected, fitness_function= fitness_function, tournament_size=2),
        "Ternary Tournament Selection": lambda population, num_selected, fitness_function: tournament_selection(population=population, num_selected=num_selected, fitness_function= fitness_function, tournament_size=3),
        "Rank Selection": lambda population, num_selected, fitness_function: rank_selection(population=population, num_selected=num_selected, fitness_function= fitness_function),
        "Stochastic Universal Selection": lambda population, num_selected, fitness_function: stochastic_universal_sampling(population=population, num_selected=num_selected, fitness_function= fitness_function),
        "Elitism Selection": lambda population, num_selected, fitness_function: elitism_selection(population=population, num_selected=num_selected, fitness_function= fitness_function)
    },
    "survivor_selection": {
        "Truncation Survivor Selection": lambda population, num_selected, fitness_function: survivor_selection.truncation_selection(population=population, num_selected=num_selected, fitness_function= fitness_function),
        "Steady State Selection": lambda old_population, new_offspring, fitness_function, num_selected: survivor_selection.steady_state_selection(old_population=old_population, new_offspring=new_offspring, fitness_function=fitness_function, num_selected=num_selected, replacement_percentage=0.5)
    }
}

def generate_experiment_combinations(configurations):
    # Extract the keys and values for each configuration category
    categories = {k: list(configurations[k].items()) for k in configurations}

    # Generate all possible combinations using itertools.product
    all_combinations = itertools.product(*categories.values())

    experiment_names = []
    experiment_functions = []

    for combination in all_combinations:
        # Create the experiment name by concatenating the keys with ' + '
        experiment_name = ' + '.join([c[0] for c in combination])
        experiment_names.append(experiment_name)

        # Collect the corresponding functions and their parameters
        experiment_funcs = [c[1] for c in combination]
        experiment_functions.append(experiment_funcs)

    return experiment_names, experiment_functions

experiment_names, experiment_functions = generate_experiment_combinations(configurations)
short_experiment_names = [ f"Combination_Experiment_{name}" for name in range(1, len(experiment_names)+1)]

# # For demonstration, printing out the first few experiment names and their corresponding functions
# for name, funcs in zip(experiment_names[:5], experiment_functions[:5]):
#     print(f"Experiment Name: {name}")
#     print(f"Functions and Parameters: {funcs}\n")

# print("Total Number of Experiments:", len(experiment_names))

# Customizable Optimization Strategy
def log_generation_results(generation, generation_number, experiment_name, root_folder="./RESULTS"):
    """
    Logs the results of a generation to a text file.

    Parameters:
    - generation (List[List[float]]): The generation of genotypes.
    - generation_number (int): The generation number.
    - experiment_name (str): The name of the experiment.
    - root_folder (str): The root folder where results will be saved.
    """
    exp_name = experiment_name.replace(" ", "_").replace("+", "and")  

    fitness_results = lift_coef_based_fitness_function_multi(generation, return_full_dict=True)
    fitness_values = [x[0] for x in fitness_results]
    results_dicts = [x[1] for x in fitness_results]

    result_logger(root_folder, exp_name, generation_number, generation, fitness_values, results_dicts)
    print(f"Generation {generation_number} results logged for experiment '{experiment_name}' in folder '{root_folder}'.")

def parent_selection(gene_list: List[List[float]], number_of_pairs: int) -> List[List[float]]:
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


#  Flexible Optimzier Algorithm
def flexible_optimizer(current_gen, population_size, crossover_method, mutation_rate, mutation_method, selection_method, survivor_selection_method):
    """
    Performs one generation of a genetic algorithm using flexible strategies.

    Parameters:
    - current_gen (List[List[float]]): The current generation of genotypes.
    - population_size (int): The size of the population/generation.
    - crossover_method (callable): Function for crossover.
    - mutation_rate (float): Mutation rate.
    - mutation_method (callable): Function for mutation.
    - selection_method (callable): Function for selection.
    - survivor_selection_method (callable): Function for survivor selection.

    Returns:
    - List[List[float]]: The next generation of genotypes.
    """
    # Check if the survivor slection is setady state
    print("Error here", flush=True)
    signature = inspect.signature(survivor_selection_method)
    is_steady_state = len(signature.parameters) == 4

    # Evaluate fitness of current generation
    fitness_scores = lift_coef_based_fitness_function_multi(current_gen)

    # Perform selection
    next_generation_parent_individuals = selection_method(current_gen, population_size, lift_coef_based_fitness_function_multi)

    # Get Parent pairs for crossover
    parent_pairs = parent_selection(next_generation_parent_individuals, population_size // 2)

    # Perform crossover
    offspring = []
    for parent1, parent2 in parent_pairs:
        offsprings = crossover_method(parent1, parent2)
        offspring.extend(offsprings)

    # Remove duplicates
    tuple_offspring = {tuple(genotype) for genotype in offspring}
    crossover_population = [list(unique_tuple) for unique_tuple in tuple_offspring]

    # Perform mutation
    mutated_population = [mutation_method(genotype, mutation_rate) for genotype in crossover_population]

    
    # For Non Steady StaeCombine and select survivors
    if not is_steady_state:
        combined_generations = current_gen + mutated_population
        final_generation = survivor_selection_method(combined_generations, population_size, lift_coef_based_fitness_function_multi)
        return final_generation
    else:
        # For Steady State
        final_generation = survivor_selection_method(current_gen, mutated_population, lift_coef_based_fitness_function, population_size)
        return final_generation


# Flexible Siimulation

def flexible_simulation(experiment_name, experiment_variables, num_generations=100, population_size=50, root_folder="./RESULTS"):
    """
    Runs a flexible genetic algorithm simulation based on experiment variables.

    Parameters:
    - experiment_name (str): The name of the experiment.
    - experiment_variables (list): A list containing the experiment variables in order.
    - num_generations (int): The number of generations to simulate.
    - population_size (int): The size of the population/generation.
    - root_folder (str): The root folder where results will be saved.
    """

    # # Create a directory for the experiment results
    # experiment_directory = os.path.join(root_folder, experiment_name.replace(" ", "_"))
    # if not os.path.exists(experiment_directory):
    #     os.makedirs(experiment_directory)

    # Unpack experiment variables
    crossover_method, mutation_rate, mutation_method, selection_method, survivor_selection_method = experiment_variables

    # Generate initial population
    initial_population = generate_population(population_size)

    # Run the genetic algorithm for the specified number of generations
    current_generation = initial_population
    for generation_number in tqdm(range(num_generations)):
        current_generation = flexible_optimizer(
            current_generation, population_size, crossover_method, mutation_rate, 
            mutation_method, selection_method, survivor_selection_method
        )
        log_generation_results(current_generation, generation_number + 1, experiment_name)

    # Evaluate fitness of final generation
    fitness_scores = lift_coef_based_fitness_function_multi(current_generation)
    return current_generation, fitness_scores

#  Test Simulation
# experiment_name = "Single Point Crossover + 1 Percent + Creep Mutation + Ternary Tournament Selection + Truncation Survivor Selection"
# experiment_name = "Combination Experiment 1"
# experiment_variables = [
#     configurations['crossover']['Single Point Crossover'],
#     configurations['mutation_rate']['1 Percent'],
#     configurations['mutation']['Creep Mutation'],
#     configurations['selection']['Ternary Tournament Selection'],
#     configurations['survivor_selection']['Truncation Survivor Selection']
# ]

# exp_num = 0
# experiment_name = short_experiment_names[exp_num]
# experiment_variables = experiment_functions[exp_num]
# long_experiment_name = experiment_names[exp_num]

# print(f"Runnign for experiment: {long_experiment_name}")
# flexible_simulation(experiment_name, experiment_variables, num_generations=100, population_size=50, root_folder="./RESULTS")

# print("Running Experiments", len(experiment_names))
start_index = 0
end_index = 50
for name, long_name, funcs in tqdm(list(zip(short_experiment_names, experiment_names, experiment_functions))[start_index:end_index]):
    print(f"Running Experiment Name: {long_name}", flush=True)
    final_generation, fitness_scores = flexible_simulation(name, funcs, num_generations=100, population_size=50)