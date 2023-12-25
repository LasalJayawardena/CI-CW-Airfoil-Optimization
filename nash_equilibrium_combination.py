from typing import *
from tqdm import tqdm
import random
import pickle

from fitness import lift_coef_based_fitness_function, lift_coef_based_fitness_function_multi
from genotype import generate_population
from selection import tournament_selection
from mutation import gaussian_mutation
from crossover import single_point_crossover

# Main Dictionary to keep track fo genotypes of nash fitness for each population
FITNESS_DICT = {}

# Define Nash Ftiness, difference keep track of Co-Evolution changes as well
def nash_fitness_function(genotype: list, population_type: str = "u", stored_fitness_dict: dict = FITNESS_DICT, base_fitness_function: Callable = lift_coef_based_fitness_function) -> float:
    """
    Calculates the fitness of a genotype using the Nash fitness function.

    Args:
    - genotype (list): The genotype to calculate the fitness of.
    - population_type (str): The population type of the genotype.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.
    - base_fitness_function (Callable): The fitness function to use.

    Returns:
    - float: The fitness of the genotype.
    """
    # Dict Key
    key = (tuple(genotype), population_type)

    # Calculate the fitness of the genotype if not in stored_fitness_dict
    if key not in stored_fitness_dict:
        stored_fitness_dict[key] = base_fitness_function(genotype)

    fitness = FITNESS_DICT[key] 

    # Return the fitness
    return fitness

def set_nash_fitness(genotype: list, population_type: str, fitness_value: float, stored_fitness_dict: dict = FITNESS_DICT) -> None:
    """
    Sets the fitness score of a genotype.

    Args:
    - genotype (list): The genotype to set the fitness score for.
    - population_type (str): The population type of the genotype.
    - fitness_value (float): The fitness value to set.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.
    """
    # Dict Key
    key = (tuple(genotype), population_type)

    # Set the fitness value in the stored_fitness_dict
    stored_fitness_dict[key] = fitness_value

def single_u_fitness(genotype: list, stored_fitness_dict: dict = FITNESS_DICT) -> float:
    """
    Calculate the fitness of a single genotype from population 'u'.

    Args:
    - genotype (list): The genotype to calculate the fitness of.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.

    Returns:
    - float: The fitness of the genotype.
    """
    return nash_fitness_function(genotype, population_type="u", stored_fitness_dict=stored_fitness_dict)


def single_v_fitness(genotype: list, stored_fitness_dict: dict = FITNESS_DICT) -> float:
    """
    Calculate the fitness of a single genotype from population 'v'.

    Args:
    - genotype (list): The genotype to calculate the fitness of.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.

    Returns:
    - float: The fitness of the genotype.
    """
    return nash_fitness_function(genotype, population_type="v", stored_fitness_dict=stored_fitness_dict)


def multi_u_fitness(genotypes: List[list], stored_fitness_dict: dict = FITNESS_DICT) -> List[float]:
    """
    Calculate the fitnesses of multiple genotypes from population 'u'.

    Args:
    - genotypes (List[list]): The genotypes to calculate the fitnesses of.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.

    Returns:
    - List[float]: The fitnesses of the genotypes.
    """
    return [nash_fitness_function(genotype, population_type="u", stored_fitness_dict=stored_fitness_dict) for genotype in genotypes]


def multi_v_fitness(genotypes: List[list], stored_fitness_dict: dict = FITNESS_DICT) -> List[float]:
    """
    Calculate the fitnesses of multiple genotypes from population 'v'.

    Args:
    - genotypes (List[list]): The genotypes to calculate the fitnesses of.
    - stored_fitness_dict (dict): The dictionary of stored fitness values.

    Returns:
    - List[float]: The fitnesses of the genotypes.
    """
    return [nash_fitness_function(genotype, population_type="v", stored_fitness_dict=stored_fitness_dict) for genotype in genotypes]

def parent_selection(gene_list: List[List[float]], number_of_pairs: int) -> List[Tuple[List[float], List[float]]]:
    """
    Given candidate parents this function returns a list of parent pairs and generates a specific number of offspring.

    Parameters:
    - gene_list (List[List[float]]): A list of candidate parents.
    - number_of_pairs (int): The number of parent pairs to be returned.

    Returns:
    - List[Tuple[List[float], List[float]]]: A list of parent pairs.
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

    # Limit the number of parent pairs to the number of offspring required
    return parent_pairs

def compete(individual_u: list, individual_v: list, fitness_function_u: Callable, fitness_function_v: Callable) -> Tuple[list, list, float]:
    """
    Determine the outcome of a competition between two individuals based on their fitness.
    
    Args:
    - individual_u (list): The genotype of the first individual.
    - individual_v (list): The genotype of the second individual.
    - fitness_function_u (Callable): The fitness function to evaluate genotypes of population u.
    - fitness_function_v (Callable): The fitness function to evaluate genotypes of population v.

    Returns:
    - Tuple[list, list, float]: The winning individual, the losing individual, and the fitness difference.
    """
    fitness_u = fitness_function_u(individual_u)
    fitness_v = fitness_function_v(individual_v)
    
    if fitness_u > fitness_v:
        return individual_u, individual_v, fitness_u - fitness_v
    else:
        return individual_v, individual_u, fitness_v - fitness_u

def co_evolution(population_u: List[list], population_v: List[list], fitness_scores: dict = FITNESS_DICT, scaling_factor: float = 0.001, use_fitness_ordering: bool = False) -> None:
    """
    Perform competitive co-evolution on two populations using scaled fitness adjustments.

    Args:
    - population_u (List[list]): The first population of genotypes.
    - population_v (List[list]): The second population of genotypes.
    - fitness_scores (dict): Dictionary holding the fitness scores of individuals.
    - scaling_factor (float): The factor by which the fitness difference is scaled.
    - use_fitness_ordering (bool): Whether to order individuals by fitness before competing.
    """
    # Optionally order populations by fitness
    if use_fitness_ordering:
        population_u = sorted(population_u, key=lambda ind: single_u_fitness(ind), reverse=True)
        population_v = sorted(population_v, key=lambda ind: single_u_fitness(ind), reverse=True)

    for individual_u in population_u:
        # Select opponent based on ordering method
        individual_v = population_v[0] if use_fitness_ordering else random.choice(population_v)
        
        winner, loser, fitness_diff = compete(individual_u, individual_v, single_u_fitness, single_v_fitness)
        
        # Update the fitness scores dictionary with the scaled adjustments
        if winner == individual_u:
            evolved_fitness_diff = single_u_fitness(individual_u) + (fitness_diff * scaling_factor)
            set_nash_fitness(individual_u, population_type="u", fitness_value=evolved_fitness_diff, stored_fitness_dict=fitness_scores)
        else:
            evolved_fitness_diff = single_v_fitness(individual_v) + (fitness_diff * scaling_factor)
            set_nash_fitness(individual_v, population_type="v", fitness_value=evolved_fitness_diff, stored_fitness_dict=fitness_scores)

        # If using fitness ordering, remove the competed individual from population_v
        if use_fitness_ordering:
            population_v.pop(0)


def check_nash_equilibrium(population_u: List[list], population_v: List[list], stored_fitness_dict: dict = FITNESS_DICT, convergence_threshold: float = 0.01) -> bool:
    """
    Check if two populations are at Nash Equilibrium.

    Args:
    - population_u (List[list]): The first population of genotypes (population U).
    - population_v (List[list]): The second population of genotypes (population V).
    - stored_fitness_dict (dict): The dictionary of stored fitness values.
    - convergence_threshold (float): The threshold for considering a change in fitness significant.

    Returns:
    - bool: True if the populations are at Nash Equilibrium, False otherwise.
    """
    for population, population_type in [(population_u, "u"), (population_v, "v")]:
        for individual in population:
            current_fitness = nash_fitness_function(individual, population_type, stored_fitness_dict)

            # Test mutations to see if any could increase the fitness significantly
            for _ in range(10):  # Check multiple mutations
                mutated_individual = gaussian_mutation(individual, mutation_rate=1, std_dev=1.0)
                mutated_fitness = nash_fitness_function(mutated_individual, population_type, stored_fitness_dict)

                # Calculate fitness difference considering only positive improvements
                fitness_difference = max(mutated_fitness - current_fitness, 0)

                if fitness_difference > convergence_threshold:
                    return False  # Significant improvement found, not at Nash Equilibrium

    return True  # No significant improvements found, populations are at Nash Equilibrium



def save_fitness_dict(fitness_dict: dict, filename: str) -> None:
    """
    Save the fitness dictionary to a file.

    Args:
    - fitness_dict (dict): The fitness dictionary to save.
    - filename (str): The filename to save the dictionary to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(fitness_dict, f)


# Assume previously defined modules and methods are imported here.
def nash_genetic_algorithm(population_size: int, max_generations: int, convergence_threshold: float) -> Tuple[List[list], List[list]]:
    """
    Executes the Nash genetic algorithm to find the Nash Equilibrium in airfoil optimization.

    Args:
    - population_size (int): Size of each population.
    - max_generations (int): Maximum number of generations.
    - convergence_threshold (float): Convergence threshold to determine if Nash Equilibrium is reached.

    Returns:
    - Tuple[List[list], List[list]]: The two populations at Nash Equilibrium.
    """
    # Step 1: Initialize populations U and V (Block 1 in the diagram)
    population_u = generate_population(population_size)
    population_v = generate_population(population_size)

    for generation in tqdm(range(max_generations)):
        # Step 2: Perform tournament selection (Block 2 in the diagram)
        selected_u = tournament_selection(population_u, population_size, tournament_size=2, fitness_function = multi_u_fitness)
        selected_v = tournament_selection(population_v, population_size, tournament_size=2, fitness_function = multi_v_fitness)

        # Step 3: Apply crossover and mutation (Blocks 3.1 and 3.2 in the diagram)
        parents_u = parent_selection(selected_u, population_size)
        parents_v = parent_selection(selected_v, population_size)

        offspring_u = [single_point_crossover(parent1, parent2)[0] for parent1, parent2 in parents_u]
        offspring_v = [single_point_crossover(parent1, parent2)[0] for parent1, parent2 in parents_v]

        # Apply mutation to the offspring
        mutated_offspring_u = [gaussian_mutation(individual, mutation_rate=0.1, std_dev=1.0) for individual in offspring_u]
        mutated_offspring_v = [gaussian_mutation(individual, mutation_rate=0.1, std_dev=1.0) for individual in offspring_v]

        # Step 4: Co-evolution (Block 4 in the diagram)
        co_evolution(mutated_offspring_u, mutated_offspring_v, scaling_factor=0.1, use_fitness_ordering=False, fitness_scores=FITNESS_DICT)

        # Step 5: Evaluate populations (Block 5 in the diagram)
        fitness_u = multi_u_fitness(mutated_offspring_u, stored_fitness_dict=FITNESS_DICT)
        fitness_v = multi_v_fitness(mutated_offspring_v, stored_fitness_dict=FITNESS_DICT)

        # Step 6: Check for Nash Equilibrium (Block 6 in the diagram)
        if check_nash_equilibrium(population_u=mutated_offspring_u, population_v=mutated_offspring_v, convergence_threshold= convergence_threshold):
            print(f"Nash Equilibrium reached at generation:{generation}.")
            # Order the Genotypes according to fitness values
            mutated_offspring_u = sorted(mutated_offspring_u, key=lambda ind: single_u_fitness(ind), reverse=True)
            mutated_offspring_v = sorted(mutated_offspring_v, key=lambda ind: single_v_fitness(ind), reverse=True)
            return mutated_offspring_u, mutated_offspring_v

        # If not at Nash Equilibrium, prepare for the next generation
        population_u = mutated_offspring_u
        population_v = mutated_offspring_v


        
    # Step 7: Check if generation tolerance limit is reached (Block 7 in the diagram)
    print("Maximum generations reached without finding Nash Equilibrium.")
    # Order the Genotypes according to fitness values
    population_u = sorted(population_u, key=lambda ind: single_u_fitness(ind), reverse=True)
    population_v = sorted(population_v, key=lambda ind: single_v_fitness(ind), reverse=True)
    return population_u, population_v




configurations = {
    "crossover": {
        "Single Point Crossover": lambda parent1, parent2: single_point_crossover(parent1=parent1, parent2=parent2),
        "Uniform Crossover": lambda parent1, parent2: uniform_crossover(parent1=parent1, parent2=parent2),
        "Arithmetic Crossover": lambda parent1, parent2: arithmetic_crossover(parent1=parent1, parent2=parent2, alpha=0.4)
    },
    "mutation_rate":{
        "1 Percent": 0.01,
        "5 Percent": 0.05,
        "10 Percent": 0.1,
    },
    "mutation": {
        "Creep Mutation": lambda individual, mutation_rate: creep_mutation(individual=individual, mutation_rate=mutation_rate, creep_magnitude=0.05),
        "Gaussian Mutation": lambda individual, mutation_rate: gaussian_mutation(individual=individual, mutation_rate=mutation_rate, std_dev=1.0),
        "Uniform Mutation": lambda individual, mutation_rate: uniform_mutation(individual=individual, mutation_rate=mutation_rate, uniform_range_fraction=0.1)
    },
    "survivor_selection": {
        "Truncation Survivor Selection": lambda population, num_selected, fitness_function: survivor_selection.truncation_selection(population=population, num_selected=num_selected, fitness_function= fitness_function),
        "Steady State Selection": lambda old_population, new_offspring, fitness_function, num_selected: survivor_selection.steady_state_selection(old_population=old_population, new_offspring=new_offspring, fitness_function=fitness_function, num_selected=num_selected, replacement_percentage=0.5)
    }
}




# Example usage
population_size = 50
max_generations = 10
convergence_threshold = 0.02
nash_population_u, nash_population_v = nash_genetic_algorithm(population_size, max_generations, convergence_threshold)

save_fitness_dict(FITNESS_DICT, "./RESULTS/Nash/fitness_dict.pkl")

# # Output the Nash Equilibrium populations
# print("Nash Equilibrium population U:")
# for genotype in nash_population_u:
#     print(genotype)

# print("Nash Equilibrium population V:")
# for genotype in nash_population_v:
#     print(genotype)

# Output Top 5 Genotypes of each population along with normal fitness values, and the FITNESS_DICT fitness values

genotypes_u = nash_population_u

# Generate genotypes of population V
genotypes_v = nash_population_v

# Calculate normal fitness values for population U
fitness_values_u = lift_coef_based_fitness_function_multi(genotypes_u)

# Calculate normal fitness values for population V
fitness_values_v = lift_coef_based_fitness_function_multi(genotypes_v)

# Calculate FITNESS_DICT fitness values for population U
fitness_dict_values_u = [single_u_fitness(genotype) for genotype in genotypes_u]

# Calculate FITNESS_DICT fitness values for population V
fitness_dict_values_v = [single_v_fitness(genotype) for genotype in genotypes_v]

# # Print genotypes and fitness values
# print("Population U:")
# for genotype, fitness, dict_fitness in list(zip(genotypes_u, fitness_values_u, fitness_dict_values_u))[:5]:
#     print("Genotype:", genotype)
#     print("Normal Fitness:", fitness)
#     print("FITNESS_DICT Fitness:", dict_fitness)

# print("Population V:")
# for genotype, fitness, dict_fitness in list(zip(genotypes_v, fitness_values_v, fitness_dict_values_v))[:5]:
#     print("Genotype:", genotype)
#     print("Normal Fitness:", fitness)
#     print("FITNESS_DICT Fitness:", dict_fitness)














# ==================================================================================================================
"""
Nash Genetic Algorithm for PARSEC Airfoil Optimization

This Python file contains an implementation of a Nash genetic algorithm tailored for the optimization of airfoil shapes using the PARSEC parameterization method. The algorithm utilizes a co-evolutionary approach to optimize the shape of an airfoil for better aerodynamic performance. 

The co-evolutionary method is based on the idea of competitive evolution, where multiple populations (genotypes representing airfoil shapes) evolve not only by their performance metrics but also through direct competition with one another. Each individual's fitness is determined by the lift coefficient over a range of attack angles. During the competition phase, individuals from different populations are pitted against each other, with the winner gaining a fitness advantage. This approach promotes the evolution of robust airfoil shapes that perform well across different conditions.

Advantages of this approach include:

1. Multi-objective Optimization: By considering multiple populations that may represent different design objectives or constraints, the algorithm can search for a balance between competing requirements, leading to more holistic design solutions.

2. Robustness: The use of Nash Equilibrium concepts ensures that the solutions are stable and robust, meaning they are less likely to be adversely affected by small variations in the parameters.

3. Innovation: The competitive aspect of the co-evolutionary process can drive the populations toward innovative solutions that might not be discovered through traditional, single-population evolutionary algorithms.

4. Adaptability: The algorithm can be adapted to various other optimization problems where the interaction between different solution strategies is a key factor in determining the overall system performance.

This file defines the compete and co_evolution functions responsible for implementing the competitive co-evolutionary process and the main nash_genetic_algorithm function that orchestrates the entire optimization process.
"""