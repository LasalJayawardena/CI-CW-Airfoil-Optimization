import random

def placeholder_calculate_fitness_scores(population):
    return [random.random() for _ in population]  

def roulette_wheel_selection(population, num_selected, fitness_function):
    """
    Select individuals from the population using the roulette wheel selection method.
    
    Parameters:
    - population (list): The population of individuals.
    - num_selected (int): The number of individuals to select.
    
    Returns:
    - selected_individuals (list): The list of selected individuals.
    """
    fitness_scores = fitness_function(population)
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    
    selected_individuals = []
    for _ in range(num_selected):
        # Spin the roulette wheel
        pick = random.random()
        current = 0
        for individual, prob in zip(population, selection_probs):
            current += prob
            if current > pick:
                selected_individuals.append(individual)
                break
    
    return selected_individuals

# # Test Roulette Wheel Selection
# from genotype import generate_population

# population = generate_population(100)
# num_selected = 5  # Number of individuals to select

# selected_individuals = roulette_wheel_selection(population, num_selected, placeholder_calculate_fitness_scores)
# print("Selected Individuals:")
# for ind in selected_individuals:
#     print(ind)


def tournament_selection(population, num_selected, tournament_size, fitness_function):
    """
    Select individuals from the population using the tournament selection method.
    
    Parameters:
    - population (list): The population of individuals.
    - num_selected (int): The number of individuals to select.
    - tournament_size (int): The number of individuals to compete in each tournament.
    
    Returns:
    - selected_individuals (list): The list of selected individuals.
    """
    selected_individuals = []
    fitness_scores = fitness_function(population)
    
    for _ in range(num_selected):
        # Randomly select tournament_size individuals for the tournament
        tournament = random.sample(list(enumerate(population)), tournament_size)
        # Determine the winner of the tournament (highest fitness score)
        winner = max(tournament, key=lambda x: fitness_scores[x[0]])
        # Append the winner's genotype to the list of selected individuals
        selected_individuals.append(winner[1])
    
    return selected_individuals

# # Test Tournament Selection
# from genotype import generate_population

# population = generate_population(100)
# num_selected = 5  # Number of individuals to select
# tournament_size = 10  # Number of individuals in each tournament

# selected_individuals = tournament_selection(population, num_selected, tournament_size, placeholder_calculate_fitness_scores)
# print("Selected Individuals:")
# for ind in selected_individuals:
#     print(ind)


def truncation_selection(population, num_selected, fitness_function):
    """
    Select individuals from the population using the truncation selection method.
    
    Parameters:
    - population (list): The population of individuals.
    - num_selected (int): The number of individuals to select.
    
    Returns:
    - selected_individuals (list): The list of selected individuals.
    """
    # Calculate fitness for each individual in the population
    fitness_scores = fitness_function(population)
    
    # Pair each individual with its fitness score
    paired_population = list(zip(population, fitness_scores))
    
    # Sort the paired population by fitness score in descending order
    paired_population.sort(key=lambda x: x[1], reverse=True)
    
    # Select the top-performing individuals
    selected_individuals = [individual for individual, score in paired_population[:num_selected]]
    
    return selected_individuals

# # Test Truncation Selection
# from genotype import generate_population

# population = generate_population(100)
# num_selected = 5  # Number of individuals to select

# selected_individuals = truncation_selection(population, num_selected, placeholder_calculate_fitness_scores)
# print("Selected Individuals:")
# for ind in selected_individuals:
#     print(ind)

def elitism_selection(population, num_elites, fitness_function):
    """
    Select the top individuals from the population based on their fitness scores.
    
    Args:
    - population (list): The population of individuals.
    - num_elites (int): The number of top individuals to select.
    - fitness_function (function): The function to calculate fitness scores.
    
    Returns:
    - elites (list): The list of elite individuals.
    """
    # Calculate fitness scores for the population
    fitness_scores = fitness_function(population)
    
    # Pair each individual with its fitness score
    paired_population = list(zip(population, fitness_scores))
    
    # Sort the paired population by fitness score in descending order
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    
    # Select the top num_elites individuals
    elites = [individual for individual, score in sorted_population[:num_elites]]
    
    return elites

# # Test Elitism Selection
# from genotype import generate_population

# population = generate_population(100)
# num_elites = 5  # Number of elite individuals to select

# # Select the elite individuals
# elite_individuals = elitism_selection(population, num_elites, placeholder_calculate_fitness_scores)

# # Print the elite individuals
# print("Elite Individuals:")
# for ind in elite_individuals:
#     print(ind)



def steady_state_selection(old_population, new_offspring, fitness_function, max_population_size, replacement_percentage):
    """
    Perform steady-state selection by replacing a percentage of the worst-performing individuals
    in the old population with new offspring.
    
    Parameters:
    - old_population (list): The current population of individuals.
    - new_offspring (list): The newly generated offspring.
    - fitness_function (function): A function that calculates the fitness of an individual.
    - max_population_size (int): The maximum allowed size of the population.
    - replacement_percentage (float): The percentage of the population to be replaced by new offspring.
    
    Returns:
    - new_population (list): The updated population after steady-state selection.
    """
    # Calculate fitness for each individual in the old population
    fitness_scores = [fitness_function(individual) for individual in old_population]
    
    # Pair each individual with its fitness score
    paired_population = list(zip(old_population, fitness_scores))
    
    # Sort the paired population by fitness score in ascending order (worst first)
    paired_population.sort(key=lambda x: x[1])
    
    # Determine the number of individuals to replace
    num_to_replace = int(replacement_percentage * max_population_size)
    
    # Replace the worst-performing individuals with new offspring
    for i in range(num_to_replace):
        if i < len(new_offspring):
            paired_population[i] = (new_offspring[i], fitness_function(new_offspring[i]))
    
    # If there are more offspring than the replacement number, add them to the population
    if len(new_offspring) > num_to_replace:
        extra_offspring = new_offspring[num_to_replace:]
        paired_population.extend([(offspring, fitness_function(offspring)) for offspring in extra_offspring])
    
    # Ensure the population does not exceed the maximum size
    if len(paired_population) > max_population_size:
        paired_population = paired_population[:max_population_size]
    
    # Unpair the population to get the new list of individuals
    new_population = [individual for individual, score in paired_population]
    
    return new_population

# # Test Steady-State Selection
# from genotype import generate_population

# old_population = generate_population(100)
# new_offspring = generate_population(50)

# # Perform steady-state selection
# new_population = steady_state_selection(old_population, new_offspring, placeholder_calculate_fitness_scores, 100, 0.5)

# # Print the new population
# print("New Population:")
# for ind in new_population:
#     print(ind)


############################### PLAN ###############################

# 1. **Fitness Proportionate Selection (Roulette Wheel Selection):**
#    In this method, individuals are selected based on their fitness proportionate to the total fitness of the population. The higher the fitness, the higher the chance of being selected. This method can quickly converge to a good solution but may suffer from premature convergence if there are a few individuals with significantly higher fitness than the rest.

# 2. **Tournament Selection:**
#    A set number of individuals are randomly chosen from the population, and the one with the highest fitness within this group is selected. This process is repeated until the desired number of individuals is selected. Tournament selection can maintain diversity better than fitness proportionate selection and is less prone to premature convergence.

# 3. **Truncation Selection:**
#    Only the top-performing individuals (e.g., the top 50%) are selected for the next generation. This method is straightforward and ensures that only the best individuals are carried forward, but it can lead to a loss of genetic diversity and may cause the algorithm to get stuck in local optima.

# 4. **Elitism:**
#    A certain number of the best individuals from the current generation are guaranteed to survive to the next generation. This ensures that the best solutions found so far are not lost, which can be particularly important in maintaining progress towards the optimum. However, too much elitism can reduce diversity and lead to premature convergence.

# 5. **Steady-State Selection:**
#    Instead of generating a completely new population, only a few individuals are replaced at a time. This can be done by removing the weakest individuals and replacing them with offspring from crossover and mutation. This method can maintain a stable population with a mix of old and new individuals, which can help preserve diversity and prevent rapid loss of good solutions.

# Justifications for these strategies are based on the balance between exploration and exploitation. E

# - Fitness Proportionate Selection and Tournament Selection are good at maintaining a balance between exploration and exploitation, with tournament selection generally providing a better balance.
# - Truncation Selection is more exploitative, quickly focusing on the best solutions but risking loss of diversity.
# - Elitism ensures that the best solutions are not lost, which is crucial for exploitation, but it must be used carefully to avoid reducing exploration.
# - Steady-State Selection provides a continuous blend of old and new individuals, which can be beneficial for maintaining diversity (exploration) while still allowing for incremental improvements (exploitation).

####################################################################