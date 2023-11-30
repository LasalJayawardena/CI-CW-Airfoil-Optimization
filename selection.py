import random

def placeholder_calculate_fitness_scores(population):
    return [random.random() for _ in population]  

def roulette_wheel_selection(population, num_selected, fitness_function):
    """
    Select individuals from the population using the roulette wheel selection method.
    
    Args:
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

# Test Roulette Wheel Selection
# from genotype import generate_population

# population = generate_population(100)
# num_selected = 5  # Number of individuals to select

# selected_individuals = roulette_wheel_selection(population, num_selected)
# print("Selected Individuals:")
# for ind in selected_individuals:
#     print(ind)


def tournament_selection(population, num_selected, tournament_size, fitness_function):
    """
    Select individuals from the population using the tournament selection method.
    
    Args:
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


def rank_selection(population, num_selected, fitness_function):
    """
    Select individuals from the population using the rank selection method.
    
    Args:
    - population (list): The population of individuals.
    - num_selected (int): The number of individuals to select.
    
    Returns:
    - selected_individuals (list): The list of selected individuals.
    """
    selected_individuals = []
    fitness_scores = fitness_function(population)
    
    # Rank individuals based on fitness scores
    ranked_individuals = sorted(enumerate(population), key=lambda x: fitness_scores[x[0]])
    
    # Calculate selection probabilities based on ranks
    total_ranks = sum(range(1, len(population) + 1))
    selection_probabilities = [(rank + 1) / total_ranks for rank, _ in enumerate(ranked_individuals)]
    
    # Make a cumulative distribution
    cumulative_probabilities = [sum(selection_probabilities[:i+1]) for i in range(len(selection_probabilities))]
    
    # Select individuals based on ranks
    for _ in range(num_selected):
        r = random.random()
        for i, individual in enumerate(ranked_individuals):
            if r <= cumulative_probabilities[i]:
                selected_individuals.append(individual[1])
                break
    
    return selected_individuals

# # Test Rank Selection
# from genotype import generate_population

# population = generate_population(100)

# num_selected = 5  # Number of individuals to select

# selected_individuals = rank_selection(population, num_selected, placeholder_calculate_fitness_scores)
# print("Selected Individuals:")

# for ind in selected_individuals:
#     print(ind)


def stochastic_universal_sampling(population, num_selected, fitness_function):
    """
    Select individuals from the population using Stochastic Universal Sampling (SUS).
    
    Args:
    - population (list): The population of individuals.
    - num_selected (int): The number of individuals to select.
    - fitness_function (function): A function that calculates fitness scores for the population.
    
    Returns:
    - selected_individuals (list): The list of selected individuals.
    """
    selected_individuals = []
    fitness_scores = fitness_function(population)
    total_fitness = sum(fitness_scores)
    point_distance = total_fitness / num_selected
    start_point = random.uniform(0, point_distance)

    # Create a list of points to sample
    points = [start_point + i * point_distance for i in range(num_selected)]

    # Calculate the cumulative sum of fitness scores
    cumulative_sum = [sum(fitness_scores[:i+1]) for i in range(len(fitness_scores))]
    
    # Select individuals based on the points
    for point in points:
        for i, score in enumerate(cumulative_sum):
            if score >= point:
                selected_individuals.append(population[i])
                break

    return selected_individuals

# # Test Stochastic Universal Sampling (SUS) Selection
# from genotype import generate_population

# # Generate a population
# population = generate_population(100)  
# num_selected = 5  # Number of individuals to select

# # Perform selection using SUS
# selected_individuals = stochastic_universal_sampling(population, num_selected, placeholder_calculate_fitness_scores)

# # Print selected individuals
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



######################## PLAN #############################

# 1. **Roulette Wheel Selection (Fitness Proportionate Selection):**
#    In this method, individuals are selected based on their fitness scores. The probability of an individual being selected is proportional to its fitness relative to the total fitness of the population.

#    *Justification:* This method is simple and intuitive, giving better individuals a higher chance of being selected. However, it can lead to premature convergence if high-fitness individuals dominate the population early on.

# 2. **Tournament Selection:**
#    A set number of individuals are randomly chosen from the population, and the one with the highest fitness within this group is selected. This process is repeated until the desired number of individuals is chosen.

#    *Justification:* Tournament selection can maintain diversity within the population because it does not solely depend on fitness. It allows individuals with lower fitness to have a chance of being selected, which can be beneficial in exploring the search space more thoroughly.

# 3. **Rank Selection:**
#    Individuals are ranked based on their fitness scores, and selection is based on this ranking rather than the actual fitness values. This helps to reduce the problem of fitness scaling where differences in fitness are very large or very small.

#    *Justification:* Rank selection can prevent the best individuals from taking over the population too quickly, thus maintaining diversity and reducing the risk of premature convergence.

# 4. **Stochastic Universal Sampling (SUS):**
#    This method is a variation of roulette wheel selection that provides a more even spread of selections across the population. It involves using a single random value to sample all individuals by choosing them at evenly spaced intervals.

#    *Justification:* SUS ensures a more representative sample of the population and can be more efficient than the basic roulette wheel selection, especially in larger populations.

# 5. **Elitism:**
#    A certain number of the best individuals from the current generation are guaranteed to be passed on to the next generation. This is often used in conjunction with other selection methods.

#    *Justification:* Elitism ensures that the solutions do not get worse from one generation to the next by preserving the best individuals. It can speed up convergence but should be used carefully to avoid losing diversity too quickly.

###########################################################