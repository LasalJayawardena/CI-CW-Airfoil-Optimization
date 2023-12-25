from constraints import get_all_valid_ranges, Parameter
import random
from typing import List, Tuple


# Initialize the valid ranges for each parameter
valid_ranges = get_all_valid_ranges()
paramter_list = list(valid_ranges.keys())

def random_resetting_mutation(individual: List[float], mutation_rate: float) -> List[float]:
    """
    Perform random resetting mutation on an individual.
    
    Args:
    - individual (List[float]): The individual genotype to mutate.
    - mutation_rate (float): The probability of mutation for each gene.
    
    Returns:
    - List[float]: The mutated genotype.
    """
    # Perform mutation based on the mutation rate
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.uniform(*valid_ranges[paramter_list[i]])
    
    return mutated_individual

# # Test Random Resetting Mutation
# mutation_rate = 0.1  # Example mutation rate (10%)

# individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# mutated_individual = random_resetting_mutation(individual, mutation_rate)

# print("Original Individual:", individual)
# print("Mutated Individual:", mutated_individual)



def creep_mutation(individual: List[float], mutation_rate: float, creep_magnitude: float) -> List[float]:
    """
    Perform creep mutation on an individual.
    
    Args:
    - individual (List[float]): The individual genotype to mutate.
    - mutation_rate (float): The probability of mutation for each gene. AKA Creep Rate
    - creep_magnitude (float): The magnitude of the creep mutation.
    
    Returns:
    - List[float]: The mutated genotype.
    """
    # Perform creep mutation based on the creep rate
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            delta = random.uniform(-creep_magnitude, creep_magnitude)
            mutated_value = mutated_individual[i] + delta
            # Ensure the mutated value stays within bounds
            mutated_individual[i] = max(valid_ranges[paramter_list[i]][0], min(valid_ranges[paramter_list[i]][1], mutated_value))
    
    return mutated_individual

# # Test Creep Mutation
# creep_rate = 0.25  # Example creep mutation rate (10%)
# creep_magnitude = 0.01  # Example creep magnitude

# individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# mutated_individual = creep_mutation(individual, creep_rate, creep_magnitude)

# print("Original Individual:", individual)
# print("Mutated Individual:", mutated_individual)


def gaussian_mutation(individual: List[float], mutation_rate: float, std_dev: float) -> List[float]:
    """
    Perform Gaussian mutation on an individual.
    
    Args:
    - individual (List[float]): The individual genotype to mutate.
    - mutation_rate (float): The probability of mutation for each gene.
    - std_dev (float): The standard deviation for the Gaussian distribution.
    
    Returns:
    - List[float]: The mutated genotype.
    """
    # Perform mutation based on the mutation rate
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutation = random.gauss(0, std_dev)
            mutated_individual[i] += mutation
            # Ensure the mutated value is within the allowed range
            mutated_individual[i] = max(valid_ranges[paramter_list[i]][0], min(valid_ranges[paramter_list[i]][1], mutated_individual[i]))
    
    return mutated_individual

# # Test Gaussian Mutation
# mutation_rate = 0.1  # Example mutation rate (10%)
# std_dev = 0.01  # Example standard deviation for Gaussian distribution

# individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# mutated_individual = gaussian_mutation(individual, mutation_rate, std_dev)

# print("Original Individual:", individual)
# print("Mutated Individual:", mutated_individual)


def boundary_mutation(individual: List[float], mutation_rate: float) -> List[float]:
    """
    Perform boundary mutation on an individual.
    
    Args:
    - individual (List[float]): The individual genotype to mutate.
    - mutation_rate (float): The probability of mutation for each gene.
    
    Returns:
    - List[float]: The mutated genotype.
    """
    # Perform mutation based on the mutation rate
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Choose one of the boundaries at random for the mutation
            mutated_individual[i] = random.choice(valid_ranges[paramter_list[i]])
    
    return mutated_individual

# # Test Boundary Mutation
# # We can use prior knowledge for better reuslts in boundaries
# mutation_rate = 0.5  # Example mutation rate (10%)

# individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# mutated_individual = boundary_mutation(individual, mutation_rate)

# print("Original Individual:", individual)
# print("Mutated Individual:", mutated_individual)


def uniform_mutation(individual: List[float], mutation_rate: float, uniform_range_fraction: float = 0.1) -> List[float]:
    """
    Perform uniform mutation on an individual by slightly altering each gene within a fraction of its total range.
    
    Args:
    - individual (List[float]): The individual genotype to mutate.
    - mutation_rate (float): The probability of mutation for each gene.
    - uniform_range_fraction (float): Fraction of the total range to use for mutation.
    
    Returns:
    - List[float]: The mutated genotype.
    """
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            range_min, range_max = valid_ranges[paramter_list[i]]
            mutation_range = (range_max - range_min) * uniform_range_fraction
            delta = random.uniform(-mutation_range, mutation_range)
            mutated_individual[i] = max(range_min, min(range_max, mutated_individual[i] + delta))
    
    return mutated_individual

# # Test Uniform Mutation
# mutation_rate = 0.1  # Example mutation rate (10%)

# individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# mutated_individual = uniform_mutation(individual, mutation_rate)

# print("Original Individual:", individual)
# print("Mutated Individual:", mutated_individual)



############################## PLAN #####################################

# Remeber Range loossening is a consideration for the mutation functions.

# 1. **Random Resetting:**
#    This strategy involves selecting a gene at random and resetting it to a value within its allowed range. This is akin to a random jump in the search space for that particular parameter.

#    *Justification:* Random resetting is simple and can help escape local optima by providing a chance to explore entirely new areas of the search space. It's particularly useful when the search space is large and the optimal value for a parameter is not known a priori.

# 2. **Creep Mutation:**
#    Creep mutation involves making a small random change to a gene's value. The change is typically drawn from a normal distribution with a mean of zero and a small standard deviation.

#    *Justification:* This strategy is useful for fine-tuning solutions. In the context of airfoil design, where small changes in parameters can lead to significant performance differences, creep mutation can help refine an already good design.

# 3. **Gaussian Mutation:**
#    Gaussian mutation adds a normally distributed random value to the selected gene. The standard deviation of the distribution can be adjusted to control the magnitude of the mutation.

#    *Justification:* Gaussian mutation allows for a balance between exploration and exploitation. It can make both small and large adjustments to the gene values, which is beneficial when the landscape of the optimization problem is not well understood.

# 4. **Boundary Mutation:**
#    Boundary mutation involves setting a gene to one of the boundaries of its allowed range. This is particularly useful when there is reason to believe that optimal values may lie at the extremes of the search space.

#    *Justification:* In airfoil design, certain parameters might have optimal values at the limits of their ranges (e.g., maximum or minimum curvature). Boundary mutation can help discover such extreme solutions.

# 5. **Uniform Mutation:**
#    Uniform mutation replaces the value of a gene with a uniform random value selected from the gene's allowed range.

#    *Justification:* This strategy is useful for a broad search of the solution space and can introduce significant diversity into the population. It is less biased than other methods and can be particularly effective in the early stages of the optimization process to ensure a good coverage of the search space.

##########################################################################


# ## Old Code

# import random

# def random_resetting_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate):
#     """
#     Perform random resetting mutation on an individual.
    
#     Args:
#     - individual (list): The individual genotype to mutate.
#     - chord_length (float): The chord length of the airfoil.
#     - min_curvature (float): The minimum allowable curvature.
#     - max_curvature (float): The maximum allowable curvature.
#     - max_thickness (float): The maximum allowable thickness as a proportion of chord length.
#     - mutation_rate (float): The probability of mutation for each gene.
    
#     Returns:
#     - mutated_individual (list): The mutated genotype.
#     """
#     # Define the ranges for each parameter based on the given constraints
#     param_ranges = [
#         (0.005 * chord_length, 0.05 * chord_length),  # rLE range
#         (0, chord_length),                            # Xup range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Yup range
#         (min_curvature, max_curvature),               # YXXup range
#         (0, chord_length),                            # Xlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Ylow range
#         (min_curvature, max_curvature),               # YXXlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # yTE range
#         (0, max_thickness * chord_length),            # deltaYTE range
#         (0, 360),                                     # alphaTE range
#         (0, 360)                                      # betaTE range
#     ]
    
#     # Perform mutation based on the mutation rate
#     mutated_individual = individual[:]
#     for i in range(len(mutated_individual)):
#         if random.random() < mutation_rate:
#             mutated_individual[i] = random.uniform(*param_ranges[i])
    
#     return mutated_individual

# # # Test Random Resetting Mutation
# # chord_length = 1.0  # Example chord length
# # min_curvature = -0.1  # Example minimum curvature
# # max_curvature = 0.1  # Example maximum curvature
# # max_thickness = 0.1  # Example maximum thickness as a proportion of chord length
# # mutation_rate = 0.1  # Example mutation rate (10%)

# # individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# # mutated_individual = random_resetting_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate)

# # print("Original Individual:", individual)
# # print("Mutated Individual:", mutated_individual)



# def creep_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, creep_rate, creep_magnitude):
#     """
#     Perform creep mutation on an individual.
    
#     Args:
#     - individual (list): The individual genotype to mutate.
#     - chord_length (float): The chord length of the airfoil.
#     - min_curvature (float): The minimum allowable curvature.
#     - max_curvature (float): The maximum allowable curvature.
#     - max_thickness (float): The maximum allowable thickness as a proportion of chord length.
#     - creep_rate (float): The probability of mutation for each gene.
#     - creep_magnitude (float): The magnitude of the creep mutation.
    
#     Returns:
#     - mutated_individual (list): The mutated genotype.
#     """
#     # Define the ranges for each parameter based on the given constraints
#     param_ranges = [
#         (0.005 * chord_length, 0.05 * chord_length),  # rLE range
#         (0, chord_length),                            # Xup range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Yup range
#         (min_curvature, max_curvature),               # YXXup range
#         (0, chord_length),                            # Xlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Ylow range
#         (min_curvature, max_curvature),               # YXXlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # yTE range
#         (0, max_thickness * chord_length),            # deltaYTE range
#         (0, 360),                                     # alphaTE range
#         (0, 360)                                      # betaTE range
#     ]
    
#     # Perform creep mutation based on the creep rate
#     mutated_individual = individual[:]
#     for i in range(len(mutated_individual)):
#         if random.random() < creep_rate:
#             # Apply creep mutation within the allowed range
#             delta = random.uniform(-creep_magnitude, creep_magnitude)
#             mutated_value = mutated_individual[i] + delta
#             # Ensure the mutated value stays within bounds
#             lower_bound, upper_bound = param_ranges[i]
#             mutated_value = max(lower_bound, min(upper_bound, mutated_value))
#             mutated_individual[i] = mutated_value
    
#     return mutated_individual

# # # Test Creep Mutation
# # chord_length = 1.0  # Example chord length
# # min_curvature = -0.1  # Example minimum curvature
# # max_curvature = 0.1  # Example maximum curvature
# # max_thickness = 0.1  # Example maximum thickness as a proportion of chord length
# # creep_rate = 0.1  # Example creep mutation rate (10%)
# # creep_magnitude = 0.01  # Example creep magnitude

# # individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# # mutated_individual = creep_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, creep_rate, creep_magnitude)

# # print("Original Individual:", individual)
# # print("Mutated Individual:", mutated_individual)



# def clamp(value, min_value, max_value):
#     """Clamp the value to be within the specified range."""
#     return max(min_value, min(value, max_value))

# def gaussian_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate, std_dev):
#     """
#     Perform Gaussian mutation on an individual.
    
#     Args:
#     - individual (list): The individual genotype to mutate.
#     - chord_length (float): The chord length of the airfoil.
#     - min_curvature (float): The minimum allowable curvature.
#     - max_curvature (float): The maximum allowable curvature.
#     - max_thickness (float): The maximum allowable thickness as a proportion of chord length.
#     - mutation_rate (float): The probability of mutation for each gene.
#     - std_dev (float): The standard deviation for the Gaussian distribution.
    
#     Returns:
#     - mutated_individual (list): The mutated genotype.
#     """
#     # Define the ranges for each parameter based on the given constraints
#     param_ranges = [
#         (0.005 * chord_length, 0.05 * chord_length),  # rLE range
#         (0, chord_length),                            # Xup range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Yup range
#         (min_curvature, max_curvature),               # YXXup range
#         (0, chord_length),                            # Xlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Ylow range
#         (min_curvature, max_curvature),               # YXXlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # yTE range
#         (0, max_thickness * chord_length),            # deltaYTE range
#         (0, 360),                                     # alphaTE range
#         (0, 360)                                      # betaTE range
#     ]
    
#     # Perform mutation based on the mutation rate
#     mutated_individual = individual[:]
#     for i in range(len(mutated_individual)):
#         if random.random() < mutation_rate:
#             mutation = random.gauss(0, std_dev)
#             mutated_individual[i] += mutation
#             # Ensure the mutated value is within the allowed range
#             mutated_individual[i] = clamp(mutated_individual[i], *param_ranges[i])
    
#     return mutated_individual

# # # Test Gaussian Mutation
# # chord_length = 1.0  # Example chord length
# # min_curvature = -0.1  # Example minimum curvature
# # max_curvature = 0.1  # Example maximum curvature
# # max_thickness = 0.1  # Example maximum thickness as a proportion of chord length
# # mutation_rate = 0.1  # Example mutation rate (10%)
# # std_dev = 0.01  # Example standard deviation for Gaussian distribution

# # individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# # mutated_individual = gaussian_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate, std_dev)

# # print("Original Individual:", individual)
# # print("Mutated Individual:", mutated_individual)


# def boundary_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate):
#     """
#     Perform boundary mutation on an individual.
    
#     Args:
#     - individual (list): The individual genotype to mutate.
#     - chord_length (float): The chord length of the airfoil.
#     - min_curvature (float): The minimum allowable curvature.
#     - max_curvature (float): The maximum allowable curvature.
#     - max_thickness (float): The maximum allowable thickness as a proportion of chord length.
#     - mutation_rate (float): The probability of mutation for each gene.
    
#     Returns:
#     - mutated_individual (list): The mutated genotype.
#     """
#     # Define the ranges for each parameter based on the given constraints
#     param_ranges = [
#         (0.005 * chord_length, 0.05 * chord_length),  # rLE range
#         (0, chord_length),                            # Xup range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Yup range
#         (min_curvature, max_curvature),               # YXXup range
#         (0, chord_length),                            # Xlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Ylow range
#         (min_curvature, max_curvature),               # YXXlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # yTE range
#         (0, max_thickness * chord_length),            # deltaYTE range
#         (0, 360),                                     # alphaTE range
#         (0, 360)                                      # betaTE range
#     ]
    
#     # Perform mutation based on the mutation rate
#     mutated_individual = individual[:]
#     for i in range(len(mutated_individual)):
#         if random.random() < mutation_rate:
#             # Choose one of the boundaries at random for the mutation
#             mutated_individual[i] = random.choice(param_ranges[i])
    
#     return mutated_individual

# # # Test Boundary Mutation
# # # We can use prior knowledge for better reuslts in boundaries
# # chord_length = 1.0  # Example chord length
# # min_curvature = -0.1  # Example minimum curvature
# # max_curvature = 0.1  # Example maximum curvature
# # max_thickness = 0.1  # Example maximum thickness as a proportion of chord length
# # mutation_rate = 0.1  # Example mutation rate (10%)

# # individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# # mutated_individual = boundary_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate)

# # print("Original Individual:", individual)
# # print("Mutated Individual:", mutated_individual)


# def uniform_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate):
#     """
#     Perform uniform mutation on an individual.
    
#     Args:
#     - individual (list): The individual genotype to mutate.
#     - chord_length (float): The chord length of the airfoil.
#     - min_curvature (float): The minimum allowable curvature.
#     - max_curvature (float): The maximum allowable curvature.
#     - max_thickness (float): The maximum allowable thickness as a proportion of chord length.
#     - mutation_rate (float): The probability of mutation for each gene.
    
#     Returns:
#     - mutated_individual (list): The mutated genotype.
#     """
#     # Define the ranges for each parameter based on the given constraints
#     param_ranges = [
#         (0.005 * chord_length, 0.05 * chord_length),  # rLE range
#         (0, chord_length),                            # Xup range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Yup range
#         (min_curvature, max_curvature),               # YXXup range
#         (0, chord_length),                            # Xlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # Ylow range
#         (min_curvature, max_curvature),               # YXXlow range
#         (-0.2 * chord_length, 0.2 * chord_length),    # yTE range
#         (0, max_thickness * chord_length),            # deltaYTE range
#         (0, 360),                                     # alphaTE range
#         (0, 360)                                      # betaTE range
#     ]
    
#     # Perform mutation based on the mutation rate
#     mutated_individual = individual[:]
#     for i in range(len(mutated_individual)):
#         if random.random() < mutation_rate:
#             mutated_individual[i] = random.uniform(*param_ranges[i])
    
#     return mutated_individual

# # # Test Uniform Mutation
# # chord_length = 1.0  # Example chord length
# # min_curvature = -0.1  # Example minimum curvature
# # max_curvature = 0.1  # Example maximum curvature
# # max_thickness = 0.1  # Example maximum thickness as a proportion of chord length
# # mutation_rate = 0.1  # Example mutation rate (10%)

# # individual = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# # mutated_individual = uniform_mutation(individual, chord_length, min_curvature, max_curvature, max_thickness, mutation_rate)

# # print("Original Individual:", individual)
# # print("Mutated Individual:", mutated_individual)