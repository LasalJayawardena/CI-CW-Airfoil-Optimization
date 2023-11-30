import random

def single_point_crossover(parent1, parent2):
    """
    Perform single-point crossover between two parents.
    
    Args:
    - parent1 (list): The first parent genotype.
    - parent2 (list): The second parent genotype.
    
    Returns:
    - offspring1, offspring2 (tuple of lists): Two offspring genotypes.
    """
    # Ensure the parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Genotypes of parents must be of the same length.")
    
    # Choose a random crossover point
    crossover_point = random.randint(1, len(parent1) - 1)
    
    # Create offspring by combining the genes of the parents
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2

# Test Single Point Crossover
# parent1 = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# parent2 = [0.04, 0.6, 0.15, 0.03, 0.5, -0.05, 0.02, 0.2, 0.01, 60, 25]

# offspring1, offspring2 = single_point_crossover(parent1, parent2)

# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)


def multi_point_crossover(parent1, parent2, n_points):
    """
    Perform multi-point crossover between two parents.
    
    Args:
    - parent1 (list): The first parent genotype.
    - parent2 (list): The second parent genotype.
    - n_points (int): The number of crossover points.
    
    Returns:
    - offspring1, offspring2 (tuple of lists): Two offspring genotypes.
    """
    # Ensure the parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Genotypes of parents must be of the same length.")
    
    # Ensure the number of crossover points is valid
    if n_points <= 0 or n_points >= len(parent1):
        raise ValueError("Number of crossover points must be between 1 and the length of the genotype - 1.")
    
    # Generate unique crossover points
    crossover_points = sorted(random.sample(range(1, len(parent1)), n_points))
    
    # Create offspring by combining the genes of the parents
    offspring1, offspring2 = parent1[:], parent2[:]
    
    # Perform crossover at the specified points
    for i in range(n_points):
        if i % 2 == 0:
            offspring1[crossover_points[i]:crossover_points[i+1] if i+1 < n_points else None] = parent2[crossover_points[i]:crossover_points[i+1] if i+1 < n_points else None]
            offspring2[crossover_points[i]:crossover_points[i+1] if i+1 < n_points else None] = parent1[crossover_points[i]:crossover_points[i+1] if i+1 < n_points else None]
    
    return offspring1, offspring2

# # Test Multi Point Crossover
# parent1 = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# parent2 = [0.04, 0.6, 0.15, 0.03, 0.5, -0.05, 0.02, 0.2, 0.01, 60, 25]
# n_points = 3  # Number of crossover points

# offspring1, offspring2 = multi_point_crossover(parent1, parent2, n_points)

# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)


def uniform_crossover(parent1, parent2):
    """
    Perform uniform crossover between two parents.
    
    Args:
    - parent1 (list): The first parent genotype.
    - parent2 (list): The second parent genotype.
    
    Returns:
    - offspring1, offspring2 (tuple of lists): Two offspring genotypes.
    """
    # Ensure the parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Genotypes of parents must be of the same length.")
    
    # Initialize offspring with empty lists
    offspring1 = []
    offspring2 = []
    
    # Iterate over each gene position
    for i in range(len(parent1)):
        # Randomly choose the gene from one of the parents for each offspring
        if random.random() < 0.5:
            offspring1.append(parent1[i])
            offspring2.append(parent2[i])
        else:
            offspring1.append(parent2[i])
            offspring2.append(parent1[i])
    
    return offspring1, offspring2

# # Test Uniform Crossover
# parent1 = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# parent2 = [0.04, 0.6, 0.15, 0.03, 0.5, -0.05, 0.02, 0.2, 0.01, 60, 25]

# offspring1, offspring2 = uniform_crossover(parent1, parent2)

# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)


def blend_crossover(parent1, parent2, alpha=0.5):
    """
    Perform BLX-alpha crossover between two parents.
    
    Args:
    - parent1 (list): The first parent genotype.
    - parent2 (list): The second parent genotype.
    - alpha (float): The alpha parameter defining the range of the blend.
    
    Returns:
    - offspring1, offspring2 (tuple of lists): Two offspring genotypes.
    """
    # Ensure the parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Genotypes of parents must be of the same length.")
    
    offspring1 = []
    offspring2 = []
    
    # Iterate over each gene to create offspring
    for gene1, gene2 in zip(parent1, parent2):
        # Calculate the range for the blend
        min_gene = min(gene1, gene2)
        max_gene = max(gene1, gene2)
        range_gene = max_gene - min_gene
        
        # Calculate lower and upper bounds for the blend
        lower_bound = min_gene - alpha * range_gene
        upper_bound = max_gene + alpha * range_gene
        
        # Generate random genes for the offspring within the blend range
        offspring_gene1 = random.uniform(lower_bound, upper_bound)
        offspring_gene2 = random.uniform(lower_bound, upper_bound)
        
        # Append the new genes to the offspring
        offspring1.append(offspring_gene1)
        offspring2.append(offspring_gene2)
    
    return offspring1, offspring2

# # Test BLX-alpha Crossover
# parent1 = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# parent2 = [0.04, 0.6, 0.15, 0.03, 0.5, -0.05, 0.02, 0.2, 0.01, 60, 25]

# offspring1, offspring2 = blend_crossover(parent1, parent2, alpha=0.5)

# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)


def arithmetic_crossover(parent1, parent2, alpha=None):
    """
    Perform arithmetic crossover between two parents.
    
    Args:
    - parent1 (list): The first parent genotype.
    - parent2 (list): The second parent genotype.
    - alpha (float): The weight used for the crossover. If None, a random value between 0 and 1 is used.
    
    Returns:
    - offspring1, offspring2 (tuple of lists): Two offspring genotypes.
    """
    # Ensure the parents are of the same length
    if len(parent1) != len(parent2):
        raise ValueError("Genotypes of parents must be of the same length.")
    
    # If alpha is not provided, generate a random value between 0 and 1
    if alpha is None:
        alpha = random.uniform(0, 1)
    
    # Create offspring by taking the weighted average of the parents' genes
    offspring1 = [alpha * x + (1 - alpha) * y for x, y in zip(parent1, parent2)]
    offspring2 = [alpha * y + (1 - alpha) * x for x, y in zip(parent1, parent2)]
    
    return offspring1, offspring2

# # Test Arithmetic Crossover
# parent1 = [0.03, 0.5, 0.1, 0.02, 0.4, -0.1, 0.01, 0.15, 0.005, 45, 30]
# parent2 = [0.04, 0.6, 0.15, 0.03, 0.5, -0.05, 0.02, 0.2, 0.01, 60, 25]

# # Perform arithmetic crossover with a specific alpha value
# offspring1, offspring2 = arithmetic_crossover(parent1, parent2, alpha=0.6)

# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)









######################### PLAN ##############################

# 1. **Single-Point Crossover:**
#    In single-point crossover, a point on the parent organisms' chromosome is selected. All data beyond that point in the chromosome is swapped between the two parent organisms. This method respects the grouping of parameters, which might be beneficial if certain parameters are more related to each other.

#    *Justification:* This is a simple and widely used crossover method. It can be effective if the parameters are ordered in such a way that related parameters are close to each other, thus preserving building blocks of good solutions.

# 2. **Two-Point (or Multi-Point) Crossover:**
#    This method selects two (or more) points on the chromosome and the segments between these points are swapped between the parent organisms. This allows for more complex shuffling of the genetic material.

#    *Justification:* Two-point crossover can be useful for mixing the features of the parents more thoroughly than single-point crossover. It can help in exploring a more diverse set of solutions, which is beneficial in a complex optimization problem like airfoil design.

# 3. **Uniform Crossover:**
#    In uniform crossover, each gene is considered separately. For each gene, a random number decides whether it comes from the first or the second parent. This method creates offspring that are a mix of both parents' genes.

#    *Justification:* Uniform crossover does not assume any linkage between the genes, which can be advantageous if there is no clear relationship between the parameters. It can introduce diversity and has the potential to explore new areas of the search space.

# 4. **Blend Crossover (BLX-alpha):**
#    Blend crossover is a method used for real-valued genetic algorithms. It generates offspring whose genes are a blend of the parents' genes, within a range defined by a parameter alpha.

#    *Justification:* This method is particularly suitable for real-valued optimization problems like airfoil design. It allows for a smooth exploration of the space between parents and can generate children that are not just a simple recombination but a true blend, potentially leading to better intermediate solutions.

# 5. **Arithmetic Crossover:**
#    Arithmetic crossover combines the parents' genes using a weighted average, where the weights are typically random numbers between 0 and 1. This method is also specific to real-valued representations.

#    *Justification:* Arithmetic crossover can be useful when the design variables are highly correlated, as it allows for a linear combination of the parents' traits. This can lead to offspring that are more likely to inherit the good performance of their parents in a continuous design space.

############################################################