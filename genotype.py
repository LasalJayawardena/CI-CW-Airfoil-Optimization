import random

def generate_random_airfoil_parameters(chord_length: float = 1.0,
                                       min_curvature: float = -0.1,
                                       max_curvature: float = 0.1,
                                       max_thickness: float = 0.1) -> dict:
    """
    Generates random airfoil parameters within specified ranges.

    Parameters:
    - chord_length (float): The length of the airfoil's chord. Default is 1.0.
    - min_curvature (float): The minimum curvature value for upper and lower crest curvatures.
                             Default is -0.1.
    - max_curvature (float): The maximum curvature value for upper and lower crest curvatures.
                             Default is 0.1.
    - max_thickness (float): The maximum trailing edge thickness as a percentage of the chord length.
                             Default is 0.1.

    Returns:
    - dict: A dictionary containing randomly generated airfoil parameters:
        - 'rLE': Leading edge radius within the range of 0.5% to 5% of chord length.
        - 'Xup': Upper crest abscissa within the range of 0 to the chord length.
        - 'Yup': Upper crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - 'YXXup': Upper crest curvature within the range of min_curvature to max_curvature.
        - 'Xlow': Lower crest abscissa within the range of 0 to the chord length.
        - 'Ylow': Lower crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - 'YXXlow': Lower crest curvature within the range of min_curvature to max_curvature.
        - 'yTE': Trailing edge ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - 'deltaYTE': Trailing edge thickness within the range of 0 to max_thickness * chord length.
        - 'alphaTE': Trailing edge direction angle within the range of 0 to 360 degrees.
        - 'betaTE': Trailing edge wedge angle within the range of 0 to 360 degrees.
    """

    parameters = {
        'rLE': random.uniform(0.005 * chord_length, 0.05 * chord_length),
        'Xup': random.uniform(0, chord_length),
        'Yup': random.uniform(-0.2 * chord_length, 0.2 * chord_length),
        'YXXup': random.uniform(min_curvature, max_curvature),
        'Xlow': random.uniform(0, chord_length),
        'Ylow': random.uniform(-0.2 * chord_length, 0.2 * chord_length),
        'YXXlow': random.uniform(min_curvature, max_curvature),
        'yTE': random.uniform(-0.2 * chord_length, 0.2 * chord_length),
        'deltaYTE': random.uniform(0, max_thickness * chord_length),
        'alphaTE': random.uniform(0, 360),
        'betaTE': random.uniform(0, 360)
    }
    return parameters

# # Test Generate Random Airfoil Parameters
# random_parameters = generate_random_airfoil_parameters(chord_length=1)  
# print(random_parameters)

# from plot_utils import plot_airfoil
# plot_airfoil(**random_parameters)

def generate_random_genotype(chord_length: float = 1.0,
                            min_curvature: float = -0.1,
                            max_curvature: float = 0.1,
                            max_thickness: float = 0.1) -> list:
    
    """
    Generates a random genotype (list of airfoil parameters) based on specified ranges.

    Parameters:
    - chord_length (float): The length of the airfoil's chord. Default is 1.0.
    - min_curvature (float): The minimum curvature value for upper and lower crest curvatures.
                             Default is -0.1.
    - max_curvature (float): The maximum curvature value for upper and lower crest curvatures.
                             Default is 0.1.
    - max_thickness (float): The maximum trailing edge thickness as a percentage of the chord length.
                             Default is 0.1.

    Returns:
    - list: A list representing a random genotype of airfoil parameters:
        The list contains airfoil parameters in the following order:
        [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]
        - rLE: Leading edge radius within the range of 0.5% to 5% of chord length.
        - Xup: Upper crest abscissa within the range of 0 to the chord length.
        - Yup: Upper crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - YXXup: Upper crest curvature within the range of min_curvature to max_curvature.
        - Xlow: Lower crest abscissa within the range of 0 to the chord length.
        - Ylow: Lower crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - YXXlow: Lower crest curvature within the range of min_curvature to max_curvature.
        - yTE: Trailing edge ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - deltaYTE: Trailing edge thickness within the range of 0 to max_thickness * chord length.
        - alphaTE: Trailing edge direction angle within the range of 0 to 360 degrees.
        - betaTE: Trailing edge wedge angle within the range of 0 to 360 degrees.
    """

    # Generate random airfoil parameters
    parameters = generate_random_airfoil_parameters(chord_length, min_curvature, max_curvature, max_thickness)

    genotype = []

    # Add airfoil parameters to the genotype
    for key, value in parameters.items():
        genotype.append(value)

    return genotype

# # Test Generate Random Genotype
# random_genotype = generate_random_genotype(chord_length=1.5)
# print(random_genotype)

# Functioon to valdate genotype
def check_valid_genotype(genotype: list, chord_length: float) -> bool:
    """
    Validates if the provided genotype (list of airfoil parameters) is within specified ranges.

    Parameters:
    - genotype (list): A list representing a genotype of airfoil parameters in the following order:
        [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]
        - rLE: Leading edge radius within the range of 0.5% to 5% of chord length.
        - Xup: Upper crest abscissa within the range of 0 to the chord length.
        - Yup: Upper crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - YXXup: Upper crest curvature within the range of min_curvature to max_curvature.
        - Xlow: Lower crest abscissa within the range of 0 to the chord length.
        - Ylow: Lower crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - YXXlow: Lower crest curvature within the range of min_curvature to max_curvature.
        - yTE: Trailing edge ordinate within the range of -0.2 * chord length to 0.2 * chord length.
        - deltaYTE: Trailing edge thickness within the range of 0 to max_thickness * chord length.
        - alphaTE: Trailing edge direction angle within the range of 0 to 360 degrees.
        - betaTE: Trailing edge wedge angle within the range of 0 to 360 degrees.
    - chord_length (float): The length of the airfoil's chord.

    Returns:
    - bool: True if the genotype is valid (within specified ranges), otherwise False.
    """
    valid = True

    # Unpack the genotype list into named variables
    rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

    # Validating rLE (leading edge radius)
    if not (0.005 * chord_length <= rLE <= 0.05 * chord_length):
        valid = False

    # Validating Xup and Xlow (upper and lower crest abscissas)
    if not (0 <= Xup <= chord_length):
        valid = False
    if not (0 <= Xlow <= chord_length):
        valid = False

    # Validating Yup, Ylow, yTE (upper and lower crest ordinates, trailing edge ordinate)
    y_range = 0.2 * chord_length
    if not (-y_range <= Yup <= y_range):
        valid = False
    if not (-y_range <= Ylow <= y_range):
        valid = False
    if not (-y_range <= yTE <= y_range):
        valid = False

    # Validating YXXup and YXXlow (upper and lower crest curvatures)
    # No specific range validation provided in this example for curvature

    # Validating deltaYTE (trailing edge thickness)
    if not (0 <= deltaYTE <= 0.1 * chord_length):  # Assuming max thickness is 10% of chord length
        valid = False

    # Validating alphaTE and betaTE (trailing edge direction and wedge angle)
    if not (0 <= alphaTE <= 360):
        valid = False
    if not (0 <= betaTE <= 360):
        valid = False

    return valid

# # Test Check Valid Genotype
# # Example parameters list (replace with your generated parameters list)
# parameters_list = [0.02, 0.5, 0.1, 0.05, 0.3, -0.15, -0.03, 0.05, 0.02, 180, 270]
# parameters_list = [0.2, 0.5, 0.1, 0.05, 0.3, -0.15, -0.03, 0.05, 0.02, 180, 270]
# # Example chord length
# chord_length = 1.0

# # Checking if the parameters list forms a valid genotype
# is_valid = check_valid_genotype(parameters_list, chord_length)
# print("Are parameters valid for genotype?", is_valid)


# Generate Population of N Genotypes
def generate_population(population_size: int = 10,
                        chord_length: float = 1.0,
                        min_curvature: float = -0.1,
                        max_curvature: float = 0.1,
                        max_thickness: float = 0.1) -> list:

    population = []

    # Generate random genotypes until population size is reached
    while len(population) < population_size:
        # Generate random genotype
        genotype = generate_random_genotype(chord_length, min_curvature, max_curvature, max_thickness)

        # Check if genotype is valid
        if check_valid_genotype(genotype, chord_length):
            # Add genotype to population
            population.append(genotype)

    return population

# # Test Generate Population
# # Example population size
# population_size = 10
# # Example chord length
# chord_length = 1.0
# # Example min and max curvature
# min_curvature = -0.1
# max_curvature = 0.1
# # Example max thickness
# max_thickness = 0.1

# # Generate population
# population = generate_population(population_size, chord_length, min_curvature, max_curvature, max_thickness)
# print("Population:")
# print(population)



