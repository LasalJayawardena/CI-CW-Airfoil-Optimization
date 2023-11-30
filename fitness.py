import os
import tensorflow as tf

from airfoil_Builder import Airfoil_Builder

# Define Constants
DRAG_MODEL = tf.keras.models.load_model('model1_ANN.h5')
REYNOLDS_NUMBER = 200000
MACH_NUMBER = 1
ATTACK_ON_ANGLE = 5


def lift_coef_based_fitness_function(genotype: list) -> float: 
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

    Returns:
    - float: Return the "Estimated" lift coefficient of the airfoil.
    """
    # Unpack the genotype list into named variables
    rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

    # Create an airfoil object
    airfoil = Airfoil_Builder(rLE, Xup,Yup,YXXup,Xlow,Ylow,YXXlow,yTE,deltaYTE,alphaTE,betaTE)
    airfoil.build()
    xcoor = airfoil.XCoordinates
    yCoorUpper = airfoil.YCoordinatesUpper
    yCoorLower = airfoil.YCoordinatesLower

    features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
    features = tf.convert_to_tensor(features, dtype=tf.float32)

    # Predict the drag coefficient
    pred = DRAG_MODEL.predict(features)[0]
    cl, cd, cm = pred[0], pred[1], pred[2]

    return cl


# # Test Fitnes Function
# from genotype import generate_random_genotype

# genotype = generate_random_genotype()
# print(genotype)

# print(lift_coef_based_fitness_function(genotype))


def lift_coef_based_fitness_function_multi(genotypes: list) -> list:
    """
    Validates if the provided genotypes (lists of airfoil parameters) are within specified ranges and returns
    a list of corresponding Estimated lift coefficients for each genotype.

    Parameters:
    - genotypes (list): A list of lists, each representing a genotype of airfoil parameters in the following order:
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

    Returns:
    - list[]: Return a list of "Estimated" lift coefficients of the airfoils for each provided genotype.
    """
    lift_coefficients = []

    for genotype in genotypes:
        # Unpack the genotype list into named variables
        rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

        # Create an airfoil object
        airfoil = Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
        airfoil.build()
        xcoor = airfoil.XCoordinates
        yCoorUpper = airfoil.YCoordinatesUpper
        yCoorLower = airfoil.YCoordinatesLower

        features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
        features = tf.convert_to_tensor(features, dtype=tf.float32)

        # Predict the lift coefficient
        pred = DRAG_MODEL.predict(features)[0]
        cl, cd, cm = pred[0], pred[1], pred[2]

        lift_coefficients.append(cl)

    return lift_coefficients    


# # # Test Fitnes Function Multi
# from genotype import generate_population

# genotypes = generate_population(10)
# print(genotypes)

# print(lift_coef_based_fitness_function_multi(genotypes))