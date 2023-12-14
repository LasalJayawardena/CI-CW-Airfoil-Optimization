import os
import joblib
import numpy as np
import tensorflow as tf

from airfoil_Builder import Airfoil_Builder

# Define Constants
DRAG_MODEL = tf.keras.models.load_model('./FITNESS_MODEL/model15.h5')
MIN_MAX_SCALER = joblib.load('./FITNESS_MODEL/min_max_scaler_15point.pkl')
REYNOLDS_NUMBER = 20000
MACH_NUMBER = 1
ATTACK_ON_ANGLE = 5


def lift_coef_based_fitness_function(genotype: list, angle_range: tuple = (-10, 10), return_full_dict: bool = False) -> dict:
    """
    Calculates the lift, drag, and moment coefficients (cl, cd, cm) for a range of attack angles in a batch process.
    Optionally returns the average lift and drag coefficient.

    Parameters:
    - genotype (list): A list representing a genotype of airfoil parameters.
    - angle_range (tuple, optional): A tuple representing the range of attack angles. Default is (-10, 10).
    - return_full_dict (bool, optional): If True, returns the full dictionary of results. If False, returns the average cl, cd.

    Returns:
    - dict or tuple: A dictionary of (cl, cd, cm) for each angle, or a tuple of average cl and cd.
    """
    # Unpack the genotype list into named variables
    rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

    # Create an airfoil object
    airfoil = Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
    airfoil.build()
    xcoor = airfoil.XCoordinates
    yCoorUpper = airfoil.YCoordinatesUpper
    yCoorLower = airfoil.YCoordinatesLower

    # Prepare a batch of features for all angles
    angles = range(angle_range[0], angle_range[1] + 1)
    features_batch = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, angle] for angle in angles]
    features_batch = MIN_MAX_SCALER.transform(features_batch)
    features_batch = tf.convert_to_tensor(features_batch, dtype=tf.float32)

    # Predict coefficients for all angles in one batch
    predictions = DRAG_MODEL.predict(features_batch, verbose=0)

    # Process the results
    results = {angle: tuple(pred) for angle, pred in zip(angles, predictions)}
    sum_cl, sum_cd = np.sum(predictions[:, 0]), np.sum(predictions[:, 1])
    angle_count = len(angles)


    # Calculate and return the average cl and cd
    avg_cl = sum_cl / angle_count
    avg_cd = sum_cd / angle_count

    avg_cl_cd = avg_cl / avg_cd

    # Check if the full dictionary is to be returned
    if return_full_dict:
        return avg_cl_cd, results
    else:
        return avg_cl_cd


# Test Fitnes Function
from genotype import generate_random_genotype

genotype = generate_random_genotype()
print(genotype)

print(lift_coef_based_fitness_function(genotype))

def lift_coef_based_fitness_function_multi(genotypes: list, angle_range: tuple = (-10, 10), return_full_dict: bool = False) -> list:
    """
    Validates if the provided genotypes (lists of airfoil parameters) are within specified ranges and returns
    a list of corresponding average Estimated lift coefficients for each genotype.

    Parameters:
    - genotypes (list): A list of lists, each representing a genotype of airfoil parameters.
    - angle_range (tuple, optional): A tuple representing the range of attack angles. Default is (-10, 10).
    - return_full_dict (bool, optional): If True, returns the full dictionary of results. If False, returns the average cl, cd.

    Returns:
    - list[]: Return a list of average "Estimated" lift coefficients of the airfoils for each provided genotype.
    """
    average_lift_coefficients = []

    for genotype in genotypes:
        # Call the lift_coef_based_fitness_function for each genotype
        result = lift_coef_based_fitness_function(genotype, angle_range, return_full_dict=return_full_dict)
        average_lift_coefficients.append(result)

    return average_lift_coefficients



# # Test Fitnes Function Multi
# from genotype import generate_population

# genotypes = generate_population(10)
# print(genotypes)

# print(lift_coef_based_fitness_function_multi(genotypes))
# print(lift_coef_based_fitness_function_multi(genotypes, return_full_dict=True))

# # Define Constants
# DRAG_MODEL = tf.keras.models.load_model('model1_ANN.h5')
# REYNOLDS_NUMBER = 200000
# MACH_NUMBER = 1
# ATTACK_ON_ANGLE = 5



#  Old Code

# def lift_coef_based_fitness_function(genotype: list) -> float: 
#     """
#     Validates if the provided genotype (list of airfoil parameters) is within specified ranges.

#     Parameters:
#     - genotype (list): A list representing a genotype of airfoil parameters in the following order:
#         [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]

#     Returns:
#     - float: Return the "Estimated" lift coefficient of the airfoil.
#     """
#     # Unpack the genotype list into named variables
#     rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

#     # Create an airfoil object
#     airfoil = Airfoil_Builder(rLE, Xup,Yup,YXXup,Xlow,Ylow,YXXlow,yTE,deltaYTE,alphaTE,betaTE)
#     airfoil.build()
#     xcoor = airfoil.XCoordinates
#     yCoorUpper = airfoil.YCoordinatesUpper
#     yCoorLower = airfoil.YCoordinatesLower

#     features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
#     features = MIN_MAX_SCALER.transform(features)
#     features = tf.convert_to_tensor(features, dtype=tf.float32)

#     # Predict the drag coefficient
#     pred = DRAG_MODEL.predict(features, verbose=0)[0]
#     cl, cd, cm = pred[0], pred[1], pred[2]

#     return cl


# def lift_coef_based_fitness_function_multi(genotypes: list) -> list:
#     """
#     Validates if the provided genotypes (lists of airfoil parameters) are within specified ranges and returns
#     a list of corresponding Estimated lift coefficients for each genotype.

#     Parameters:
#     - genotypes (list): A list of lists, each representing a genotype of airfoil parameters in the following order:
#         [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]


#     Returns:
#     - list[]: Return a list of "Estimated" lift coefficients of the airfoils for each provided genotype.
#     """
#     lift_coefficients = []

#     for genotype in genotypes:
#         # Unpack the genotype list into named variables
#         rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

#         # Create an airfoil object
#         airfoil = Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
#         airfoil.build()
#         xcoor = airfoil.XCoordinates
#         yCoorUpper = airfoil.YCoordinatesUpper
#         yCoorLower = airfoil.YCoordinatesLower

#         features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
#         features = MIN_MAX_SCALER.transform(features)
#         features = tf.convert_to_tensor(features, dtype=tf.float32)

#         # Predict the lift coefficient
#         pred = DRAG_MODEL.predict(features, verbose=0)[0]
#         cl, cd, cm = pred[0], pred[1], pred[2]

#         lift_coefficients.append(cl)

#     return lift_coefficients    

# def lift_coef_based_fitness_function(genotype: list) -> float: 
#     """
#     Validates if the provided genotype (list of airfoil parameters) is within specified ranges.

#     Parameters:
#     - genotype (list): A list representing a genotype of airfoil parameters in the following order:
#         [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]
#         - rLE: Leading edge radius within the range of 0.5% to 5% of chord length.
#         - Xup: Upper crest abscissa within the range of 0 to the chord length.
#         - Yup: Upper crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - YXXup: Upper crest curvature within the range of min_curvature to max_curvature.
#         - Xlow: Lower crest abscissa within the range of 0 to the chord length.
#         - Ylow: Lower crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - YXXlow: Lower crest curvature within the range of min_curvature to max_curvature.
#         - yTE: Trailing edge ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - deltaYTE: Trailing edge thickness within the range of 0 to max_thickness * chord length.
#         - alphaTE: Trailing edge direction angle within the range of 0 to 360 degrees.
#         - betaTE: Trailing edge wedge angle within the range of 0 to 360 degrees.

#     Returns:
#     - float: Return the "Estimated" lift coefficient of the airfoil.
#     """
#     # Unpack the genotype list into named variables
#     rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

#     # Create an airfoil object
#     airfoil = Airfoil_Builder(rLE, Xup,Yup,YXXup,Xlow,Ylow,YXXlow,yTE,deltaYTE,alphaTE,betaTE)
#     airfoil.build()
#     xcoor = airfoil.XCoordinates
#     yCoorUpper = airfoil.YCoordinatesUpper
#     yCoorLower = airfoil.YCoordinatesLower

#     features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
#     features = tf.convert_to_tensor(features, dtype=tf.float32)

#     # Predict the drag coefficient
#     pred = DRAG_MODEL.predict(features)[0]
#     cl, cd, cm = pred[0], pred[1], pred[2]

#     return cl


# # # Test Fitnes Function
# # from genotype import generate_random_genotype

# # genotype = generate_random_genotype()
# # print(genotype)

# # print(lift_coef_based_fitness_function(genotype))


# def lift_coef_based_fitness_function_multi(genotypes: list) -> list:
#     """
#     Validates if the provided genotypes (lists of airfoil parameters) are within specified ranges and returns
#     a list of corresponding Estimated lift coefficients for each genotype.

#     Parameters:
#     - genotypes (list): A list of lists, each representing a genotype of airfoil parameters in the following order:
#         [rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE]
#         - rLE: Leading edge radius within the range of 0.5% to 5% of chord length.
#         - Xup: Upper crest abscissa within the range of 0 to the chord length.
#         - Yup: Upper crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - YXXup: Upper crest curvature within the range of min_curvature to max_curvature.
#         - Xlow: Lower crest abscissa within the range of 0 to the chord length.
#         - Ylow: Lower crest ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - YXXlow: Lower crest curvature within the range of min_curvature to max_curvature.
#         - yTE: Trailing edge ordinate within the range of -0.2 * chord length to 0.2 * chord length.
#         - deltaYTE: Trailing edge thickness within the range of 0 to max_thickness * chord length.
#         - alphaTE: Trailing edge direction angle within the range of 0 to 360 degrees.
#         - betaTE: Trailing edge wedge angle within the range of 0 to 360 degrees.

#     Returns:
#     - list[]: Return a list of "Estimated" lift coefficients of the airfoils for each provided genotype.
#     """
#     lift_coefficients = []

#     for genotype in genotypes:
#         # Unpack the genotype list into named variables
#         rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = genotype

#         # Create an airfoil object
#         airfoil = Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
#         airfoil.build()
#         xcoor = airfoil.XCoordinates
#         yCoorUpper = airfoil.YCoordinatesUpper
#         yCoorLower = airfoil.YCoordinatesLower

#         features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
#         features = tf.convert_to_tensor(features, dtype=tf.float32)

#         # Predict the lift coefficient
#         pred = DRAG_MODEL.predict(features)[0]
#         cl, cd, cm = pred[0], pred[1], pred[2]

#         lift_coefficients.append(cl)

#     return lift_coefficients    


# # # # Test Fitnes Function Multi
# # from genotype import generate_population

# # genotypes = generate_population(10)
# # print(genotypes)

# # print(lift_coef_based_fitness_function_multi(genotypes))