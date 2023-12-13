from enum import Enum
from typing import Tuple, Dict

class Parameter(Enum):
    rLE = ('rLE', (0.0085, 0.0126))
    Xup = ('Xup', (0.41, 0.46))
    Yup = ('Yup', (0.11, 0.13))
    YXXup = ('YXXup', (-0.9, -0.7))
    Xlow = ('Xlow', (0.20, 0.26))
    Ylow = ('Ylow', (-0.023, -0.015))
    YXXlow = ('YXXlow', (0.05, 0.20))
    yTE = ('yTE', (-0.006, -0.003))
    deltaYTE = ('deltaYTE', (0.0025, 0.0050))
    alphaTE = ('alphaTE', (7.0, 10.0))
    betaTE = ('betaTE', (10.0, 14.0))

    def __init__(self, param_name: str, valid_range: Tuple[float, float]):
        self.param_name = param_name
        self.valid_range = valid_range

def get_valid_range(parameter: Parameter) -> Tuple[float, float]:
    """
    Get the valid range for a given parameter.

    Args:
    - parameter (Parameter): The enum member representing the parameter.

    Returns:
    - Tuple[float, float]: The valid range for the parameter.
    """
    return parameter.valid_range

def get_all_valid_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get a dictionary of all valid ranges for parameters.

    Returns:
    - Dict[str, Tuple[float, float]]: A dictionary with parameter names as keys and their valid ranges as values.
    """
    return {param.param_name: param.valid_range for param in Parameter}

# # Test usage
# print("Valid range for rLE:", get_valid_range(Parameter.rLE))
# print("All valid ranges:", get_all_valid_ranges())

# valid_ranges = {
#         'rLE': (0.0085, 0.0126),
#         'Xup': (0.41, 0.46),
#         'Yup': (0.11, 0.13),
#         'YXXup': (-0.9, -0.7),
#         'Xlow': (0.20, 0.26),
#         'Ylow': (-0.023, -0.015),
#         'YXXlow': (0.05, 0.20),
#         'yTE': (-0.006, -0.003),
#         'deltaYTE': (0.0025, 0.0050),
#         'alphaTE': (7.0, 10.0),
#         'betaTE': (10.0, 14.0)
#     }

# print(valid_ranges == get_all_valid_ranges())













# # Old Code

# from enum import Enum, auto

# # Option 1
# class Parameter(Enum):
#     rLE = auto()       # Leading edge radius
#     Xup = auto()       # Upper crest abscissa
#     Yup = auto()       # Upper crest ordinate
#     YXXup = auto()     # Upper crest curvature
#     Xlow = auto()      # Lower crest abscissa
#     Ylow = auto()      # Lower crest ordinate
#     YXXlow = auto()    # Lower crest curvature
#     yTE = auto()       # Trailing edge ordinate
#     deltaYTE = auto()  # Trailing edge thickness
#     alphaTE = auto()   # Trailing edge direction angle
#     betaTE = auto()    # Trailing edge wedge angle

# class Constraint(Enum):
#     CHORD_LENGTH = 1.0
#     MIN_CURVATURE = -0.1
#     MAX_CURVATURE = 0.1
#     MAX_THICKNESS = 0.1


# # Option 2 - with Flexibility
# class Parameter(Enum):
#     def __new__(cls, min_value, max_value, soft_min=None, soft_max=None):
#         obj = object.__new__(cls)
#         obj._value_ = obj  # Unique value for each member
#         obj.min_value = min_value
#         obj.max_value = max_value
#         obj.soft_min = soft_min if soft_min is not None else min_value
#         obj.soft_max = soft_max if soft_max is not None else max_value
#         return obj

#     rLE = (0.005, 0.05)       # Leading edge radius
#     Xup = (0, 1.0)            # Upper crest abscissa
#     Yup = (-0.2, 0.2)         # Upper crest ordinate
#     YXXup = (-0.1, 0.1)       # Upper crest curvature
#     Xlow = (0, 1.0)           # Lower crest abscissa
#     Ylow = (-0.2, 0.2)        # Lower crest ordinate
#     YXXlow = (-0.1, 0.1)      # Lower crest curvature
#     yTE = (-0.2, 0.2)         # Trailing edge ordinate
#     deltaYTE = (0, 0.1)       # Trailing edge thickness
#     alphaTE = (0, 360)        # Trailing edge direction angle
#     betaTE = (0, 360)         # Trailing edge wedge angle



# # # To Use

# # from constraints import Constraint, Parameter


# # chord_length = Constraint.CHORD_LENGTH.value
# # min_curvature = Constraint.MIN_CURVATURE.value
# # max_curvature = Constraint.MAX_CURVATURE.value
# # max_thickness = Constraint.MAX_THICKNESS.value


# # for param in Parameter:
# #     print(f"Parameter: {param.name}")

# # for constraint in Constraint:
# #     print(f"Constraint: {constraint.name}, Value: {constraint.value}")

# # print(f"Chord length: {Constraint.CHORD_LENGTH.value}")


# # print(f"rLE min: {Parameter.rLE.min_value}, max: {Parameter.rLE.max_value}")
# # print(f"Xup soft min: {Parameter.Xup.soft_min}, soft max: {Parameter.Xup.soft_max}")