import os
import datetime

def result_logger(root_folder: str, experiment_name: str, generation_number: int, genotypes: list, fitness_values: list, results_dicts: list):
    """
    Logs the results of an experiment in a text file.

    Parameters:
    - root_folder (str): The root directory where the experiment folder will be created.
    - experiment_name (str): The name of the experiment.
    - generation_number (int): The generation number of the experiment.
    - genotypes (list): List of genotypes.
    - fitness_values (list): Corresponding fitness values for each genotype.
    - results_dicts (list): A list of dictionaries with detailed results for each genotype.
    """
    # Create experiment directory if it doesn't exist
    experiment_path = os.path.join(root_folder, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Create a file name with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Results_Gen_{generation_number}_{timestamp}.txt"
    file_path = os.path.join(experiment_path, filename)

    # Write results to the file
    with open(file_path, 'w') as file:
        file.write(f"Experiment: {experiment_name}\n")
        file.write(f"Generation: {generation_number}\n")
        file.write(f"Timestamp: {timestamp}\n\n")
        
        file.write("Genotypes and their Fitness Values:\n")
        for genotype, fitness in zip(genotypes, fitness_values):
            file.write(f"{genotype} - Fitness: {fitness}\n")
        
        file.write("\nDetailed Results for Each Genotype:\n")
        for genotype, result_dict in zip(genotypes, results_dicts):
            file.write(f"Genotype {genotype}:\n")
            for angle, values in result_dict.items():
                file.write(f"  Angle {angle}: cl={values[0]}, cd={values[1]}, cm={values[2]}\n")

    print(f"Results saved to {file_path}")

# Example usage:
# result_logger('path/to/root_folder', 'Experiment1', 1, [[genotype1], [genotype2]], [fitness1, fitness2], [{angle1: (cl1, cd1, cm1)}, {angle2: (cl2, cd2, cm2)}])

def read_experiment_results(file_path: str):
    """
    Reads the experiment results from a text file and extracts genotypes, fitness values, and detailed results.

    Parameters:
    - file_path (str): The path to the results file.

    Returns:
    - Tuple of three elements:
      1. List of genotypes
      2. Dictionary of genotype fitness values
      3. Dictionary of detailed results for each genotype
    """
    genotypes = []
    fitness_values = {}
    detailed_results = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        reading_genotypes = False
        reading_detailed_results = False
        current_genotype = None

        for line in lines:
            line = line.strip()

            # Check for sections in the file
            if "Genotypes and their Fitness Values:" in line:
                reading_genotypes = True
                reading_detailed_results = False
                continue
            elif "Detailed Results for Each Genotype:" in line:
                reading_genotypes = False
                reading_detailed_results = True
                continue

            # Process the appropriate section
            if reading_genotypes and '-' in line:
                genotype_str, fitness_str = line.split('- Fitness:')
                genotype = eval(genotype_str.strip())
                fitness = float(fitness_str.strip())
                genotypes.append(genotype)
                fitness_values[str(genotype)] = fitness
            elif reading_detailed_results and 'Genotype' in line:
                genotype_str = line.split('Genotype ')[1].split(':')[0].strip()
                current_genotype = eval(genotype_str)
                detailed_results[str(current_genotype)] = {}
            elif reading_detailed_results and 'Angle' in line:
                parts = line.split(':')
                angle = int(parts[0].split()[1])
                values_str = parts[1].split(',')
                cl = float(values_str[0].split('=')[1])
                cd = float(values_str[1].split('=')[1])
                cm = float(values_str[2].split('=')[1])
                detailed_results[str(current_genotype)][angle] = (cl, cd, cm)

    return genotypes, fitness_values, detailed_results

# # Example usage:
# genotypes, fitness_values, detailed_results = read_experiment_results("./RESULTS/Combination_Experiment_1/Results_Gen_100_20231225_150840.txt")
# print(genotypes)
# print(fitness_values)
# print(detailed_results)
# print(fitness_values.values())