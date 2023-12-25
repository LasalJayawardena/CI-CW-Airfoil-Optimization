from PySide6.QtWidgets import QMainWindow, QLabel, QComboBox, QCheckBox, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, \
    QGridLayout, QPushButton, QFileDialog, QInputDialog
from PySide6.QtCharts import QLineSeries, QChart, QChartView,QXYSeries
from PySide6.QtGui import QPainter
from PySide6.QtCore import QPointF, Slot, QObject, Signal, QCoreApplication
import pyqtgraph as pg

import re as regex
import os
from random import randint

from tqdm import tqdm
import shutil

# AIRFOIL
import airfoil_Builder
PointConfig = QXYSeries.PointConfiguration

# OPTIMIZATION
import optimization

class ChartWindow(QMainWindow):
    updateGeneration = Signal()

    # initial airfoil

    # xcoor = airfoil.XCoordinates
    # yCoorUpper = airfoil.YCoordinatesUpper
    # yCoorLower = airfoil.YCoordinatesLower

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parsec Airfoil")

        # initialize parameters
        self.rLE, self.Xup, self.Yup, self.YXXup, self.Xlow, self.Ylow, self.YXXlow, self.yTE, self.deltaYTE, self.alphaTE, self.betaTE = 0.0147, 0.3015, 0.0599, -0.4360, 0.2996, -0.06, 0.4406, 0, 0, 0, 14.67
        airfoil = airfoil_Builder.Airfoil_Builder(self.rLE, self.Xup, self.Yup, self.YXXup, self.Xlow, self.Ylow, self.YXXlow, self.yTE, self.deltaYTE, self.alphaTE, self.betaTE, 100)
        airfoil.build()

        self.xcoor = airfoil.XCoordinates
        self.yCoorUpper = airfoil.YCoordinatesUpper
        self.yCoorLower = airfoil.YCoordinatesLower

        self.gen_number = [0]
        self.highest_fitness = [0]

        self.fitness = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        self.aoa = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]

        # initialize plot
        self._fitness_graph_Widget = pg.PlotWidget()
        self._fitness_graph_Widget.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line = self._fitness_graph_Widget.plot(self.gen_number, self.highest_fitness, pen=pen)
        self._fitness_graph_Widget.setLabel('bottom', 'Generation Number')
        self._fitness_graph_Widget.setLabel('left', 'Highest Fitness')

        self._expanded_fitness_graph_Widget = pg.PlotWidget()
        self._expanded_fitness_graph_Widget.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0))
        self.expanded_fitness_data_line1 = self._expanded_fitness_graph_Widget.plot(self.aoa, self.fitness[0], pen=pen)
        self._expanded_fitness_graph_Widget.setLabel('bottom', 'Angle of Attack (AOA)')
        self._expanded_fitness_graph_Widget.setLabel('left', 'Fitness Value')

        self._airfoil_shape_graph_Widget = pg.PlotWidget()
        self._airfoil_shape_graph_Widget.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0))
        self.airfoil_shape_graph_line_yUpper = self._airfoil_shape_graph_Widget.plot(self.xcoor, self.yCoorUpper, pen=pen)
        self.airfoil_shape_graph_line_yLower = self._airfoil_shape_graph_Widget.plot(self.xcoor, self.yCoorLower, pen=pen)
        self._airfoil_shape_graph_Widget.setLabel('bottom', 'X')
        self._airfoil_shape_graph_Widget.setLabel('left', 'Y')

        # Labels
        advance_control_label = QLabel("advanced controls")
        crossover_label = QLabel("Crossover : ")
        mutation_rate_label = QLabel("Mutation rate : ")
        mutation_label = QLabel("Mutation : ")
        selection_label = QLabel("Selection : ")
        survivor_selection_label = QLabel("Survivor selection : ")

        fitness_parameters_label = QLabel("optimizer controls")
        reynold_label = QLabel("Reynold Number: ")
        mach_label = QLabel("Mach Number: ")

        rLE_label = QLabel("rLE: ")
        Xup_label = QLabel("Xup: ")
        Yup_label = QLabel("Yup: ")
        YXXup_label = QLabel("YXXup: ")
        Xlow_label = QLabel("Xlow: ")
        Ylow_label = QLabel("Ylow: ")
        YXXlow_label = QLabel("YXXlow: ")
        yTE_label = QLabel("yTE: ")
        deltaYTE_label = QLabel("deltaYTE: ")
        alphaTE_label = QLabel("alphaTE: ")
        betaTE_label = QLabel("betaTE: ")

        # properties
        self._crossover_combobox = QComboBox()
        self._mutation_rate_combobox = QComboBox()
        self._mutation_combobox = QComboBox()
        self._selection_combobox = QComboBox()
        self._survivor_selection_combobox = QComboBox()

        self._reynold_combobox = QComboBox()
        self._mach_combobox = QComboBox()
        self._optimization_button = QPushButton("Optimize")
        self._export_button = QPushButton("Export")

        self._rLE_lineedit = QLineEdit(str(self.rLE))
        self._Xup_lineedit = QLineEdit(str(self.Xup))
        self._Yup_lineedit = QLineEdit(str(self.Yup))
        self._YXXup_lineedit = QLineEdit(str(self.YXXup))
        self._Xlow_lineedit = QLineEdit(str(self.Xlow))
        self._Ylow_lineedit = QLineEdit(str(self.Ylow))
        self._YXXlow_lineedit = QLineEdit(str(self.YXXlow))
        self._yTE_lineedit = QLineEdit(str(self.yTE))
        self._deltaYTE_lineedit = QLineEdit(str(self.deltaYTE))
        self._alphaTE_lineedit = QLineEdit(str(self.alphaTE))
        self._betaTE_lineedit = QLineEdit(str(self.betaTE))


        for co in [ "Single Point Crossover", "Two Point Crossover",  "Uniform Crossover", "Blend Crossover", "Arithmetic Crossover"]:
            self._crossover_combobox.addItem(str(co), co)

        for mr in [0.01, 0.05, 0.10]:
            self._mutation_rate_combobox.addItem(str(mr), mr)

        for m in ["Creep Mutation", "Gaussian Mutation",  "Uniform Mutation"]:
            self._mutation_combobox.addItem(str(m), m)

        for s in ["Roulette Wheel Selection", "Binary Tournament Selection", "Ternary Tournament Selection", "Rank Selection", "Stochastic Universal Selection","Elitism Selection"]:
            self._selection_combobox.addItem(str(s), s)

        for ss in ["Truncation Survivor Selection", "Steady State Selection"]:
            self._survivor_selection_combobox.addItem(str(ss), ss)

        for re in [100000, 200000, 300000, 400000, 500000]:
            self._reynold_combobox.addItem(str(re), re)

        for mach in [0.1,0.2,0.3]:
            self._mach_combobox.addItem(str(mach), mach)

        # connections
        self._crossover_combobox.activated.connect(self._set_crossover)
        self._mutation_rate_combobox.activated.connect(self._set_mutation_rate)
        self._mutation_combobox.activated.connect(self._set_mutation)
        self._selection_combobox.activated.connect(self._set_selection)
        self._survivor_selection_combobox.activated.connect(self._set_survivor_selection)

        self._reynold_combobox.activated.connect(self._set_reynold)
        self._mach_combobox.activated.connect(self._set_mach)

        self._rLE_lineedit.editingFinished.connect(self._set_custom_label)
        self._Xup_lineedit.editingFinished.connect(self._set_custom_label)
        self._Yup_lineedit.editingFinished.connect(self._set_custom_label)
        self._YXXup_lineedit.editingFinished.connect(self._set_custom_label)
        self._Xlow_lineedit.editingFinished.connect(self._set_custom_label)
        self._Ylow_lineedit.editingFinished.connect(self._set_custom_label)
        self._YXXlow_lineedit.editingFinished.connect(self._set_custom_label)
        self._yTE_lineedit.editingFinished.connect(self._set_custom_label)
        self._deltaYTE_lineedit.editingFinished.connect(self._set_custom_label)
        self._alphaTE_lineedit.editingFinished.connect(self._set_custom_label)
        self._betaTE_lineedit.editingFinished.connect(self._set_custom_label)

        self._optimization_button.clicked.connect(self._run_optimizer)
        self._export_button.clicked.connect(self._export_results)

        self.updateGeneration.connect(self._update_generation)

        # layout
        airfoil_widget = QWidget(self)
        airfoil_layout = QGridLayout(airfoil_widget)
        airfoil_layout.addWidget(self._airfoil_shape_graph_Widget)

        control_widget = QWidget(self)
        control_layout = QGridLayout(control_widget)
        control_layout.setColumnStretch(0, 1)

        control_layout.addWidget(rLE_label, 1, 0)
        control_layout.addWidget(self._rLE_lineedit, 1, 1)
        control_layout.addWidget(Xup_label, 2, 0)
        control_layout.addWidget(self._Xup_lineedit, 2, 1)
        control_layout.addWidget(Yup_label, 3, 0)
        control_layout.addWidget(self._Yup_lineedit, 3, 1)
        control_layout.addWidget(YXXup_label, 4, 0)
        control_layout.addWidget(self._YXXup_lineedit, 4, 1)
        control_layout.addWidget(Xlow_label, 5, 0)
        control_layout.addWidget(self._Xlow_lineedit, 5, 1)
        control_layout.addWidget(Ylow_label, 6, 0)
        control_layout.addWidget(self._Ylow_lineedit, 6, 1)
        control_layout.addWidget(YXXlow_label, 7, 0)
        control_layout.addWidget(self._YXXlow_lineedit, 7, 1)
        control_layout.addWidget(yTE_label, 8, 0)
        control_layout.addWidget(self._yTE_lineedit, 8, 1)
        control_layout.addWidget(deltaYTE_label, 9, 0)
        control_layout.addWidget(self._deltaYTE_lineedit, 9, 1)
        control_layout.addWidget(alphaTE_label, 10, 0)
        control_layout.addWidget(self._alphaTE_lineedit, 10, 1)
        control_layout.addWidget(betaTE_label, 11, 0)
        control_layout.addWidget(self._betaTE_lineedit, 11, 1)

        optimize_control_widget = QWidget(self)
        optimize_control_layout = QGridLayout(optimize_control_widget)
        optimize_control_layout.setColumnStretch(0, 1)

        optimize_control_layout.addWidget(advance_control_label, 0, 0)
        optimize_control_layout.addWidget(crossover_label, 1, 0)
        optimize_control_layout.addWidget(self._crossover_combobox, 1, 1)
        optimize_control_layout.addWidget(mutation_rate_label, 2, 0)
        optimize_control_layout.addWidget(self._mutation_rate_combobox, 2, 1)
        optimize_control_layout.addWidget(mutation_label, 3, 0)
        optimize_control_layout.addWidget(self._mutation_combobox, 3, 1)
        optimize_control_layout.addWidget(selection_label, 4, 0)
        optimize_control_layout.addWidget(self._selection_combobox, 4, 1)
        optimize_control_layout.addWidget(survivor_selection_label, 5, 0)
        optimize_control_layout.addWidget(self._survivor_selection_combobox,5, 1)

        optimize_control_layout.addWidget(fitness_parameters_label, 6, 0)
        optimize_control_layout.addWidget(reynold_label, 7, 0)
        optimize_control_layout.addWidget(self._reynold_combobox, 7, 1)
        optimize_control_layout.addWidget(mach_label, 8, 0)
        optimize_control_layout.addWidget(self._mach_combobox, 8, 1)

        optimize_control_layout.addWidget(self._optimization_button,9,0)
        optimize_control_layout.addWidget(self._export_button,9,1)

        parameter_widget = QWidget(self)
        parameter_layout = QGridLayout(parameter_widget)
        parameter_layout.addWidget(control_widget,0,0)
        parameter_layout.addWidget(optimize_control_widget,0,1)

        airfoil_layout.addWidget(parameter_widget,1,0)

        charts_widget = QWidget(self)
        charts_layout = QGridLayout(charts_widget)
        charts_layout.setColumnStretch(0, 1)

        charts_layout.addWidget(self._fitness_graph_Widget,0,0)
        charts_layout.addWidget(self._expanded_fitness_graph_Widget,1,0)

        main_widget = QWidget(self)
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(airfoil_widget)
        main_layout.addWidget(charts_widget)
        main_layout.setStretch(0, 1)
        self.setCentralWidget(main_widget)


    @Slot(int)
    def _set_crossover(self, index: int):
        crossover = self._crossover_combobox.itemData(index)
        print(crossover)

    @Slot(int)
    def _set_mutation_rate(self, index: int):
        mutation_rate = self._mutation_rate_combobox.itemData(index)
        print(mutation_rate)

    @Slot(int)
    def _set_mutation(self, index: int):
        mutation = self._mutation_combobox.itemData(index)
        print(mutation)

    @Slot(int)
    def _set_selection(self, index: int):
        selection = self._selection_combobox.itemData(index)
        print(selection)

    @Slot(int)
    def _set_survivor_selection(self, index: int):
        survivor_selection = self._survivor_selection_combobox.itemData(index)
        print(survivor_selection)

    @Slot(int)
    def _set_reynold(self, index: int):
        print("optimizer has to be wrapped around a class to add the reynold feature")
        reynold = self._reynold_combobox.itemData(index)
        print(f"Reynold Number set to: {reynold}")
        with open('./RESULTS/Reynold_and_Mach_Inputs/REYNOLD.txt', 'w') as reynold_file:
            reynold_file.write(f'{reynold}')

    @Slot(int)
    def _set_mach(self, index: int):
            print("optimizer has to be wrapped around a class to add the mach feature")
            mach = self._mach_combobox.itemData(index)
            print(f"Mach Number set to: {mach}")
            with open('./RESULTS/Reynold_and_Mach_Inputs/MACH.txt', 'w') as mach_file:
                mach_file.write(f'{mach}')

    @Slot()
    def _set_custom_label(self):
        print("These signals will not be used.")

    def _export_results(self):
        file_dialog = QFileDialog()
        export_path, _ = file_dialog.getSaveFileName(self, 'Export File', '', 'All Files (*)')
        if export_path:
            source_path = './RESULTS/CurrentOptimizationCycle/optimization_cycle.txt'  # Replace with the actual path
            shutil.copy(source_path, export_path)

    @Slot()
    def _run_optimizer(self):
        # clear data
        self.fitness = []
        self.highest_fitness = []
        self.gen_number = []

        # clear plots
        self._expanded_fitness_graph_Widget.clear()
        self._fitness_graph_Widget.clear()

        # Run optimization_strategy_one for 100 Generations
        # Generate initial population
        # make separate folder and open a file, inthe below fic add the current fie data to this file
        # Specify the path of the new folder
        folder_path = "./RESULTS/CurrentOptimizationCycle"

        # Create the new folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        initial_population = optimization.generate_population(10)
        current_generation = initial_population
        optimization.log_genration_results(current_generation, 0)
        self.updateGeneration.emit()
        for i in tqdm(list(range(50))):
            current_generation = optimization.optimization_strategy_one(current_generation, 10)
            optimization.log_genration_results(current_generation, i + 1)
            self.updateGeneration.emit()

        # Evaluate fitness of final generation
        fitness_scores = optimization.lift_coef_based_fitness_function_multi(current_generation)
        #return current_generation, fitness_scores

    @Slot()
    def _update_generation(self):
        #self._airfoil_shape_graph_Widget.clear()

        # Specify the file name and path within the new folder, This is for the export part
        optimization_cycle_file_name = "optimization_cycle.txt"
        optimization_cycle_file_path = os.path.join('./RESULTS/CurrentOptimizationCycle/', optimization_cycle_file_name)

        folder_path = "./RESULTS/Experiment1"
        files = os.listdir(folder_path)
        text_files = [file for file in files if file.endswith(".txt")]

        max_timestamp_file = ""
        if text_files:
            max_timestamp_file = max(text_files, key=self.extract_timestamp)

        full_path = os.path.join(folder_path, max_timestamp_file)

        with open(full_path, 'r') as file:
            content = file.read()
            # Extract generation number
            generation_match = regex.search(r"Generation: (\d+)", content)
            generation_number = int(generation_match.group(1)) if generation_match else None

            # Extract the first array in Genotypes and their Fitness Values
            genotype_match = regex.search(r"\[([^\]]+)] - Fitness:", content)
            genotype_array = [float(x) for x in genotype_match.group(1).split(",")] if genotype_match else None

        rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = \
            genotype_array[0], genotype_array[1], genotype_array[2], genotype_array[3], genotype_array[4], \
                genotype_array[5], \
                genotype_array[6], genotype_array[7], genotype_array[8], genotype_array[9], genotype_array[10]

        fitness_pattern = regex.compile(r"Fitness: ([\d.]+)")
        fitness_match = fitness_pattern.search(content)
        if fitness_match:
            fitness_value = float(fitness_match.group(1))
        print('\n\nGen Number: '+str(generation_number) +' , Fitness Value: '+str(fitness_value)+'\n\n')

        ## Getting the list of cl_cd ratios
        lines_list = content.split('\n')
        genotype_details_set_index = lines_list.index('Detailed Results for Each Genotype:')

        # Lists to store x and y coordinates
        cl_cd_ratio_list = []

        # Regular expression to extract angle, cl, and cd values
        pattern = regex.compile(r"  Angle (-?\d+): cl=(-?\d+\.\d+), cd=(-?\d+\.\d+), cm=(-?\d+\.\d+)")

        # Iterate over each line in the input
        for input_text in lines_list[genotype_details_set_index+2:genotype_details_set_index+2+21]:
            # Match the pattern in the input text
            match = pattern.match(input_text)

            # If there is a match, extract values
            if match:
                angle = int(match.group(1))
                cl = float(match.group(2))
                cd = float(match.group(3))

                # Calculate the ratio of Cl to Cd
                if cd != 0:
                    cl_cd_ratio = cl / cd
                else:
                    cl_cd_ratio = 0

                # Append values to the lists
                cl_cd_ratio_list.append(cl_cd_ratio)

        print(cl_cd_ratio_list)

        # below one is for the export part
        # Create a new file in the specified path
        if generation_number == 1:
            with open(optimization_cycle_file_path, 'w') as file:
                # You can write content to the file if needed
                file.write(f'#######################################     Generation Number: {generation_number}     ##########################################\n{content}\n\n')
        else:
            with open(optimization_cycle_file_path, 'a') as file:
                # You can write content to the file if needed
                file.write(f'#######################################     Generation Number: {generation_number}     ##########################################\n{content}\n\n')

        airfoil = airfoil_Builder.Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE,
                                                  betaTE,100)
        airfoil.build()
        xcoor = airfoil.XCoordinates
        yCoorUpper = airfoil.YCoordinatesUpper
        yCoorLower = airfoil.YCoordinatesLower

        self._rLE_lineedit.setText(str(rLE))
        self._Xup_lineedit.setText(str(Xup))
        self._Yup_lineedit.setText(str(Yup))
        self._YXXup_lineedit.setText(str(YXXup))
        self._Xlow_lineedit.setText(str(Xlow))
        self._Ylow_lineedit.setText(str(Ylow))
        self._YXXlow_lineedit.setText(str(YXXlow))
        self._yTE_lineedit.setText(str(yTE))
        self._deltaYTE_lineedit.setText(str(deltaYTE))
        self._alphaTE_lineedit.setText(str(alphaTE))
        self._betaTE_lineedit.setText(str(betaTE))

        # self._series = QLineSeries()
        # self._series.append([QPointF(0, 0),
        #                      QPointF(xcoor[0], yCoorUpper[0]),
        #                      QPointF(xcoor[1], yCoorUpper[1]),
        #                      QPointF(xcoor[2], yCoorUpper[2]),
        #                      QPointF(xcoor[3], yCoorUpper[3]),
        #                      QPointF(xcoor[4], yCoorUpper[4]),
        #                      QPointF(xcoor[5],yCoorUpper[5]),
        #                      QPointF(xcoor[6],yCoorUpper[6]),
        #                      QPointF(xcoor[7],yCoorUpper[7]),
        #                      QPointF(xcoor[8],yCoorUpper[8]),
        #                      QPointF(xcoor[9],yCoorUpper[9]),
        #                      QPointF(xcoor[10], yCoorUpper[10]),
        #                      QPointF(xcoor[11], yCoorUpper[11]),
        #                      QPointF(xcoor[12], yCoorUpper[12]),
        #                      QPointF(xcoor[13], yCoorUpper[13]),
        #                      QPointF(xcoor[14], yCoorUpper[14]),
        #                      QPointF(1, 0),
        #
        #                      QPointF(xcoor[14], yCoorLower[14]),
        #                      QPointF(xcoor[13], yCoorLower[13]),
        #                      QPointF(xcoor[12], yCoorLower[12]),
        #                      QPointF(xcoor[11], yCoorLower[11]),
        #                      QPointF(xcoor[10], yCoorLower[10]),
        #                      QPointF(xcoor[9], yCoorLower[9]),
        #                      QPointF(xcoor[8], yCoorLower[8]),
        #                      QPointF(xcoor[7], yCoorLower[7]),
        #                      QPointF(xcoor[6], yCoorLower[6]),
        #                      QPointF(xcoor[5], yCoorLower[5]),
        #                      QPointF(xcoor[4], yCoorLower[4]),
        #                      QPointF(xcoor[3], yCoorLower[3]),
        #                      QPointF(xcoor[2], yCoorLower[2]),
        #                      QPointF(xcoor[1], yCoorLower[1]),
        #                      QPointF(xcoor[0], yCoorLower[0]),
        #                      QPointF(0, 0)])

        # update properties
        #gen_number = generation_number
        f = cl_cd_ratio_list
        self.fitness.append(f)

        # update airfoil chart.
        # self._chart.removeAllSeries()
        # self._chart.addSeries(self._series)

        # update the airfoil plot
        pen1 = pg.mkPen(color=(255, 0, 0))
        self.airfoil_data_line_yUpper = self._airfoil_shape_graph_Widget.plot(airfoil.XCoordinates, airfoil.YCoordinatesUpper, pen=pen1)
        self.airfoil_data_line_yLower = self._airfoil_shape_graph_Widget.plot(airfoil.XCoordinates, airfoil.YCoordinatesLower, pen=pen1)

        # update the highest fitness value got in each fitness value
        self.gen_number.append(generation_number)
        print(self.gen_number)
        self.highest_fitness.append(fitness_value)
        print(self.highest_fitness)
        pen2 = pg.mkPen(color=(255, 0, 0))
        self.data_line = self._fitness_graph_Widget.plot(self.gen_number, self.highest_fitness, pen=pen2)

        # cl,cd values respect to AOA for each generation.
        for i in range(len(self.fitness)):
            pen = pg.mkPen(color=(255, 0, 0))
            self.expanded_fitness_data_line2 = self._expanded_fitness_graph_Widget.plot( self.aoa,self.fitness[i] , pen=pen)

        # force to update the pending actions
        QCoreApplication.processEvents()
        print("a new generation started.")

    def extract_timestamp(self, file_name):
        time_stamp = file_name.split("_")[-1].split(".")[0]
        return int(time_stamp)



