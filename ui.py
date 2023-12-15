from PySide6.QtWidgets import QMainWindow, QLabel, QComboBox, QCheckBox, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, \
    QGridLayout, QPushButton
from PySide6.QtCharts import QLineSeries, QChart, QChartView,QXYSeries
from PySide6.QtGui import QPainter
from PySide6.QtCore import QPointF, Slot
import pyqtgraph as pg

from random import randint
from tqdm import tqdm

# AIRFOIL
import airfoil_Builder
PointConfig = QXYSeries.PointConfiguration

# OPTIMIZATION
import optimization

class ChartWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parsec Airfoil")

        # lakindu - code starts here
        ######################

        rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE = 0.0147,0.3015,0.0599,-0.4360,0.2996,-0.06,0.4406,0,0,0,14.67
        airfoil = airfoil_Builder.Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE,betaTE)
        airfoil.build()
        xcoor = airfoil.XCoordinates
        yCoorUpper = airfoil.YCoordinatesUpper
        yCoorLower = airfoil.YCoordinatesLower

        self.graphWidget = pg.PlotWidget()
        self.x = list(range(100))  # 100 time points
        self.y = [randint(0, 100) for _ in range(100)]  # 100 data points
        self.graphWidget.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)
        self.timer = pg.QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        # Labels
        fitness_parameters_label = QLabel("fitness parameters")
        reynold_label = QLabel("Reynold Number: ")
        mach_label = QLabel("Mach Number: ")
        label_visibility_label = QLabel("Label Visibility: ")

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
        self._reynold_combobox = QComboBox()
        self._mach_combobox = QComboBox()
        self.optimization_button = QPushButton("Optimize")
        #self.optimization_button.move(100, 100)

        self._label_visibility_checkbox = QCheckBox()

        self._rLE_lineedit = QLineEdit(str(rLE))
        self._Xup_lineedit = QLineEdit(str(Xup))
        self._Yup_lineedit = QLineEdit(str(Yup))
        self._YXXup_lineedit = QLineEdit(str(YXXup))
        self._Xlow_lineedit = QLineEdit(str(Xlow))
        self._Ylow_lineedit = QLineEdit(str(Ylow))
        self._YXXlow_lineedit = QLineEdit(str(YXXlow))
        self._yTE_lineedit = QLineEdit(str(yTE))
        self._deltaYTE_lineedit = QLineEdit(str(deltaYTE))
        self._alphaTE_lineedit = QLineEdit(str(alphaTE))
        self._betaTE_lineedit = QLineEdit(str(betaTE))

        self._chart = QChart()

        self._series = QLineSeries()
        self._series.setName("Airfoil")
        self._series.setPointsVisible(True)
        self._series.append([QPointF(0,0),
                             QPointF(xcoor[0],yCoorUpper[0]),
                             QPointF(xcoor[1],yCoorUpper[1]),
                             QPointF(xcoor[2],yCoorUpper[2]),
                             QPointF(xcoor[3],yCoorUpper[3]),
                             QPointF(xcoor[4],yCoorUpper[4]),
                             QPointF(xcoor[5],yCoorUpper[5]),
                             QPointF(xcoor[6],yCoorUpper[6]),
                             QPointF(xcoor[7],yCoorUpper[7]),
                             QPointF(xcoor[8],yCoorUpper[8]),
                             QPointF(xcoor[9],yCoorUpper[9]),
                             QPointF(xcoor[10], yCoorUpper[10]),
                             QPointF(xcoor[11], yCoorUpper[11]),
                             QPointF(xcoor[12], yCoorUpper[12]),
                             QPointF(xcoor[13], yCoorUpper[13]),
                             QPointF(xcoor[14], yCoorUpper[14]),

                             QPointF(xcoor[0], yCoorLower[0]),
                             QPointF(xcoor[1], yCoorLower[1]),
                             QPointF(xcoor[2], yCoorLower[2]),
                             QPointF(xcoor[3], yCoorLower[3]),
                             QPointF(xcoor[4], yCoorLower[4]),
                             QPointF(xcoor[5], yCoorLower[5]),
                             QPointF(xcoor[6], yCoorLower[6]),
                             QPointF(xcoor[7], yCoorLower[7]),
                             QPointF(xcoor[8], yCoorLower[8]),
                             QPointF(xcoor[9], yCoorLower[9]),
                             QPointF(xcoor[10], yCoorLower[10]),
                             QPointF(xcoor[11], yCoorLower[11]),
                             QPointF(xcoor[12], yCoorLower[12]),
                             QPointF(xcoor[13], yCoorLower[13]),
                             QPointF(xcoor[14], yCoorLower[14])
                             ])

        self._chart.addSeries(self._series)
        self._chart.createDefaultAxes()

        for re in [100000, 200000, 300000, 400000, 500000]:
            self._reynold_combobox.addItem(str(re), re)

        for mach in [0.1,0.2,0.3]:
            self._mach_combobox.addItem(str(mach), mach)

        # connections
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

        self.optimization_button.clicked.connect(self._run_optimizer)
        chart_view = QChartView(self._chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        control_widget = QWidget(self)
        control_layout = QGridLayout(control_widget)
        control_layout.setColumnStretch(1, 1)

        control_layout.addWidget(fitness_parameters_label, 0, 0)
        control_layout.addWidget(reynold_label, 1, 0)
        control_layout.addWidget(self._reynold_combobox, 1, 1)

        control_layout.addWidget(mach_label, 2, 0)
        control_layout.addWidget(self._mach_combobox, 2, 1)

        genotype_widget = QWidget(control_widget)
        control_layout.addWidget(genotype_widget, 4, 0)
        genotype_layout = QGridLayout(genotype_widget)
        # genotype_layout.setColumnStretch(1, 1)

        genotype_layout.addWidget(rLE_label, 1, 0)
        genotype_layout.addWidget(self._rLE_lineedit, 1, 1)
        genotype_layout.addWidget(Xup_label, 2, 0)
        genotype_layout.addWidget(self._Xup_lineedit, 2, 1)
        genotype_layout.addWidget(Yup_label, 3, 0)
        genotype_layout.addWidget(self._Yup_lineedit, 3, 1)
        genotype_layout.addWidget(YXXup_label, 4, 0)
        genotype_layout.addWidget(self._YXXup_lineedit, 4, 1)
        genotype_layout.addWidget(Xlow_label, 5, 0)
        genotype_layout.addWidget(self._Xlow_lineedit, 5, 1)
        genotype_layout.addWidget(Ylow_label, 6, 0)
        genotype_layout.addWidget(self._Ylow_lineedit, 6, 1)
        genotype_layout.addWidget(YXXlow_label, 7, 0)
        genotype_layout.addWidget(self._YXXlow_lineedit, 7, 1)
        genotype_layout.addWidget(yTE_label, 8, 0)
        genotype_layout.addWidget(self._yTE_lineedit, 8, 1)
        genotype_layout.addWidget(deltaYTE_label, 9, 0)
        genotype_layout.addWidget(self._deltaYTE_lineedit, 9, 1)
        genotype_layout.addWidget(alphaTE_label, 10, 0)
        genotype_layout.addWidget(self._alphaTE_lineedit, 10, 1)
        genotype_layout.addWidget(betaTE_label, 11, 0)
        genotype_layout.addWidget(self._betaTE_lineedit, 11, 1)

        genotype_layout.addWidget(self.optimization_button,12,1)

        # label_checkbox = self._label_checkbox
        # label_checkbox.clicked.connect(self._set_label_visibility)
        # control_layout.addWidget(selected_point_index_label, 0, 0)
        # control_layout.addWidget(self._selected_point_index_lineedit, 0, 1)

        control_layout.addWidget(label_visibility_label, 3, 0)
        control_layout.addWidget(self._label_visibility_checkbox, 3, 1, 1, 2)

        main_widget = QWidget(self)
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(chart_view)
        main_layout.addWidget(control_widget)
        control_layout.addWidget(self.graphWidget)
        main_layout.setStretch(0, 1)
        self.setCentralWidget(main_widget)

    @Slot(int)
    def _set_reynold(self, index: int):
        print(index)

    @Slot(int)
    def _set_mach(self, index: int):
            print("index")

    @Slot(bool)
    def _set_label_visibility(self, checked: bool):
        print("")

    @Slot()
    def _set_custom_label(self):
        print("...value changed")

    @Slot()
    def _run_optimizer(self):

        # Run optimization_strategy_one for 100 Generations

        # Generate initial population
        initial_population = optimization.generate_population(10)

        # Run optimization_strategy_one for 100 generations
        current_generation = initial_population
        for i in tqdm(list(range(2))):
            current_generation = optimization.optimization_strategy_one(current_generation, 10)
            optimization.log_genration_results(current_generation, i + 1)

        # Evaluate fitness of final generation
        fitness_scores = optimization.lift_coef_based_fitness_function_multi(current_generation)
        return current_generation, fitness_scores

    def update_plot_data(self):

        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first
        self.y.append(randint(0, 100))  # Add a new random value.

        self.data_line.setData(self.x, self.y)  # Update the data.


