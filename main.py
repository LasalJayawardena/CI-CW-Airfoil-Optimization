# UI
import sys
from PySide6.QtWidgets import QApplication
from ui import ChartWindow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = QApplication(sys.argv)
    main_window = ChartWindow()
    main_window.resize(1440, 1480)
    main_window.show()
    sys.exit(a.exec())

# # Now you can use train_and_save_model in your main.py
# train_and_save_model(
#     'model1_ANN.keras',
#     '/Users/ak/PycharmProjects/PARSEC_OPT/RESOURCES/fitness/NACA6408.csv',
#     ['Cl', 'Cd', 'Cm'],
#     ['yU_1', 'yU_2','yU_3', 'yU_4','yU_5', 'yU_6','yU_7', 'yU_8','yU_9', 'yU_10','yL_1', 'yL_2','yL_3', 'yL_4','yL_5', 'yL_6','yL_7', 'yL_8','yL_9', 'yL_10', 'ReynoldsNumber','MachNumber','alpha'],
#     'This is Model 1.'
# )
