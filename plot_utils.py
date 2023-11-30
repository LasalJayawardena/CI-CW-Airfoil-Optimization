import matplotlib.pyplot as plt
import airfoil_Builder  # Assuming airfoil_Builder module is available

def plot_airfoil(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE):
    # Create an Airfoil_Builder object with the given parameters
    airfoil = airfoil_Builder.Airfoil_Builder(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
    airfoil.build()
    
    # Get coordinates
    xcoor = airfoil.XCoordinates
    yCoorUpper = airfoil.YCoordinatesUpper
    yCoorLower = airfoil.YCoordinatesLower

    # Create a scatter plot of the points
    plt.figure(figsize=(8, 6))  # Set figure size (optional)
    plt.scatter(xcoor, yCoorUpper, label="Upper Coordinates", color="blue", marker="o")
    plt.scatter(xcoor, yCoorLower, label="Lower Coordinates", color="red", marker="o")

    # Add labels and title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Airfoil Coordinates")

    # Add a legend
    # plt.legend()

    # Show the plot
    plt.grid(True)  # Show grid (optional)
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal (optional)
    plt.show()


if __name__ == '__main__':
    # Example parameters (replace these with your desired values)
    rLE = 0.0147
    Xup = 0.3015
    Yup = 0.0599
    YXXup = -0.4360
    Xlow = 0.2996
    Ylow = -0.06
    YXXlow = 0.4406
    yTE = 0
    deltaYTE = 0
    alphaTE = 0
    betaTE = 14.67

    # Call the function to plot the airfoil coordinates
    plot_airfoil(rLE, Xup, Yup, YXXup, Xlow, Ylow, YXXlow, yTE, deltaYTE, alphaTE, betaTE)
