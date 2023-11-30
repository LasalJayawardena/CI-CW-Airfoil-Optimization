import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# @param model_path : name of the model to be saved "model.H5"
# @param dataset_path : path of the dataset
# @param target_columns : list of dependent parameters
# @param feature_columns : list of independent parameters
# @param description : para about the model
def train_and_save_model(model_path, dataset_path, target_columns, feature_columns, description):
    df = pd.read_csv(dataset_path)

    features = df[feature_columns]
    targets = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(target_columns))  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

    # Save the model to a specific folder
    model.save(model_path)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error for each output
    mse_list = [mean_squared_error(y_test[col], y_pred[:, idx]) for idx, col in enumerate(target_columns)]

    log_file = open('/Users/ak/PycharmProjects/PARSEC3/LOGS/models_log.txt', 'a+')
    log_file.write(f"Model Name: {model_path}\n")
    log_file.write(f"Dataset Path: {dataset_path}\n")
    log_file.write(f"Size of Dataset: {len(df)}\n")
    log_file.write(f"Train Size: {len(X_train)}\n")
    log_file.write(f"Test Size: {len(X_test)}\n")
    log_file.write(f"Columns of Dataset: {', '.join(df.columns)}\n")
    log_file.write(f"Target Columns: {', '.join(target_columns)}\n")
    log_file.write(f"Feature Columns: {', '.join(feature_columns)}\n")
    log_file.write(f"MSE for each target: {', '.join(map(str, mse_list))}\n")
    log_file.write(f"Description: {description}\n\n")


if __name__ == "__main__":
    train_and_save_model(
    '../model1_ANN.h5',
    '../RESOURCES/fitness/NACA6408.csv',
    ['Cl', 'Cd', 'Cm'],
    ['yU_1', 'yU_2','yU_3', 'yU_4','yU_5', 'yU_6','yU_7', 'yU_8','yU_9', 'yU_10','yL_1', 'yL_2','yL_3', 'yL_4','yL_5', 'yL_6','yL_7', 'yL_8','yL_9', 'yL_10', 'ReynoldsNumber','MachNumber','alpha'],
    'This is Model 1.'
)


