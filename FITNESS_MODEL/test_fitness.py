import tensorflow as tf
import joblib

min_max_scaler = joblib.load('./min_max_scaler.pkl')

# Define Constants
DRAG_MODEL = tf.keras.models.load_model('./model.h5')
REYNOLDS_NUMBER = 100000
MACH_NUMBER = 0.1
ATTACK_ON_ANGLE = -10
# attack on angle from -10 to 10

def test_model_inference_func() -> float: 
    
    yCoorUpper = [0.016659, 0.024604, 0.021796, 0.012770, 0.003794]
    yCoorLower = [-0.016659, -0.024604, -0.021796, -0.012770, -0.003794]

    features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
    
    features = min_max_scaler.transform(features)
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)

    # Predict the drag coefficient
    pred = DRAG_MODEL.predict(features)[0]
    cl, cd, cm = pred[0], pred[1], pred[2]

    return cl


print(test_model_inference_func())