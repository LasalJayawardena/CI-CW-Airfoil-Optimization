import tensorflow as tf
import joblib
import pickle



# Define Constants
DRAG_MODEL = tf.keras.models.load_model('best_model15.h5')
with open("best_scaler15.obj", "rb") as f:
    MIN_MAX_SCALER = pickle.load(f)

REYNOLDS_NUMBER = 100000
MACH_NUMBER = 0.1
ATTACK_ON_ANGLE = -10
# attack on angle from -10 to 10

def test_model_inference_func() -> float: 
    
    yCoorUpper = [2.07692721e-02,  3.93104182e-02,  5.46992442e-02,
        6.60399430e-02,  7.26857201e-02,  7.44701301e-02,  7.17682176e-02,
        6.53878200e-02,  5.63418071e-02,  4.57425800e-02,  3.45854691e-02,
        2.37791669e-02,  1.41758894e-02,  6.57918271e-03,  1.68970028e-03]
    yCoorLower = [-2.07692721e-02, -3.93104182e-02, -5.46992442e-02, -6.60399430e-02,
       -7.26857201e-02, -7.44701301e-02, -7.17682176e-02, -6.53878100e-02,
       -5.63418071e-02, -4.57425800e-02, -3.45854691e-02, -2.37791669e-02,
       -1.41758894e-02, -6.57918271e-03, -1.68970028e-03]

    features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
    
    features = MIN_MAX_SCALER.transform(features)
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)

    # Predict the drag coefficient
    pred = DRAG_MODEL.predict(features)[0]
    cl, cd, cm = pred[0], pred[1], pred[2]

    return cl


print(test_model_inference_func())