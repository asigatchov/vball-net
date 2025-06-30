import argparse
import tensorflow as tf
import tf2onnx
import onnx
from model.VballNetV1 import MotionPromptLayer, FusionLayerTypeA
from utils import custom_loss

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert Keras model to ONNX with optional resolution change')
parser.add_argument('--model_path', type=str, required=True, help='Path to the Keras model file')
parser.add_argument('--height', type=int, default=288, help='New input height')
parser.add_argument('--width', type=int, default=512, help='New input width')
args = parser.parse_args()

# Define custom objects
custom_objects = {
    'MotionPromptLayer': MotionPromptLayer,
    'FusionLayerTypeA': FusionLayerTypeA,
    'custom_loss': custom_loss
}

# Load the Keras model
try:
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define input signature with new resolution
input_shape = (None, 9, args.height, args.width)
input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')]

# Convert the model to ONNX
try:
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13
    )
    print("Model converted to ONNX successfully")
except Exception as e:
    print(f"Error during ONNX conversion: {e}")
    exit(1)

# Save the ONNX model (replace .keras with .onnx in the output path)
onnx_model_path = args.model_path.replace('.keras', f'_h{args.height}_w{args.width}.onnx')
onnx.save_model(onnx_model, onnx_model_path)

# Verify the ONNX model
try:
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model is valid and saved as {onnx_model_path}")
except Exception as e:
    print(f"ONNX model verification failed: {e}")