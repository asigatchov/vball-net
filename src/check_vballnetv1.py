import tensorflow as tf
import numpy as np
import os
import logging
from model.VballNetV1 import VballNetV1

# Constants
IMG_HEIGHT = 288
IMG_WIDTH = 512
SEQ = 9  # Number of input/output frames for grayscale mode
BATCH_SIZE = 4
MODEL_DIR = "models"  # Directory for saving the model
MODEL_NAME = "VballNetV1_seq9_grayscale"


def setup_logging(debug=False):
    """
    Configure logging with specified level based on debug flag.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


def compile_and_test_model():
    """
    Compile VballNetV1 model, test its output with random grayscale input data,
    and save the model to disk.
    """
    logger = setup_logging(debug=True)
    logger.info("Starting model compilation, testing, and saving")

    # Create model save directory
    model_save_dir = os.path.join(MODEL_DIR, MODEL_NAME)
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Model save directory created: {model_save_dir}")

    # Initialize model
    try:
        model = VballNetV1(
            height=IMG_HEIGHT,
            width=IMG_WIDTH,
            in_dim=SEQ,  # 9 grayscale frames
            out_dim=SEQ,  # 9 output heatmaps
            fusion_layer_type="TypeA",
        )
        logger.info("Model VballNetV1 created successfully")
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise

    # Compile model
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        logger.info("Model compiled with Adam optimizer, MSE loss, and MAE metric")
    except Exception as e:
        logger.error(f"Failed to compile model: {str(e)}")
        raise

    # Print model summary
    model.summary(print_fn=lambda x: logger.info(x))

    # Create random input data
    try:
        input_data = np.random.rand(BATCH_SIZE, SEQ, IMG_HEIGHT, IMG_WIDTH).astype(
            np.float32
        )
        logger.info(f"Generated random input data with shape: {input_data.shape}")
    except Exception as e:
        logger.error(f"Failed to generate input data: {str(e)}")
        raise

    # Test model output
    try:
        predictions = model.predict(input_data, batch_size=BATCH_SIZE)
        logger.info(f"Model prediction shape: {predictions.shape}")
        expected_shape = (BATCH_SIZE, SEQ, IMG_HEIGHT, IMG_WIDTH)
        if predictions.shape == expected_shape:
            logger.info(f"Output shape is correct: {expected_shape}")
        else:
            logger.error(
                f"Unexpected output shape: {predictions.shape}, expected: {expected_shape}"
            )
            raise ValueError(
                f"Output shape mismatch: got {predictions.shape}, expected {expected_shape}"
            )

        # Verify prediction values are in [0, 1] due to sigmoid activation
        if np.all((predictions >= 0) & (predictions <= 1)):
            logger.info(
                "Prediction values are within [0, 1] as expected for sigmoid activation"
            )
        else:
            logger.error("Prediction values are outside [0, 1]")
            raise ValueError("Prediction values are outside expected range [0, 1]")
    except Exception as e:
        logger.error(f"Failed to predict with model: {str(e)}")
        raise

    # Save model to disk
    try:
        model_save_path = os.path.join(model_save_dir, f"{MODEL_NAME}_initial.keras")
        model.save(model_save_path)
        logger.info(f"Model saved successfully to: {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

    logger.info("Model compilation, testing, and saving completed successfully")


if __name__ == "__main__":
    compile_and_test_model()
