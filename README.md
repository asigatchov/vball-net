# VballNet Repository

## Overview
VballNet is a specialized deep learning framework designed for volleyball tracking, built upon the foundation of TrackNetV4. This repository includes two primary models, **VballNetV1** and **VballNetFastV1**, optimized for efficient detection and training on consumer-grade hardware. By leveraging lightweight architectures and depthwise separable convolutions, VballNet achieves high performance with reduced computational requirements, making it suitable for deployment on resource-constrained devices such as OpenVINO GPUs.

Watch a demo of VballNet in action:

[![VballNet Demo](https://img.youtube.com/vi/72ksyX5v8uU/0.jpg)](https://www.youtube.com/watch?v=72ksyX5v8uU)

## Installation

To set up the VballNet repository, follow these steps to install the required dependencies and tools. This project uses `uv` for dependency management and supports conversion of models to ONNX format for efficient inference.

### Prerequisites
- **Python**: Version 3.11 .
- **Hardware**: A consumer-grade GPU is recommended for training and inference, though CPU is sufficient for lightweight models like VballNetFastV1.
- **Optional**: OpenVINO for optimized inference on compatible hardware.

### Step 1: Install `uv`
`uv` is a fast Python package and dependency manager. Install it by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install Dependencies
Sync the project dependencies using `uv` and install the `tensorflow-onnx` package for model conversion:

```bash
uv sync
uv pip install git+https://github.com/onnx/tensorflow-onnx
```



### Step 3: Verify Installation
To confirm that the dependencies are correctly installed, run:

```bash
python -c "import tensorflow, tf2onnx; print('Dependencies installed successfully')"
```

If no errors are displayed, your environment is ready.

### Step 4: Download Demo Models
Download pre-trained VballNet models for inference from one of the following links:
- [Google Drive](https://drive.google.com/file/d/1ItIxekyN47Rn2gBtq4tF-Tfe-74ez33j/view?usp=drive_link)
- [Yandex Disk](https://disk.yandex.ru/d/6Fm_JcbmF8rNug)

Place the downloaded models in the `vb-models/` directory for use in inference.

### Notes
- The `tensorflow-onnx` package is required to convert Keras models to ONNX format for optimized inference, particularly on hardware accelerators like OpenVINO GPUs.
- Ensure your Python environment is isolated (e.g., using a virtual environment or `uv`) to avoid dependency conflicts.

## Run Example Inference

Run inference with a pre-trained model using the following command:

```bash
uv run src/inference_onnx.py --video_path volleyball-backline-video.mp4 --model_path vb-models/VballNetFastV1_155_h288_w512.onnx --output_dir ./predicts/
```

## Key Features
- **Based on TrackNetV4**: Inherits the robust motion tracking capabilities of TrackNetV4, tailored specifically for volleyball detection.
- **Optimized Network Architectures**:
  - **VballNetV1**: A U-Net-like model with motion attention via a custom `MotionPromptLayer` and depthwise separable convolutions, balancing accuracy and efficiency.
  - **VballNetFastV1**: A lightweight model inspired by TrackNetV3Nano, further optimized for speed with a reduced number of channels and streamlined layers.
- **Depthwise Separable Convolutions**: Both models use depthwise separable convolutions to significantly reduce computational complexity while maintaining detection accuracy.
- **Motion Attention Mechanism**: VballNetV1 incorporates a `MotionPromptLayer` to generate attention maps from consecutive RGB frames, enhancing ball tracking by focusing on motion cues.
- **Channels-First Data Format**: Both models adopt a channels-first (N, C, H, W) format for compatibility with optimized hardware accelerators.
- **Efficient Training and Inference**: Designed to enable faster training and inference on consumer hardware, including GPUs and edge devices.

## Model Details

### VballNetV1
- **Architecture**: A U-Net-like structure with an encoder-decoder pipeline, augmented by a `MotionPromptLayer` and `FusionLayerTypeB` for motion-guided feature integration.
- **Input**: Three consecutive RGB frames (9 channels, shape: `(batch_size, 9, height, width)`).
- **Output**: Heatmaps for ball positions (shape: `(batch_size, 3, height, width)`).
- **Key Components**:
  - **MotionPromptLayer**: Generates attention maps from frame differences to emphasize motion, improving tracking precision.
  - **FusionLayerTypeB**: Combines feature maps with attention maps, weighting frames based on motion cues for enhanced detection.
  - **Depthwise Separable Convolutions**: Reduces parameter count and computational load, enabling efficient processing.
- **Advantages**:
  - High accuracy in volleyball tracking due to motion-guided attention.
  - Optimized for deployment on resource-constrained devices like OpenVINO GPUs.
  - Flexible architecture supporting various input resolutions.

### VballNetFastV1
- **Architecture**: A streamlined U-Net-like model inspired by TrackNetV3Nano, designed for maximum efficiency with fewer channels and layers.
- **Input**: Three consecutive RGB frames (9 channels, shape: `(batch_size, 9, height, width)`).
- **Output**: Heatmaps for ball positions (shape: `(batch_size, 3, height, width)`).
- **Key Components**:
  - **Single2DConv Blocks**: Simplified depthwise separable convolution blocks with batch normalization and ReLU activation.
  - **Reduced Channel Count**: Uses 8, 16, 32, and 64 channels in the encoder, minimizing memory and computational requirements.
  - **Skip Connections**: Concatenates encoder and decoder features to preserve spatial information.
- **Advantages**:
  - Extremely lightweight, enabling faster training and inference on consumer hardware.
  - Maintains competitive accuracy with a significantly reduced model size.
  - Ideal for real-time applications on edge devices.

## Advantages of VballNet
- **Performance Optimization**: Both models are tailored for consumer-grade hardware, offering faster training and inference compared to TrackNetV4.
- **Lightweight Design**: Depthwise separable convolutions reduce the computational footprint, making the models suitable for edge devices.
- **Motion-Guided Tracking**: VballNetV1’s attention mechanism improves detection accuracy by focusing on ball motion.
- **Scalability**: Supports various input resolutions, adaptable to different use cases and hardware constraints.
- **Ease of Deployment**: Channels-first format and compatibility with frameworks like OpenVINO ensure seamless integration into production environments.

## Usage
To use VballNetV1 or VballNetFastV1, load the models as follows:

```python
from VballNetV1 import VballNetV1
from VballNetFastV1 import VballNetFastV1

# VballNetV1
model_v1 = VballNetV1(height=288, width=512, in_dim=9, out_dim=3, fusion_layer_type="TypeB")

# VballNetFastV1
model_fast = VballNetFastV1(input_height=288, input_width