import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D,
    concatenate, Reshape, Layer
)
from tensorflow.keras.models import Model

# Utility functions (unchanged)
def rearrange_tensor(input_tensor, order):
    """
    Rearranges the dimensions of a tensor according to the specified order.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    return tf.transpose(input_tensor, [order.index(dim) for dim in "BTCHW"])

def power_normalization(input, a, b):
    """
    Power normalization function for attention map generation.
    """
    return 1 / (1 + tf.exp(-(5 / (0.45 * tf.abs(tf.tanh(a)) + 1e-1)) * (tf.abs(input) - 0.6 * tf.tanh(b))))

# MotionPromptLayer (updated for variable frames and mode)
class MotionPromptLayer(Layer):
    """
    A custom Keras layer for generating attention maps from video sequences.
    Supports RGB and Grayscale modes with variable number of frames.
    """
    def __init__(self, num_frames=3, mode='rgb', penalty_weight=0.0, **kwargs):
        super(MotionPromptLayer, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ['rgb', 'grayscale'], "Mode must be 'rgb' or 'grayscale'"
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB" if self.mode == 'rgb' else None
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
        self.pn = power_normalization
        self.a = self.add_weight(shape=(), initializer=tf.constant_initializer(0.1), trainable=True, name='a')
        self.b = self.add_weight(shape=(), initializer=tf.constant_initializer(0.0), trainable=True, name='b')
        self.lambda1 = penalty_weight

    def call(self, video_seq):
        loss = tf.constant(0.0)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45

        if self.mode == 'rgb':
            idx_list = [self.color_map[idx] for idx in self.input_color_order]
            gray_scale_tensor = tf.gather(self.gray_scale, idx_list)
            weights = tf.cast(gray_scale_tensor, dtype=norm_seq.dtype)
            grayscale_video_seq = tf.einsum("btcwh,c->btwh", norm_seq, weights)
        else:  # grayscale mode
            grayscale_video_seq = norm_seq[:, :, 0, :, :]  # Single channel per frame

        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]
        attention_map = self.pn(frame_diff, self.a, self.b)
        norm_attention = tf.expand_dims(attention_map, axis=2)

        if self.trainable:
            B, T, H, W = grayscale_video_seq.shape
            if B is None:
                B = 1
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = tf.reduce_sum(tf.square(temp_diff)) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss
            self.add_loss(loss)

        return attention_map, loss

# FusionLayerTypeA (updated for variable frames and output dimensions)
class FusionLayerTypeA(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 1.
    Supports variable number of frames and output dimensions.
    """
    def __init__(self, num_frames=3, out_dim=3, **kwargs):
        super(FusionLayerTypeA, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.out_dim = out_dim

    def call(self, inputs):
        feature_map, attention_map = inputs
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            if t == 0:
                outputs.append(feature_map[:, t, :, :])  # No attention for first frame
            else:
                outputs.append(feature_map[:, t, :, :] * attention_map[:, t-1, :, :])
        return tf.stack(outputs, axis=1)

# FusionLayerTypeB (updated for variable frames and output dimensions)
class FusionLayerTypeB(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 2.
    Supports variable number of frames and output dimensions.
    """
    def __init__(self, num_frames=3, out_dim=3, **kwargs):
        super(FusionLayerTypeB, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.out_dim = out_dim

    def call(self, inputs):
        feature_map, attention_map = inputs
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            if t == 0:
                outputs.append(feature_map[:, t, :, :] * attention_map[:, t, :, :])
            elif t == self.num_frames - 1:
                outputs.append(feature_map[:, t, :, :] * attention_map[:, t-1, :, :])
            else:
                outputs.append(feature_map[:, t, :, :] * ((attention_map[:, t-1, :, :] + attention_map[:, t, :, :])/2))
        return tf.stack(outputs, axis=1)

def VballNetV1(height, width, in_dim=9, out_dim=3, fusion_layer_type="TypeA"):
    """
    Constructs the VballNetV1 model, a lightweight neural network for volleyball tracking.
    Supports RGB (3 frames * 3 channels input, 3 channels output) and Grayscale (seq channels input, seq channels output) modes.

    Args:
        height (int): Height of the input frames in pixels.
        width (int): Width of the input frames in pixels.
        in_dim (int, optional): Number of input channels (e.g., 9 for RGB, seq for grayscale). Defaults to 9.
        out_dim (int, optional): Number of output channels (e.g., 3 for RGB, seq for grayscale). Defaults to 3.
        fusion_layer_type (str, optional): Type of fusion layer ('TypeA' or 'TypeB'). Defaults to 'TypeA'.

    Returns:
        Model: A Keras model instance with input shape (batch_size, in_dim, height, width) and
               output shape (batch_size, out_dim, height, width).

    Notes:
        - RGB mode: in_dim=9 (3 frames × 3 channels), out_dim=3.
        - Grayscale mode: in_dim=seq, out_dim=seq.
        - Uses a U-Net-like architecture with motion attention via MotionPromptLayer.
    """
    assert fusion_layer_type in ["TypeA", "TypeB"], "Fusion layer must be 'TypeA' or 'TypeB'"

    # Determine mode and number of frames
    mode = 'grayscale' if in_dim == out_dim else 'rgb'
    num_frames = in_dim if mode == 'grayscale' else in_dim // 3  # For RGB, in_dim=9 means 3 frames

    # Select fusion layer
    fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim) if fusion_layer_type == "TypeA" else FusionLayerTypeB(num_frames=num_frames, out_dim=out_dim)

    # Input layer
    imgs_input = Input(shape=(in_dim, height, width))
    channels_per_frame = in_dim // num_frames
    motion_input = Reshape((num_frames, channels_per_frame, height, width))(imgs_input)

    # Motion prompt layer
    residual_maps, _ = MotionPromptLayer(num_frames=num_frames, mode=mode)(motion_input)

    # Encoder
    x = SeparableConv2D(32, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                        padding='same', data_format='channels_first')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(32, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                         padding='same', data_format='channels_first')(x)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x1)

    x = SeparableConv2D(64, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                        padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x2 = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x2)

    x = SeparableConv2D(128, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                        padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Decoder
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x2], axis=1)

    x = SeparableConv2D(64, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                        padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x1], axis=1)

    x = SeparableConv2D(32, (3, 3), depthwise_initializer='random_uniform', pointwise_initializer='random_uniform',
                        padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Output layer
    x = Conv2D(out_dim, (1, 1), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = fusion_layer([x, residual_maps])
    x = Activation('sigmoid')(x)

    # Model creation
    model = Model(inputs=imgs_input, outputs=x)
    return model