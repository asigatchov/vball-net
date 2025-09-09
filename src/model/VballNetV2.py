from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    SeparableConv2D,
    Activation,
    BatchNormalization,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Reshape,
    Layer, 
            Lambda,  # Добавьте Lambda

)
from tensorflow.keras.models import Model


# Utility functions
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
    return 1 / (
        1
        + tf.exp(
            -(5 / (0.45 * tf.abs(tf.tanh(a)) + 1e-1))
            * (tf.abs(input) - 0.8 * tf.tanh(b))
        )
    )


# MotionPromptLayer
class MotionPromptLayer(Layer):
    """
    A custom Keras layer for generating attention maps from video sequences.
    Uses central differences for motion detection to align with current frame.
    Supports grayscale (N frames) and RGB (N×3 channels) modes.
    """

    def __init__(self, num_frames, mode="grayscale", penalty_weight=0.0, **kwargs):
        super(MotionPromptLayer, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"], "Mode must be 'rgb' or 'grayscale'"
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB" if self.mode == "rgb" else None
        self.color_map = {"R": 0, "G": 1, "B": 2}
        self.gray_scale = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
        self.pn = power_normalization
        self.a = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.1), trainable=True, name="a"
        )
        self.b = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.0), trainable=True, name="b"
        )
        self.lambda1 = penalty_weight

    def call(self, video_seq):
        loss = tf.constant(0.0)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45

        if self.mode == "rgb":
            idx_list = [self.color_map[idx] for idx in self.input_color_order]
            gray_scale_tensor = tf.gather(self.gray_scale, idx_list)
            weights = tf.cast(gray_scale_tensor, dtype=norm_seq.dtype)
            grayscale_video_seq = tf.einsum("btcwh,c->btwh", norm_seq, weights)
        else:  # grayscale mode
            grayscale_video_seq = norm_seq[:, :, 0, :, :]  # Single channel per frame

        # Compute central differences for frames t=1 to t=num_frames-2
        attention_map = []
        for t in range(self.num_frames):
            if t == 0:
                # Forward difference for first frame
                frame_diff = grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t]
            elif t == self.num_frames - 1:
                # Backward difference for last frame
                frame_diff = grayscale_video_seq[:, t] - grayscale_video_seq[:, t - 1]
            else:
                # Central difference for intermediate frames
                frame_diff = (
                    grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t - 1]
                ) / 2
            attention_map.append(self.pn(frame_diff, self.a, self.b))
        attention_map = tf.stack(
            attention_map, axis=1
        )  # Shape: (batch, num_frames, height, width)
        norm_attention = tf.expand_dims(attention_map, axis=2)

        if self.trainable:
            B, T, H, W = grayscale_video_seq.shape
            if B is None:
                B = 1
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = tf.reduce_sum(tf.square(temp_diff)) / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss
            self.add_loss(loss)

        return attention_map, loss


# FusionLayerTypeA
class FusionLayerTypeA(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 1.
    Applies attention map of current frame t to feature map of frame t.
    """

    def __init__(self, num_frames, out_dim, **kwargs):
        super(FusionLayerTypeA, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.out_dim = out_dim

    def call(self, inputs):
        feature_map, attention_map = inputs
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            outputs.append(
                feature_map[:, t, :, :] * attention_map[:, t, :, :]
            )  # Use attention map of current frame
        return tf.stack(outputs, axis=1)

def spatial_attention(x):
    # x имеет форму (None, channels, height, width), например (None, 128, 72, 128)
    
    # Переключаемся на channels_last: (None, height, width, channels)
    x = keras.layers.Permute((2, 3, 1))(x)  # (None, 72, 128, 128)
    
    # Вычисляем среднее и максимум по пространственным осям
    avg_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)  # (None, channels)
    max_pool = keras.layers.GlobalMaxPooling2D(data_format='channels_last')(x)    # (None, channels)
    
    # Добавляем единичные размерности
    avg_pool = keras.layers.Reshape((1, 1, x.shape[-1]))(avg_pool)  # (None, 1, 1, channels)
    max_pool = keras.layers.Reshape((1, 1, x.shape[-1]))(max_pool)  # (None, 1, 1, channels)
    
    # Конкатенируем по канальной оси (axis=-1 в channels_last)
    concat = keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])  # (None, 1, 1, 2*channels)
    
    # Восстанавливаем пространственные размерности
    concat = keras.layers.UpSampling2D(
        size=(x.shape[1], x.shape[2]), data_format='channels_last'
    )(concat)  # (None, 72, 128, 2*channels)
    
    # Применяем свертку
    attention = keras.layers.Conv2D(
        1, (7, 7), padding='same', activation='sigmoid', data_format='channels_last'
    )(concat)  # (None, 72, 128, 1)
    
    # Умножаем на входной тензор
    output = x * attention  # (None, 72, 128, channels)
    
    # Возвращаемся к channels_first: (None, channels, height, width)
    output = keras.layers.Permute((3, 1, 2))(output)  # (None, 128, 72, 128)
    
    return output

def VballNetV2(height, width, in_dim, out_dim, fusion_layer_type="TypeA"):
  
    assert fusion_layer_type in [
        "TypeA",
        "TypeB",
    ], "Fusion layer must be 'TypeA' or 'TypeB'"

    # Determine mode and number of frames
    mode = "grayscale" if in_dim == out_dim else "rgb"
    num_frames = in_dim if mode == "grayscale" else in_dim // 3

    # Select fusion layer
    fusion_layer = (
        FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)
       
    )

    # Input layer
    imgs_input = Input(shape=(in_dim, height, width))
    channels_per_frame = in_dim // num_frames
    motion_input = Reshape((num_frames, channels_per_frame, height, width))(imgs_input)

    # Motion prompt layer
    residual_maps, _ = MotionPromptLayer(num_frames=num_frames, mode=mode)(motion_input)

    # Encoder
    x = SeparableConv2D(
        32,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(imgs_input)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(
        32,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x1 = Activation("relu")(x1)
    x1 = BatchNormalization()(x1)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x1)

    x = SeparableConv2D(
        64,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = Activation("relu")(x)
    x2 = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x2)

    x = SeparableConv2D(
        128,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = spatial_attention(x)  # Apply spatial attention

    # Decoder
    x = concatenate([UpSampling2D((2, 2), data_format="channels_first")(x), x2], axis=1)

    x = SeparableConv2D(
        64,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = concatenate([UpSampling2D((2, 2), data_format="channels_first")(x), x1], axis=1)

    x = SeparableConv2D(
        32,
        (3, 3),
        depthwise_initializer="random_uniform",
        pointwise_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # Output layer
    x = Conv2D(
        out_dim,
        (1, 1),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)

    x = fusion_layer([x, residual_maps])
    x = Activation("sigmoid")(x)

    # Model creation
    model = Model(inputs=imgs_input, outputs=x)
    return model
