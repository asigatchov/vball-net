import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Layer

class ChannelShuffle(Layer):
    """Custom Keras Layer for Channel Shuffle operation"""
    def __init__(self, groups, name=None, trainable=True, dtype=None):
        super(ChannelShuffle, self).__init__(name=name, trainable=trainable, dtype=dtype)
        self.groups = groups

    def call(self, x):
        _, C, H, W = x.shape
        x = tf.reshape(x, [-1, self.groups, C // self.groups, H, W])
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        x = tf.reshape(x, [-1, C, H, W])
        return x

    def get_config(self):
        config = super(ChannelShuffle, self).get_config()
        config.update({
            "groups": self.groups
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def depthwise_block(x, channels, name):
    """Optimized block with depthwise and pointwise convolutions"""
    input_channels = x.shape[1]  # For channels_first (N, C, H, W)

    x = Conv2D(
        filters=input_channels,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        groups=input_channels,
        kernel_initializer='glorot_uniform',
        data_format='channels_first',
        name=f'{name}_depthwise'
    )(x)
    x = BatchNormalization(axis=1)(x)
    
    x = Conv2D(
        filters=channels,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        data_format='channels_first',
        name=f'{name}_pointwise'
    )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    
    x = ChannelShuffle(groups=4, name=f'{name}_shuffle')(x)
    return x

def PlayerNetFastV1(input_shape=(9, 288, 512), output_channels=3):
    """
    Optimized model for object detection with 3 output channels
    
    Args:
        input_shape (tuple): Input tensor shape (channels_first)
        output_channels (int): Number of output channels (3 for three heatmaps)
        
    Returns:
        tf.keras.Model: Optimized model
    """
    inputs = Input(shape=input_shape, name='input')
    
    # Encoder
    x1 = depthwise_block(inputs, 16, 'enc1')
    x = MaxPooling2D((2,2), data_format='channels_first')(x1)
    
    x2 = depthwise_block(x, 32, 'enc2')
    x = MaxPooling2D((2,2), data_format='channels_first')(x2)
    
    x3 = depthwise_block(x, 64, 'enc3')
    x = MaxPooling2D((2,2), data_format='channels_first')(x3)
    
    # Bottleneck
    x = depthwise_block(x, 128, 'bottleneck')
    
    # Decoder
    x = UpSampling2D((2,2), data_format='channels_first')(x)
    x = Concatenate(axis=1)([x, x3])
    x = depthwise_block(x, 64, 'dec1')
    
    x = UpSampling2D((2,2), data_format='channels_first')(x)
    x = Concatenate(axis=1)([x, x2])
    x = depthwise_block(x, 32, 'dec2')
    
    x = UpSampling2D((2,2), data_format='channels_first')(x)
    x = Concatenate(axis=1)([x, x1])
    x = depthwise_block(x, 16, 'dec3')
    
    # Output layer with 3 channels
    outputs = Conv2D(
        output_channels,
        (1,1),
        activation='sigmoid',
        data_format='channels_first',
        name='output'
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)