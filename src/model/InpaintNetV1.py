import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=5000, d_model=128):
        super(PositionalEncoding, self).__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.positional_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:, :seq_len, :]

class Conv1DBlock(layers.Layer):
    def __init__(self, out_dim, kernel_size=3, strides=1, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = layers.Conv1D(out_dim, kernel_size=kernel_size, padding='same',
                                  strides=strides, use_bias=True)
        self.bn = layers.BatchNormalization()
        self.relu = layers.LeakyReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class Double1DConv(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super(Double1DConv, self).__init__(**kwargs)
        self.conv_1 = Conv1DBlock(out_dim)
        self.conv_2 = Conv1DBlock(out_dim)

    def call(self, x, training=False):
        x = self.conv_1(x, training=training)
        x = self.conv_2(x, training=training)
        return x

class TemporalAttention(layers.Layer):
    def __init__(self, dim, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.pos_encoding = PositionalEncoding(d_model=dim)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=dim // 4)

    def call(self, x):
        x = self.pos_encoding(x)
        return self.attention(x, x)

class InpaintNetV1(tf.keras.Model):
    def __init__(self, output_channels=2, task='regression', dropout_rate=0.3, **kwargs):
        super(InpaintNetV1, self).__init__(**kwargs)
        self.down_1 = Conv1DBlock(32, strides=2)   # seq_len/2
        self.down_2 = Conv1DBlock(64, strides=2)   # seq_len/4
        self.down_3 = Conv1DBlock(128, strides=2)  # seq_len/8
        self.attention = TemporalAttention(128)
        self.bottleneck = Double1DConv(256)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.up_sample_1 = layers.UpSampling1D(size=2)
        self.up_1 = Conv1DBlock(128)
        self.up_sample_2 = layers.UpSampling1D(size=2)
        self.up_2 = Conv1DBlock(64)
        self.up_sample_3 = layers.UpSampling1D(size=2)
        self.up_3 = Conv1DBlock(32)
        self.dropout2 = layers.Dropout(dropout_rate / 2)
        self.mask_conv = Conv1DBlock(8)
        # Add alignment convolutions for skip connections
        self.align_d3 = layers.Conv1D(256, kernel_size=1, padding='same')  # Align d3 to 256 channels
        self.align_d2 = layers.Conv1D(64, kernel_size=1, padding='same')   # Align d2 to 64 channels
        self.align_d1 = layers.Conv1D(32, kernel_size=1, padding='same')   # Align d1 to 32 channels
        activation = 'linear' if task == 'regression' else 'softmax'
        self.predictor = layers.Conv1D(output_channels, kernel_size=3, padding='same', activation=activation)

    def call(self, inputs, training=None):
        # inputs теперь кортеж: (input, mask)
        x, mask = inputs  # вместо x = inputs['input'], mask = inputs['mask']
        m = tf.cast(mask, tf.float32)
        
        m_feat = self.mask_conv(m, training=training)
        x = tf.concat([x, m_feat], axis=-1)
        
        d1 = self.down_1(x, training=training)   # Shape: [?, 8, 32]
        d2 = self.down_2(d1, training=training)  # Shape: [?, 4, 64]
        d3 = self.down_3(d2, training=training)  # Shape: [?, 2, 128]

        attn = self.attention(d3)                # Shape: [?, 2, 128]
        bottleneck = self.bottleneck(attn, training=training)  # Shape: [?, 2, 256]
        bottleneck = self.dropout1(bottleneck, training=training)

        u1 = self.up_sample_1(bottleneck)        # Shape: [?, 4, 256]
        d3_aligned = self.align_d3(d3)           # [?, 2, 256]
        # Исправлено: tf.image.resize требует size из двух элементов (height, width)
        # Для 1D Conv: увеличиваем только длину (time/seq), channels не трогаем
        d3_aligned_up = tf.repeat(d3_aligned, repeats=tf.shape(u1)[1] // tf.shape(d3_aligned)[1], axis=1)  # [?, 4, 256]
        u1 = self.up_1(u1 + d3_aligned_up, training=training)  # [?, 4, 128]

        u2 = self.up_sample_2(u1)                # [?, 8, 128]
        d2_aligned = self.align_d2(d2)           # [?, 4, 64]
        d2_aligned_up = tf.repeat(d2_aligned, repeats=tf.shape(u2)[1] // tf.shape(d2_aligned)[1], axis=1)  # [?, 8, 64]
        u2 = self.up_2(u2 + d2_aligned_up, training=training)  # [?, 8, 64]

        u3 = self.up_sample_3(u2)                # [?, 16, 64]
        d1_aligned = self.align_d1(d1)           # [?, 8, 32]
        d1_aligned_up = tf.repeat(d1_aligned, repeats=tf.shape(u3)[1] // tf.shape(d1_aligned)[1], axis=1)  # [?, 16, 32]
        u3 = self.up_3(u3 + d1_aligned_up, training=training)  # [?, 16, 32]

        u3 = self.dropout2(u3, training=training)
        out = self.predictor(u3)                 # [?, 16, 2]

        # Ensure output sequence length matches input (SEQUENCE_LENGTH)
        out = tf.image.resize(out, [SEQUENCE_LENGTH, 2], method='bilinear')
        return out

    def get_config(self):
        config = super(InpaintNetV1, self).get_config()
        config.update({
            'output_channels': self.predictor.filters,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)