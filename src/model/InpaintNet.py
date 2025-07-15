import numpy as np
import tensorflow as tf
import glob
import pandas as pd
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- Определение модели ---
class Conv1DBlock(layers.Layer):
    """Conv1D + BatchNormalization + LeakyReLU"""
    def __init__(self, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = layers.Conv1D(out_dim, kernel_size=3, padding='same', use_bias=True)
        self.bn = layers.BatchNormalization()
        self.relu = layers.LeakyReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double1DConv(layers.Layer):
    """Conv1DBlock x 2"""
    def __init__(self, out_dim, **kwargs):
        super(Double1DConv, self).__init__(**kwargs)
        self.conv_1 = Conv1DBlock(out_dim)
        self.conv_2 = Conv1DBlock(out_dim)

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class TemporalAttention(layers.Layer):
    """Temporal Self-Attention"""
    def __init__(self, dim, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=dim // 4)

    def call(self, x):
        return self.attention(x, x)

class InpaintNet(Model):
    def __init__(self, **kwargs):
        super(InpaintNet, self).__init__(**kwargs)
        self.down_1 = Conv1DBlock(32)
        self.down_2 = Conv1DBlock(64)
        self.down_3 = Conv1DBlock(128)
        self.attention = TemporalAttention(128)
        self.bottleneck = Double1DConv(256)
        self.dropout = layers.Dropout(0.3)
        self.up_1 = Conv1DBlock(128)
        self.up_2 = Conv1DBlock(64)
        self.up_3 = Conv1DBlock(32)
        self.predictor = layers.Conv1D(2, kernel_size=3, padding='same', activation='linear')

    def call(self, inputs):
        x, m = inputs
        x = tf.concat([x, m], axis=-1)  # (N, seq_len, 3)
        x1 = self.down_1(x)  # (N, seq_len, 32)
        x2 = self.down_2(x1)  # (N, seq_len, 64)
        x3 = self.down_3(x2)  # (N, seq_len, 128)
        x = self.attention(x3)  # (N, seq_len, 128)
        x = self.bottleneck(x)  # (N, seq_len, 256)
        x = self.dropout(x)
        x = tf.concat([x, x3], axis=-1)  # (N, seq_len, 384)
        x = self.up_1(x)  # (N, seq_len, 128)
        x = tf.concat([x, x2], axis=-1)  # (N, seq_len, 192)
        x = self.up_2(x)  # (N, seq_len, 64)
        x = tf.concat([x, x1], axis=-1)  # (N, seq_len, 96)
        x = self.up_3(x)  # (N, seq_len, 32)
        x = self.predictor(x)  # (N, seq_len, 2)
        return x

    def get_config(self):
        config = super(InpaintNet, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)