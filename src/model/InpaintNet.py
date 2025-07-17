import tensorflow as tf
from tensorflow.keras import layers, Model

class Conv1DBlock(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = layers.Conv1D(out_dim, kernel_size=3, padding='same', use_bias=True)
        self.relu = layers.LeakyReLU()
    
    def call(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
    def get_config(self):
        config = super(Conv1DBlock, self).get_config()
        config.update({
            'in_dim': self.in_dim,
            'out_dim': self.out_dim
        })
        return config

class Double1DConv(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Double1DConv, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
    def get_config(self):
        config = super(Double1DConv, self).get_config()
        config.update({
            'in_dim': self.in_dim,
            'out_dim': self.out_dim
        })
        return config

class InpaintNet(Model):
    def __init__(self, **kwargs):
        super(InpaintNet, self).__init__(**kwargs)
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.bottleneck = Double1DConv(128, 128)
        self.up_1 = Conv1DBlock(256, 64)  # 128 + 128 = 256
        self.up_2 = Conv1DBlock(128, 32)  # 64 + 64 = 128
        self.up_3 = Conv1DBlock(64, 16)   # 32 + 32 = 64
        self.predictor = layers.Conv1D(2, 3, padding='same')
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        x, m = inputs
        # Входные данные: x shape (batch, 16, 2), m shape (batch, 16, 1)
        x = tf.concat([x, m], axis=2)  # shape: (batch, 16, 3)
        
        # Применяем свертки по временной оси
        x1 = self.down_1(x)      # shape: (batch, 16, 32)
        x2 = self.down_2(x1)     # shape: (batch, 16, 64)
        x3 = self.down_3(x2)     # shape: (batch, 16, 128)
        
        x = self.bottleneck(x3)  # shape: (batch, 16, 128)
        
        # Skip connections по каналам (axis=2)
        x = tf.concat([x, x3], axis=2)  # shape: (batch, 16, 256)
        x = self.up_1(x)                # shape: (batch, 16, 64)
        
        x = tf.concat([x, x2], axis=2)  # shape: (batch, 16, 128)
        x = self.up_2(x)                # shape: (batch, 16, 32)
        
        x = tf.concat([x, x1], axis=2)  # shape: (batch, 16, 64)
        x = self.up_3(x)                # shape: (batch, 16, 16)
        
        x = self.predictor(x)           # shape: (batch, 16, 2)
        x = self.sigmoid(x)
        
        return x  # shape: (batch, 16, 2)
    
    def get_config(self):
        config = super(InpaintNet, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)