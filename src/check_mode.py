
import tensorflow as tf
from utils import custom_loss

# Загрузка модели
model = tf.keras.models.load_model("models/VballNetFastV1/VballNetFastV1/VballNetFastV1_145.keras", custom_objects={'custom_loss': custom_loss})

# Вывод структуры модели
print("Model summary:")
model.summary()

# Проверка входного слоя
input_layer = model.input
print("Input layer shape:", input_layer.shape)
print("Input layer configuration:", input_layer.get_config())