import tensorflow as tf
import tensorflowjs as tfjs
from utils import custom_loss

# Загрузка модели Keras
model = tf.keras.models.load_model(
    "models/VballNetFastV1/VballNetFastV1/VballNetFastV1_145.keras",
    #"models/VballNetV1_82.keras"
    custom_objects={"custom_loss": custom_loss},
)


# Создание новой модели с явным InputLayer
input_shape = (9, 288, 512)  # channels-first: (channels, height, width)
inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
outputs = model(inputs)  # Передаём вход через существующую модель
new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Проверка входной формы
print("New model input shape:", new_model.input.shape)
new_model.summary()

# Проверка конфигурации входного слоя
input_layer = new_model.layers[0]
print("Input layer configuration:", input_layer.get_config())

# Сохранение исправленной модели
new_model.save("tracknet_fixed.keras")

# Конвертация в TensorFlow.js
tfjs.converters.save_keras_model(new_model, "tfjs_model")
