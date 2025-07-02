import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Установить неинтерактивный бэкенд
import matplotlib.pyplot as plt
from model.PlayerNetFastV1 import PlayerNetFastV1

# Инициализация модели
input_shape = (9, 288, 512)
output_channels = 3
model = PlayerNetFastV1(input_shape=input_shape, output_channels=output_channels)

# Вывод структуры модели
model.summary()

# Создание тестовых данных
batch_size = 2
test_input = np.random.rand(batch_size, 9, 288, 512).astype(np.float32)
print("Test input shape:", test_input.shape)

# Компиляция модели (опционально)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Инференс
predictions = model.predict(test_input)
print("Prediction shape:", predictions.shape)

# Визуализация
for i in range(batch_size):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Input Frame {i+1}")
    plt.imshow(test_input[i, 0, :, :], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Mask {i+1}")
    plt.imshow(predictions[i, 0, :, :], cmap='gray')
    plt.axis('off')
    plt.savefig(f'output_{i+1}.png', bbox_inches='tight')
    plt.close()

model.save('models/PlayerNetFastV1_01.keras')