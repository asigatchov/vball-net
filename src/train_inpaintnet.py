import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model.InpaintNet import InpaintNet

# Константы
SEQ_LEN = 16
WIDTH = 512
HEIGHT = 288
TRAIN_OUTPUT_DIR = 'data/frames/train'
TEST_OUTPUT_DIR = 'data/frames/test'
CHECKPOINT_DIR = 'checkpoints'
EPOCHS = 300
BATCH_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(data_dir):
    data = []
    mask_counts = {'0': 0, '1': 0}
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            sequence = df.values[:SEQ_LEN, 1:4]  # Visibility, X, Y
            if sequence.shape == (SEQ_LEN, 3):
                data.append(sequence)
                # Подсчет значений маски
                unique, counts = np.unique(sequence[:, 0], return_counts=True)
                for val, count in zip(unique, counts):
                    mask_counts[str(int(val))] += count
    print(f"Mask distribution in {data_dir}: {mask_counts}")
    return np.array(data, dtype=np.float32)


def prepare_data(train_dir, test_dir):
    train_data = load_data(train_dir)
    test_data = load_data(test_dir)
    
    normalization_factor = np.array([WIDTH, HEIGHT], dtype=np.float32)
    
    # Входные данные: координаты X, Y с пропусками (где Visibility=0)
    X_train = train_data[:, :, 1:3] / normalization_factor  # X, Y (столбцы 2, 3)
    m_train = train_data[:, :, 0:1]  # Visibility (столбец 1)
    X_train = X_train * m_train  # Обнуляем координаты, где Visibility=0
    y_train = train_data[:, :, 1:3] / normalization_factor  # Полные координаты X, Y
    
    X_test = test_data[:, :, 1:3] / normalization_factor  # X, Y (столбцы 2, 3)
    m_test = test_data[:, :, 0:1]  # Visibility (столбец 1)
    X_test = X_test * m_test  # Обнуляем координаты, где Visibility=0
    y_test = test_data[:, :, 1:3] / normalization_factor  # Полные координаты X, Y
    

    X_train, m_train = augment_data(X_train, m_train, missing_prob=0.2)

    return (X_train, m_train, y_train), (X_test, m_test, y_test)


def augment_data(X, m, missing_prob=0.2):
    mask = np.random.choice([0, 1], size=m.shape, p=[missing_prob, 1 - missing_prob])
    m_augmented = m * mask  # Новые пропуски
    X_augmented = X * m_augmented  # Обнуляем координаты для новых пропусков
    return X_augmented, m_augmented

# Применение в prepare_data

def masked_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)  # Ошибка для всех координат
    return tf.reduce_mean(error)


def train_model(model, X_train, m_train, y_train, X_val, m_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model.compile(optimizer='adam', loss=masked_mse, metrics=['mae'])
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "inpaintnet_{epoch:03d}.keras")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_freq='epoch'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        try:
            model.load_weights(latest_checkpoint)
            print(f"Loaded weights from checkpoint: {latest_checkpoint}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    
    history = model.fit(
        [X_train, m_train],
        y_train,
        validation_data=([X_val, m_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback, early_stopping],
        verbose=1
    )
    
    return history

def create_dummy_input(model):
    """Создает фиктивные входные данные для инициализации модели"""
    dummy_coords = np.zeros((1, SEQ_LEN, 2), dtype=np.float32)
    dummy_mask = np.zeros((1, SEQ_LEN, 1), dtype=np.float32)
    _ = model([dummy_coords, dummy_mask])
    return model

if __name__ == "__main__":
    print(f"Training configuration:")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Input dimensions: {WIDTH}x{HEIGHT}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Test size: {TEST_SIZE}")
    
    # Загрузка и подготовка данных
    (X_train, m_train, y_train), (X_test, m_test, y_test) = prepare_data(TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR)
    
    print(f"Data shapes:")
    print(f"  Train: X={X_train.shape}, m={m_train.shape}, y={y_train.shape}")
    print(f"  Test: X={X_test.shape}, m={m_test.shape}, y={y_test.shape}")
    
    # Разделение на обучающую и валидационную выборки
    X_train, X_val, m_train, m_val, y_train, y_val = train_test_split(
        X_train, m_train, y_train, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Split data shapes:")
    print(f"  Train: X={X_train.shape}, m={m_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, m={m_val.shape}, y={y_val.shape}")
    
    # Создание и инициализация модели
    model = InpaintNet()
    model = create_dummy_input(model)
    
    # Обучение модели


    history = train_model(
        model,
        X_train, m_train, y_train,
        X_val, m_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Оценка на тестовой выборке
    test_loss, test_mae = model.evaluate([X_test, m_test], y_test, verbose=0)
    print(f"\nFinal test results:")
    print(f"  Test loss: {test_loss:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    
    # Сохранение финальной модели
    final_model_path = os.path.join(CHECKPOINT_DIR, "inpaintnet_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")