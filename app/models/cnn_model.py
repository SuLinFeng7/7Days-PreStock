from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')
    return model

def train_cnn_model(X_train, y_train, X_test, y_test, progress_callback=None):
    model = create_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, 200)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ProgressCallback() if progress_callback else None
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[cb for cb in callbacks if cb],
        verbose=1
    )
    
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
