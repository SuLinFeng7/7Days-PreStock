import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Layer, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]), 
                                initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],), 
                                initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W)
        a = tf.matmul(q, inputs, transpose_b=True)
        attention_weights = tf.nn.softmax(a, axis=-1)
        return tf.matmul(attention_weights, inputs)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        LayerNormalization(),
        Dropout(0.1),
        
        AttentionLayer(),
        LayerNormalization(),
        
        LSTM(units=64, return_sequences=True),
        LayerNormalization(),
        Dropout(0.1),
        
        LSTM(units=32, return_sequences=False),
        LayerNormalization(),
        Dropout(0.1),
        
        Dense(units=64, activation='relu'),
        BatchNormalization(),
        Dense(units=32, activation='relu'),
        BatchNormalization(),
        Dense(units=1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, progress_callback=None):
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, 70)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ProgressCallback() if progress_callback else None
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=70,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[cb for cb in callbacks if cb],
        verbose=1
    )
    
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
