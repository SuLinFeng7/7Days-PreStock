from tensorflow.keras import layers, models

def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_transformer_model(X_train, y_train, X_test, y_test):
    model = create_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1), y_test.reshape(-1, 1), model
