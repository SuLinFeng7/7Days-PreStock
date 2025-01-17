MODEL_VERSIONS = {
    'LSTM': {
        'version': 'V1.0',
        'parameters': {
            'units': 50,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mse'
        }
    },
    'CNN': {
        'version': 'V1.0',
        'parameters': {
            'filters': [64, 32, 16],
            'kernel_size': 3,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mse'
        }
    },
    'Transformer': {
        'version': 'V1.0',
        'parameters': {
            'n_heads': 4,
            'num_layers': 2,
            'd_model': 64,
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mse'
        }
    }
} 