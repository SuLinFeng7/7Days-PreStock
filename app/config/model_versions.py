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
        'version': 'V2.0',
        'parameters': {
            'n_heads': 8,
            'num_layers': 3,
            'd_model': 128,
            'd_ff': 512,
            'dropout': 0.15,
            'attention_dropout': 0.1,
            'epochs': 150,
            'batch_size': 64,
            'warmup_steps': 4000,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'learning_rate': 0.0001,
            'loss': 'huber',
            'clip_grad_norm': 1.0
        }
    }
} 