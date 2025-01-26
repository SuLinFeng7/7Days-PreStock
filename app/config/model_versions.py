MODEL_VERSIONS = {
    'LSTM': {
        'version': 'V2.0',
        'parameters': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'bidirectional': True,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'patience': 15,
            'reduce_lr_patience': 8,
            'reduce_lr_factor': 0.5,
            'min_lr': 1e-6
        }
    },
    'CNN': {
        'version': 'V2.0',
        'parameters': {
            'filters': 64,
            'kernel_size': 3,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 15,
            'reduce_lr_patience': 8,
            'reduce_lr_factor': 0.5,
            'min_lr': 1e-6,
            'optimizer': 'adam',
            'loss': 'huber'
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