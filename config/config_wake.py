
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 64,
    'ni': 200,
    'enc_nh': 512,
    'dec_nh': 512,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 128,
    'epochs': 32,
    'test_nepoch': 4,
    'train_data': 'datasets/wake_data/wake.train.txt.gz',
    'val_data': 'datasets/wake_data/wake.valid.txt.gz',
    'test_data': 'datasets/wake_data/wake.test.txt.gz',
    'bpemb': {
        'lang': 'en',
        'vs': 25000,
        'dim': 200
    }
}
