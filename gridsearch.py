from main import train_and_evaluate
from itertools import product
from argparse import Namespace
import json


# Grid search param grid
param_grid = {
    'num_hidden': [64, 128],
    'num_layers': [1, 2],
    'dropout': [0.2, 0.5],
    'lr': [1e-3, 1e-4],
    'encoder_layer': ['dualcata-softmax-4', 'dualcata-tanh-4'],
    'rnn': ['gru', 'lstm'],
    'rnn_agg': ['last', 'mean'],
    'oversample': [None, 1.0],
}

keys, values = zip(*param_grid.items())
all_combinations = [dict(zip(keys, v)) for v in product(*values)]

# Default params
from argparse import Namespace

args = Namespace(
    # Dataset & xử lý dữ liệu
    data='ethers_data',
    data_name='PyG_BTC_2015',
    use_unlabeled='SEMI',
    scale='minmax',
    sort_by='a',
    length=32,
    anomaly_rate=None,

    # RNN
    rnn_feat='e',
    lstm_norm='ln',
    emb_first=1,
    rnn_in_channels=8,

    # GNN
    gnn_norm='bn',
    graph_op=None,
    graph_type='MultiDi',
    feature_type='node',
    neighbor_size=20,
    fan_out='10,25',
    aggr='add',
    directed=True,

    # Decoder & tổng thể mô hình
    num_outputs=2,
    decoder_layers=2,
    concat_feature=0,
    train_rate=0.5,

    # Huấn luyện
    num_epochs=10,
    batch_size=128,
    weight_lr=0.001,
    reweight=False,
    undersample=False,
    patience=10,
    random_state=5211,

    # GPU & sampling
    gpu=0,
    num_workers=16,
    sample_gpu=False,
    inductive=False
)

if __name__ == "__main__":
    
    results = []

    for config in all_combinations:
        full_args = Namespace(**vars(args), **config)
        print(f"Running confing: {full_args}")
        metrics = train_and_evaluate(full_args)
        result_entry = {"config": full_args, "metrics": metrics}
        results.append(result_entry)
        
    with open('/kaggle/working/grid_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # In top-k theo F1
    top_k = sorted(results, key=lambda x: x['metrics']['Best_F1_mean'], reverse=True)[:5]
    for entry in top_k:
        print(entry) 
    