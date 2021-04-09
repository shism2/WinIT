import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import pickle as pkl
from captum.attr import Saliency, IntegratedGradients, DeepLift

from xgboost_model import XGBPytorchStub
from run_experiment import get_fit_attributions, get_inverse_fit_attributions, get_tsr_attributions
from FIT.TSX.models import StateClassifier
from FIT.TSX.utils import train_model as train_single_label_model
from xgboost_model import loader_to_np


def load_to_np(ds_name):
    # Keeping these here to avoid dependency issues
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    from sktime.utils.data_processing import from_nested_to_3d_numpy

    train_x, train_y = load_from_tsfile_to_dataframe(f'{ds_name}/{ds_name}_TRAIN.ts')
    test_x, test_y = load_from_tsfile_to_dataframe(f'{ds_name}/{ds_name}_TEST.ts')

    train_x = from_nested_to_3d_numpy(train_x)
    test_x = from_nested_to_3d_numpy(test_x)

    np.random.seed(0)
    train_shuffle = np.random.permutation(train_y.shape[0])
    train_x = train_x[train_shuffle]
    train_y = train_y[train_shuffle]
    np.random.seed(0)

    if ds_name == 'Heartbeat':
        train_y = train_y != 'normal'
        test_y = test_y != 'normal'
    elif ds_name == 'FaceDetection':
        train_y = train_y.astype(np.float)
        test_y = test_y.astype(np.float)
    elif ds_name == 'FingerMovements':
        train_y = train_y != 'left'
        test_y = test_y != 'left'
    else:
        raise Exception('Dataset name unrecognized')

    np.save(f'{ds_name}/train_x.npy', train_x)
    np.save(f'{ds_name}/train_y.npy', train_y)
    np.save(f'{ds_name}/test_x.npy', test_x)
    np.save(f'{ds_name}/test_y.npy', test_y)


def load_data(ds_name, batch_size=64):
    train_x = np.load(f'{ds_name}/train_x.npy')
    train_y = np.load(f'{ds_name}/train_y.npy')
    test_x = np.load(f'{ds_name}/test_x.npy')
    test_y = np.load(f'{ds_name}/test_y.npy')

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()), batch_size=batch_size)
    attr_loader = DataLoader(TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()), batch_size=batch_size)

    return train_loader, test_loader, attr_loader


if __name__ == '__main__':
    dataset = 'FingerMovements'  # ['FingerMovements', 'Heartbeat', 'FaceDetection']
    model = 'rnn'  # ['rnn', 'xgb']
    method = 'fit'  # ['fit', 'grad_tsr', 'grad', 'IG', 'DL']
    train_model = True
    train_gen = True
    window_size = 10
    top_k = 300
    skip_saliency = False
    load_from_ts_file = False

    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if load_from_ts_file:
        load_to_np(dataset)

    train_loader, test_loader, attr_loader = load_data(dataset)
    if model == 'rnn':
        model = StateClassifier(feature_size=next(iter(test_loader))[0].shape[1], n_state=2, hidden_size=200, rnn='GRU')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        if train_model:
            train_single_label_model(model=model, train_loader=train_loader, valid_loader=test_loader,
                           optimizer=optimizer, n_epochs=200, device=device, experiment=dataset)
        else:
            model.load_state_dict(torch.load(f"ckpt/mimic/{dataset}_0.pt"))
    elif model == 'xgb':
        model = XGBPytorchStub(train_loader, test_loader, window_size, 0, 0, f'Models/{dataset}_{window_size}.model', train=train_model)
    else:
        raise Exception(f'Model {model} unrecognized')

    if not skip_saliency:
        if method == 'fit':
            attributions = get_fit_attributions(model, train_loader, attr_loader, next(iter(test_loader))[0].shape[1],
                                                dataset, train=train_gen, mock_generator=False,
                                                activation=torch.nn.Softmax(-1) if model == "rnn" else lambda x: x)
        elif method == 'grad_tsr':
            attributions = get_tsr_attributions(Saliency(model), attr_loader)
        elif method in ['grad', 'IG', 'DL']:
            x, y = loader_to_np(attr_loader)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            baseline_single = torch.from_numpy(np.random.random(x.shape)).to(device).float()

            if method == 'grad':
                attributions = Saliency(model).attribute(x, target=y).data.cpu().numpy()
            elif method == 'IG':
                attributions = IntegratedGradients(model).attribute(x, baselines=baseline_single, target=y).data.cpu().numpy()
            elif method == 'DL':
                attributions = DeepLift(model).attribute(x, baselines=baseline_single, target=y).data.cpu().numpy()
        else:
            raise Exception(f'Method {method} unrecognized')

        with open(f'Results/{dataset}_xgb_{window_size}_{method}_importance', 'wb') as f:
            pkl.dump(attributions, f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with open(f'Results/{dataset}_xgb_{window_size}_{method}_importance', 'rb') as f:
            attributions = pkl.load(f)

    labels = []
    preds = []
    masked_preds = []
    for i, (x, y) in enumerate(attr_loader):
        labels.append(y)
        preds.append(model(x).detach())

        for j in range(len(y)):
            idx = i * len(y) + j
            flat_sorted_imp = np.argsort(attributions[idx], axis=None)
            for k in range(top_k):
                imp_idx = flat_sorted_imp[-k - 1]
                ft, ts = imp_idx // x.shape[-1], imp_idx % x.shape[-1]
                x[j, ft, ts] = x[j, ft, ts - 1]  # TODO: What to use here?
        masked_preds.append(model(x).detach())

    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    masked_preds = np.concatenate(masked_preds, axis=0)

    auc_drop = metrics.roc_auc_score(labels, preds[:, 1]) - metrics.roc_auc_score(labels, masked_preds[:, 1])
    print(f'{dataset} AUC Drop: {auc_drop}')
