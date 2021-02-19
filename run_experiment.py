import pathlib
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn import metrics
from captum.attr import Saliency

from FIT.TSX.models import StateClassifier
from FIT.TSX.utils import train_model_rt
from FIT.TSX.explainers import FITExplainer
from FIT.TSX.generator import JointFeatureGenerator
from TSR.Scripts.Plotting.plot import plotExampleBox
from TSR.Scripts.tsr import get_tsr_saliency
from TSR.Scripts.train_models import train_model
from TSR.Scripts.Models.LSTM import LSTM
from TSR.Scripts.Models.TCN import TCN


class MockFitGenerator:
    def eval(self):
        pass

    def to(self, device):
        pass

    def get_sample(self, shape):
        return torch.zeros(shape)

    def forward_conditional(self, past, current, sig_inds):
        """AFAIK, sig_inds is a list of feature indices, and we want to mask all BUT those features"""
        sig_inds_comp = list(set(range(past.shape[-2]))-set(sig_inds))
        if len(current.shape) == 1:
            current = current.unsqueeze(0)
        full_sample = current.clone()
        full_sample[:, sig_inds_comp] = self.get_sample([full_sample.shape[0], len(sig_inds_comp)])
        return full_sample, None


def generate_data(experiment, num_samples, batch_size, trainer_type):
    assert trainer_type in ("FIT", "TSR")

    num_features = experiment['num_features']
    num_timesteps = experiment['num_timesteps']

    data = experiment['noise']((num_samples, num_features, num_timesteps))
    signal = experiment['signal']((num_samples, num_features, num_timesteps))
    labels = np.random.choice((0, 1), num_samples)

    ground_truth_importance = np.zeros((num_samples, num_features, num_timesteps))
    labels_in_time = np.zeros((num_samples, num_timesteps))

    for i in range(num_samples):
        if labels[i]:
            ground_truth_importance[i] = experiment['importance_map']((num_features, num_timesteps))
            first_important_timestep = np.min(np.argwhere(ground_truth_importance[i])[:, 1])
            labels_in_time[i, first_important_timestep:] = 1

    data[np.argwhere(ground_truth_importance)] = signal[np.argwhere(ground_truth_importance)]

    data = torch.from_numpy(data).float() if trainer_type == "FIT" else torch.from_numpy(data).double()
    tsr_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels)), batch_size=batch_size)
    fit_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels_in_time)), batch_size=batch_size)
    return tsr_loader, fit_loader, ground_truth_importance


def get_fit_attributions(model, train_loader, test_loader, num_features, name, train, mock_generator):
    if mock_generator:
        generator = MockFitGenerator()
        FIT = FITExplainer(model, generator=generator, ft_dim_last=False)
    else:
        generator = JointFeatureGenerator(num_features, latent_size=100, data=name)
        if train:
            FIT = FITExplainer(model, ft_dim_last=False)
            FIT.fit_generator(generator, train_loader, test_loader, n_epochs=50)
        else:
            generator.load_state_dict(torch.load(f'ckpt/{name}/joint_generator_0.pt'))
            FIT = FITExplainer(model, generator=generator, ft_dim_last=False)

    fit_attributions = []

    for X, y in test_loader:
        fit_attributions.append(FIT.attribute(X, y))

    fit_attributions = np.concatenate(fit_attributions, 0)
    return fit_attributions


def get_tsr_attributions(saliency, test_loader):
    tsr_attributions = []
    for X, y in test_loader:
        tsr_attributions.append(get_tsr_saliency(saliency, X, y, ft_dim_last=False))
    tsr_attributions = np.concatenate(tsr_attributions, 0)
    return tsr_attributions


def run_experiment(experiment, method, trainer_type, train, train_generator, mock_fit_generator=False, reset_metrics_file=False):
    train_samples = 1000
    test_samples = 100
    batch_size = 10

    assert trainer_type in ("TSR", "FIT")

    if mock_fit_generator and method == 'fit':
        print(f"Running {experiment['name']} with mock FIT generator")

    method_name = f"{method}_TrainerType_{trainer_type}"
    model_path = f"Models/TCN/{experiment['name']}_BEST.pkl" if trainer_type == "TSR" else f"ckpt/simulation/{experiment['name']}_0.pt"

    num_features = experiment['num_features']
    num_timesteps = experiment['num_timesteps']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_tsr_loader, train_fit_loader, train_gt = \
        generate_data(experiment, train_samples, batch_size, trainer_type)
    test_tsr_loader, test_fit_loader, test_gt = \
        generate_data(experiment, test_samples, batch_size, trainer_type)

    if trainer_type == "FIT":
        model = StateClassifier(feature_size=num_features, n_state=2, hidden_size=200, rnn='GRU')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        if train:
            train_model_rt(model=model, train_loader=train_fit_loader, valid_loader=test_fit_loader,
                           optimizer=optimizer, n_epochs=50, device=device, experiment=experiment['name'])
        else:
            model.load_state_dict(torch.load(model_path))
    elif trainer_type == "TSR":
        if train:
            num_chans = [5] * (3 - 1) + [num_timesteps]
            model = TCN(num_features, 2, num_chans, 4, 0.1, ft_dim_last=False).to(device)
            train_model(model, "TCN", experiment['name'], torch.nn.CrossEntropyLoss(), train_tsr_loader, test_tsr_loader, device,
                        num_timesteps, num_features, 500, experiment['name'], 0.01)
        model = torch.load(model_path, map_location=device)
    else:
        raise Exception(f"Trainer type {trainer_type} unrecognized")

    if method == 'fit':
        attributions = get_fit_attributions(model, train_tsr_loader, test_tsr_loader, num_features,
                                            experiment['name'], train_generator, mock_fit_generator)
    elif method == 'grad_tsr':
        attributions = get_tsr_attributions(Saliency(model), test_tsr_loader)
    else:
        raise Exception(f"Method {method} unrecognized")

    auc_score = metrics.roc_auc_score(test_gt.flatten(), attributions.flatten())
    aupr_score = metrics.average_precision_score(test_gt.flatten(), attributions.flatten())

    result_path = f'experiment_results/{experiment["name"]}'
    plots_path = result_path + "/plots"
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

    metrics_file = f'{result_path}/metrics.csv'
    if reset_metrics_file and os.path.exists(metrics_file):
        os.remove(metrics_file)

    metrics_file_exists = os.path.exists(metrics_file)
    with open(metrics_file, 'a') as file:
        if not metrics_file_exists:
            file.write(f"{experiment['name']}, AUC, AUPR\n")
        file.write(f"{method_name}, {auc_score}, {aupr_score}\n")

    for i in range(10):
        plotExampleBox(attributions[i], f'{plots_path}/{method_name}_{i}',greyScale=True)
        plotExampleBox(test_gt[i], f'{plots_path}/gt_{i}',greyScale=True)


"""
Experiments need
- 'name': str - identifier
- 'num_features': int
- 'num_timesteps': int
- 'noise': function(shape) - returns np array of size shape of background noise 
- 'signal': function(shape) - returns np array of size shape of 'important' data
- 'importance_map': function(shape) - returns a boolean np array where True values are 'important'
"""


def basic_rare_time(num_features, num_timesteps, noise_mean=0, signal_mean=2, moving=False):
    def get_importance_map(shape):
        assert len(shape) == 2, f"Importance maps are size (num_features, num_timesteps)"
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        if moving:
            imp_ts = np.random.randint(0, num_timesteps)
            importance_map[:, imp_ts] = True
        else:
            imp_ts = num_timesteps // 2
            importance_map[:, imp_ts] = True
        return importance_map

    return {
        'name': f'{"Moving" if moving else ""}GaussianRareTime_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'noise': lambda shape: np.random.normal(noise_mean, 1, shape),
        'signal': lambda shape: np.random.normal(signal_mean, 1, shape),
        'importance_map': get_importance_map
    }


def basic_rare_feature(num_features, num_timesteps, noise_mean=0, signal_mean=2, moving=False):
    def get_importance_map(shape):
        assert len(shape) == 2, f"Importance maps are size (num_features, num_timesteps)"
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        if moving:
            imp_ft = np.random.randint(0, num_features)
            importance_map[imp_ft, :] = True
        else:
            imp_ft = num_features // 2
            importance_map[imp_ft, :] = True
        return importance_map

    return {
        'name': f'{"Moving" if moving else ""}GaussianRareFeature_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'noise': lambda shape: np.random.normal(noise_mean, 1, shape),
        'signal': lambda shape: np.random.normal(signal_mean, 1, shape),
        'importance_map': get_importance_map
    }


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # Change these to run different experiments
    experiments = [basic_rare_time(20, 20, signal_mean=10, moving=True)]
    methods = ['grad_tsr', 'fit']
    trainer_types = ['TSR', 'FIT']
    train = True
    train_generator = True
    mock_fit_generator = False
    reset_metrics_file = True

    for i, experiment in enumerate(experiments):
        for j, trainer_type in enumerate(trainer_types):
            for k, method in enumerate(methods):
                train_now = train and k == 0
                train_generator_now = train_generator and j == 0
                reset_metrics_now = reset_metrics_file and j == 0 and k == 0

                run_experiment(experiment, method, trainer_type, train_now, train_generator_now,
                               mock_fit_generator=mock_fit_generator, reset_metrics_file=reset_metrics_now)
