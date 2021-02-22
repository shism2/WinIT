import pathlib
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn import metrics
from captum.attr import Saliency
import timesynth as ts

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

    data = np.zeros((num_samples, num_features, num_timesteps))
    ground_truth_importance = np.zeros((num_samples, num_features, num_timesteps))
    labels = np.random.choice((0, 1), num_samples)
    labels_in_time = np.zeros((num_samples, num_timesteps))

    for i in range(num_samples):
        data[i], ground_truth_importance[i] = experiment['generate_sample']((num_features, num_timesteps), labels[i])
        if np.any(ground_truth_importance[i]):
            first_important_timestep = np.min(np.argwhere(ground_truth_importance[i])[:, 1])
            labels_in_time[i, first_important_timestep:] = 1

    data = torch.from_numpy(data).float() if trainer_type == "FIT" else torch.from_numpy(data).double()
    tsr_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels)), batch_size=batch_size)
    fit_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels_in_time)), batch_size=batch_size)
    return tsr_loader, fit_loader, data, ground_truth_importance


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


def run_experiment(experiment, method, trainer_type, train, train_generator, reset_metrics_file=False):
    train_samples = 500
    test_samples = 50
    batch_size = 10

    assert trainer_type in ("TSR", "FIT")

    print(f"Running {experiment['name']} with {method} and trainer type {trainer_type}")

    method_name = f"{method}_TrainerType_{trainer_type}"
    model_path = f"Models/TCN/{experiment['name']}_BEST.pkl" if trainer_type == "TSR" else f"ckpt/simulation/{experiment['name']}_0.pt"

    num_features = experiment['num_features']
    num_timesteps = experiment['num_timesteps']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_fit_generator = False
    if method == 'mock_fit':
        method = 'fit'
        mock_fit_generator = True

    train_tsr_loader, train_fit_loader, train_data, train_gt = \
        generate_data(experiment, train_samples, batch_size, trainer_type)
    test_tsr_loader, test_fit_loader, test_data, test_gt = \
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
        plotExampleBox(test_data[i], f'{plots_path}/data_{i}',greyScale=True)


"""
Experiments need
- 'name': str - identifier
- 'num_features': int
- 'num_timesteps': int
- 'generate_sample': function(shape, label) - returns (sample, boolean importance map)
"""


def generate_TSR_sample(num_features, num_timesteps, generation_type, sampler="irregular", has_noise=False):
    if generation_type == "Gaussian":
        return np.random.normal(0, 1, (num_features, num_timesteps))

    time_sampler = ts.TimeSampler(stop_time=20)
    sample = np.zeros([num_features, num_timesteps])

    if sampler == "regular":
        time = time_sampler.sample_regular_time(num_points=num_timesteps * 2, keep_percentage=50)
    else:
        time = time_sampler.sample_irregular_time(num_points=num_timesteps * 2, keep_percentage=50)

    for i in range(num_features):
        if generation_type == "Harmonic":
            signal = ts.signals.Sinusoidal(frequency=2.0)

        elif generation_type == "GaussianProcess":
            signal = ts.signals.GaussianProcess(kernel="Matern", nu=3. / 2)

        elif generation_type == "PseudoPeriodic":
            signal = ts.signals.PseudoPeriodic(frequency=2.0, freqSD=0.01, ampSD=0.5)

        elif generation_type == "AutoRegressive":
            signal = ts.signals.AutoRegressive(ar_param=[0.9])

        elif generation_type == "CAR":
            signal = ts.signals.CAR(ar_param=0.9, sigma=0.01)

        elif generation_type == "NARMA":
            signal = ts.signals.NARMA(order=10)

        if has_noise:
            noise = ts.noise.GaussianNoise(std=0.3)
            timeseries = ts.TimeSeries(signal, noise_generator=noise)
        else:
            timeseries = ts.TimeSeries(signal)

        feature, signals, errors = timeseries.sample(time)
        sample[i, :] = feature
    return sample


def basic_spike(num_features, num_timesteps, generation_type="Gaussian", noise_mean=0, signal_mean=2):
    def generate_spike_sample(shape, label):
        assert len(shape) == 2
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        data = generate_TSR_sample(num_features, num_timesteps, generation_type) + noise_mean

        if label:
            imp_ts = np.random.randint(0, num_timesteps)
            imp_ft = np.random.randint(0, num_features)
            importance_map[imp_ft, imp_ts] = True
            data[imp_ft, imp_ts] += signal_mean - noise_mean

        return data, importance_map

    return {
        'name': f'Moving{generation_type}Spike{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_sample': generate_spike_sample
    }


def basic_rare_time(num_features, num_timesteps, generation_type="Gaussian", noise_mean=0, signal_mean=2, moving=False):
    def generate_raretime_sample(shape, label):
        assert len(shape) == 2
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        data = generate_TSR_sample(num_features, num_timesteps, generation_type) + noise_mean

        if label:
            if moving:
                imp_ts = np.random.randint(0, num_timesteps)
            else:
                imp_ts = num_timesteps // 2
            importance_map[:, imp_ts] = True
            data[:, imp_ts] += signal_mean - noise_mean

        return data, importance_map

    return {
        'name': f'{"Moving" if moving else ""}{generation_type}RareTime_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_sample': generate_raretime_sample
    }


def basic_rare_feature(num_features, num_timesteps, generation_type="Gaussian", noise_mean=0, signal_mean=2, moving=False):
    def generate_rarefeature_sample(shape, label):
        assert len(shape) == 2
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        data = generate_TSR_sample(num_features, num_timesteps, generation_type) + noise_mean

        if label:
            if moving:
                imp_ft = np.random.randint(0, num_features)
            else:
                imp_ft = num_features // 2
            importance_map[imp_ft, :] = True
            data[imp_ft, :] += signal_mean - noise_mean

        return data, importance_map

    return {
        'name': f'{"Moving" if moving else ""}{generation_type}RareFeature_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_sample': generate_rarefeature_sample
    }


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # Change these to run different experiments
    generation_types = ["Gaussian", "AutoRegressive", "CAR", "GaussianProcess", "Harmonic", "NARMA", "PseudoPeriodic"]
    experiments = []
    for generation_type in generation_types:
        experiments.append(basic_rare_feature(20, 20, generation_type=generation_type, moving=True, signal_mean=10))
    methods = ['grad_tsr', 'mock_fit', 'fit']
    trainer_types = ['TSR', 'FIT']
    train = True
    train_generator = True
    reset_metrics_file = True

    for i, experiment in enumerate(experiments):
        for j, trainer_type in enumerate(trainer_types):
            for k, method in enumerate(methods):
                train_now = train and k == 0
                train_generator_now = train_generator and j == 0
                reset_metrics_now = reset_metrics_file and j == 0 and k == 0

                run_experiment(experiment, method, trainer_type, train_now, train_generator_now, reset_metrics_file=reset_metrics_now)