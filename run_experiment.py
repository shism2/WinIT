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
from TSR.Scripts.Models.LSTMWithInputCellAttention import LSTMWithInputCellAttention
from TSR.Scripts.Models.TCN import TCN
from inverse_fit import inverse_fit_attribute
from xgboost_model import XGBPytorchStub


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


def generate_data(experiment, num_samples, batch_size, model_type):
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

    data = torch.from_numpy(data).float() if model_type == "FIT" else torch.from_numpy(data).double()
    tsr_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels)), batch_size=batch_size)
    fit_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels_in_time)), batch_size=batch_size)
    return (fit_loader if model_type in ["FIT", "XGB"] else tsr_loader), data, ground_truth_importance


def get_fit_attributions(model, train_loader, test_loader, num_features, name, train, mock_generator, activation=lambda x: x):
    if mock_generator:
        generator = MockFitGenerator()
        FIT = FITExplainer(model, generator=generator, ft_dim_last=False, activation=activation)
    else:
        generator = JointFeatureGenerator(num_features, latent_size=100, data=name)
        if train:
            FIT = FITExplainer(model, ft_dim_last=False, activation=activation)
            FIT.fit_generator(generator, train_loader, test_loader, n_epochs=300)
        else:
            generator.load_state_dict(torch.load(f'ckpt/{name}/joint_generator_0.pt'))
            FIT = FITExplainer(model, generator=generator, ft_dim_last=False, activation=activation)

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


def get_inverse_fit_attributions(model, test_loader, model_type):
    activation = torch.nn.Softmax(-1) if model_type == "FIT" else None
    ifit_attributions = []
    for x, _ in test_loader:
        ifit_attributions.append(inverse_fit_attribute(x, model, activation))
    ifit_attributions = np.concatenate(ifit_attributions, 0)
    return ifit_attributions


def run_experiment(experiment, method, model_type, train, train_generator, reset_metrics_file=False):
    train_samples = 500
    test_samples = 50
    batch_size = 10

    print(f"Running {experiment['name']} with {method} and trainer type {model_type}")

    method_name = f"{method}_TrainerType_{model_type}"
    model_path = f"ckpt/simulation/{experiment['name']}_0.pt" if model_type == "FIT" else f"Models/{model_type}/{experiment['name']}_BEST.pkl"

    num_features = experiment['num_features']
    num_timesteps = experiment['num_timesteps']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mock_fit_generator = False
    if method == 'mock_fit':
        method = 'fit'
        mock_fit_generator = True

    train_loader, train_data, train_gt = experiment['generate_data'](train_samples, batch_size, model_type) \
        if 'generate_data' in experiment.keys() else generate_data(experiment, train_samples, batch_size, model_type)
    test_loader, test_data, test_gt = experiment['generate_data'](test_samples, batch_size, model_type) \
        if 'generate_data' in experiment.keys() else generate_data(experiment, test_samples, batch_size, model_type)

    if model_type == "FIT":
        model = StateClassifier(feature_size=num_features, n_state=2, hidden_size=200, rnn='GRU')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        if train:
            train_model_rt(model=model, train_loader=train_loader, valid_loader=test_loader,
                           optimizer=optimizer, n_epochs=5, device=device, experiment=experiment['name'])
        else:
            model.load_state_dict(torch.load(model_path))
    elif model_type == "TCN":
        if train:
            num_chans = [5] * (3 - 1) + [num_timesteps]
            model = TCN(num_features, 2, num_chans, 4, 0.1, ft_dim_last=False).to(device)
            train_model(model, "TCN", experiment['name'], torch.nn.CrossEntropyLoss(), train_loader, test_loader, device,
                        num_timesteps, num_features, 50, experiment['name'], 0.01)
        model = torch.load(model_path, map_location=device)
    elif model_type == "LSTMWithInputCellAttention":
        if train:
            model = LSTMWithInputCellAttention(num_features, 5, 2, 0.1, 10, 50, ft_dim_last=False).to(device)
            train_model(model, "LSTMWithInputCellAttention", experiment['name'], torch.nn.CrossEntropyLoss(),
                        train_loader, test_loader, device,
                        num_timesteps, num_features, 50, experiment['name'], 0.01)
        model = torch.load(model_path, map_location=device)
    elif model_type == "XGB":
        model = XGBPytorchStub(train_loader, test_loader, 10, 0, 1, f'ckpt/{experiment["name"]}/XGB.model', True)
    else:
        raise Exception(f"Trainer type {model_type} unrecognized")

    if method == 'fit':
        attributions = get_fit_attributions(model, train_loader, test_loader, num_features,
                                            experiment['name'], train_generator, mock_fit_generator,
                                            torch.nn.Softmax(-1) if model_type == "FIT" else lambda x: x)
    elif method == 'grad_tsr':
        attributions = get_tsr_attributions(Saliency(model), test_loader)
    elif method == 'ifit':
        attributions = get_inverse_fit_attributions(model, test_loader, model_type)
    else:
        raise Exception(f"Method {method} unrecognized")

    attributions = np.nan_to_num(attributions)

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

    print(f"{experiment['name']}: AUC {auc_score}, AUPR {aupr_score}\n")

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
OR
- 'generate_data': function(num_samples, batch_size, model_type) - returns (loader, data, ground_truth_importance)
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


def basic_rare_time(num_features, num_timesteps, generation_type="Gaussian", noise_mean=0, signal_mean=2, moving=False,
                    posneg=False):
    def generate_raretime_sample(shape, label):
        assert len(shape) == 2
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        data = generate_TSR_sample(num_features, num_timesteps, generation_type) + noise_mean

        if label or posneg:
            if moving:
                imp_ts = np.random.randint(0, num_timesteps)
            else:
                imp_ts = num_timesteps // 2
            importance_map[:, imp_ts] = True
            data[:, imp_ts] += (signal_mean - noise_mean) * 1 if label or not posneg else -1

        return data, importance_map

    return {
        'name': f'{"Moving" if moving else ""}{generation_type}RareTime{"PosNeg" if posneg else ""}_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_sample': generate_raretime_sample
    }


def basic_rare_feature(num_features, num_timesteps, generation_type="Gaussian", noise_mean=0, signal_mean=2, moving=False,
                       posneg=False):
    def generate_rarefeature_sample(shape, label):
        assert len(shape) == 2
        num_features, num_timesteps = shape

        importance_map = np.zeros(shape, dtype=bool)
        data = generate_TSR_sample(num_features, num_timesteps, generation_type) + noise_mean

        if label or posneg:
            if moving:
                imp_ft = np.random.randint(0, num_features)
            else:
                imp_ft = num_features // 2
            importance_map[imp_ft, :] = True
            data[imp_ft, :] += (signal_mean - noise_mean) * 1 if label or not posneg else -1

        return data, importance_map

    return {
        'name': f'{"Moving" if moving else ""}{generation_type}RareFeature{"PosNeg" if posneg else ""}_{noise_mean}_{signal_mean}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_sample': generate_rarefeature_sample
    }


def and_experiment(num_features, num_timesteps, noise, signal):
    def generate_data(num_samples, batch_size, model_type):
        assert model_type == 'FIT' or model_type == 'XGB'
        data = np.random.choice([noise, signal], (num_samples, num_features, num_timesteps))
        labels = np.all(data == signal, axis=1)
        gt_imp = np.repeat(labels[:, :, None], num_features, axis=2).swapaxes(1, 2)
        labels[:, 1:num_timesteps] = labels[:, 0:num_timesteps - 1]
        loader = DataLoader(TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels)), batch_size=batch_size)
        return loader, data, gt_imp

    return {
        'name': f'AndExperiment_{noise}_{signal}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_data': generate_data
    }


def delay_experiment(num_features, num_timesteps, noise, signal, delay_amount=1):
    def generate_data(num_samples, batch_size, model_type):
        assert model_type == 'FIT' or model_type == 'XGB'
        data = np.full((num_samples, num_features, num_timesteps), fill_value=noise)
        for sample in range(num_samples):
            imp_ft = np.random.randint(0, num_features)
            imp_ts = np.random.randint(0, num_timesteps)
            data[sample, imp_ft, imp_ts] = signal

        labels = np.zeros((num_samples, num_timesteps))
        labels[:, 1 + delay_amount:num_timesteps] = np.any(data == signal, axis=1)[:, 0:num_timesteps - 1 - delay_amount]

        gt_imp = data == signal

        loader = DataLoader(TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels)),
                            batch_size=batch_size)

        return loader, data, gt_imp

    return {
        'name': f'DelayExperiment_{noise}_{signal}',
        'num_features': num_features,
        'num_timesteps': num_timesteps,
        'generate_data': generate_data
    }


if __name__ == '__main__':
    # Change these to run different experiments
    experiments = [delay_experiment(2, 50, 0, 1)]
    methods = ['fit']
    model_types = ['XGB']
    train = False
    train_generator = False
    reset_metrics_file = False

    for i, experiment in enumerate(experiments):
        for j, model_type in enumerate(model_types):
            for k, method in enumerate(methods):
                np.random.seed(0)
                torch.manual_seed(0)

                train_now = train and k == 0
                train_generator_now = train_generator and j == 0
                reset_metrics_now = reset_metrics_file and j == 0 and k == 0

                run_experiment(experiment, method, model_type, train_now, train_generator_now, reset_metrics_file=reset_metrics_now)
