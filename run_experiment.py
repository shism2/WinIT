import pathlib

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


def generate_data(num_samples, num_features, num_timesteps, batch_size, trainer_type):
    assert trainer_type in ("FIT", "TSR")

    data = np.random.normal(0, 1, (num_samples, num_features, num_timesteps))
    labels = np.random.choice((0, 1), num_samples)
    imp_ts = np.random.randint(0, num_timesteps, num_samples)
    imp_ft = np.random.randint(0, num_features, num_samples)

    imp_data = labels * 100
    ground_truth_importance = np.zeros((num_samples, num_features, num_timesteps))
    labels_in_time = np.zeros((num_samples, num_timesteps))

    for i in range(num_samples):
        data[i, imp_ft[i], imp_ts[i]] += imp_data[i]
        ground_truth_importance[i, imp_ft[i], imp_ts[i]] = labels[i]
        labels_in_time[i, imp_ts[i]:] = labels[i]

    data = torch.from_numpy(data).float() if trainer_type == "FIT" else torch.from_numpy(data).double()
    tsr_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels)), batch_size=batch_size)
    fit_loader = DataLoader(TensorDataset(data, torch.from_numpy(labels_in_time)), batch_size=batch_size)
    return tsr_loader, fit_loader, ground_truth_importance


def get_fit_attributions(model, train_loader, test_loader, num_features, train, mock_generator):
    if mock_generator:
        generator = MockFitGenerator()
        FIT = FITExplainer(model, generator=generator, ft_dim_last=False)
    else:
        generator = JointFeatureGenerator(num_features, latent_size=100, data='top_level_experiment')
        if train:
            FIT = FITExplainer(model, ft_dim_last=False)
            FIT.fit_generator(generator, train_loader, test_loader, n_epochs=50)
        else:
            generator.load_state_dict(torch.load('ckpt/top_level_experiment/joint_generator_0.pt'))
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


def main():
    # Modify these
    train = True
    trainer_type = "FIT"

    # model_path = "Models/TCN/top_level_experiment_BEST.pkl"
    model_path = "ckpt/simulation/top_level_experiment_0.pt"

    methods = ['grad_tsr', 'fit']

    mock_fit_generator = False

    train_samples = 1000
    test_samples = 100
    num_features = 3
    num_timesteps = 50
    batch_size = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_tsr_loader, train_fit_loader, train_gt = \
        generate_data(train_samples, num_features, num_timesteps, batch_size, trainer_type)
    test_tsr_loader, test_fit_loader, test_gt = \
        generate_data(test_samples, num_features, num_timesteps, batch_size, trainer_type)

    if trainer_type == "FIT":
        model = StateClassifier(feature_size=num_features, n_state=2, hidden_size=200, rnn='GRU')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        if train:
            model_path = train_model_rt(model=model, train_loader=train_fit_loader, valid_loader=test_fit_loader,
                                    optimizer=optimizer, n_epochs=50, device=device, experiment='top_level_experiment')
            print(f"Model saved in {model_path}")
        else:
            model.load_state_dict(torch.load(model_path))
    elif trainer_type == "TSR":
        if train:
            num_chans = [5] * (3 - 1) + [num_timesteps]
            model = TCN(num_features, 2, num_chans, 4, 0.1, ft_dim_last=False).to(device)
            model_path = train_model(model, "TCN", "top_level_experiment", torch.nn.CrossEntropyLoss(),
                                     train_tsr_loader, test_tsr_loader, device, num_timesteps, num_features, 500,
                                     'spike', 0.01)
            print(f"Model saved in {model_path}")
        model = torch.load(model_path, map_location=device)

    for method in methods:
        if method == 'fit':
            attributions = get_fit_attributions(model, train_tsr_loader, test_tsr_loader, num_features, train, mock_fit_generator)
        elif method == 'grad_tsr':
            attributions = get_tsr_attributions(Saliency(model), test_tsr_loader)
        else:
            raise Exception(f"Method {method} unrecognized")

        auc_score = metrics.roc_auc_score(test_gt.flatten(), attributions.flatten())
        aupr_score = metrics.average_precision_score(test_gt.flatten(), attributions.flatten())

        print(method, 'auc:', auc_score, ' aupr:', aupr_score)

        pathlib.Path('experiment_plots').mkdir(parents=True, exist_ok=True)

        for i in range(10):
            plotExampleBox(attributions[i], f'experiment_plots/{method}_{i}',greyScale=True)
            plotExampleBox(test_gt[i], f'experiment_plots/gt_{i}',greyScale=True)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
