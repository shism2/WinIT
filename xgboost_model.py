import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn import metrics


def generate_sliding_window_data(inputs, labels, window_size, buffer_size, prediction_window_size):
    batch_size, num_fts, num_ts = inputs.shape
    assert num_ts >= window_size + buffer_size + prediction_window_size
    windows = []
    window_labels = []
    for t in range(num_ts - window_size - buffer_size - prediction_window_size + 1):
        windows.append(inputs[:, :, t:t + window_size])
        # TODO: Figure out what we want here
        if len(labels.shape) > 1:
            new_label = np.max(labels[:, t + window_size + buffer_size:t + window_size + buffer_size + prediction_window_size], axis=1)
        else:
            new_label = labels[:]
        window_labels.append(new_label)
    return np.concatenate(windows), np.concatenate(window_labels)


def get_model(X_train, y_train, X_test, y_test, filename, train):
    _, num_fts, num_ts = X_train.shape

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = XGBClassifier()

    if train:
        model.fit(
            X_train,
            y_train,
            eval_metric="rmse",
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True,
            early_stopping_rounds=10)
        model.save_model(filename)
    else:
        model.load_model(filename)

    test_predictions = model.predict(X_test)
    AUC = metrics.roc_auc_score(y_test.flatten(), test_predictions.flatten())
    AUPR = metrics.average_precision_score(y_test.flatten(), test_predictions.flatten())
    print(f'XGB Model AUC: {AUC}, AUPR: {AUPR}')

    np.set_printoptions(precision=3)
    print("XGB feature importance matrix:")
    print(get_importance_matrix(model, num_fts, num_ts))

    return model


def get_importance_matrix(model, num_fts, window_size):
    # Get a matrix of feature importances
    ft_imp_dict = model.get_booster().get_score(importance_type="gain")
    ft_imp_matrix = np.zeros((num_fts, window_size))
    for i in range(num_fts):
        for j in range(window_size):
            key = f"f{i * window_size + j}"
            if key in ft_imp_dict.keys():
                ft_imp_matrix[i, j] = ft_imp_dict[f"f{i * window_size + j}"]
            else:
                ft_imp_matrix[i, j] = 0
    return ft_imp_matrix / np.max(ft_imp_matrix)


def simple_experiment_data(shape, imp_ft, noise_mean=0, signal_mean=10):
    batch_size, num_fts, num_ts = shape
    inputs = np.random.normal(noise_mean, 1, shape)
    labels = np.zeros((batch_size, num_ts))

    for i in range(batch_size):
        transition_ts = np.random.randint(0, num_ts)
        inputs[i, imp_ft, transition_ts:] += signal_mean
        labels[i, transition_ts:] = 1

    return inputs, labels


def loader_to_np(loader):
    X_batches = []
    y_batches = []
    for X, y in loader:
        X_batches.append(X)
        y_batches.append(y)
    return np.concatenate(X_batches), np.concatenate(y_batches)


class XGBPytorchStub():
    def __init__(self, train_loader, test_loader, window_size, buffer_size, prediction_window_size, filename, train):
        self.num_fts = next(iter(train_loader))[0].shape[1]
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.prediction_window_size = prediction_window_size

        X_train, y_train = generate_sliding_window_data(*loader_to_np(train_loader), self.window_size, self.buffer_size, self.prediction_window_size)
        X_test, y_test = generate_sliding_window_data(*loader_to_np(test_loader), self.window_size, self.buffer_size, self.prediction_window_size)

        self.model = get_model(X_train, y_train, X_test, y_test, filename, train)

    def __call__(self, inputs):
        # Best we can do is run the model on the last window of input, if the input is long enough
        if inputs.shape[2] >= self.window_size:
            window = inputs[:, :, -self.window_size:].cpu().detach().numpy().reshape(inputs.shape[0], -1)
            prediction = torch.from_numpy(self.model.predict_proba(window))
            return prediction
        else:
            return torch.zeros(inputs.shape[0], 2)

    def eval(self):
        return self

    def to(self, device):
        return self

    def train(self):
        pass

    @property
    def imp_matrix(self):
        return get_importance_matrix(self.model, self.num_fts, self.window_size)


def main():
    train_size = 1000
    test_size = 100

    num_ts = 100
    num_fts = 3
    imp_ft = 1
    window_size = 5
    buffer_size = 0
    prediction_window_size = 1

    np.random.seed(0)
    X_train, y_train = simple_experiment_data((train_size, num_fts, num_ts), imp_ft)
    X_test, y_test = simple_experiment_data((test_size, num_fts, num_ts), imp_ft)

    X_train, y_train = generate_sliding_window_data(X_train, y_train, window_size, buffer_size, prediction_window_size)
    X_test, y_test = generate_sliding_window_data(X_test, y_test, window_size, buffer_size, prediction_window_size)

    model = train_model(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
