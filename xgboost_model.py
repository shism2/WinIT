import numpy as np
import torch
from xgboost import XGBRegressor
from sklearn import metrics


def generate_sliding_window_data(inputs, labels, window_size, buffer_size, prediction_window_size):
    batch_size, num_fts, num_ts = inputs.shape
    assert num_ts >= window_size + buffer_size + prediction_window_size
    windows = []
    window_labels = []
    for t in range(num_ts - window_size - buffer_size - prediction_window_size + 1):
        windows.append(inputs[:, :, t:t + window_size])
        # TODO: Figure out what we want here
        new_label = np.max(labels[:, t + window_size + buffer_size:t + window_size + buffer_size + prediction_window_size], axis=1)
        window_labels.append(new_label)
    return np.concatenate(windows), np.concatenate(window_labels)


def train_model(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=0)

    model.fit(
        X_train,
        y_train,
        eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
        early_stopping_rounds=10)

    test_predictions = model.predict(X_test)
    AUC = metrics.roc_auc_score(y_test.flatten(), test_predictions.flatten())
    AUPR = metrics.average_precision_score(y_test.flatten(), test_predictions.flatten())
    print(f'XGB Model AUC: {AUC}, AUPR: {AUPR}')

    return model


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
    def __init__(self, train_loader, test_loader, window_size, buffer_size, prediction_window_size):
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.prediction_window_size = prediction_window_size

        X_train, y_train = generate_sliding_window_data(*loader_to_np(train_loader), self.window_size, self.buffer_size, self.prediction_window_size)
        X_test, y_test = generate_sliding_window_data(*loader_to_np(test_loader), self.window_size, self.buffer_size, self.prediction_window_size)

        self.model = train_model(X_train, y_train, X_test, y_test)

    def __call__(self, inputs):
        # Best we can do is run the model on the last window of input, if the input is long enough
        if inputs.shape[2] >= self.window_size:
            window = inputs[:, :, -self.window_size:].detach().numpy().reshape(inputs.shape[0], -1)
            return torch.from_numpy(self.model.predict(window))
        else:
            return torch.zeros(inputs.shape[0])

    def eval(self):
        return self

    def to(self, device):
        return self

    def train(self):
        pass


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
