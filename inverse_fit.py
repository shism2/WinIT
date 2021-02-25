import numpy as np
import torch


def inverse_fit_attribute(x, model, activation=None):
    def model_predict(x):
        if activation is not None:
            return activation(model(x))
        else:
            return model(x)

    model.eval()

    batch_size, num_features, num_timesteps = x.shape
    score = torch.zeros(x.shape)

    for t in range(num_timesteps):
        p_y = model_predict(x[:, :, :t + 1])
        for f in range(num_features):
            x_hat = x[:, :, :t + 1].clone()
            x_hat[:, f, -1] = x[0, 0, 0]
            p_y_hat = model_predict(x_hat)
            div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
            score[:, f, t] = 2. / (1 + torch.exp(-5 * div)) - 1
    return score.detach().numpy()
