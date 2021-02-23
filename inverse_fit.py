import torch


def inverse_fit_attribute(x, model):
    batch_size, num_features, num_timesteps = x.shape
    score = torch.zeros(x.shape)

    for t in range(num_timesteps):
        p_y = model(x[:, :, :t + 1])
        for f in range(num_features):
            x_hat = x[:, :, :t + 1].clone()
            x_hat[:, f, -1] = 0
            p_y_hat = model(x_hat)
            div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
            score[:, f, t] = 2. / (1 + torch.exp(-5 * div)) - 1
    return score.detach().numpy()
