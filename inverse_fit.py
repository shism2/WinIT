import numpy as np
import torch


def inverse_fit_attribute(x, model, activation=None, ft_dim_last=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    def model_predict(x):
        if ft_dim_last:
            x = x.permute(0, 2, 1)
        if activation is not None:
            return activation(model(x))
        else:
            return model(x)

    model.eval()

    if ft_dim_last:
        x = x.permute(0, 2, 1)

    batch_size, num_features, num_timesteps = x.shape
    score = torch.zeros(x.shape)

    for t in range(num_timesteps):
        p_y = model_predict(x[:, :, :t + 1])
        for f in range(num_features):
            x_hat = x[:, :, :t + 1].clone()
            x_hat[:, f, -1] = torch.mean(x)
            p_y_hat = model_predict(x_hat)
            div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
            score[:, f, t] = 2. / (1 + torch.exp(-5 * div)) - 1

    if ft_dim_last:
        score = score.permute(0, 2, 1)

    return score.detach().numpy()


def wfit_attribute(x, model, N, activation=None, ft_dim_last=False, single_label=False, collapse=False, inverse=False, generator=None):
    assert not single_label or not collapse

    if N == 1 and inverse:
        return inverse_fit_attribute(x, model, activation, ft_dim_last)

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    def model_predict(x):
        if ft_dim_last:
            x = x.permute(0, 2, 1)
        if activation is not None:
            return activation(model(x))
        else:
            return model(x)

    model.eval()

    if ft_dim_last:
        x = x.permute(0, 2, 1)

    batch_size, num_features, num_timesteps = x.shape
    scores = []

    start = num_timesteps - 1 if single_label else 0

    for t in range(start, num_timesteps):
        window_size = min(t, N)
        score = torch.zeros(batch_size, num_features, window_size)

        if t == 0:
            if ft_dim_last:
                score = score.permute(0, 2, 1)
            scores.append(score.detach().numpy())
            continue

        p_y = model_predict(x[:, :, :t + 1])
        p_tm1 = model_predict(x[:, :, :t])

        for n in range(window_size):
            for f in range(num_features):
                x_hat = x[:, :, :t + 1].clone()

                masked_f = [f] if inverse else list(set(range(num_features)) - {f})
                unmasked_f = [f] if not inverse else list(set(range(num_features)) - {f})
                if generator is None:
                    # Carry forward
                    x_hat[:, masked_f, t - n:t + 1] = x_hat[:, masked_f, t - n - 1, None]
                else:
                    for mask_t in range(t - n, t + 1):
                        x_hat[:, :, mask_t], _ = generator.forward_conditional(x_hat[:, :, :mask_t].float(), x_hat[:, :, mask_t].float(), unmasked_f)

                p_y_hat = model_predict(x_hat)

                if inverse:
                    div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
                else:
                    div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_tm1), p_y), -1) - \
                             torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)

                acc_score = 2. / (1 + torch.exp(-5 * div)) - 1
                score[:, f, window_size - n - 1] = acc_score - score[:, f, window_size - n:].sum(axis=-1) if n > 0 else acc_score

        if ft_dim_last:
            score = score.permute(0, 2, 1)

        scores.append(score.detach().numpy())

    if single_label:
        scores = scores[0]
    elif collapse:
        scores = absmax_collapse(scores)

    return scores


def absmax_collapse(attributions):
    combined_attrs = np.zeros((attributions[0].shape[0], attributions[0].shape[1], len(attributions)))
    for pred in range(len(attributions)):
        attributions[pred] = np.nan_to_num(attributions[pred])
        start = pred - attributions[pred].shape[-1] + 1
        end = pred + 1
        combined_attrs[:, :, start:end] = np.where(np.abs(combined_attrs[:, :, start:end]) > np.abs(attributions[pred]),
                                                   combined_attrs[:, :, start:end], attributions[pred])
    return combined_attrs
