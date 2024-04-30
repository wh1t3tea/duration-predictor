import torch


def concordance_cc(y_true, y_pred, mask):
    ccc_metric = 0
    for i in range(y_true.shape[0]):
        mask_ = torch.sum(mask[i], dim=-1)

        mean_true = torch.mean(y_true[i][:mask_])
        mean_pred = torch.mean(y_pred[i][:mask_])

        var_true = torch.var(y_true[i][:mask_])
        var_pred = torch.var(y_pred[i][:mask_])
        pair = torch.stack((y_true[i][:mask_], y_pred[i][:mask_]), dim=1).T

        cov = torch.cov(pair)[0, 1]

        ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) * 2)
        ccc_metric += ccc

    return ccc_metric / y_true.shape[0]


def masked_mae(y_true, y_pred, mask):
    masked_mae_sum = torch.sum(torch.abs(y_pred - y_true) * mask, dim=-1)
    non_padding_count = torch.sum(mask, dim=-1)
    mae = masked_mae_sum / non_padding_count
    return torch.mean(mae)


class ThresholdMaskedMAE:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.idx2weight = sorted([value for value in thresholds.values()], reverse=True)

    def mask_thresholds(self, sequence):
        masks = [torch.sum(sequence[sequence <= thresh]) for thresh, value in self.thresholds.items()]
        return masks

    def __call__(self, y_true, y_pred, mask):
        masked_mae_sum = torch.sum(torch.abs(y_pred - y_true) * mask, dim=-1)
        masks = self.mask_thresholds(masked_mae_sum)
        metric = sum([(masks[idx] - masks[idx - 1]) * self.idx2weight[idx] for idx in range(len(masks))])
        return torch.tensor(metric, dtype=torch.float32) / y_true.shape[0]
