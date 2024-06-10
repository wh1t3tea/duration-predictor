import torch
from torch import nn


class ConcordanceCC(nn.Module):
    def forward(self, y_true, y_pred, mask):
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
    mae = nn.L1Loss()
    y_true, y_pred = y_true, y_pred
    mask_ = 0
    for i in range(len(y_pred)):
        mask_ += mae(y_pred[i][:torch.sum(mask[i], dim=-1)], y_true[i][:torch.sum(mask[i], dim=-1)])
    return mask_ / y_pred.shape[0]


class ThresholdMaskedMAE:
    def __init__(self, thresholds, mode="hard"):
        self.mode = mode
        self.thresholds = thresholds
        if thresholds is not None:
            self.idx2weight = sorted([value for value in thresholds.values()], reverse=True)

    def mask_thresholds(self, sequence, ):
        masks = []
        for thresh in self.thresholds.keys():
            mask = torch.where(sequence <= thresh, torch.tensor(1), torch.tensor(0))
            if len(masks) == 0:
                masks.append(torch.sum(mask))
            else:
                masks.append(torch.sum(mask) - sum(masks))
        return masks

    def __call__(self, y_true, y_pred, mask):
        masked_mae = torch.abs(y_pred - y_true) * mask
        if self.mode == "hard":
            weighted_m = masked_mae.pow(-1)
            metric = torch.where(weighted_m >= 1, 1, weighted_m)
            metric = torch.sum(metric) / mask.shape[-1]
            return metric / y_true.shape[0]
        masks = self.mask_thresholds(masked_mae)
        metric = torch.sum(torch.tensor([masks[idx] * self.idx2weight[idx] for idx in range(len(masks))]), dim=-1) / \
                 mask.shape[-1]
        return torch.tensor(metric, dtype=torch.float32) / y_true.shape[0]
