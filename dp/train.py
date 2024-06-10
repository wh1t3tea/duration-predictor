import json
import torch
from box import Box
from model import DurationPredictor
from dataset import PhoneDataset
from metrics import masked_mae, ConcordanceCC
from loss import masked_mse_loss
from torch import optim
from torch.optim import lr_scheduler
import argparse
from torch.utils.data import DataLoader
from logs import setup_train_logging
from metrics import ThresholdMaskedMAE


def train(arg):

    arg = json.load(open(arg.config))
    arg = Box(arg)

    logger = setup_train_logging(arg.log_file)
    device = arg.device

    train_set = PhoneDataset(arg.root_dir, arg.train_files)
    val_set = PhoneDataset(arg.root_dir, arg.val_files)
    test_set = PhoneDataset(arg.root_dir, arg.test_files)

    train_loader = DataLoader(train_set, batch_size=arg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=arg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

    model = DurationPredictor(
        vocab_size=len(train_set.vocab),
        in_channels=arg.embedding_size,
        filter_channels=arg.filter_size,
        hidden_dim=arg.hidden,
        speaker_emb_size=arg.speaker_embedding_size,
        n_convs=arg.n_conv_layers
    ).to(device)

    loss_fn = masked_mse_loss

    assert arg.optim in [None, "adam", "adamw"]

    if arg.optim == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                                arg.lr,
                                weight_decay=arg.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               arg.lr,
                               weight_decay=arg.weight_decay)

    assert arg.scheduler in [None, "exponential", "cosine", "step"]

    if arg.scheduler == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                               gamma=arg.decay)
    elif arg.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=arg.step_decay,
                                        gamma=arg.decay)
    else:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=arg.step_decay,
                                                             eta_min=1e-5)

    concordance_cc = ConcordanceCC()
    mae = masked_mae
    mae_thresh = ThresholdMaskedMAE(None, mode="hard")

    metrics = {
        "train":
            {
                "mae": [],
                "ccc": [],
                "mae_thresh": []
            },
        "validation":
            {
                "mae": [],
                "ccc": [],
                "mae_thresh": []
            },
        "test":
            {
                "mae": [],
                "ccc": [],
                "mae_thresh": []
            }
    }
    loss = {
        "train": [],
        "validation": [],
        "test": []
    }

    logger.info(msg="Training starts:")

    test_loss = 0

    for epoch in range(arg.max_epochs):
        epoch_loss = epoch_mae = epoch_ccc = epoch_weighted_mae = 0
        stage = "train"
        model.train()
        for batch in train_loader:
            tokens, lengths, mask, sp_e = batch["tokens"], batch["lengths"], batch["mask"], batch["sp_e"]
            tokens, lengths, mask, sp_e = tokens.to(device), lengths.to(device), mask.to(device), sp_e.to(device)

            optimizer.zero_grad()
            predicted_durations = model(tokens, sp_e)
            lengths = lengths.float()

            results = [lengths,
                       predicted_durations.squeeze(-1),
                       mask]

            train_loss = loss_fn(*results)
            train_loss.backward()
            epoch_loss += train_loss.detach()

            results[1] = results[1].detach()

            optimizer.step()

            epoch_mae += mae(*results)
            epoch_ccc += concordance_cc(*results)
            epoch_weighted_mae += mae_thresh(*results)

        scheduler.step()

        loss["train"].append(epoch_loss / len(train_loader))
        metrics["train"]["mae"].append(epoch_mae / len(train_loader))
        metrics["train"]["ccc"].append(epoch_ccc / len(train_loader))
        metrics["train"]["mae_thresh"].append(epoch_weighted_mae / len(train_loader))

        logger.info(msg=
                    f"Epoch: {epoch + 1} | Stage: {stage} | loss: {loss[stage][-1]} | mae: {metrics[stage]['mae'][-1]} | "
                    f"concordance_cc: {metrics[stage]['ccc'][-1]}, mae_thresh: {metrics[stage]['mae_thresh'][-1]}")

        epoch_loss = epoch_mae = epoch_ccc = epoch_weighted_mae = 0

        stage = "validation"

        val_loss = 0

        model.eval()
        with torch.inference_mode():
            for batch in val_loader:
                tokens, lengths, mask, sp_e = batch["tokens"], batch["lengths"], batch["mask"], batch["sp_e"]
                tokens, lengths, mask, sp_e = tokens.to(device), lengths.to(device), mask.to(device), sp_e.to(device)

                predicted_durations = model(tokens, sp_e)
                lengths = lengths.float()

                results = [lengths,
                           predicted_durations.squeeze(-1),
                           mask]

                val_loss += loss_fn(*results)

                epoch_mae += mae(*results)
                epoch_ccc += concordance_cc(*results)
                epoch_weighted_mae += mae_thresh(*results)

        loss[stage].append(val_loss / len(val_loader))
        metrics[stage]["mae"].append(epoch_mae / len(val_loader))
        metrics[stage]["ccc"].append(epoch_ccc / len(val_loader))
        metrics[stage]["mae_thresh"].append(epoch_weighted_mae / len(val_loader))

        logger.info(msg=
                    f"Epoch: {epoch + 1} | Stage: {stage} | loss: {loss[stage][-1]} | mae: {metrics[stage]['mae'][-1]} | "
                    f"concordance_cc: {metrics[stage]['ccc'][-1]}, mae_thresh: {metrics[stage]['mae_thresh'][-1]}")

    epoch_loss = epoch_mae = epoch_ccc = epoch_weighted_mae = 0

    stage = "test"

    with torch.inference_mode():
        for batch in test_loader:
            tokens, lengths, mask, sp_e = batch["tokens"], batch["lengths"], batch["mask"], batch["sp_e"]
            tokens, lengths, mask, sp_e = tokens.to(device), lengths.to(device), mask.to(device), sp_e.to(device)

            predicted_durations = model(tokens, sp_e)
            lengths = lengths.float()

            results = [lengths,
                       predicted_durations.squeeze(-1),
                       mask]

            test_loss += loss_fn(*results)

            epoch_mae += mae(*results)
            epoch_ccc += concordance_cc(*results)
            epoch_weighted_mae += mae_thresh(*results)

    loss[stage].append(epoch_loss / len(test_loader))
    metrics[stage]["mae"].append(epoch_mae / len(test_loader))
    metrics[stage]["ccc"].append(epoch_ccc / len(test_loader))
    metrics[stage]["mae_thresh"].append(epoch_weighted_mae / len(test_loader))

    logger.info(msg=
                f"Stage: Test | loss: {loss[stage][-1]} | mae: {metrics[stage]['mae'][-1]} | "
                f"concordance_cc: {metrics[stage]['ccc'][-1]}, mae_thresh: {metrics[stage]['mae_thresh'][-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train config")
    parser.add_argument("config",
                        type=str,
                        help="path to your config")
    train(parser.parse_args())
