import torch

from model import DurationPredictor
from dataset import PhoneDataset
from metrics import masked_mae, сoncordance_cc
from loss import masked_mse_loss
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logs import setup_train_logging


def train(arg):
    logger = setup_train_logging(arg.log_file)
    device = arg.device

    dataset = PhoneDataset(arg.root_dir)
    train_set, val_set = train_test_split(dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_set, batch_size=arg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=arg.batch_size, shuffle=False)

    model = DurationPredictor(voc_size=len(dataset.vocab),
                              in_channels=arg.embedding_size,
                              filter_channels=arg.filter_size,
                              hidden_dim=arg.hidden,
                              speaker_emb_size=arg.speaker_embedding_size,
                              n_convs=arg.n_conv_layers).to(device)

    loss_fn = masked_mse_loss

    assert arg.optim in [None, "adam", "adamw"]

    if arg.optim == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                                arg.lr,
                                arg.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               arg.lr,
                               arg.weight_decay)
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
    concordance_cc = сoncordance_cc
    mae = masked_mae

    metrics = {"train":
        {
            "mae": [],
            "ccc": []
        },
        "validation":
            {
                "mae": [],
                "ccc": []
            }
    }
    loss = {"train": [],
            "validation": []}

    logger.log("Training starts:")

    for epoch in range(arg.max_epochs):
        epoch_loss = epoch_mae = epoch_ccc = 0
        stage = "train"
        model.train()
        for batch in train_loader:
            tokens, lengths, mask, sp_e = batch["tokens"], batch["lenghts"], batch["mask"], batch["sp_embd"]
            tokens, lengths, mask, sp_e = tokens.to(device), lengths.to(device), mask.to(device), sp_e.to("cuda")

            optimizer.zero_grad()
            predicted_durations = model(tokens, sp_e)
            lengths = lengths.float()

            results = [lengths, predicted_durations, mask]

            train_loss = loss_fn(*results)
            train_loss.backward()
            epoch_loss += train_loss.detach()

            optimizer.step()

            epoch_mae += mae(*results)
            epoch_ccc += concordance_cc(*results)

        scheduler.step()

        loss["train"].append(epoch_loss.mean())
        metrics["train"]["mae"].append(epoch_mae.mean())
        metrics["train"]["ccc"].append(epoch_ccc.mean())

        logger.log(
            f"Epoch: {epoch+1} | Stage: {stage} | loss: {loss[stage][-1]} | mae: {metrics[stage]['mae'][-1]} | concordance_cc: {metrics[stage]['ccc'][-1]}")

        epoch_loss = epoch_mae = epoch_ccc = 0

        stage = "validation"

        model.eval()
        with torch.inference_mode():
            for batch in train_loader:
                tokens, lengths, mask, sp_e = batch["tokens"], batch["lenghts"], batch["mask"], batch["sp_embd"]
                tokens, lengths, mask, sp_e = tokens.to(device), lengths.to(device), mask.to(device), sp_e.to("cuda")
                predicted_durations = model(tokens, sp_e)
                lengths = lengths.float()
                results = [lengths, predicted_durations, mask]

                train_loss = loss_fn(*results)
                epoch_loss += train_loss

                epoch_mae += mae(*results)
                epoch_ccc += concordance_cc(*results)

        loss[stage].append(epoch_loss.mean())
        metrics[stage]["mae"].append(epoch_mae.mean())
        metrics[stage]["ccc"].append(epoch_ccc.mean())

        logger.log(
            f"Epoch: {epoch + 1} | Stage: {stage} | loss: {loss[stage][-1]} | mae: {metrics[stage]['mae'][-1]} | concordance_cc: {metrics[stage]['ccc'][-1]}")

    logger.log("Training ends.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train config")
    parser.add_argument("config",
                        type=str,
                        help="path to your config")
    train(parser.parse_args())