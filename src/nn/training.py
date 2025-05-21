from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.util import input_listener

from .optimization import Optimization


def log_statistic(*, name: str, value) -> None:
    print(f"{name}: {value:4g}")


def loss_plot(
    *,
    train_losses,
    validation_losses=None,
    epoch: int = 0,
    freq: int = 1,
    loss_plot_name="losses.png",
    path: str | None = None,
) -> None:
    file = f"{path}/{loss_plot_name}" if path else loss_plot_name
    if epoch % freq == 0:
        plt.close("all")
        plt.rcdefaults()
        plt.plot(
            np.arange(1, len(train_losses) + 1), train_losses, alpha=0.8, label="train"
        )
        if validation_losses is not None:
            plt.plot(
                np.arange(1, len(validation_losses) + 1),
                validation_losses,
                alpha=0.8,
                label="validation",
            )
        plt.legend()
        plt.savefig(file)
        plt.close("all")


def number_of_parameters(nn: torch.nn.Module) -> int:
    return sum(p.numel() for p in nn.parameters())


def print_number_of_parameters(nn: torch.nn.Module) -> None:
    print(
        "\033[1;34m" + f"Number of parameters: {number_of_parameters(nn)}" + "\x1b[0m"
    )


def nn_training(
    *,
    train_set: Tensor,
    optimization: Optimization,
    loss_function: Callable,
    nn: torch.nn.Module,
    device: str | None = None,
) -> tuple[list[float], list[float]]:

    # TODO: copied code
    tensor_train_set = TensorDataset(train_set.to(device))

    nn = torch.compile(nn)

    nn.to(train_set.device)

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=optimization.batch_size,
        shuffle=True,
        drop_last=True,
    )
    nn.train(True)

    optimizer = optimization.build_optimizer(nn.parameters())

    if optimization.lr_scheduler_class is not None:
        lr_scheduler = optimization.build_lr_scheduler(optimizer)  # Example scheduler

    train_epochs_losses = []
    validation_epochs_losses = []
    best_validation_loss = torch.inf
    best_model = None
    epochs_since_best = 0
    n_batches = len(train_loader)

    input_happened = input_listener() if optimization.interactive else False

    for epoch in range(optimization.epochs):
        if input_happened:
            break
        log_statistic(name="epoch", value=epoch + 1)
        print(f'learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
        losses = np.zeros(n_batches)
        validation_losses = np.zeros(n_batches)

        pbar = tqdm.tqdm(total=n_batches)
        optimizer.zero_grad()

        for i, X_batch in enumerate(train_loader):

            if input_happened:
                break

            loss = loss_function(X=X_batch[0])
            if isinstance(loss, tuple):
                validation_losses[i] = loss[1]
                loss = loss[0]

            (loss / optimization.grad_accumulation_steps).backward()

            if optimization.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

            if ((i + 1) % optimization.grad_accumulation_steps == 0) or (
                i == n_batches - 1
            ):
                optimizer.step()
                optimizer.zero_grad()

            losses[i] = loss.detach()
            pbar.set_postfix(loss=f"{losses[i]:.2f}")
            pbar.update(1)
        pbar.close()

        if optimization.lr_scheduler_class is not None:
            lr_scheduler.step()

        train_epochs_losses.append(np.mean(losses))

        log_statistic(name="train loss mean", value=train_epochs_losses[-1])
        if validation_losses.any():
            validation_epochs_losses.append(validation_losses.mean())
            log_statistic(name="validation loss", value=validation_epochs_losses[-1])
            if (
                optimization.restore_best_validation_model
                and validation_epochs_losses[-1] < best_validation_loss
            ):
                best_validation_loss = validation_epochs_losses[-1]
                best_model = nn.state_dict()
                epochs_since_best = 0
            epochs_since_best += 1
        # log_statistic(name="train loss std", value=losses.std())

        print()

        loss_plot(
            train_losses=train_epochs_losses,
            validation_losses=validation_epochs_losses,
            epoch=epoch,
            freq=1 + optimization.epochs // 10,
        )

        if epoch == 0:
            print_number_of_parameters(nn)

        if (
            optimization.patience is not None
            and epochs_since_best > optimization.patience
        ):
            print(
                f"Early stopping (validation loss not improving since last {epochs_since_best} epochs)"
            )
            break

    loss_plot(
        train_losses=train_epochs_losses,
        validation_losses=validation_epochs_losses,
        epoch=epoch,
        freq=1,
    )

    if optimization.restore_best_validation_model and best_model is not None:
        print("restoring best model...")
        nn.load_state_dict(best_model)

    nn.eval()
    return train_epochs_losses, validation_epochs_losses


def get_warmup_lr(
    *, epoch: int, warmup_lr: float = 1e-8, final_lr: float, warmup_epochs: int = 3
) -> float:
    if epoch > warmup_epochs:
        return final_lr
    return warmup_lr + epoch * (final_lr - warmup_lr) / warmup_epochs
