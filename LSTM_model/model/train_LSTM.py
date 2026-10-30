from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as distributed
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from LSTM_model.model.config import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    LR_GAMMA,
    LR_SCHEDULER,
    LR_STEP_SIZE,
    LOOKBACK,
    MODEL_NAME,
    NUM_EPOCHS,
    NUM_LAYERS,
    OUTPUTPATH,
    OUTPUT_SIZE,
    TARGET_REGION,
    TRAINING_PERIOD,
    AwesomeLSTM,
)
from LSTM_model.model.dataset import (
    GridSequenceDataset,
    calculate_training_statistics,
)

logger = logging.getLogger(__name__)


class train_LSTM_model:
    def train(self) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")

        initialized_here = False
        if world_size > 1 and not distributed.is_initialized():
            backend = "nccl" if use_cuda else "gloo"
            distributed.init_process_group(backend=backend)
            initialized_here = True

        is_distributed = distributed.is_initialized()
        rank = distributed.get_rank() if is_distributed else 0

        try:
            if use_cuda:
                torch.cuda.set_device(device)
                logger.info("Training on %s", torch.cuda.get_device_name(device))
            else:
                logger.info("Training on CPU")

            os.makedirs(OUTPUTPATH, exist_ok=True)
            statistics = calculate_training_statistics()
            output_dir = Path(OUTPUTPATH)
            stats_path = output_dir / f"meanstd_{TARGET_REGION}_{MODEL_NAME}.pkl"
            temporary_stats_path = output_dir / f".{stats_path.name}.tmp"
            if rank == 0:
                with temporary_stats_path.open("wb") as stream:
                    pickle.dump(statistics, stream)
                os.replace(temporary_stats_path, stats_path)

            dataset = GridSequenceDataset(
                target_start=LOOKBACK,
                target_end=TRAINING_PERIOD,
                statistics=statistics,
            )
            sampler = DistributedSampler(dataset, shuffle=True, seed=0)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                sampler=sampler,
                pin_memory=use_cuda,
            )

            model = AwesomeLSTM(
                INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT
            ).to(device)
            if is_distributed:
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank] if use_cuda else None,
                )

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = (
                torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
                )
                if LR_SCHEDULER
                else None
            )
            epoch_losses: list[float] = []

            for epoch in range(NUM_EPOCHS):
                sampler.set_epoch(epoch)
                model.train()
                batch_losses: list[float] = []

                for inputs, target in dataloader:
                    inputs = inputs.to(device, non_blocking=use_cuda)
                    target = target.to(device, non_blocking=use_cuda)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(inputs).flatten()
                    loss = criterion(prediction, target)
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.detach().clone()
                    if is_distributed:
                        distributed.all_reduce(batch_loss)
                        batch_loss /= distributed.get_world_size()
                    batch_losses.append(batch_loss.item())

                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)
                if scheduler is not None:
                    scheduler.step()

                if rank == 0:
                    logger.info(
                        "Epoch %d/%d - MSE %.6f",
                        epoch + 1,
                        NUM_EPOCHS,
                        epoch_loss,
                    )
                    model_path = output_dir / f"{TARGET_REGION}_{MODEL_NAME}.pt"
                    temporary_model_path = output_dir / f".{model_path.name}.tmp"
                    torch.save(
                        model.module.state_dict()
                        if is_distributed
                        else model.state_dict(),
                        temporary_model_path,
                    )
                    os.replace(temporary_model_path, model_path)

            if rank == 0:
                fig, ax = plt.subplots()
                ax.plot(range(len(epoch_losses)), epoch_losses)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("MSE")
                fig.savefig(output_dir / f"{TARGET_REGION}_{MODEL_NAME}.png")
                plt.close(fig)
        finally:
            if initialized_here and distributed.is_initialized():
                distributed.destroy_process_group()
