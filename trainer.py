import tqdm
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainerParams:
    """
    Configuration parameters for the model trainer.

    Attributes:
        model: Language model to train
        train_data: Training data loader
        optimizer: Optimizer for training
        gpu_id: GPU device ID
        save_every: Frequency of saving model checkpoints
        loss_fn: Loss function for training
    """
    model: nn.Module
    train_data: DataLoader
    optimizer: torch.optim.Optimizer
    gpu_id: int = 0
    save_every: int = 10
    loss_fn: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss())


class Trainer:
    def __init__(self, params: TrainerParams) -> None:
        self.gpu_id = params.gpu_id
        self.model = params.model
        self.model = self.model.to(self.gpu_id)
        self.train_data = params.train_data
        self.optimizer = params.optimizer
        self.loss_fn = params.loss_fn
        self.save_every = params.save_every
        self.train_losses = []
        self.train_steps = []

    def _run_epoch(self, epoch:int, steps:int):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data):,}")
        for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = self.loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.train_losses.append(loss.item())
            self.train_steps.append(i+1)
            if (i+1) % 200 == 0:
                print(f"Step {i+1}/{len(self.train_data)} | Loss: {loss.item():.3f} | Norm: {norm:.3f}")
            if (i+1) == steps:
                break

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int, steps: int = 1):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch, steps)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        return self.model