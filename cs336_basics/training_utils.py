import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader


# class LLMDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, context_length=128, shuffle=False):
#         super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
#         self.context_length = context_length
#     def collate_fn(self, batch):
#         # Implement logic to create input-target pairs for language modeling
#         inputs = []
#         targets = []
#         for item in batch:
#             # Assuming item is a list of token indices
#             for i in range(0, len(item) - self.context_length):
#                 inputs.append(item[i:i+self.context_length])
#                 targets.append(item[i+1:i+self.context_length+1])
#         return torch.tensor(inputs), torch.tensor(targets)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and their next-token targets from dataset.

    Args:
        dataset: 1D numpy integer array of token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        Tuple (x, y) of LongTensors with shape (batch_size, context_length),
        where y[i] = x[i] shifted right by one token.
    """
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return x_tensor, y_tensor


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer whose state to save.
        epoch: The current epoch number (for record-keeping).
        path: The file path to save the checkpoint to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)


def load_checkpoint(src, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        src: The file path to load the checkpoint from.
        model: The PyTorch model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
