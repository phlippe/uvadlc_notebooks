import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def rank_print(text: str):
    """
    Prints a statement with an indication of what node rank is sending it
    """
    rank = dist.get_rank()
    # Keep the print statement as a one-liner to guarantee that
    # one single process prints all the lines
    print(f"Rank: {rank}, {text}.")


def disk(
    matrix: torch.Tensor,
    center: tuple[int, int] = (1, 1),
    radius: int = 1,
    value: float = 1.0,
):
    """
    Places a disk with a certain radius and center in a matrix. The value given to the disk must be defined.
    Something like this:
    0 0 0 0 0
    0 0 1 0 0
    0 1 1 1 0
    0 0 1 0 0
    0 0 0 0 0

    Arguments:
     - matrix: the matrix where to place the shape.
     - center: a tuple indicating the center of the disk
     - radius: the radius of the disk in pixels
     - value: the value to write where the disk is placed
    """
    device = matrix.get_device()
    shape = matrix.shape

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(center, shape)]
    x_coords, y_coords = np.mgrid[grid]
    mask = torch.tensor(
        ((x_coords / radius) ** 2 + (y_coords / radius) ** 2 <= 1), device=device
    )
    matrix = matrix * (~mask) + mask * value

    return matrix, mask


def square(
    matrix: torch.tensor,
    topleft: tuple[int, int] = (0, 0),
    length: int = 1,
    value: float = 1.0,
):
    """
    Places a square starting from the given top-left position and having given side length.
    The value given to the disk must be defined.
    Something like this:
    0 0 0 0 0
    0 1 1 1 0
    0 1 1 1 0
    0 1 1 1 0
    0 0 0 0 0

    Arguments:
     - matrix: the matrix where to place the shape.
     - topleft: a tuple indicating the top-left-most vertex of the square
     - length: the side length of the square
     - value: the value to write where the square is placed
    """
    device = matrix.get_device()
    shape = matrix.shape
    grid = [slice(-x0, dim - x0) for x0, dim in zip(topleft, shape)]
    x_coords, y_coords = np.mgrid[grid]
    mask = torch.tensor(
        (
            (x_coords <= length)
            & (x_coords >= 0)
            & (y_coords >= 0)
            & (y_coords <= length)
        ),
        device=device,
    )
    matrix = matrix * (~mask) + mask * value

    return matrix, mask


def plot_matrix(
    matrix: torch.Tensor,
    rank: int,
    title: str = "Matrix",
    name: str = "image",
    folder: Optional[str] = None,
    storefig: bool = True,
):
    """
    Helper function to plot the images more easily. Can store them or visualize them right away.
    """
    plt.figure()
    plt.title(title)
    plt.imshow(matrix.cpu(), cmap="tab20", vmin=0, vmax=19)
    plt.axis("off")
    if folder:
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
    else:
        folder = Path(".")

    if storefig:
        plt.savefig(folder / Path(f"rank_{rank}_{name}.png"))
    else:
        plt.show()
    plt.close()


def setup_distrib(
    rank: int,
    world_size: int,
    init_method: str = "file:///scratch/sharedfile",
    backend: str = "nccl",
):
    # select the correct device for this process
    torch.cuda.set_device(rank)

    # initialize the processing group
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, init_method=init_method, rank=rank
    )

    # return the current device
    return torch.device(f"cuda:{rank}")
