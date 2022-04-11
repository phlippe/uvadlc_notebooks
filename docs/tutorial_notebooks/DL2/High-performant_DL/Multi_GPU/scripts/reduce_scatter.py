from utils import setup_distrib, disk, square, rank_print, plot_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

OPERATION = dist.ReduceOp.MAX


def main_process(rank, world_size=2):
    device = setup_distrib(rank, world_size)

    image = torch.zeros((11, 11), device=device)

    input_tensors = []

    if rank == 0:
        input_tensors.append(disk(image, (4, 5), 2, rank + 1)[0])
        input_tensors.append(square(image, (5, 5), 3, rank + 1)[0])
    elif rank == 1:
        input_tensors.append(disk(image, (7, 6), 2, rank + 1)[0])
        input_tensors.append(square(image, (0, 2), 4, rank + 1)[0])

    output = torch.zeros_like(image, device=device)

    plot_matrix(
        input_tensors[0],
        rank,
        title=f"Rank {rank}",
        name="before_reduce_scatter_0",
        folder="reduce_scatter",
    )
    plot_matrix(
        input_tensors[1],
        rank,
        title=f"",
        name="before_reduce_scatter_1",
        folder="reduce_scatter",
    )
    plot_matrix(
        output, rank, title=f"", name="before_reduce_scatter", folder="reduce_scatter"
    )

    # The main operation
    dist.reduce_scatter(output, input_tensors, op=OPERATION)
    plot_matrix(
        output, rank, title=f"", name="after_reduce_scatter", folder="reduce_scatter"
    )


if __name__ == "__main__":
    mp.spawn(main_process, nprocs=2, args=())
