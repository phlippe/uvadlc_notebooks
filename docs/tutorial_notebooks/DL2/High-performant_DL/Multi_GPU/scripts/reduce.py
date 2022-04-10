from utils import setup_distrib, disk, square, rank_print, plot_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

DESTINATION_RANK = 0
OPERATION = dist.ReduceOp.MAX


def main_process(rank, world_size=2):
    device = setup_distrib(rank, world_size)

    image = torch.zeros((11, 11), device=device)

    if rank == 0:
        rank_image, rank_mask = disk(image, (4, 5), 2, rank + 1)
    elif rank == 1:
        rank_image, rank_mask = square(image, (3, 3), 2, rank + 1)

    plot_matrix(
        rank_image,
        rank,
        title=f"Rank {rank} Before Reduce",
        name="before_reduce",
        folder="reduce",
    )

    # The main operation
    dist.reduce(rank_image, dst=DESTINATION_RANK, op=OPERATION)
    plot_matrix(
        rank_image,
        rank,
        title=f"Rank {rank} After Reduce Operation: {OPERATION}",
        name="after_reduce",
        folder="reduce",
    )


if __name__ == "__main__":
    mp.spawn(main_process, nprocs=2, args=())
