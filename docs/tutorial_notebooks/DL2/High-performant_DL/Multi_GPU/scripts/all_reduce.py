from utils import setup_distrib, disk, square, rank_print, plot_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

# operation performed by the reduce
OPERATION = dist.ReduceOp.MAX


def main_process(rank: int, world_size: int = 2):
    device = setup_distrib(rank, world_size)
    rank_print("test")
    image = torch.zeros((11, 11), device=device)

    if rank == 0:
        rank_image, rank_mask = disk(image, (3, 3), 2, rank + 1)
    elif rank == 1:
        rank_image, rank_mask = square(image, (3, 3), 2, rank + 1)

    plot_matrix(
        rank_image,
        rank,
        title=f"Rank {rank} Before All Reduce",
        name="before_all_reduce",
        folder="all_reduce",
    )

    # The main operation
    dist.all_reduce(rank_image, op=OPERATION)
    plot_matrix(
        rank_image,
        rank,
        title=f"Rank {rank} After All Reduce Operation: {OPERATION}",
        name="after_all_reduce",
        folder="all_reduce",
    )


if __name__ == "__main__":
    mp.spawn(main_process, nprocs=2, args=())
