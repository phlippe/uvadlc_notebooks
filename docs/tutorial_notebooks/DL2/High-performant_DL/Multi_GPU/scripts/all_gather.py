from utils import setup_distrib, disk, square, rank_print, plot_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
import torch


def main_process(rank, world_size=2):
    device = setup_distrib(rank, world_size)

    image = torch.zeros((11, 11), device=device)

    rank_images = []

    if rank == 0:
        rank_images.append(disk(image, (4, 5), 2, rank + 1)[0])
    elif rank == 1:
        rank_images.append(disk(image, (7, 6), 2, rank + 1)[0])

    output_tensors = []
    for _ in range(world_size):
        output_tensors.append(torch.zeros_like(image, device=device))

    plot_matrix(
        output_tensors[0],
        rank,
        title=f"Rank {rank}",
        name="before_gather_0",
        folder="all_gather",
    )
    plot_matrix(
        output_tensors[1], rank, title=f"", name="before_gather_1", folder="all_gather"
    )

    # The main operation
    dist.all_gather(output_tensors, rank_images[0])
    plot_matrix(
        output_tensors[0],
        rank,
        title=f"Rank {rank}",
        name="after_gather_0",
        folder="all_gather",
    )
    plot_matrix(
        output_tensors[1], rank, title=f"", name="after_gather_1", folder="all_gather"
    )


if __name__ == "__main__":
    mp.spawn(main_process, nprocs=2, args=())
