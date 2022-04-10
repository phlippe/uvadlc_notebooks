import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
from utils import rank_print

DSET_FOLDER = "/scratch/"


def main_process(rank, world_size=2):
    print(f"Process for rank: {rank} has been spawned")

    # Setup the distributed processing
    device = setup_distrib(rank, world_size)

    # Load the dataset in all processes download only in the first one
    if rank == 0:
        dset = torchvision.datasets.CIFAR10(DSET_FOLDER, download=True)
    # Make sure download has finished
    dist.barrier()

    # Load the dataset
    dset = torchvision.datasets.CIFAR10(DSET_FOLDER)

    input_size = 3 * 32 * 32  # [channel size, height, width]
    per_gpu_batch_size = 128
    num_classes = 10
    if dist.get_rank() == 0:
        weights = torch.rand((input_size, num_classes), device=device)
    else:
        weights = torch.zeros((input_size, num_classes), device=device)

    # Distribute weights to all GPUs
    handle = dist.broadcast(tensor=weights, src=0, async_op=True)
    rank_print(f"Weights received.")

    # Flattened images
    cur_input = torch.zeros((per_gpu_batch_size, input_size), device=device)
    # One-Hot encoded target
    cur_target = torch.zeros((per_gpu_batch_size, num_classes), device=device)
    for i in range(per_gpu_batch_size):
        rank_print(f"Loading image {i+world_size*rank} into GPU...")
        image, target = dset[i + world_size * rank]
        cur_input[i] = transforms.ToTensor()(image).flatten()
        cur_target[i, target] = 1.0

    # Compute the linear part of the layer
    output = torch.matmul(cur_input, weights)
    rank_print(f"\nComputed output: {output}, Size: {output.size()}.")

    # Define the activation function of the output layer
    logsoftm = torch.nn.LogSoftmax(dim=1)

    # Apply activation function to output layer
    output = logsoftm(output)
    rank_print(f"\nLog-Softmaxed output: {output}, Size: {output.size()}.")

    loss = output.sum(dim=1).mean()
    rank_print(f"Loss: {loss}, Size: {loss.size()}")

    # Here the GPUs need to be synched again
    handle = dist.reduce(tensor=loss, dst=0, op=dist.ReduceOp.SUM)

    rank_print(f"Final Loss: {loss/world_size}")


if __name__ == "__main__":
    mp.spawn(main_process, nprocs=2, args=())
