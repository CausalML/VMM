import itertools
import random

import torch


class BatchIter(object):
    def __init__(self, num, batch_size):
        self.num = int(num)
        self.batch_size = int(batch_size)
        self.num_batches = self.num // self.batch_size
        if self.num % self.batch_size > 0:
            self.num_batches += 1
        self.indices = list(range(self.num))
        self.batch_i = 0
        self.batch_cycle = None

    def __next__(self):
        if self.batch_i == 0:
            random.shuffle(self.indices)
            self.batch_cycle = itertools.cycle(self.indices)
        elif self.batch_i == self.num_batches:
            self.batch_i = 0
            raise StopIteration
        self.batch_i += 1
        return [next(self.batch_cycle) for _ in range(self.batch_size)]

    def __iter__(self):
        return self


def np_to_tensor(data_array, cuda=False, device=None):
    data_tensor = torch.from_numpy(data_array).float()
    if cuda:
        data_tensor = (data_tensor.cuda() if device is None
                       else data_tensor.cuda(device))
    return data_tensor


def torch_softplus(x, sharpness=1):
    x_s = sharpness * x
    return ((torch.log(1 + torch.exp(-torch.abs(x_s)))
             + torch.max(x_s, torch.zeros_like(x_s))) / sharpness)


def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float64")


def debug():
    batch_iter = BatchIter(100, 100)
    for _ in range(3):
        for batch_idx in batch_iter:
            print(batch_idx)
        print("")


if __name__ == "__main__":
    debug()
