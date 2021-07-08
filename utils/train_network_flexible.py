import torch

from utils.torch_utils import BatchIter


def train_network_flexible(loss_function, parameters, data_tuple, n,
                           data_tuple_dev=None, max_epochs=10000,
                           batch_size=128, max_no_improve=20):
    optim = torch.optim.Adam(parameters)
    batch_iter = BatchIter(n, batch_size)
    min_dev_loss = float("inf")
    num_no_improve = 0
    # dev_mse = calc_dev_mse(g, x_dev, y_dev, batch_size=batch_size)
    # print(dev_mse)
    for epoch_i in range(max_epochs):
        # iterate through all minibatches for this epoch
        for batch_idx in batch_iter:
            data_tuple_batch = [d_[batch_idx] for d_ in data_tuple]
            loss = loss_function(*data_tuple_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # calculate MSE on dev data
        if (data_tuple_dev is not None) and (max_no_improve > 0):
            dev_loss = float(loss_function(*data_tuple_dev))
            if dev_loss < min_dev_loss:
                num_no_improve = 0
                min_dev_loss = dev_loss
            else:
                num_no_improve += 1
                if num_no_improve == max_no_improve:
                    break
