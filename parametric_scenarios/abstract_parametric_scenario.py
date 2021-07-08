import random
import numpy as np
import os
import torch


class ParametricDataset(object):
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.size = None

    def to_tensor(self):
        self.x = torch.as_tensor(self.x).double()
        self.z = torch.as_tensor(self.z).double()

    def to_2d(self):
        n_data = self.y.shape[0]
        if len(self.x.shape) > 2:
            self.x = self.x.reshape(n_data, -1)
        if len(self.z.shape) > 2:
            self.z = self.z.reshape(n_data, -1)

    def info(self, verbose=False):
        for name, x in [("x", self.x), ("z", self.z)]:
            print("  " + name + ":", x.__class__.__name__,
                  "(" + str(x.dtype) + "): ",
                  "x".join([str(d) for d in x.shape]))
            if verbose:
                print("      min: %.2f" % x.min(), ", max: %.2f" % x.max())

    def as_tuple(self):
        return self.x, self.z

    def as_dict(self, prefix=""):
        d = {"x": self.x, "z": self.z}
        return {prefix + k: v for k, v in d.items()}

    def to_numpy(self):
        self.x = self.x.data.numpy()
        self.z = self.z.data.numpy()

    def to_cuda(self):
        self.x = self.x.cuda()
        self.z = self.z.cuda()


class AbstractParametricScenario(object):
    def __init__(self, rho_dim, theta_dim, z_dim, filename=None):
        self.splits = {"test": None, "train": None, "dev": None}

        self.initialized = False
        if filename is not None:
            self.from_file(filename)

        self.rho_dim = rho_dim
        self.theta_dim = theta_dim
        self.z_dim = z_dim

    def generate_data(self, num_data, split):
        raise NotImplementedError()

    def get_true_parameter_vector(self):
        raise NotImplementedError()

    def get_true_psi(self):
        raise NotImplementedError()

    def get_rho_generator(self):
        raise NotImplementedError()

    def get_rho_dim(self):
        return self.rho_dim

    def get_theta_dim(self):
        return self.theta_dim

    def get_z_dim(self):
        return self.z_dim

    def calc_test_risk(self, x_test, z_test, predictor):
        raise NotImplementedError()

    def to_cuda(self):
        for split in self.splits.values():
            split.to_cuda()

    def to_tensor(self):
        for split in self.splits.values():
            split.to_tensor()

    def to_numpy(self):
        for split in self.splits.values():
            split.to_numpy()

    def to_2d(self):
        """
        flatten x and z to 2D
        """
        for split in self.splits.values():
            split.to_2d()

    def setup(self, num_train, num_dev=0, num_test=0):
        """
        draw data internally, without actually returning anything
        """
        for split, num_data in (("train", num_train),
                                ("dev", num_dev),
                                ("test", num_test)):
            if num_data > 0:
                x, z = self.generate_data(num_data, split)
                self.splits[split] = ParametricDataset(x, z)
        self.initialized = True

    def to_file(self, filename):
        all_splits = {"splits": list()}
        for split, dataset in self.splits.items():
            if dataset is not None:
                all_splits.update(dataset.as_dict(split + "_"))
                all_splits["splits"].append(split)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **all_splits)

    def from_file(self, filename):
        data = np.load(filename)
        for split in data["splits"].tolist():
            self.splits[split] = ParametricDataset(
                *(data[split + "_" + var] for var in ["x", "z"]))
        self.initialized = True

    def info(self):
        for split, dataset in self.splits.items():
            print(split)
            dataset.info(verbose=(split == "train"))

    def get_data(self, split):
        if self.initialized is False:
            raise LookupError(
                "trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no training data to get")
        return self.splits[split].as_tuple()

    def get_train_data(self):
        return self.get_data("train")

    def get_dataset(self, split):
        if self.initialized is False:
            raise LookupError(
                "trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no training data to get")
        return self.splits[split]

    def get_dev_data(self):
        return self.get_data("dev")

    def get_test_data(self):
        return self.get_data("test")

    def iterate_data(self, split, batch_size):
        """
        iterator over training data, using given batch size
        each iteration returns batch as tuple (x, z, y, g, w)
        """
        if self.initialized is False:
            raise LookupError(
                "trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no " + split + " data to iterate over")
        x, z = self.splits[split].as_tuple()
        n = x.shape[0]
        idx = self._get_random_index_order(n, batch_size)
        num_batches = len(idx) // batch_size
        for batch_i in range(num_batches):
            yield self._get_batch(batch_i, batch_size, x, z, idx)

    def iterate_train_data(self, batch_size):
        return self.iterate_data("train", batch_size)

    def iterate_dev_data(self, batch_size):
        return self.iterate_data("dev", batch_size)

    def iterate_test_data(self, batch_size):
        return self.iterate_data("test", batch_size)

    @staticmethod
    def _get_batch(batch_num, batch_size, x, z, index_order):
        l = batch_num * batch_size
        u = (batch_num + 1) * batch_size
        idx = index_order[l:u]
        return x[idx], z[idx]

    @staticmethod
    def _get_random_index_order(num_data, batch_size):
        idx = list(range(num_data))
        idx.extend(random.sample(idx, num_data % batch_size))
        random.shuffle(idx)
        return idx

