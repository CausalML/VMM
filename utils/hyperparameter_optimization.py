from itertools import product


class HyperparameterPlaceholder(object):
    def __init__(self, name):
        self.name = name
        self.__name__ = "HyperparameterPlaceholder(%s)" % name

    def get_name(self):
        return self.name

class GlobalSetupVal(object):
    def __init__(self, name):
        self.name = name
        self.__name__ = "GlobalValue(%s)" % name

    def get_name(self):
        return self.name


def fill_global_values(args, setup):
    filled_args = {}
    for k, v in args.items():
        if isinstance(v, GlobalSetupVal):
            name = v.get_name()
            try:
                filled_args[k] = setup[name]
            except KeyError:
                raise KeyError("Key '%s' missing from global setup params"
                               % name)
        elif isinstance(v, dict):
            filled_args[k] = fill_global_values(v, setup)
        else:
            filled_args[k] = v
    return filled_args


def fill_placeholders(args, placeholder_values):
    filled_args = {}
    for k, v in args.items():
        if isinstance(v, HyperparameterPlaceholder):
            name = v.get_name()
            filled_args[k] = placeholder_values[name]
        elif isinstance(v, dict):
            filled_args[k] = fill_placeholders(v, placeholder_values)
        else:
            filled_args[k] = v
    return filled_args


def iterate_placeholder_values(placeholder_options):
    if not placeholder_options:
        yield {}
        return
    placeholder_options = list(sorted(placeholder_options.items()))
    placeholder_names = [k for k, _ in placeholder_options]
    placeholder_value_lists = [v for _, v in placeholder_options]
    values_iter = product(*placeholder_value_lists)
    while True:
        try:
            values = next(values_iter)
            yield {k: v for k, v in zip(placeholder_names, values)}
        except StopIteration:
            return


def debug():
    args = {
        "alpha": HyperparameterPlaceholder("alpha"),
        "k_z_class": "GaussianKernel",
        "k_z_args": {"sigma": 3.6},
        "g_network_class": "MLPModel",
        "g_network_args": {
            "input_dim": 1,
            # "layer_widths": HyperparameterPlaceholder("g-layer-widths"),
            "layer_widths": [20, 50],
            "activation": "torch.nn.LeakyReLU",
        },
        "optimizer_class": "torch.optim.Adam",
        "optimizer_args": {
            # "lr": HyperparameterPlaceholder("lr"),
            "lr": 1e-3,
        },
        "batch_size": 128,
        "num_cycles": 5,
        "iterations_per_cycle": 500,
    }
    print(args["alpha"].__name__)
    placeholder_options = {
        "alpha": [1e-2, 1e0, 1e2],
        # "lr": [1e-4, 5e-3, 1e-3],
        # "g-layer-widths": [(50, 20), (50,), (100, 100, 100)]
    }

    for placeholder_values in iterate_placeholder_values(placeholder_options):
        print("placeholder values:")
        print(placeholder_values)
        print("filled in args:")
        print(fill_placeholders(args, placeholder_values))
        print("")


if __name__ == "__main__":
    debug()
