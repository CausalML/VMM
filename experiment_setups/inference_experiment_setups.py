import torch

from inference_methods.kernel_inference import KernelInferenceMethod
from inference_methods.neural_inference import NeuralInferenceMethod
from utils.oadam import OAdam
from utils.models import ModularMLPModel
from estimation_methods.double_neural_vmm import DoubleNeuralVMM
from estimation_methods.single_kernel_vmm import SingleKernelVMM
from scenarios.heteroskedastic_iv_scenario import \
    HeteroskedasticIVScenario
from scenarios.simple_iv_scenario import SimpleIVScenario
from utils.hyperparameter_optimization import GlobalSetupVal, \
    HyperparameterPlaceholder
from utils.kernels import TripleMedianKernel


estimation_methods = [
    {
        "class": SingleKernelVMM,
        "name": "SingleKernelVMM",
        "placeholder_options": {
            "alpha": [1e-8, 1e-4, 1e0],
        },
        "args": {
            "verbose": GlobalSetupVal("verbose"),
            "alpha": HyperparameterPlaceholder("alpha"),
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
            "num_iter": 2,
        },
    },
    {
        "class": DoubleNeuralVMM,
        "name": "DoubleNeuralVMM",
        "placeholder_options": {},
        "args": {
            "kernel_lambda": 0,
            "l2_lambda": 0,
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
            "rho_optim_class": OAdam,
            "rho_optim_args": {
                "lr": 5e-4,
                "betas": (0.5, 0.9),
            },
            "f_network_class": ModularMLPModel,
            "f_network_args": {
                "input_dim": GlobalSetupVal("z_dim"),
                "layer_widths": [50, 20],
                "activation": torch.nn.LeakyReLU,
                "num_out": GlobalSetupVal("rho_dim"),
            },
            "f_optim_class": OAdam,
            "f_optim_args": {
                "lr": 5 * 5e-4,
                "betas": (0.5, 0.9),
            },
            "batch_size": 200,
            "max_num_epochs": 50000,
            "burn_in_cycles": 5,
            "eval_freq": 2000,
            "max_no_improve": 3,
            "pretrain": False,
            "verbose": False,
        }
    },
]


inference_methods = [
    {
        "class": KernelInferenceMethod,
        "name": "KernelInferenceMethod",
        "placeholder_options": {
            "alpha": [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]
        },
        "args": {
            "alpha": HyperparameterPlaceholder("alpha"),
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
        },
    },
    {
        "class": NeuralInferenceMethod,
        "name": "NeuralInferenceMethod",
        "placeholder_options": {
            "l2_lambda": [0, 1e-4, 1e-2, 1e0]
        },
        "args": {
            "kernel_lambda": 0,
            "l2_lambda": 1e-2,
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
            "learning_stage_args": [
                {
                    "gamma_optim_class": OAdam,
                    "gamma_optim_args": {
                        "lr": 5e-2,
                        "betas": (0.5, 0.9),
                    },
                    "f_optim_class": OAdam,
                    "f_optim_args": {
                        "lr": 1.0 * 5e-2,
                        "betas": (0.5, 0.9),
                    },
                    "num_iter": 3000,
                    "batch_size": 200,
                },
            ],
            "f_network_class": ModularMLPModel,
            "f_network_args": {
                "input_dim": GlobalSetupVal("z_dim"),
                "layer_widths": [50, 20],
                "activation": torch.nn.LeakyReLU,
                "num_out": GlobalSetupVal("rho_dim"),
            },
            "num_print": 10,
            "num_smooth_epoch": 100,
            "verbose": False,
        },
    },
]


n_range_low = [200]
n_range_high = [2000]
num_test = 20000
num_reps = 200
num_procs = 1
dev_z_kernel_class = TripleMedianKernel
dev_z_kernel_args = {}


simple_inference_setup_low = {
    "setup_name": "simple_inference_low",
    "scenario": {
        "class": SimpleIVScenario,
        "args": {},
    },
    "n_range": n_range_low,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": TripleMedianKernel,
    "dev_z_kernel_args": {},
    "num_reps": num_reps,
    "num_procs": num_procs,
    "estimation_methods": estimation_methods,
    "inference_methods": inference_methods,
}

heteroskedastic_inference_setup_low = {
    "setup_name": "heteroskedastic_inference_low",
    "scenario": {
        "class": HeteroskedasticIVScenario,
        "args": {},
    },
    "n_range": n_range_low,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": TripleMedianKernel,
    "dev_z_kernel_args": {},
    "num_reps": num_reps,
    "num_procs": num_procs,
    "estimation_methods": estimation_methods,
    "inference_methods": inference_methods,
}

simple_inference_setup_high = {
    "setup_name": "simple_inference_high",
    "scenario": {
        "class": HeteroskedasticIVScenario,
        "args": {},
    },
    "n_range": n_range_high,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": TripleMedianKernel,
    "dev_z_kernel_args": {},
    "num_reps": num_reps,
    "num_procs": num_procs,
    "estimation_methods": estimation_methods,
    "inference_methods": inference_methods,
}

heteroskedastic_inference_setup_high = {
    "setup_name": "heteroskedastic_inference_high",
    "scenario": {
        "class": HeteroskedasticIVScenario,
        "args": {},
    },
    "n_range": n_range_high,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": TripleMedianKernel,
    "dev_z_kernel_args": {},
    "num_reps": num_reps,
    "num_procs": num_procs,
    "estimation_methods": estimation_methods,
    "inference_methods": inference_methods,
}
