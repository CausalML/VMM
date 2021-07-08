import torch

from utils.oadam import OAdam
from utils.models import ModularMLPModel
from parametric_methods.double_neural_vmm import DoubleNeuralVMM
from parametric_methods.kernel_mmr import KernelMMR
from parametric_methods.non_causal_baseline import NonCausalBaseline
from parametric_methods.sieve_minimum_distance import SMDIdentity, \
    SMDHomoskedastic, SMDHeteroskedastic
from parametric_methods.single_kernel_vmm import SingleKernelVMM
from parametric_scenarios.heteroskedastic_iv_scenario import \
    HeteroskedasticIVScenario
from parametric_scenarios.policy_learning_scenario import PolicyLearningScenario
from parametric_scenarios.simple_iv_scenario import SimpleIVScenario
from utils.hyperparameter_optimization import GlobalSetupVal, \
    HyperparameterPlaceholder
from utils.kernels import TripleMedianKernel
from utils.sieve_basis import MultiOutputPolynomialSplineBasis

parametric_experiment_methods = [
    {
        "class": NonCausalBaseline,
        "name": "NonCausalBaseline",
        "placeholder_options": {},
        "args": {},
    },
    {
        "class": SMDIdentity,
        "name": "SMDIdentity",
        "placeholder_options": {},
        "args": {
            "basis_class": MultiOutputPolynomialSplineBasis,
            "basis_args": {
                "num_knots": 5,
                "degree": 2,
                "z_dim": GlobalSetupVal("z_dim"),
                "num_out": GlobalSetupVal("rho_dim"),
            },
        }
    },
    {
        "class": SMDHomoskedastic,
        "name": "SMDHomoskedastic",
        "placeholder_options": {},
        "args": {
            "basis_class": MultiOutputPolynomialSplineBasis,
            "basis_args": {
                "num_knots": 5,
                "degree": 2,
                "z_dim": GlobalSetupVal("z_dim"),
                "num_out": GlobalSetupVal("rho_dim"),
            },
            "num_iter": 2,
        }
    },
    {
        "class": SMDHeteroskedastic,
        "name": "SMDHeteroskedastic",
        "placeholder_options": {},
        "args": {
            "basis_class": MultiOutputPolynomialSplineBasis,
            "basis_args": {
                "num_knots": 5,
                "degree": 2,
                "z_dim": GlobalSetupVal("z_dim"),
                "num_out": GlobalSetupVal("rho_dim"),
            },
            "num_iter": 2,
            "z_dim": GlobalSetupVal("z_dim"),
        }
    },
    {
        "class": KernelMMR,
        "name": "KernelMMR",
        "placeholder_options": {},
        "args": {
            "verbose": GlobalSetupVal("verbose"),
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
        },
    },
    {
        "class": DoubleNeuralVMM,
        "name": "DoubleNeuralVMM",
        "placeholder_options": {
            "lambda": [0, 1e-4, 1e-2, 1e0]
        },
        "args": {
            "kernel_lambda": 0,
            "l2_lambda": HyperparameterPlaceholder("lambda"),
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
            "verbose": GlobalSetupVal("verbose"),
        }
    },
    {
        "class": SingleKernelVMM,
        "name": "SingleKernelVMM",
        "placeholder_options": {
            "alpha": [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e0],
        },
        "args": {
            "verbose": GlobalSetupVal("verbose"),
            "alpha": HyperparameterPlaceholder("alpha"),
            "k_z_class": TripleMedianKernel,
            "k_z_args": {},
            "num_iter": 2,
        },
    },
]

n_range = [200, 500, 1000, 2000, 5000, 10000]
num_test = 20000
num_reps = 50
num_procs = 1
dev_z_kernel_class = TripleMedianKernel
dev_z_kernel_args = {}

simple_iv_setup = {
    "setup_name": "simple_iv_experiment",
    "scenario": {
        "class": SimpleIVScenario,
        "args": {},
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": dev_z_kernel_class,
    "dev_z_kernel_args": dev_z_kernel_args,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": parametric_experiment_methods,
}

heteroskedastic_iv_setup = {
    "setup_name": "heteroskedastic_iv_experiment",
    "scenario": {
        "class": HeteroskedasticIVScenario,
        "args": {},
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": dev_z_kernel_class,
    "dev_z_kernel_args": dev_z_kernel_args,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": parametric_experiment_methods,
}

policy_learning_setup = {
    "setup_name": "policy_learning_experiment",
    "scenario": {
        "class": PolicyLearningScenario,
        "args": {},
    },
    "n_range": n_range,
    "num_test": num_test,
    "verbose": False,
    "dev_z_kernel_class": dev_z_kernel_class,
    "dev_z_kernel_args": dev_z_kernel_args,
    "num_reps": num_reps,
    "num_procs": num_procs,
    "methods": parametric_experiment_methods,
}
