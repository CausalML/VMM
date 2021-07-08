import json
import os
from collections import defaultdict
from multiprocessing import Queue, Process

import numpy as np

from experiment_setups.parametric_experiment_setups import simple_iv_setup, \
    heteroskedastic_iv_setup, policy_learning_setup
from utils.hyperparameter_optimization import iterate_placeholder_values, \
    fill_placeholders, fill_global_values


setup_list = [simple_iv_setup, heteroskedastic_iv_setup, policy_learning_setup]
save_dir = "results_parametric"


def main():
    for setup in setup_list:
        run_experiment(setup)


def run_experiment(setup):
    results = []

    n_range = sorted(setup["n_range"], reverse=True)
    num_procs = setup["num_procs"]
    num_reps = setup["num_reps"]
    num_jobs = len(n_range) * num_reps

    if num_procs == 1:
        # run jobs sequentially
        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                results.extend(do_job(setup, n, rep_i, verbose=True))
    else:
        # run jobs in separate processes using queue'd system
        jobs_queue = Queue()
        results_queue = Queue()

        for n in n_range:
            for rep_i in range(setup["num_reps"]):
                jobs_queue.put((setup, n, rep_i))

        procs = []
        for i in range(num_procs):
            p = Process(target=run_jobs_loop, args=(jobs_queue, results_queue))
            procs.append(p)
            jobs_queue.put("STOP")
            p.start()

        num_done = 0
        while num_done < num_jobs:
            results.extend(results_queue.get())
            num_done += 1
        for p in procs:
            p.join()

    # build aggregate results
    aggregate_results = build_aggregate_results(results)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "%s_results.json" % setup["setup_name"])
    with open(save_path, "w") as f:
        output = {"results": results, "setup": setup,
                  "aggregate_results": aggregate_results}
        json.dump(output, f, default=lambda c_: c_.__name__,
                  indent=2, sort_keys=True)


def run_jobs_loop(jobs_queue, results_queue):
    for job_args in iter(jobs_queue.get, "STOP"):
        results = do_job(*job_args)
        results_queue.put(results)


def do_job(setup, n, rep_i, verbose=False):
    results = []
    print("setting up scenario for %s setup (n=%d, rep=%d)"
          % (setup["setup_name"], n, rep_i))
    scenario_class = setup["scenario"]["class"]
    scenario_args = setup["scenario"]["args"]
    scenario = scenario_class(**scenario_args)
    scenario.setup(num_train=n, num_dev=n,
                   num_test=setup["num_test"])

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    k_z_class = setup["dev_z_kernel_class"]
    k_z_args = setup["dev_z_kernel_args"]
    rho_dim = scenario.get_rho_dim()

    setup["rho_dim"] = scenario.get_rho_dim()
    setup["z_dim"] = scenario.get_z_dim()

    if isinstance(k_z_class, list):
        k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
    else:
        k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
    for k_z in k_z_list:
        k_z.train(train.z)

    for method in setup["methods"]:
        if verbose:
            print("running iv_methods %s under %s setup (n=%d, rep=%d)"
                  % (method["name"], setup["setup_name"], n, rep_i))
        placeholder_options = method["placeholder_options"]
        for placeholder_values in iterate_placeholder_values(
                placeholder_options):
            if placeholder_values:
                print("using placeholder values", placeholder_values)
            rho_generator = scenario.get_rho_generator()
            args = fill_global_values(method["args"], setup)
            args = fill_placeholders(args, placeholder_values)
            predictor = method["class"](rho_generator=rho_generator,
                                        rho_dim=rho_dim, **args)
            predictor.fit(x=train.x, z=train.z, x_dev=dev.x, z_dev=dev.z)
            predicted_params = predictor.get_fitted_parameter_vector()
            true_params = scenario.get_true_parameter_vector()
            sq_error = float(((predicted_params - true_params) ** 2).sum())
            param_dict = predictor.get_fitted_parameter_dict()
            dev_mmr_loss = predictor.calc_mmr_loss(k_z_list, dev.x, dev.z)
            risk = scenario.calc_test_risk(test.x, test.z, predictor)
            row = {
                "n": n,
                "dev_mmr_loss": dev_mmr_loss,
                "method": method["name"],
                "rep": rep_i,
                "sq_error": sq_error,
                "predicted_params": param_dict,
                "placeholder_values": placeholder_values,
                "risk": risk,
            }
            results.append(row)
            if verbose:
                print(json.dumps(row, sort_keys=True, indent=2))
    if verbose:
        print("")
    return results


def build_aggregate_results(results):
    se_list_collection = defaultdict(lambda: defaultdict(list))
    risk_list_collection = defaultdict(lambda: defaultdict(list))

    for row in results:
        method = row["method"]
        n = row["n"]
        key = "%05d::%s" % (n, method)
        hyperparam_values = tuple(sorted(row["placeholder_values"].items()))
        se_list_collection[key][hyperparam_values].append(row["sq_error"])
        risk_list_collection[key][hyperparam_values].append(row["risk"])

    aggregate_results = {}
    for key in sorted(se_list_collection.keys()):
        n = int(key.split("::")[0])
        method = key.split("::")[1]
        print("aggregate results for n=%d, method: %s" % (n, method))
        aggregate_results[key] = []
        for hyperparam_values in sorted(se_list_collection[key].keys()):
            sq_error_list = se_list_collection[key][hyperparam_values]
            se_mean = float(np.mean(sq_error_list))
            se_std = float(np.std(sq_error_list))
            se_max = float(np.max(sq_error_list))
            risk_list = risk_list_collection[key][hyperparam_values]
            risk_mean = float(np.mean(risk_list))
            risk_std = float(np.std(risk_list))
            risk_max = float(np.max(risk_list))
            print("%r: mse = %f ± %f (max %f)  ---  risk = %f ± %f (max %f)"
                  % (hyperparam_values, se_mean, se_std, se_max,
                     risk_mean, risk_std, risk_max))
            result = {
                "hyperparam_values": dict(hyperparam_values),
                "mean_square_error": se_mean,
                "std_square_error": se_std,
                "max_square_error": se_max,
                "mean_risk": risk_mean,
                "std_risk": risk_std,
                "max_risk": risk_max,
            }
            aggregate_results[key].append(result)

    return aggregate_results


if __name__ == "__main__":
    main()