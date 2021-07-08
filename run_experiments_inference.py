import json
import os
from collections import defaultdict
from multiprocessing import Queue, Process

import numpy as np

from experiment_setups.inference_experiment_setups import \
    simple_inference_setup_high, heteroskedastic_inference_setup_high, \
    simple_inference_setup_low, heteroskedastic_inference_setup_low
from utils.hyperparameter_optimization import iterate_placeholder_values, \
    fill_placeholders, fill_global_values


setup_list = [simple_inference_setup_high, heteroskedastic_inference_setup_high,
              simple_inference_setup_low, heteroskedastic_inference_setup_low]
save_dir = "results_inference"


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
    theta_dim = scenario.get_theta_dim()

    setup["rho_dim"] = scenario.get_rho_dim()
    setup["z_dim"] = scenario.get_z_dim()

    if isinstance(k_z_class, list):
        k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
    else:
        k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
    for k_z in k_z_list:
        k_z.train(train.z)

    for method in setup["estimation_methods"]:
        if verbose:
            print("running iv_methods %s under %s setup (n=%d, rep=%d)"
                  % (method["name"], setup["setup_name"], n, rep_i))
        placeholder_options = method["placeholder_options"]
        for placeholders in iterate_placeholder_values(
                placeholder_options):
            if placeholders:
                print("using placeholder values", placeholders)
            rho_generator = scenario.get_rho_generator()
            args = fill_global_values(method["args"], setup)
            args = fill_placeholders(args, placeholders)
            predictor = method["class"](rho_generator=rho_generator,
                                        rho_dim=rho_dim, **args)
            predictor.fit(x=train.x, z=train.z, x_dev=dev.x, z_dev=dev.z)
            rho = predictor.get_rho()
            pred_psi = predictor.get_pred_psi()
            true_psi = scenario.get_true_psi()
            print("predicted-psi=%.3f, true-psi=%.3f" % (pred_psi, true_psi))

            for inf_method in setup["inference_methods"]:
                inf_placeholder_options = inf_method["placeholder_options"]
                for inf_placeholders in iterate_placeholder_values(
                        inf_placeholder_options):
                    if inf_placeholders:
                        print("performing  inference using %s method %r"
                              % (inf_method["name"], inf_placeholders))
                    else:
                        print("performing  inference using %s method"
                              % inf_method["name"])
                    inf_args = fill_global_values(inf_method["args"], setup)
                    inf_args = fill_placeholders(inf_args, inf_placeholders)
                    inference = inf_method["class"](
                        rho=rho, rho_dim=rho_dim,
                        theta_dim=theta_dim, **inf_args)
                    pred_avar = inference.estimate_avar(train.x, train.z)
                    ci_width = 1.96 * (pred_avar ** 0.5) / (n ** 0.5)
                    ci = (pred_psi - ci_width, pred_psi + ci_width)
                    coverage_success = ((ci[0] <= true_psi)
                                        and (ci[1] >= true_psi))
                    row = {
                        "n": n,
                        "pred_psi": pred_psi,
                        "true_psi": true_psi,
                        "pred_avar": pred_avar,
                        "pred_std": (pred_avar ** 0.5) / (n ** 0.5),
                        "pred_ci_lower": ci[0],
                        "pred_ci_upper": ci[1],
                        "coverage_success": coverage_success,
                        "estimation_method": method["name"],
                        "inference_method": inf_method["name"],
                        "rep": rep_i,
                        "estimation_placeholders": placeholders,
                        "inference_placeholders": inf_placeholders,
                    }
                    results.append(row)
                    if verbose:
                        print(json.dumps(row, sort_keys=True, indent=2))
    if verbose:
        print("")
    return results


def create_method_key(method_name, placeholders):
    parts = [method_name]
    for k, v in sorted(placeholders.items()):
        if isinstance(v, float):
            parts.append("%s=%e" % (k, v))
        else:
            parts.append("%s=%r" % (k, v))
    return "::".join(parts)


def build_aggregate_results(results):
    success_results = defaultdict(list)
    prediction_results = defaultdict(list)
    ci_lower_results = defaultdict(list)
    ci_upper_results = defaultdict(list)
    bias_results = defaultdict(list)
    true_psi_array = defaultdict(list)
    pred_std_results = defaultdict(list)

    for row in results:
        est_key = create_method_key(row["estimation_method"],
                                    row["estimation_placeholders"])
        inf_key = create_method_key(row["inference_method"],
                                    row["inference_placeholders"])
        key = (est_key, inf_key)
        success = float(row["coverage_success"])
        success_results[key].append(success)
        prediction_results[key].append(row["pred_psi"])
        ci_lower_results[key].append(row["pred_ci_lower"])
        ci_upper_results[key].append(row["pred_ci_upper"])
        bias_results[key].append(row["pred_psi"] - row["true_psi"])
        true_psi_array[key].append(row["true_psi"])
        pred_std_results[key].append(row["pred_std"])

    aggregate_results = []
    for key in sorted(success_results.keys()):
        bias = float(np.mean(bias_results[key]))
        bc_upper = np.array(ci_upper_results[key]) - bias
        bc_lower = np.array(ci_lower_results[key]) - bias
        true_psi = np.array(true_psi_array[key])
        bc_coverage = float(((bc_lower <= true_psi)
                             & (bc_upper >= true_psi)).mean())
        coverage_rate = float(np.mean(success_results[key]))
        std_pred_psi = float(np.std(prediction_results[key]))
        pred_std_05 = float(np.percentile(pred_std_results[key], q=5))
        pred_std_50 = float(np.percentile(pred_std_results[key], q=50))
        pred_std_95 = float(np.percentile(pred_std_results[key], q=95))
        est_method, inf_method = key
        print("aggregate results for estimation method: %s" % est_method)
        print("    inference method: %s" % inf_method)
        print("    coverage-rate = %f" % coverage_rate)
        print("    bias-corrected-coverage-rate = %f" % bc_coverage)
        print("    pred-std-05 = %f" % pred_std_05)
        print("    pred-std-50 = %f" % pred_std_50)
        print("    pred-std-95 = %f" % pred_std_95)
        print("    empirical-psi-std = %f" % std_pred_psi)
        print("")
        result = {
            "estimation method": est_method,
            "inference_method": inf_method,
            "coverage_rate": coverage_rate,
            "coverage_rate_bc": bc_coverage,
            "pred_std_05": pred_std_05,
            "pred_std_50": pred_std_50,
            "pred_std_95": pred_std_95,
            "empirical-psi-std": std_pred_psi,
        }
        aggregate_results.append(result)

    return aggregate_results


if __name__ == "__main__":
    main()
