# VMM
Code for reproducing experiments for paper "The Variational Method of Moments".

Estimation or experiments can be run by executing 
"run_experiments_TYPE.py", where TYPE is "estimation" or "inference"
respectively.

Default directories for saving results for both kinds of experiments
can be changed by editing the value of "save_dir" at the top of the
respective "run_exepriments_TYPE.py" script.

Details of what methods and hyperparameters to use for the
experiments can be changed by editing the experiments configuration
files within the "experiment_setups" directory (this includes for
example the option of running experiments in parallel using multiple
processes by changing the value of "num_procs" to something greater
than one.) In addition, you can change which estimation or inference
experiments to run by editing the value of "setup_list" at the top
of the respective "run_experiments_TYPE.py" script. 
 
Results from experiment(s) will be stored within JSON files created
upon experiment completion, in the respective save directory. The
results files contain 3 keys:
- "aggregate_results": contains aggregate statistics from the
    experiments, including all numbers presented in the results tables
    within the paper and supplement
- "results": contains detailed individual results for each experiment
    replication (from which different aggregate statistics or plots
    could be created without re-running experiments)
- "setup": a summary of the setup configuration used for
    the corresponding experiment

Experiments were run using Python 3.8.5. Versions of libraries used
are summarized in the included "requirements.txt" file.
