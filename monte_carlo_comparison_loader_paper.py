"""
Robust adaptive control via multiplicative noise from bootstrapped uncertainty estimates.
"""
# Authors: Ben Gravell and Tyler Summers


from plotting import multi_plot_paper
import os
from monte_carlo_comparison_loader import load_results, load_problem_data, get_last_dir


if __name__ == "__main__":

    experiment_folder = 'experiments'

    experiment = 'last'

    if experiment == 'last':
        experiment = get_last_dir(experiment_folder)

    dirname_in = os.path.join(experiment_folder, experiment)

    training_type = 'offline'
    testing_type = 'online'

    # Load the problem data into the main workspace (not needed for plotting)
    (training_type, testing_type, Ns, Nb, T, system_idx, seed,
    bisection_epsilon, t_start_estimate, t_explore, u_explore_var, u_exploit_var,
    n, m, A, B, Q, R, W) = load_problem_data(dirname_in)

    # Load results
    output_dict, cost_are_true, t_hist, t_start_estimate = load_results(dirname_in, training_type, testing_type)

    # # Print the log data, if it exists
    # try:
    #     for log_str in output_dict['robust']['log_str']:
    #         print(log_str)
    # except:
    #     pass

    # Plotting
    multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, save_plots=True, dirname_out=dirname_in)