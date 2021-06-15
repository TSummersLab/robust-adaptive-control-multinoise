# robust-adaptive-control-multinoise
## Robust adaptive control via multiplicative noise from bootstrapped uncertainty estimates
The code in this repository implements the algorithms and ideas from our paper:

"Robust Learning-Based Control via Bootstrapped Multiplicative Noise"

* [Proceedings of Machine Learning Research (PMLR) / Learning for Dynamics & Control (L4DC) 2020](http://proceedings.mlr.press/v120/gravell20a.html)
* [arXiv](https://arxiv.org/abs/2002.10069)

There is no formal installation procedure; simply download the files and run them in a Python console.

## Dependencies
* NumPy
* SciPy
* Matplotlib

## Code
### monte_carlo_comparison.py
Perform the simulation of the adaptive control algorithms.
Note that Ns = 100000 Monte Carlo samples will take several hours to run on a quadcore CPU.

### monte_carlo_comparison_loader_paper.py
Generate plots as shown in the paper from saved simulation results.

### monte_carlo_comparison_loader.py
Generate plots for diagnostic viewing from saved simulation results.
