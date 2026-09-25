# `examples/charge_deposition/`: coarse GPR on a coax delamination sweep

A standalone, small-data GPR surrogate for the charge deposited in the center
conductor of a coaxial cable with a delamination void. The data are 11
SCEPTRE cases. This directory does not use the diffusion sandbox or POD.

| File | Role |
|---|---|
| `coarse_gpr_delam.py` | Load cases, fit GPR, leave-one-out cross-validation, parity and 1D slice plots |
| `sceptre_case_key.txt` | Case id → $(\theta_0, \Delta\theta, \delta)$ |
| `Charge_Deposition_Total_Block_Case{0..10}.txt` | Per-block total deposited charge for each case |
| `gpr_coarse_out/` | `loo_predictions.csv`, `parity_loo.png`, `slices_nominal.png` |

(`*:Zone.Identifier` files are Windows download metadata and can be ignored;
`.gitignore` excludes them.)

## 1. Problem

Inputs, $x = (\theta_0, \Delta\theta, \delta)$:

- $\theta_0$: angle of the void centroid (deg), 0 or 90 in the data;
- $\Delta\theta$: void half-width (deg), 30 to 75;
- $\delta$: void radial thickness (cm), 0.002 to 0.01.

Output: $Q_\text{center}$, the total charge deposited in block
`center_conductor_Al` (C), read from the `Value` column of each
`Charge_Deposition_Total_Block_Case*.txt`.

Case 1 ($\theta_0=0$, $\Delta\theta=1$, $\delta=10^{-4}$) is effectively the
no-void baseline and is dropped (`DROP_CASES = {1}`), leaving $N = 10$. The
design is not space-filling: most cases vary one input at a time around
$(90, 30, 0.002)$ or $(0, \cdot, 0.002)$.

## 2. Model

One scikit-learn GP with the ARD kernel used across the repo,

$$
k(x,x') = \sigma_f^2\exp\!\Big(-\tfrac12\sum_{j=1}^3\frac{(x_j-x'_j)^2}{\ell_j^2}\Big) + \sigma_n^2\,\delta_{xx'} ,
$$

with bounds adjusted for $N = 10$: $\ell_j\in[10^{-2}, 10^2]$,
$\sigma_n^2\in[10^{-10}, 1]$, and 20 optimizer restarts. Inputs are min-max
scaled to $[0,1]$. The raw scales differ by about four orders of magnitude
($\delta\sim10^{-3}$ vs. $\theta\sim10^2$), and scaling keeps a single
initial length scale reasonable for all three. Outputs are z-scored. The
math is in
[sklearn_implementation/README.md §2](../../sklearn_implementation/README.md#2-standard-gpr-on-pod-coefficients-gpr_podpy).

## 3. Leave-one-out cross-validation

With 10 points a held-out test set isn't practical, so `loo_cv()` refits the
GP $N$ times, each time on $N-1$ points. The scaling and standardization are
refit on each training fold, so nothing from the held-out point leaks into
its own prediction. For each case it records the prediction, the predictive
standard deviation, the absolute error, and the standardized residual

$$
z_i = \frac{y_i - \hat y_{-i}}{\hat\sigma_{-i}} .
$$

If the predictive uncertainty is calibrated, the $z_i$ look like draws from a
standard normal.

**Results** (`gpr_coarse_out/loo_predictions.csv`): all $Q_\text{center}$
values lie between $-6.53\times10^{-9}$ and $-6.30\times10^{-9}$ C. The LOO
absolute errors range from $2.4\times10^{-12}$ to $1.4\times10^{-10}$ C (0.04%
to 2.2% of the value), mean $3.4\times10^{-11}$. The largest error is case 3,
the largest $\delta$, which is an extrapolation along $\delta$. Three of ten
$\lvert z_i\rvert$ exceed 2 (cases 0, 7, and 8; largest 3.46 for case 8). For
a calibrated Gaussian about one in twenty would, so the GP's predictive
standard deviation is too small for this data set and its error bars should
not be taken at face value.

## 4. Slices

`plot_slices()` fits on all $N$ cases and plots the posterior mean and
$\pm2\sigma$ along each input over its sampled range, with the other two held
at the per-input median of the design. Training points whose other
coordinates are close to that point are overlaid.
These show what the surrogate assumes about each direction; with this few
points they are a sanity check, not a result.

## 5. Running

```bash
cd examples/charge_deposition
python3 coarse_gpr_delam.py                   # data in the current directory
python3 coarse_gpr_delam.py --data-dir PATH   # or elsewhere
```
