# Reframing the project as state-conditional survival analysis

Notes following the empirical finding (May 2026) that the closed-loop
forward-invariant set is empty for hopper-base under the SAC expert:
0 / 2000 episodes survive 10⁶ steps; the 15 that survive 10⁴ all fail
between 10 008 and 15 544. The "estimate the forward-invariant set
from samples" framing therefore has nothing to estimate. This note
sketches survival analysis as a replacement framing.


## Why the existing framing breaks

A one-class detector trained on "survivors" implicitly assumes
$\{\text{survived } T\}$ is a sample from a forward-invariant set
$\mathcal{S}_\infty = \{x : \Pr^\pi(\tau = \infty \mid x_0 = x) > 0\}$.
Under generic stochastic dynamics and a non-Lyapunov-stable policy,
$\mathcal{S}_\infty$ is empty or measure-zero w.r.t. the init
distribution — every state has a positive per-step killing rate, so
$\Pr(\tau > t) \to 0$. What "survivors" actually sample is
$\{x : T_\text{fail}(x) > T\}$, a *right-tail* of the time-to-failure
distribution, contaminated by pre-failure transients. See
[true_failure_dist.py](../scripts/misc/true_failure_dist.py) and
[survivors_extreme.py](../scripts/misc/survivors_extreme.py) for the
empirics.

The reframe: stop asking "is $x$ safe?" (binary, requires invariance)
and start asking "what is the distribution of $T_\text{fail} \mid x$?"
(continuous, well-defined with right-censoring). This is the standard
setting of **survival analysis**.


## The object of interest

Let $T_\text{fail}(x)$ be the (random) first time the closed-loop
trajectory leaves the healthy set, starting from $x$. The
state-conditional **survival function**
$$S(t \mid x) = \Pr^\pi\!\big(T_\text{fail}(x) > t\big)$$
and the **hazard rate**
$$h(t \mid x) = \lim_{\Delta \to 0} \frac{1}{\Delta}\,
  \Pr\!\big(T_\text{fail}(x) \in [t, t+\Delta) \mid T_\text{fail}(x) > t\big)$$
are the two equivalent representations
($S(t \mid x) = \exp(-\int_0^t h(s \mid x)\, ds)$). The detector's
job becomes: estimate one of these from data, with right-censoring
handled honestly.


## Data construction

For each trajectory $i$ and timestep $t \in [0, \min(T_{\text{fail},i},
T_\text{cap}))$, build a sample
$$(x_{i,t},\ T_{i,t},\ \delta_{i,t}),
  \qquad T_{i,t} = T_{\text{fail},i} - t,
  \quad \delta_{i,t} \in \{0, 1\}.$$
$\delta = 1$ if $T_{i,t}$ is observed (trajectory failed),
$\delta = 0$ if right-censored at the rollout cap. With the artifacts
we have:

| source | $T_{\text{fail},i}$ | $\delta$ |
| -- | -- | -- |
| `fail < 1000` (in-window failure) | `fail[i]` | 1 |
| survivor with continuation failure (`true_fail.npz`) | `true_fail[i]` | 1 |
| survivor still alive at cap | cap | 0 |

For hopper-base at cap = 10⁶: 1985 uncensored, 15 censored. With $T$
samples per trajectory, ~2 M state-level samples — well-posed.


## Two estimators, ML-level

### Kaplan–Meier (population baseline, no state)

Non-parametric estimator of an unconditional $\hat{S}(t)$. Sort all
observed failure times $t_1 < t_2 < \cdots$. At each $t_k$,
$$\hat{S}(t_k) = \hat{S}(t_{k-1}) \cdot \Big(1 - \frac{d_k}{n_k}\Big),$$
where $d_k$ is the number of failures at $t_k$ and $n_k$ is the at-risk
population. Censored samples leave $n_k$ without contributing a $d_k$.

Use for: sanity check, stratified comparisons (e.g. survival curves
split by quantile of CD score), and to fit the baseline hazard $h_0$
that Cox-style models multiply against.

### Cox proportional hazards (the workhorse)

$$h(t \mid x) = h_0(t)\, \exp\!\big(f_\theta(x)\big).$$
$h_0(t)$ left non-parametric (estimated post-hoc, KM-style);
$f_\theta(x)$ is a scalar **log-risk score** — linear in classical
Cox, a neural net in DeepSurv, a *Christoffel-Darboux polynomial in
our setting*.

Fit by minimising the negative partial log-likelihood
$$\mathcal{L}(\theta) = -\sum_{i:\, \delta_i = 1}
  \Bigg[\,f_\theta(x_i)\ -\ \log\!\!\sum_{j \in R(T_i)} e^{f_\theta(x_j)}\,\Bigg],$$
where $R(T_i) = \{j : T_j \ge T_i\}$ is the at-risk set at time $T_i$.

It's literally a **softmax-over-the-at-risk-set, summed over actual
failure events**:

- It's a *ranking* loss — only the order of $f_\theta$ across the
  at-risk set matters, not its scale. Cf. listwise learning-to-rank.
- Censored samples never appear as the numerator (no failure event)
  but do appear in the denominators (they're at-risk). So they
  contribute "this sample also survived past $t$" information without
  forcing a value for their unknown $T$.
- The baseline $h_0(t)$ drops out of $\mathcal{L}$ entirely — Cox's
  original trick. Estimated separately afterwards via Breslow.

Predictions: per state $x$, $f_\theta(x)$ is a calibrated log-risk
that's monotone-decreasing in expected $T_\text{fail}$. Paired with
$\hat{h}_0(t)$ you recover a full survival curve $\hat{S}(t \mid x)$.

In code, < 30 lines:
```python
from lifelines import CoxPHFitter, KaplanMeierFitter     # linear / non-param
# or
from pycox.models import CoxPH                            # NN, DeepSurv-style
```

## Plugging into the existing CD pipeline

The current detector computes $\mathrm{CD}_\mu(x) = \phi(x)^\top
M[\mu]^{-1} \phi(x)$ (or its kernelised variant). Two ways to
re-purpose it under the survival framing:

1. **CD as feature.** Use $\log \mathrm{CD}_\mu(x)$ (and a few
   summary statistics) as the input vector to a linear Cox model.
   Cheapest experiment; tests whether the CD score is already a useful
   risk score.

2. **CD as risk function $f_\theta$.** The polynomial coefficients
   inside $M[\mu]^{-1}$ are the trainable parameters; fit them
   directly with the partial-likelihood loss. The architecture is
   unchanged; only the loss switches from "match the empirical
   moments of nominal data" to "rank pre-failure states above
   long-survival states, conditional on the at-risk set." Censored
   trajectories enter the loss for the first time — no more discarding
   information.

(2) is the bigger ask but the more interesting result: the same
function class, given a probabilistically meaningful loss.


## Bridge to the QSD framing already in [research_directions.md](research_directions.md)

The QSD framing centers on $\lambda_\theta$, the dominant
sub-eigenvalue of the killed transition kernel, with the interpretation
$\Pr(\tau > t \mid X_0 \sim \nu_\theta) = \lambda_\theta^t$. That's
exactly an *unconditional, constant-hazard* survival curve with hazard
$-\log \lambda_\theta$. Survival analysis generalises this in two ways
we care about:

- It lets the hazard be **time-varying** ($h_0(t)$ via KM/Breslow),
  capturing the empirically obvious fact that hopper hazard spikes
  shortly after the recorded window (most survivors fail in
  $[1000, 1100]$).
- It lets the hazard be **state-conditional** ($\exp f_\theta(x)$),
  which the QSD framing handles only through the spatial profile of
  $\nu_\theta$ on the survivors' manifold.

So survival analysis is the *empirically estimable* counterpart to the
QSD picture: the partial-likelihood loss is what you run instead of
trying to spectrally decompose an unobserved sub-Markov kernel.


## Evaluation

MSE on $T$ is wrong (breaks under censoring). Standard survival
metrics:

- **Concordance index** (Harrell's C): the probability that the model
  orders two comparable samples correctly w.r.t. their true
  $T_\text{fail}$. Censoring-aware (compares only pairs where the
  ordering is identifiable). 0.5 = random, 1.0 = perfect.
- **Time-dependent ROC / AUC at horizon $H$**: for the binary task
  "fail within $H$ steps?", standard ROC on $f_\theta(x)$. Lets you
  report per-horizon performance.
- **Integrated Brier score**: calibration of $\hat{S}(t \mid x)$
  against the empirical Bernoulli outcome at each $t$. Penalises both
  miscalibration and poor discrimination.

The existing FPR / detection-rate / median-TTD metrics map cleanly
onto these (FPR ↔ time-dependent specificity; detection rate ↔
time-dependent sensitivity; median-TTD ↔ a calibration check on
$\hat{S}^{-1}(0.5 \mid x)$).


## Does max-conformal calibration survive the reframe?

Yes, with cleaner semantics. Currently, max-conformal returns a
threshold $\tau$ such that for a fresh nominal trajectory, the maximum
score stays below $\tau$ with marginal probability $1 - \alpha$. Under
the survival framing the same calibration says: with probability
$1 - \alpha$, the max risk score along a nominal trajectory stays
below $\tau$. The covered event is still about trajectory-level
behaviour, no invariance claim required. If we want a horizon-$H$
statement instead ("the score exceeds $\tau$ before the trajectory
truly fails"), conformal time-to-event quantile regression (Candès
et al.) is the appropriate generalisation.


## Concrete first experiments

1. **KM baseline.** Fit unconditional $\hat{S}(t)$ on the full
   hopper-base dataset using `true_fail.npz`. Plot vs $t$. Check
   shape: pure-geometric (constant hazard) or heavy early hazard
   followed by a slower tail? The shape directly informs whether a
   constant-hazard QSD approximation is even plausible.
2. **Linear Cox on CD score.** Compute $\mathrm{CD}_\mu(x)$ for every
   nominal state. Fit a Cox model with that scalar (and maybe a few
   handcrafted features) as input. Report c-index against
   `true_fail`-derived per-state labels. Beats current detector ranked
   by raw CD score?
3. **CD-as-$f_\theta$.** Re-derive the CD polynomial coefficients to
   minimise the partial-likelihood loss instead of moment-matching.
   Compare c-index and time-dependent AUC against (2). Is the
   density-based fit already nearly optimal, or does survival-loss
   training shift the polynomial materially?
4. **Stratified KM by detector regime.** Split states by quintile of
   $\mathrm{CD}_\mu(x)$, fit a KM curve per stratum. If detector ↑ ⇒
   $\hat{S}$ ↓ monotonically, the current detector is already a useful
   risk score; if not, the survival loss is doing real work.


## Open questions

- **Independence violations.** State samples within one trajectory are
  highly correlated; the partial likelihood assumes independent
  observations. Cluster-robust variance (Wei–Lin–Weissfeld) or
  trajectory-level subsampling needed for honest uncertainty.
- **Choice of $T_\text{cap}$.** With ~99.25% of trajectories failing
  by 10 000, $T_\text{cap}$ in [10 000, 20 000] is probably enough;
  the 0.75% censored tail is harmless for Cox (handled by construction)
  but matters for KM at large $t$.
- **Domain randomisation.** The non-base hopper datasets vary
  mass/friction/damping per trajectory. A natural extension is a
  *frailty model* (random effects on the hazard per task parameter)
  or just appending the task parameter to $x$ and letting $f_\theta$
  absorb it.
- **Connection to the value function.** $\mathbb{E}[T_\text{fail} \mid x]
  = \int_0^\infty S(t \mid x)\, dt$ is precisely a value function for
  the time-to-failure reward. Survival analysis and TD learning are
  estimating the same object via different losses — worth a section
  on which loss has better sample efficiency.
