# Research directions: CD polynomials and quasi-stationary distributions

Notes from a conceptual discussion about where to take the project.
Captures definitions, key relations, conceptual insights, and concrete
questions to pursue. Not formal — no theorem statements.


## Core objects

### Closed-loop sub-Markov chain
A policy $\pi$ acting on an environment with task parameter $\theta$
(e.g. mass scale) induces a Markov chain $(X_t)$ on the state space $S$.
With termination absorbing into a failure set $F \subset S$, the chain
restricted to $\bar S = S \setminus F$ has a **sub-Markov** transition
kernel:
$$P_\theta(x, A) = \Pr(X_{t+1} \in A,\ \tau > t+1 \mid X_t = x),
  \quad x, A \subset \bar S.$$
The "sub-" reflects that row sums are $\le 1$ — the deficit is the
per-step probability of falling into $F$.

### Quasi-stationary distribution (QSD)
A measure $\nu_\theta$ on $\bar S$ satisfying $\nu_\theta P_\theta =
\lambda_\theta\, \nu_\theta$ with $\lambda_\theta \in (0, 1]$. The Yaglom
limit says that for many initial distributions,
$\mathrm{Law}(X_t \mid \tau > t) \to \nu_\theta$ as $t \to \infty$.
$\nu_\theta$ is the "limit cycle conditional on survival."

### Killing rate / dominant sub-eigenvalue
$\lambda_\theta$ is the Perron–Frobenius (dominant) sub-eigenvalue of
$P_\theta$. From $\nu_\theta P_\theta = \lambda_\theta \nu_\theta$:
$$\Pr(\tau > t \mid X_0 \sim \nu_\theta) = \lambda_\theta^t,$$
so survival time from the QSD is geometric with parameter
$1 - \lambda_\theta$. This is the **asymptotic per-step killing rate**.

### Moment matrix
For a measure $\mu$ on $\mathbb R^n$ and a polynomial basis
$\phi = (\phi_1, \ldots, \phi_N)$ up to degree $d$:
$$M[\mu] = \int \phi(x)\, \phi(x)^\top\, d\mu(x) \in \mathbb R^{N\times N}.$$
Empirically: $M = \frac1n \sum_i \phi(x_i)\phi(x_i)^\top$. **The map
$\mu \mapsto M[\mu]$ is linear** — this is the structural fact that
makes mixture / weighting interpretations work.

### Christoffel-Darboux polynomial
$$p_d(x) = \phi(x)^\top M^{-1} \phi(x), \qquad
  \Lambda_d(x) = 1 / p_d(x).$$
$\Lambda_d$ is the **Christoffel function**; under regularity it
approximates the density of $\mu$. Sublevel sets $\{x : p_d(x) \le c\}$
estimate the support of $\mu$.


## Key relations

- **QSD eigenproblem**: $\nu_\theta P_\theta = \lambda_\theta \nu_\theta$.
- **Survival**: $\Pr(\tau > t \mid \nu_\theta) = \lambda_\theta^t$, so
  $\lambda_\theta = \exp(-\text{killing rate})$.
- **Long-time asymptotics**: from generic start,
  $\Pr(\tau > t \mid X_0 = x) \sim C(x)\, \lambda_\theta^t$.
- **Moment-matrix linearity**: $M[\sum_i w_i \mu_i] = \sum_i w_i\, M[\mu_i]$.
- **Mixture of QSDs**: for a prior $\pi$ on $\theta$,
  $\nu_\pi = \int \nu_\theta\, d\pi(\theta)$, with
  $M[\nu_\pi] = \int M[\nu_\theta]\, d\pi(\theta)$.
- **Christoffel approximates inverse density**:
  $p_d(x) \approx 1/\nu(x)$ on the support (under regularity).


## Conceptual insights

### Two orthogonal axes of the problem
A policy + environment + parameter induces *two* independent objects:

| Object               | What it is                              | Estimated by                     |
|----------------------|-----------------------------------------|----------------------------------|
| **QSD $\nu_\theta$** | shape of behavior conditional on alive  | safe-rollout limit cycle, CD     |
| **Killing rate**     | per-step failure probability long-run   | empirical fail rate, survival   |

A policy can be **shape-robust** (QSD varies smoothly in $\theta$) but
**amplitude-brittle** (killing rate collapses sharply), or vice versa.
Looking at only one axis hides this — e.g., for Hopper, QSDs across mass
$\in [0.6, 1.05]$ look near-identical, but the killing rate spans many
orders of magnitude.

### Survivorship bias in safe-rollout fitting
Conditioning on survival to time $T$ samples from a *weighted* version
of $\nu_\theta$ (weighted by survival probability). At extreme $\theta$,
this set may be small and unrepresentative — the CD fit becomes a
high-variance estimator on a non-typical subset of policy behavior.

### Moment-matrix weighting interpretations
Different choices of $w_\theta \ge 0$ in $M = \sum_\theta w_\theta\,
M[\nu_\theta]$ correspond to different probabilistic objects:

| Weight                            | Interpretation                                    |
|-----------------------------------|---------------------------------------------------|
| $w_\theta = \pi(\theta)$          | mixture of QSDs (population-level)                |
| $w_\theta \propto \lambda_\theta^T$ | unconditional density at time $T$ (survival-weighted) |
| $w_\theta \propto r_\theta / n_\theta$ | unbiased pre-rejection density                  |
| $w_\theta = \pi_{\rm tgt}/\pi_{\rm src}$ | importance-weighted target distribution     |

The survival-weighted choice is the cleanest: it makes the CD polynomial
encode *both* shape and amplitude — total mass = survival probability,
shape = QSD-mixture.


## Directions to pursue

### A. Approximation theory: CD as QSD estimator
- How well does $\Lambda_d \approx \nu_\theta$ when $\nu_\theta$ is the
  QSD of an absorbing chain (not an i.i.d. sample)?
- QSDs often live on low-dimensional invariant manifolds (Floquet-like
  attractors). Does the manifold structure give better convergence rates
  than ambient-dimension bounds?
- What's the right notion of regularity of $\nu_\theta$ for these rates?
- Concrete payoff: turn "CD level set ≈ safe boundary" into a
  quantitative claim with finite-sample guarantees.

### B. Spectral lifting: estimate $\lambda_\theta$ alongside $\nu_\theta$
- The lifted (Koopman / transfer) operator $K_\theta$ on the polynomial
  subspace evolves moment matrices via $M_{t+1} = K_\theta\, M_t$.
- On the safe-restricted chain, $K_\theta$ is sub-stochastic and its
  dominant eigenvalue is $\lambda_\theta$.
- The Yaglom limit gives $M_{t+1} \approx \lambda_\theta\, M_t$ for
  late-time survivor distributions — i.e., $\lambda_\theta$ is read off
  the *ratio* of consecutive moment matrices.
- This is essentially **EDMD for absorbing chains**. The QSD and
  killing rate fall out of one moment-matrix computation, with no
  rejection sampling.
- Unifies the CD detector with Koopman-based safety analysis (drill-down
  below).

### C. Parameter-indexed measures with family-level guarantees
- Take the moment-matrix-as-linear-functional fact seriously: build CDs
  for *families* $\{\nu_\theta\}_\theta$, not point estimates.
- *Minimum-volume CD* over a family: an SDP over the convex hull of
  $\{M[\nu_\theta]\}$ — yields a safety boundary robust by construction.
- *PAC-Bayes / posterior-weighted CD*: $\pi(\theta \mid \text{data})$ as
  the weight, with credible regions on the level set.
- *Importance-weighted moment matrices* for domain adaptation: target
  CD from source data via $w_\theta = \pi_{\rm tgt}(\theta)/\pi_{\rm src}(\theta)$,
  with a closed-form bias-variance tradeoff.

### D. CD as a control object
- $p_d$ is a smooth positive polynomial — naturally a Lyapunov-like
  function on state space.
- **Data-driven control barrier function (CBF)**: when does
  $\{x : p_d(x) \le q_\alpha\}$ admit a control input keeping
  trajectories inside? The CBF and the policy that generated it would
  be consistent by construction.
- **IS proposal for rare-event simulation**: sample from $1/p_d$ to
  estimate $1 - \lambda_\theta$ at extreme $\theta$ without burning 60k
  rollouts on rejection.
- **Safety filter**: solve $\min_u \|u - \pi(x)\|^2$ subject to
  $p_d(f(x, u)) \le q_\alpha$ — a polynomial QCQP, tractable to
  optimality.


## Koopman-based safety analysis (drill-down on direction B)

Koopman analysis lifts a (possibly nonlinear, stochastic) dynamical
system to a *linear* operator on a space of observables. For the
sub-Markov chain $P_\theta$:

- **Transfer operator** (left action on measures): $\mu \mapsto \mu P_\theta$
  — pushes distributions forward in time.
- **Koopman operator** (right action on functions, adjoint of transfer):
  $$(P_\theta h)(x) = \mathbb E[h(X_{t+1})\,\mathbf 1\{\tau > t+1\} \mid X_t = x].$$

Both operators share the same spectrum. The dominant eigenpair has a
beautiful dual structure:

| Side       | Eigenobject         | Meaning |
|------------|---------------------|---------|
| Transfer   | left eigenmeasure $\nu_\theta$    | the QSD — *where* the policy lives, conditional on survival |
| Koopman    | right eigenfunction $h_\theta(x)$ | the **survival function** — $\Pr(\tau > t \mid x) \sim h_\theta(x)\, \lambda_\theta^t$ |

The CD polynomial estimates $1/\nu_\theta$ (transfer side). **Koopman-based
safety analysis estimates $h_\theta$ (function side).** They are dual
halves of the same spectral problem.

### Data-driven fitting (EDMD for absorbing chains)
Given transition pairs $\{(x_t, x_{t+1})\}$ from non-terminated steps,
form
$$G = \tfrac1N \sum_t \phi(x_t)\,\phi(x_t)^\top, \qquad
  A = \tfrac1N \sum_t \phi(x_t)\,\phi(x_{t+1})^\top \cdot \mathbf 1\{\tau > t+1\},$$
and approximate Koopman by $K = G^{-1} A$.
Note: $G$ **is** the moment matrix $M$ used to build the CD polynomial.
Only $A$ is new, and equally cheap. Then:

- Dominant eigenvalue of $K$ = $\lambda_\theta$ → killing rate read off
  the spectrum, no rejection sampling.
- Dominant right eigenvector $v$ → $h_\theta(x) = \phi(x)^\top v$, a
  polynomial survival function.
- Subdominant eigenvalues → mixing time to the QSD vs killing rate
  (separates "fast killer" from "slow drift" failure modes).

### Why $h_\theta$ is the right safety certificate
1. *Lyapunov-like decrease*: by construction,
   $\mathbb E[h_\theta(X_{t+1})\,\mathbf 1\{\tau > t+1\} \mid x] = \lambda_\theta\, h_\theta(x)$,
   so $h_\theta$ decreases in expectation at the **exact killing rate** —
   a stochastic Lyapunov function whose decay rate is identified with a
   physically meaningful quantity.
2. *Spatial survival bound*: superlevel sets $\{x : h_\theta(x) \ge \alpha\}$
   are exactly the regions with asymptotic survival probability $\ge \alpha/\|h_\theta\|_\infty$.
   A *certified* safe set, where CD level sets alone are not — CD says
   "$x$ is in QSD support," $h_\theta$ says "$x$ has high asymptotic
   survival."

### Joint detector / certificate
The natural detector uses both axes:

- $p_d(x) \approx 1/\nu_\theta(x)$ — shape: is $x$ in the QSD support?
- $h_\theta(x)$ — amplitude: given that it is, how survivable is it?

A state can be at low $p_d$ (typical QSD location) but low $h_\theta$
(locally high killing rate) — exactly the **survivorship-bias blind
spot** a safe-rollout-only detector misses.

### Doob $h$-transform: closing the loop
Conditioning the sub-Markov chain on long survival is equivalent to
running a new, *fully Markov* chain — the **$h$-transformed chain** —
with kernel
$$P^h(x, dy) = \frac{h_\theta(y)}{\lambda_\theta\, h_\theta(x)}\, P_\theta(x, dy).$$
It has $\nu_\theta(x)\, h_\theta(x)$ as its stationary distribution. So
once $h_\theta$ is estimated, **you can simulate the $h$-chain to sample
the QSD without rejection** — collapsing the "60k rollouts for 113 safe
ones at mass=1.05" problem into a polynomial-time computation.

### What Koopman analysis unlocks here
1. $\lambda_\theta$ vs $\theta$ — free from the same moment-matrix
   machinery, no separate rejection sweeps.
2. Two-axes detector: $(p_d(x), h_\theta(x))$.
3. Direct QSD sampling at extreme parameters via the $h$-chain.
4. Rare-event killing-rate estimation: IS with $1/h_\theta$ as proposal.
5. Spectral diagnostics: subdominant eigenvalues separate dynamically
   distinct failure modes (mixing vs killing) — invisible to any
   static density estimator.

The takeaway: **CD and Koopman are not alternatives, they're the same
spectral problem from opposite sides.** Doing both gives
$(\nu_\theta, \lambda_\theta, h_\theta)$ — the full data-driven Yaglom
package — from one moment-matrix-style computation.


## Resolved questions

### Q1. Right cross-parameter object?
**Family $\{(\nu_\theta,\lambda_\theta)\}_\theta$ is strictly more
informative than the mixture-QSD; informationally equivalent to the
joint $\rho(x,\theta)$ once a parameter prior $\pi$ of full support is
fixed.**
- Joint built from family: $\rho(x,\theta) := \nu_\theta(x)\pi(\theta)$.
- Family recovered from joint: $\nu_\theta(x) = \rho(x,\theta)/\pi(\theta)$
  if $\pi$ has full support.
- Mixture from family: $\nu_\pi = \int \nu_\theta\,d\pi$.
- Family **not** recoverable from mixture: distinct families can produce
  identical mixtures (e.g., $\{\delta_a,\delta_b\}$ and
  $\{\tfrac12(\delta_a+\delta_b),\tfrac12(\delta_a+\delta_b)\}$ both give
  $\tfrac12(\delta_a+\delta_b)$).

Informational hierarchy: **family $\equiv$ joint $\supsetneq$ mixture**.

*Pragmatic verdict.* Estimate the joint $\rho(x,\theta)$ via the tensor
basis $\phi(x)\otimes\psi(\theta)$. The moment matrix
$M[\rho] = \mathbb E_\rho[(\phi\otimes\psi)(\phi\otimes\psi)^\top]$ has
block structure encoding $\{M[\nu_\theta]\}$ via $\psi$-projection;
conditioning on $\theta$ is a polynomial operation. Mixture and
marginals are derived.

### Q2. Smoothness of $\lambda_\theta$ in $\theta$?
**Generically analytic; non-smoothness only at exceptional points.**

*Theorem (Kato perturbation).* If $\theta\mapsto P_\theta$ is an analytic
family of bounded operators on a Banach space and the dominant
sub-eigenvalue $\lambda_{\theta_0}$ is *simple* and *isolated* (positive
spectral gap), then $\theta\mapsto\lambda_\theta$ is analytic on a
neighborhood of $\theta_0$.

*Proof sketch.* Resolvent $(zI-P_\theta)^{-1}$ jointly analytic in
$(\theta,z)$ off the spectrum; contour-integrate around $\lambda_{\theta_0}$
to get an analytic projection onto a 1-D invariant subspace; hence an
analytic eigenvalue (Kato 1995, Thm VII.1.7). □

*For our setting.* Closed-loop $P_\theta$ depends smoothly on mass;
Perron–Frobenius / Krein–Rutman gives simplicity under
irreducibility+aperiodicity. So $\lambda_\theta$ is generically analytic.

*Disproof of universal smoothness.* Take $P_\theta = (1-\theta)A + \theta B$
for substochastic $A,B$ with distinct dominant eigenvectors and
crossing dominant eigenvalues — $\lambda_\theta = \max(\lambda_A,\lambda_B)$
has a kink at the crossing.

*Hopper cliff is not non-smoothness.* Fail rate $F_\theta \approx 1 - C\lambda_\theta^T$:
- $F_{1.00}\approx 0.42$, $T=800$ ⇒ $\lambda_{1.00}\approx 0.9993$.
- $F_{1.05}\approx 0.998$, $T=800$ ⇒ $\lambda_{1.05}\approx 0.9922$.

A factor-12 change in killing rate over $\Delta\theta = 0.05$ — large
but consistent with smooth $\lambda_\theta$. The apparent cliff in
$F_\theta$ is the exponential amplification $F = 1-\lambda^T$, not a
non-smoothness of $\lambda$. To verify empirically: plot
$-\log\lambda_\theta = -(1/T)\log(1-F_\theta)$ vs $\theta$ — should be
smooth.

### Q3. Does the QSD concentrate on a manifold? Constrained-CD feasible?
**Concentration: yes under attractor + small-noise hypotheses.
Constrained CD: feasible formally, methodologically open.**

*Theorem (small-noise QSD concentration).* If the noiseless closed-loop
map admits a hyperbolic periodic attractor $\gamma$ (smooth 1-D
submanifold) and noise has scale $\epsilon$, then for $\epsilon$ small
enough,
$$\nu_\theta^\epsilon\big(\{x : \mathrm{dist}(x,\gamma)>\delta\}\big)
  \le C\exp(-c\delta^2/\epsilon^2)$$
for $\delta = O(\epsilon^{1/2})$ — exponential concentration in an
$\epsilon$-tube around $\gamma$.

*Proof sketch.* Wentzell–Freidlin LDP for randomly perturbed maps. The
quasi-potential $V(x)$ grows quadratically transverse to $\gamma$
(hyperbolicity); exponential tilting by $V/\epsilon^2$ gives the bound.
Discrete-time version: Kifer (1988). □

*Disproof of universality.* Not every closed-loop system has a periodic
attractor: fixed points give QSDs concentrated on a point; chaotic
attractors give QSDs on fractal sets; multistable systems give
multimodal QSDs. The result is conditional on dynamical structure.

*For Hopper-SAC empirically:* the (θ, z) phase plots show clear closed
curves with a thin noise tube around them — strong evidence for the
periodic-attractor case. Confirmable by checking closed-curve structure
on other 2D slices of the 11-D state.

*Constrained moment methods (feasibility).* For an algebraic variety
$V = \{g_1=\dots=g_k=0\}$, a measure has $\mathrm{supp}(\mu)\subset V$
iff its moments satisfy linear constraints $\langle M, g_i q\rangle = 0$
for all polynomials $q$ of appropriate degree. These reduce to linear
equalities on $M$, giving an SDP-tractable constrained estimator
(Lasserre 2010, Schmüdgen 2017).

*Open obstacles for Hopper.* The limit cycle isn't a known algebraic
variety; learning $g_i$ from data propagates bias; misspecified $g_i$
makes constrained CD provably worse than unconstrained.

### Q4. Right detector test statistic given the two-axes picture?
**The Neyman–Pearson log-likelihood ratio, which decomposes into QSD
shift + transition shift + survival drift.**

*Theorem (NP-optimal LR for sub-Markov OOD).* Among level-$\alpha$ tests
of $H_0:\theta=\theta^*$ vs simple $H_1:\theta=\theta'$ based on a
non-terminated trajectory $X_{0:t}$, the UMP test rejects when
$$\Lambda_t = \log\frac{\nu_{\theta'}(X_0)}{\nu_{\theta^*}(X_0)}
  + \sum_{s=0}^{t-1}\log\frac{p_{\theta'}(X_s,X_{s+1})}{p_{\theta^*}(X_s,X_{s+1})}
  - t\log\frac{\lambda_{\theta'}}{\lambda_{\theta^*}} > c_\alpha.$$

*Proof.* Conditional joint law given $\{\tau>t\}$ and $X_0\sim\nu_\theta$
factors as
$L_\theta = \nu_\theta(X_0)\prod_s p_\theta(X_s,X_{s+1})/\lambda_\theta^t$.
Log-ratio gives $\Lambda_t$; Neyman–Pearson optimality follows. □

*Three terms, three meanings:*
- $\log\nu_{\theta'}/\nu_{\theta^*}$ — **QSD shift** (CD estimates this).
- $\sum_s\log p_{\theta'}/p_{\theta^*}$ — **transition shift** (Koopman /
  EDMD estimates this).
- $-t\log\lambda_{\theta'}/\lambda_{\theta^*}$ — **survival drift**
  (dominant sub-eigenvalue gives this).

*Composite $H_1$:* GLR replaces numerator with $\sup_{\theta'}L_{\theta'}$.
Bayesian: integrate against a prior on $\theta'$.

*Two-axes structure.* Killing the transition term ⇒ CD-only detector
(blind to dynamics shift). Killing the QSD term ⇒ survival-only
detector (blind to spatial shift). NP-optimality requires both, by
construction.

### Q5. Connections — LDP, SoS density, conformal time-series
Substantive content rather than yes/no.

*Donsker–Varadhan variational form.* For sub-stochastic $P_\theta$,
$$-\log\lambda_\theta = \inf_{\mu\in\mathcal P(\bar S)}
  \Big\{H(\mu\Vert\mu P_\theta) + \int(1-P_\theta(x,\bar S))\,d\mu\Big\},$$
with infimum at $\mu=\nu_\theta$ (Hennion–Hervé 2001). Gives (i) a
variational estimator of $\lambda_\theta$ alternative to spectral
methods; (ii) a rate function for LDP control over how often the LR
detector crosses thresholds under $H_0$ — i.e., the route to finite
sample FP/FN bounds.

*CD is canonical SoS.* Diagonalize $M^{-1} = \sum_i \sigma_i^{-1}u_iu_i^\top$:
$$p_d(x) = \phi(x)^\top M^{-1}\phi(x) = \sum_i \sigma_i^{-1}(u_i^\top\phi(x))^2
  = \sum_i q_i(x)^2.$$
So CD is the SoS density estimator with data-driven squares. The
Lasserre hierarchy gives moment-relaxation bounds for $\lambda_\theta$
in the same machinery — density estimation and eigenvalue estimation
unified.

*Conformal time-series.* A QSD-stationary sub-Markov chain has
exchangeable marginal-at-time-$t$ (conditional on survival, after
burn-in). Conformal prediction with CD scores gives finite-sample FP
control under $H_0$. For $H_1$ conformal $p$-values aren't exchangeable
and the bound fails — but a *change-point* variant (alarm on persistently
small empirical $p$-values) gives ARL-type control under $H_0$ without
distributional assumptions. **Combining GLR (Q4) for power with
conformal thresholding for FP control is the route to a certified
detector with finite-sample guarantees.**


## Remaining open problem

The genuinely open methodological question (from Q3): **how to identify
the QSD's underlying manifold from data without circularly using CD**.
Candidate approaches — non-linear dimensionality reduction (diffusion
maps, kernel PCA on the QSD samples), then polynomial regression to get
$g_i$; learning $g_i$ via score-matching; or estimating tangent spaces
locally and integrating. The bias-variance question (when does
constrained CD with learned manifold beat unconstrained CD?) needs both
theory and experiment.


## Diagnostic question

Where this project ultimately lands depends on which of the following
the goal is:

1. A *certified safety filter* (control direction): CD as CBF, safety
   guarantees on closed-loop dynamics.
2. A *certified anomaly detector* (statistical direction): CD test
   statistic with finite-sample false-positive / false-negative bounds.
3. A *theory paper about data-driven analysis of controlled sub-Markov
   chains* (mathematical direction): CD + Koopman + QSD theory unified.

Each ordering of A–D above serves a different one.


## JOINT-CD: an end-to-end algorithm

A concrete pipeline synthesizing the insights — moment-matrix linearity,
spectral lifting, dual eigenobjects, NP-optimal LR, conformal calibration,
Doob $h$-transform — into one detector with finite-sample guarantees.

### Setup
- Training data $\mathcal D = \{(\theta^{(i)}, X_{0:T_i}^{(i)}, \tau_i)\}_{i=1}^N$
  with $\theta^{(i)} \sim \pi_{\text{train}}$.
- Polynomial bases $\phi: \mathbb R^n \to \mathbb R^{N_x}$ on state and
  $\psi: \Theta \to \mathbb R^{N_\theta}$ on parameter.
- Burn-in $T_b$, significance $\alpha$, candidate alternatives grid
  $\{\theta'\}\subset\Theta$, nominal $\theta^*$.

### Output
- Functionals $\theta\mapsto \hat M(\theta), \hat\lambda_\theta, \hat h_\theta(\cdot)$
  (polynomial in $\theta$).
- Calibrated trajectory-level threshold $c_\alpha$ with FP rate $\le\alpha+O(1/M)$
  under $H_0$.

### Phase 1 — Polynomial regression of moment matrices
Fit each entry of the moment matrix as a polynomial in $\theta$ (uses
moment-matrix linearity in $\mu$ and linearity in measure family
parameter):
```
For each transition (θ^(i), x_t^(i), x_{t+1}^(i)), t ≥ T_b, τ_i > t+1:
  Y_M[i,j] = φ_i(x_t) · φ_j(x_t)              # moment-matrix entry
  Y_A[i,j] = φ_i(x_t) · φ_j(x_{t+1})          # time-shifted entry
Regress each Y_M[i,j], Y_A[i,j] on ψ(θ^(i))   # one regression per entry
→ polynomials m_{ij}(θ), a_{ij}(θ)
```
Embarrassingly parallel; one-shot over the entire training set.

### Phase 2 — Spectral package at any query $\theta_0$
```
M(θ_0) := [m_{ij}(θ_0)]    A(θ_0) := [a_{ij}(θ_0)]
K(θ_0) := (M(θ_0) + εI)^{-1} A(θ_0)            # ridge-regularized Koopman
eigvals, R, L := generalized_eig(K(θ_0))
j := argmax_modulus(eigvals)
λ_{θ_0} := eigvals[j]                            # killing rate complement
h_{θ_0}(x) := φ(x)^T R[:,j]                      # survival function (Koopman-side)
p_d^{θ_0}(x) := φ(x)^T M(θ_0)^{-1} φ(x)         # CD polynomial (transfer-side)
```
The CD polynomial estimates $1/\nu_{\theta_0}$; the dual right
eigenfunction $h_{\theta_0}$ estimates the survival function; their
shared eigenvalue is the killing rate complement.

### Phase 3 — Two-axes log-likelihood ratio (NP-optimal)
For test trajectory $X_{0:t}$, candidate $\theta'$ vs nominal $\theta^*$:
```
Λ_t(θ'; θ*) = log(p_d^{θ*}(X_0)) - log(p_d^{θ'}(X_0))                       # QSD shift
            + Σ_{s=1..t} [log(p_d^{θ*}(X_s)) - log(p_d^{θ'}(X_s))]          # density drift
            - t · log(λ_{θ'} / λ_{θ*})                                       # survival drift
Λ_t = max_{θ' ∈ grid} Λ_t(θ'; θ*)                                            # GLR
```
The two axes appear as distinct terms: density drift (shape/CD-side)
and survival drift (amplitude/Koopman-side). Setting either to zero
recovers a one-axis detector with the corresponding blind spot.

### Phase 4 — Conformal calibration
Held-out nominal trajectories $\{X_{0:t}^{(j)}\}_{j=1}^M$ from
$\theta=\theta^*$:
```
calibration_scores := [Λ_t(traj^(j)) for j = 1..M]
c_α := (1 - α)-quantile of calibration_scores
```
Trajectory-level FP guarantee:
$\Pr_{H_0}(\Lambda_t > c_\alpha) \le \alpha + O(1/M)$, by conformal
exchangeability after QSD burn-in.

### Phase 5 — Online sequential detection
```
Λ ← 0
For x in stream:
  Λ ← incremental_update(Λ, x)        # O(N_x^2) per timestep
  If Λ > c_α: ALARM
  If Λ < 0:  Λ ← 0                    # Page CUSUM reset
```

### Phase 6 (optional) — Doob $h$-transformed sampling
Sample the QSD at any $\theta$ without rejection (replaces "60k rollouts
for 113 survivors at mass=1.05"):
```
x ← initial state
For step = 1..N:
  x_prop ← forward_simulate(θ, x)
  weight ← h_θ(x_prop) / (λ_θ · h_θ(x))
  Accept x_prop with prob min(1, weight)
  Yield x                              # samples from ν_θ · h_θ;
                                       # reweight by 1/h_θ for ν_θ
```

### How insights map onto phases
| Insight                                  | Phase                                |
|------------------------------------------|--------------------------------------|
| Moment matrix linear in measure          | 1 (polynomial regression of entries) |
| Joint $\rho(x,\theta)$ as natural object | 1 (tensorized basis / regression)    |
| Dual eigenobjects (QSD ↔ survival fn)    | 2 (left/right of same eigenproblem)  |
| Killing rate from sub-eigenvalue         | 2 (no rejection needed)              |
| NP-optimal LR has three terms            | 3 (QSD + density + survival drifts)  |
| Two-axes structure                       | 3 (terms are dynamically distinct)   |
| Conformal finite-sample FP control       | 4                                    |
| Doob $h$-transform for sampling          | 6                                    |

### Cost
- Training (1–2): $O(N_x^2 N_\theta \sum_i T_i)$ regressions; parallel.
- Query (2 spectral): $O(N_x^3)$ per $\theta$; cacheable.
- Online (5): $O(N_x^2)$ per timestep; real-time at robotics scale.

### What it provides over the current approach
1. **Both axes** — survival drift catches "in-QSD but locally
   low-survival" states that CD-only misses.
2. **Smooth across $\theta$** — polynomial-regressed moments avoid
   per-bin retraining.
3. **Rejection-free** — $\lambda_\theta$ from the spectrum; QSD samples
   from the $h$-chain (Phase 6).
4. **Finite-sample FP guarantee** — conformal, not just empirical q90.
5. **Failure-mode identification** — relative size of density vs
   survival drift terms distinguishes *shape change* from *amplitude
   change* in the closed-loop chain. Two distinct dynamical failure
   modes, separately observable.
