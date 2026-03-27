# Complete Pseudocode for the Repo-Aligned Mean-Variance CTRL Stack

This note gives the **implementation-target pseudocode** for the repository.
It is intentionally aligned with:

- **Zhou–Li (2000)** for the synthetic-data oracle benchmark,
- **Huang–Jia–Zhou (2025)** for the theorem-aligned CTRL baseline,
- **Huang–Jia–Zhou (2022)** and the **2025 E-companion** for practical online improvements.

It does **not** make **Wang–Zhou (2019/2020)** EMV a required implementation layer. That paper remains a mathematical and derivational reference for:

- entropy-regularized exploratory control,
- Gaussian exploratory structure,
- the role of the outer-loop $w$ update.

## Source map

Primary paper-to-code map:

- **Zhou–Li (2000)**: closed-form oracle benchmark under known coefficients.
- **Huang–Jia–Zhou (2025)**: martingale critic conditions, stochastic actor updates, deterministic execution policy, convergence/regret Black–Scholes benchmark.
- **Huang–Jia–Zhou (2022)**: leverage control, risky-only projection, rebalance/update schedule separation, TD($\lambda$)-style traces.
- **Wang–Zhou (2019/2020)**: reference only for exploratory Gaussian intuition and outer-loop $w$ handling.

Repository source anchors:

- `references/portfolio_mv_papers_algorithm_summary.md`, Sections 1, 3, 4, and 6
- `references/portfolio_mv_papers_companion_implementation_notes.md`, Sections 1, 3, and 4

Note on citations:

- This environment has limited local PDF text extraction.
- The note below cites the paper names and the repository summary sections that already distill the relevant formulas.
- Exact paper page numbers should be manually confirmed against the PDFs in `references/papers/` before publication-quality documentation is finalized.

---

## 1. Core notation

Let:

- $x_t$ be wealth,
- $F_t$ be optional factor / context information,
- $u_t \in \mathbb{R}^d$ be the **dollar allocation** vector in risky assets,
- $r$ be the risk-free rate,
- $\mu, \sigma$ be the synthetic-data coefficients when known,
- $z$ be the configured target terminal wealth, corresponding to `reward.target_return`,
- $w$ be the outer-loop mean-variance target / Lagrange parameter used in the RL formulation,
- $\gamma_{\text{embed}}$ be the auxiliary classical embedding scalar from the Zhou–Li formulation; it is a derived oracle quantity, not a config field,
- $T$ be the horizon,
- $\Delta t$ be the discrete rebalance/update interval,
- `entropy_temp` be the entropy regularization temperature from `reward.entropy_temp`,
- `discount` be the discrete-time engineering discount from `reward.discount` if a discounted implementation is introduced,
- $\pi_\phi(\cdot \mid t, x_t, F_t; w)$ be the stochastic behavior policy,
- $\hat u_\phi(t, x_t, F_t; w)$ be the deterministic execution policy.

Baseline actor / critic structure for the Black–Scholes synthetic benchmark:

$$
J_\theta(t,x;w)
=
(x-w)^2 e^{-\theta_3(T-t)}
+\theta_2(t^2-T^2)
+\theta_1(t-T)
-(w-z)^2,
$$

$$
\pi_\phi(\cdot \mid t,x;w)
=
\mathcal N\!\left(
-\phi_1(x-w),
\ \phi_2 e^{\phi_3(T-t)}
\right),
$$

with deterministic execution policy

$$
\hat u_\phi(t,x;w) = -\phi_1(x-w).
$$

This is the implementation target described in the 2025 paper summary and companion notes.

---

## 2. Algorithm A: Synthetic Oracle Benchmark

Purpose:

- provide a model-based upper-reference policy in synthetic known-parameter experiments,
- validate the simulator,
- check whether the learning algorithm recovers the correct structure.

For the multi-asset synthetic benchmark, use the Zhou–Li auxiliary-control solution:

$$
\bar u(t,x)
=
[\sigma(t)\sigma(t)^\top]^{-1} B(t)^\top
\left(\gamma_{\text{embed}} e^{-\int_t^T r(s)\,ds} - x\right),
$$

where $B(t) = b(t) - r(t)\mathbf{1}$ and $\gamma_{\text{embed}} = \lambda / (2\mu)$ in the classical embedding.

Implementation note:

- in repo config terms, `reward.target_return` maps to $z$;
- $\gamma_{\text{embed}}$ is an oracle-side derived quantity and should not be confused with `reward.target_return`.

### Pseudocode

```text
Algorithm A: OracleMVBenchmark
Input:
  synthetic coefficients (r, b, sigma)
  horizon T, grid {t_k}
  initial wealth x_0
  target-return / tradeoff setting
Output:
  oracle policy handle
  oracle wealth paths
  oracle summary metrics

1. Compute B = b - r * 1 and the required oracle control coefficients.
2. Build the closed-form oracle policy u_oracle(t, x).
3. For each evaluation episode:
4.   Set x <- x_0.
5.   For k = 0, ..., K-1:
6.     Evaluate u_k = u_oracle(t_k, x_k).
7.     Step the synthetic environment with u_k.
8.     Store wealth, action, and diagnostics.
9. Aggregate wealth paths and evaluation metrics.
10. Save oracle outputs so learned policies can be compared against them.
```

Implementation requirement:

- expose the oracle strategy through configuration so synthetic experiments can switch between `oracle`, `ctrl_baseline`, and later `ctrl_online`.

---

## 3. Algorithm B: Theorem-Aligned CTRL Baseline

Purpose:

- implement the main Huang–Jia–Zhou (2025) research baseline,
- learn directly from trajectories without estimating coefficients,
- preserve the stochastic learning policy / deterministic execution-policy split.

### Mathematical backbone

Use the martingale policy-evaluation condition described in the 2025 summary:

$$
\mathbb{E}\left[
\int_0^T I(t)\{dJ + \texttt{entropy\_temp} \cdot \log \pi \, dt\}
\right] = 0,
$$

together with actor updates for:

- actor mean parameter $\phi_1$,
- actor covariance (or covariance inverse) parameterization,
- outer-loop $w$ update enforcing the mean target.

### Pseudocode

```text
Algorithm B: CTRLBaseline
Input:
  stochastic synthetic environment
  critic parameters theta
  actor parameters phi
  outer-loop parameter w
  step sizes {a_n}, {a_w,n}
  config for rollout length, batching, evaluation cadence
Output:
  trained critic, actor, and w

1. Initialize theta, phi, w, replay/log buffers, and evaluation state.
2. For n = 0, 1, ..., N-1:
3.   Reset the environment and collect one trajectory under the stochastic actor:
4.     for k = 0, ..., K-1:
5.       sample u_k ~ pi_phi(. | t_k, x_k, F_k; w_n)
6.       optionally apply configured action constraints
7.       step the environment and record:
8.         (t_k, x_k, F_k, u_k, log pi_phi(u_k | ...), x_{k+1})
9.   Build the critic residual terms from the discrete approximation of:
10.      dJ + entropy_temp * log pi * dt
11.  Update theta using the martingale moment / critic equation.
12.  Update actor mean parameters using the actor-gradient integral estimate.
13.  Update actor covariance parameters using a numerically safe covariance representation.
14.  Update w with terminal-wealth error:
15.      w_{n+1} <- Projection(w_n - a_w,n * (x_T - z))
16.  Form deterministic execution policy:
17.      u_hat(t, x, F; w) <- actor mean / greedy executor
18.  On evaluation cadence:
19.      evaluate the deterministic execution policy
20.      compare against the oracle benchmark when available
21.  Save checkpoints, scalar diagnostics, and plotting data.
```

Implementation notes:

- keep the stochastic behavior policy and deterministic execution policy separate in code,
- keep covariance updates numerically safe,
- treat theorem-backed update forms separately from engineering stabilizers,
- do not reuse `discount` notation for the entropy term; in this repo `entropy_temp` and `discount` are distinct config fields,
- log behavior-policy and execution-policy performance independently.

---

## 4. Algorithm C: Practical Online CTRL With 2022/2025 Improvements

Purpose:

- add the practical improvements after the baseline CTRL path is stable,
- preserve the 2025 baseline as the reference algorithm,
- layer in operational improvements from the 2022 paper and the 2025 E-companion.

### Practical modifications to include

- leverage control as a modular post-processing layer,
- optional risky-only projection for benchmark matching,
- separate **rebalance cadence** from **parameter-update cadence**,
- TD($\lambda$)-style trace/test-function accumulation,
- mini-batch averaging of one-step updates,
- online parameter updates from streaming trajectories.

### Pseudocode

```text
Algorithm C: CTRLOnlinePractical
Input:
  environment
  baseline theta, phi, w
  rebalance interval R
  parameter-update interval U
  trace decay lambda_trace
  online mini-batch size M
  optional leverage / risky-only settings
Output:
  updated theta, phi, w and diagnostic outputs

1. Initialize online buffers, traces, and logging state.
2. Reset the environment.
3. For each environment step k:
4.   If k is a rebalance step:
5.     sample behavior action u_k from pi_phi(. | t_k, x_k, F_k; w)
6.     apply optional leverage or risky-only projection if configured
7.   Else:
8.     reuse the current portfolio allocation
9.   Step the environment and record the one-step transition.
10.  Update critic and actor trace objects:
11.      I_k <- trace update from critic features
12.      H_k <- trace update from score / policy-gradient features
13.  If k is an update step:
14.      form one-step gradient estimators G_theta, G_phi1, G_phi2inv
15.      aggregate over the current mini-batch
16.      update theta, phi, and w
17.  Periodically evaluate the deterministic execution policy.
18.  Save checkpoints, online metrics, and plot-ready summaries.
```

Engineering requirement:

- make every practical modification optional and configuration-driven,
- keep the baseline CTRL path reproducible without those extras.

---

## 5. Required outputs and plotting hooks

Every training / evaluation algorithm should emit enough structured data to support configurable plots from YAML.

Minimum plotting hooks:

- critic / actor losses over episodes or update steps,
- gradient norms over time,
- training time,
- memory pressure when available,
- wealth trajectories versus oracle and baseline comparators,
- portfolio-weight trajectories,
- evaluation-metric overlays on saved plots.

These plotting hooks should be saved as run artifacts and rendered by a separate plotting module / script rather than only by the live trainer.
