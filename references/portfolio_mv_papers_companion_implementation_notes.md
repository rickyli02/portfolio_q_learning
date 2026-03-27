# Companion Implementation Notes for Continuous-Time Portfolio RL Papers

This note is a **companion** to the earlier summary file. Its purpose is narrower:

- collect **implementation-relevant details that are easy to miss** on a first read,
- record the **practical modifications** introduced in later or companion papers,
- point out **notation clashes**, **update-order issues**, and **places where the papers leave degrees of freedom**.

It covers the following papers and related companions:

1. **Wang and Zhou (2020)** — *Continuous-Time Mean–Variance Portfolio Selection: A Reinforcement Learning Framework*  
2. **Huang, Jia, Zhou (2022)** — *Achieving Mean–Variance Efficiency by Continuous-Time Reinforcement Learning*  
3. **Huang, Jia, Zhou (2025)** — *Mean Variance Portfolio Selection by Continuous Time Reinforcement Learning* together with its E-companion  
4. **Gao, Li, Zhou (2025)** — *Reinforcement Learning for Jump-Diffusions, with Financial Applications*  
5. **Dai, Dong, Jia, Zhou** — *Data-Driven Merton’s Strategies via Policy Randomization*  

The goal is to help turn the mathematics into code.

---

## 1. Cross-paper implementation map

A useful mental map is:

- **Wang–Zhou (2020)** gives the first compact exploratory MV algorithm (**EMV**) in a single-risky-asset setting.
- **Huang–Jia–Zhou (2022)** explains how to turn continuous-time actor–critic ideas into a **practical constrained portfolio algorithm**, with leverage control, rebalancing, and TD($\lambda$)-style traces.
- **Huang–Jia–Zhou (2025)** gives the more systematic **CTRL** framework, with theorem-backed baseline updates and a more practical **modified online algorithm** in the E-companion.
- **Gao–Li–Zhou (2025)** shows that for **jump-diffusions**, the same temporal-difference/q-learning algorithms can still be used, although parametrization may depend on jumps in non-LQ settings.
- **Dai et al.** is not an MV paper, but it contains very useful implementation lessons on **Gaussian randomization**, **bias–variance tradeoff in the exploration level**, and **wealth-normalized test functions**.

If you are building a repo, the natural order is:

1. classical oracle benchmark,  
2. EMV,  
3. ICAIF/CTRL practical modifications,  
4. 2025 theorem-backed baseline CTRL,  
5. modified online CTRL,  
6. jump-diffusion portability,  
7. optional Merton-style utility extension.

---

## 2. Wang–Zhou (2020): extra details for implementing EMV

### 2.1 What the algorithm actually needs as data

The EMV paper is simple in structure, but one implementation detail is easy to miss:

- it does **not** require explicit estimation of $\mu$ or $\sigma$,
- it assumes you can simulate or observe a trajectory under the **current Gaussian exploratory policy**,
- then it fits the critic/policy parameters by minimizing the discretized Bellman/TD-style objective.

The sample dataset is

$$
D=\{(t_i,x_i)\}_{i=0}^{\ell+1},
$$

constructed by:
1. starting from $(0,x_0)$,
2. at each $t_i$, sampling $u_i \sim \pi^\phi_{t_i}$,
3. observing the next wealth $x_{i+1}$,
4. assembling the trajectory.  

So the algorithm is fundamentally **trajectory-based**, not a closed-form parameter update.

### 2.2 Parametric tying is part of the algorithm, not just theory

The value parametrization is

$$
V^\theta(t,x)= (x-w)^2 e^{-\theta_3(T-t)}+\theta_2 t^2+\theta_1 t+\theta_0.
$$

The entropy of the Gaussian policy is parameterized as

$$
H(\pi_t^\phi)=\phi_1+\phi_2(T-t).
$$

Then the paper imposes the structural relations

$$
\sigma^2=\lambda \pi e^{1-2\phi_1},
\qquad
\theta_3 = 2\phi_2 = \rho^2,
$$

and the improved Gaussian policy becomes

$$
\pi(u;t,x,w)
=
\mathcal N\!\left(
-\sqrt{\frac{2\phi_2}{\lambda\pi}}\,e^{(2\phi_1-1)/2}(x-w),
\ \frac{1}{2\pi}e^{2\phi_2(T-t)+2\phi_1-1}
\right).
$$

This means that in code, **$\theta_3$ should not be treated as an independently free parameter** once you adopt this reduced parametrization. Likewise, $\theta_0$ is reset from the terminal condition:

$$
\theta_0 = -\theta_2T^2-\theta_1T-(w-z)^2.
$$

So the critic update is only partly a gradient step; part of it is a **constraint enforcement / projection back to the theoretical manifold**.

### 2.3 The actual order of EMV updates

A faithful implementation should use roughly this order inside each iteration:

1. sample or collect a trajectory under current $\pi^\phi$,
2. compute $C(\theta,\phi)$,
3. update $\theta_1,\theta_2$,
4. reset $\theta_3 \leftarrow 2\phi_2$,
5. reset $\theta_0$ from terminal condition,
6. update $\phi_1,\phi_2$,
7. reconstruct the Gaussian policy,
8. every $N$ iterations, update $w$.

This is not just cosmetic. If you update $\theta$ and $\phi$ “simultaneously” without reimposing the structural equations, your implementation drifts away from the intended actor/critic family.

### 2.4 The $w$-update is an outer loop

The Lagrange multiplier update is

$$
w_{n+1}=w_n-\alpha_n(X_T-z),
$$

or in stabilized form,

$$
w \leftarrow w-\alpha\left(\frac1N\sum_{j=1}^N x_T^j-z\right).
$$

This should be implemented as a **slow outer update**, not mixed inside every timestep of a trajectory. The paper’s logic is that $w$ is a constraint-enforcing primal-dual variable, not just another network parameter.

### 2.5 Practical ambiguities in EMV

These are not fully specified in the paper:

- batch size / whether one trajectory or multiple trajectories should be used per update,
- whether $w$ should be updated every episode or every $N$ episodes,
- gradient clipping or stabilization,
- how small $\Delta t$ must be before performance plateaus,
- what to do in multi-asset settings.

So for a repo, you need to choose these explicitly.

---

## 3. ICAIF 2022 paper: what it adds beyond theory

The 2022 ICAIF paper is important because it translates continuous-time theory into a **practical portfolio algorithm**.

### 3.1 Parametrization used in practice

The value function is parameterized as

$$
J_\theta(t,x;w)
=
(x-w)^2 e^{-\theta_3(T-t)}
+\theta_2(t^2-T^2)+\theta_1(t-T)-(w-z)^2.
$$

The stochastic policy is parameterized as

$$
\pi_\phi(\cdot\mid t,x;w)
=
\mathcal N\!\left(\cdot\mid -\phi_1(x-w),\ \phi_2 e^{\phi_3(T-t)}\right).
$$

The entropy term is

$$
\hat p(t,\phi)
=
-\frac12 \log\!\big((2\pi e)^d \det(\phi_2 e^{\phi_3(T-t)})\big).
$$

This is the same broad Gaussian/quadratic structure that later appears in the 2025 CTRL paper.

### 3.2 Rebalancing is separated from parameter updates

A subtle but important implementation choice:

- **portfolio rebalancing** can happen at a lower frequency,
- **parameter updates** can still happen at every finer time step.

This means the algorithm conceptually separates:
- the **economic action schedule**,
- the **learning schedule**.

That is a very useful design principle for your repo.

### 3.3 Leverage constraint formula

Given a portfolio vector $a_t$, leverage cap $\ell$, and current wealth $x_t$, the leverage-constrained portfolio is

$$
a_t' =
\frac{a_t}{\sum_{i=1}^d a_t^i}\,x_t\,\ell \cdot 1_{\{\sum_i a_t^i > x_t\ell\}}
+
a_t \cdot 1_{\{\sum_i a_t^i \le x_t\ell\}}.
$$

So if total risky allocation exceeds allowable leverage, the portfolio is rescaled back to the leverage boundary.

In code:
- compute total risky exposure,
- if it exceeds $x_t \ell$, scale the vector proportionally,
- otherwise leave it unchanged.

### 3.4 Risk-free asset exclusion trick

For empirical comparison, the paper excludes the risk-free asset and projects the unconstrained portfolio onto purely risky allocations:

$$
\hat u(t) := \frac{u(t)}{\sum_{i=1}^d u_i(t)}x(t).
$$

This is **not** part of the underlying theoretical control problem; it is a practical benchmarking choice. If you use it, document clearly that you are changing the feasible action space.

### 3.5 TD($\lambda$) / eligibility traces

The paper recommends history-dependent test functions:

$$
I(t)=\int_0^t \lambda^{\,t-s}\frac{\partial J}{\partial \theta}(s,x(s);w;\theta)\,ds,
\qquad
H(t)=\int_0^t \lambda^{\,t-s}\frac{\partial}{\partial\phi}\log\pi(u(s)\mid s,x(s);w;\phi)\,ds.
$$

This is a continuous-time analogue of TD($\lambda$).

Implementation consequence:
- maintain running traces for critic and actor,
- decay them over time,
- use them to weight the new gradient signal.

That is more informative than using only current-time gradients.

### 3.6 Relation to generic RL algorithms

The paper explicitly interprets the algorithm as:
- **soft actor–critic-like**, because of entropy regularization and stochastic policies,
- **advantage actor–critic-like**, because the policy gradient involves a Hamiltonian / advantage-type term.

That is a helpful conceptual anchor if you want to compare with PPO/DDPG/SAC in your repo docs.

---

## 4. Huang–Jia–Zhou (2025) and E-companion: the details you actually need for CTRL

This is where many important implementation details live.

## 4.1 Baseline theorem-backed algorithm

The paper uses:
- critic parameter $\theta$,
- actor mean parameter $\phi_1$,
- actor covariance parameter $\phi_2$,
- Lagrange multiplier $w$.

The Gaussian behavior policy is

$$
\pi(\cdot\mid t,x;w;\phi)
=
\mathcal N\big(-\phi_1(x-w),\ \phi_2 e^{\phi_3(T-t)}\big).
$$

The baseline updates are

$$
\theta_{n+1} \leftarrow \Pi_{K_{\theta,n}}(\theta_n + a_n \cdot \text{critic increment}),
$$

$$
\phi_{1,n+1} \leftarrow \Pi_{K_{1,n}}(\phi_{1,n} - a_n Z_{1,n}(T)),
$$

$$
\phi_{2,n+1} \leftarrow \Pi_{K_{2,n}}(\phi_{2,n} + a_n Z_{2,n}(T)),
$$

$$
w_{n+1} \leftarrow \Pi_{K_{w,n}}(w_n-a_{w,n}(x_n(T)-z)).
$$

### Why this matters
The **projection sets** are not decorative. They are used to keep iterates bounded while the admissible bounds expand over time. If your implementation ignores them entirely, you are no longer implementing the theorem-backed algorithm.

## 4.2 Inverse covariance update trick

The paper updates in $\phi_2^{-1}$, not directly in $\phi_2$.

This is a major implementation detail.

Reason:
- it simplifies the stochastic approximation structure,
- it is important in the convergence proof,
- it is usually numerically more stable than unconstrained direct covariance updates.

So if $\phi_2$ is a covariance matrix, your code should consider storing:
- either a precision matrix,
- or a Cholesky-type parameterization,
- and reconstruct covariance only when needed.

Direct free updates of a covariance matrix are risky.

## 4.3 U-shaped exploration–variance phenomenon

A central insight of the 2025 paper is that the variance of the update direction for $\phi_1$ is U-shaped in the exploration level governed by $\phi_2$:

- too little exploration $\Rightarrow$ near-deterministic policy, weak policy-improvement signal,
- too much exploration $\Rightarrow$ overly noisy updates.

This is not just philosophical. It should influence hyperparameter tuning:
- exploration variance should be tuned like a genuine optimization knob,
- you should expect both very small and very large covariance to train poorly.

## 4.4 Stochastic training, deterministic execution

The paper explicitly separates:
- **stochastic Gaussian behavior policy** for learning,
- **deterministic greedy policy** for actual execution.

This is very important for a portfolio repo.

A clean software design is:
- `behavior_policy`: stochastic, used for data generation,
- `target_policy`: deterministic, used for reported portfolio performance.

Do not blur the two in code or backtests.

## 4.5 Modified online algorithm: one-step updates

The E-companion gives one-step gradient increments:

$$
G_{\theta,k}
=
\frac{\partial J}{\partial \theta}(t_k,x(t_k);w;\theta_k)
\left[J(t_{k+1},x(t_{k+1});w;\theta_k)-J(t_k,x(t_k);w;\theta_k)+\gamma \hat p(t_k,\phi_k)\Delta t\right],
$$

$$
G_{\phi_1,k}
=
\frac{\partial \log \pi}{\partial \phi_1}(u(t_k)\mid t_k,x(t_k);w;\phi_k)
\left[J(t_{k+1},x(t_{k+1});w;\theta_k)-J(t_k,x(t_k);w;\theta_k)+\gamma \hat p(t_k,\phi_k)\Delta t\right]
+\gamma \frac{\partial \hat p}{\partial \phi_1}(t_k,\phi_k)\Delta t,
$$

$$
G_{\phi_2^{-1},k}
=
\frac{\partial \log \pi}{\partial \phi_2^{-1}}(u(t_k)\mid t_k,x(t_k);w;\phi_k)
\left[J(t_{k+1},x(t_{k+1});w;\theta_k)-J(t_k,x(t_k);w;\theta_k)+\gamma \hat p(t_k,\phi_k)\Delta t\right]
+\gamma \frac{\partial \hat p}{\partial \phi_2^{-1}}(t_k,\phi_k)\Delta t.
$$

Then with historical weighting:

$$
\theta_{k+1}
\leftarrow
\Pi_{K_{\theta,n}}
\left(\theta_k + a_n(w_{\text{prev}} G_{\theta,k-1}+w_{\text{curr}} G_{\theta,k})\right),
$$

$$
\phi_{1,k+1}
\leftarrow
\Pi_{K_{1,n}}
\left(\phi_{1,k} - a_n(w_{\text{prev}} G_{\phi_1,k-1}+w_{\text{curr}} G_{\phi_1,k})\right),
$$

$$
\phi_{2,k+1}
\leftarrow
\Pi_{K_{2,n}}
\left(\phi_{2,k} + a_n(w_{\text{prev}} G_{\phi_2^{-1},k-1}+w_{\text{curr}} G_{\phi_2^{-1},k})\right).
$$

This is effectively a **short-memory trace / weighted two-step update**.

### Implementation implication
You need to cache previous gradient increments. The modified online algorithm is not purely memoryless.

## 4.6 Modified online algorithm loop structure

A faithful implementation of the modified online algorithm is:

1. initialize $\theta,\phi,w$ from offline pretraining,
2. for each iteration / episode:
   - set current wealth $x=x_0$,
   - at each time $k$:
     - if $k \bmod f = 0$, compute deterministic greedy target action and rebalance,
     - otherwise hold current assets,
     - generate $n$ random behavior-policy actions,
     - simulate next wealths for those behavior actions,
     - compute $G_{\theta,k}^j, G_{\phi_1,k}^j, G_{\phi_2^{-1},k}^j$,
     - average over mini-batch,
     - update $\theta,\phi$ using (39)/(40),
   - every $M$ iterations, update $w$ using average terminal wealth.

So the paper effectively combines:
- online updates,
- off-policy data,
- deterministic execution,
- mini-batch averaging.

## 4.7 Practical hyperparameters reported

The implementation section reports one concrete practical setup:
- $w$-learning rate: $0.05$,
- $\theta,\phi$-learning rates: $0.005$,
- normalized initial wealth: $1$,
- target: $1.15$,
- horizon $T=1$,
- $\Delta t = 1/252$,
- entropy temperature $\gamma=0.1$,
- 20,000 training iterations,
- update $w$ every 10 iterations,
- batch size $16$,
- initial values $\theta_1=\theta_2=\theta_3=\phi_3=w=1$,
- $\phi_2 = I_d$,
- $\phi_1$ initialized as an all-ones vector.

These should not be treated as universal defaults, but they are useful sanity-check values.

## 4.8 Important ambiguity: notation collision for $\lambda$

In these papers, $\lambda$ is overloaded:

- in some contexts it is the **exploration/entropy-related parameter**,
- in TD($\lambda$) contexts it is the **trace-decay parameter**,
- in jump models it may also denote **jump intensity**.

In a codebase, do **not** call all of them `lambda`.
Use names like:
- `entropy_temp`,
- `trace_decay`,
- `jump_intensity`.

This will save you trouble.

---

## 5. Jump-diffusion paper: what changes and what does not

The jump-diffusion paper gives one of the cleanest practical messages in the whole line of work:

### 5.1 What does not change
The temporal-difference / q-learning algorithms can be used **without first deciding whether the data come from a pure diffusion or a jump-diffusion**.

This is because the Hamiltonian/q-function still enters via temporal differences of the value function, and the implemented learning rules remain the same.

### 5.2 What may change
The **parametrization** may change when jumps are present.

In general non-LQ problems:
- the optimal exploratory policy may cease to be Gaussian,
- or may not even exist in a simple Gaussian family.

### 5.3 Special case: MV portfolio selection
For the MV portfolio problem under jump-diffusion stock dynamics, the paper says the LQ structure is strong enough that:
- the optimal exploratory policy remains Gaussian,
- the value function remains quadratic,
- the same parametrizations used in the diffusion case still work.

This is a big implementation convenience.

### 5.4 Practical repo consequence
If your repo is MV-only, you can likely reuse the same actor/critic family for:
- GBM diffusion environments,
- Merton jump-diffusion environments,

and treat the environment simulator as the changed component.

If you later move to hedging or utility problems, that invariance should **not** be assumed.

---

## 6. Merton policy-randomization paper: implementation lessons worth borrowing

Although it solves a utility problem, this paper contains implementation ideas that are useful beyond Merton.

## 6.1 Gaussian randomization as a technical device

The admissible policy class is

$$
\pi^{(\lambda)}(\cdot\mid t,w,x)
=
\mathcal N\!\left(u(t,w,x),\ \frac{\lambda}{\gamma \sigma(t,x)^2}\right).
$$

The mean of the optimal Gaussian randomized policy solves the original deterministic Merton problem.

This is conceptually important:
- randomization is not only for exploration in the informational sense,
- it can also be a **technical device** that makes actor–critic learning possible.

## 6.2 Equivalent relative wealth loss (ERWL)

The cost of randomization is quantified explicitly:

$$
\mathrm{ERWL}(\pi^{(\lambda)\,*}) = 1-e^{-\lambda T/2} \approx \lambda T/2.
$$

This is a very useful calibration heuristic:
- larger $\lambda$ helps learning signal variance,
- but it directly imposes a utility cost.

## 6.3 Wealth-normalized test functions

The paper proposes

$$
\xi_t =
\frac{\partial_\psi \hat\phi_\psi(t,X_t)}
{(1-\gamma)\hat V^\psi(t,W_t,X_t)+1},
\qquad
\eta_t =
\frac{\partial_\theta \hat u_\theta(t,X_t)}
{(1-\gamma)\hat V^\psi(t,W_t,X_t)+1}.
$$

Interpretation:
- this replaces raw TD errors by a **relative** TD error,
- it uses the homothetic structure of CRRA utility,
- it helps normalize wealth growth and nonstationarity.

This is a strong idea if you later move beyond mean-variance into utility-based portfolio RL.

## 6.4 Bias–variance tradeoff in randomization level

The paper’s Black–Scholes analysis shows:

- the **bias** due to time discretization is linear in $\Delta t$ and also grows with $\lambda$,
- the **variance** of the learning signal has a term of order $\lambda^{-1}$.

So:
- small $\lambda$ reduces bias,
- large $\lambda$ reduces variance,
- practical tuning requires a balance.

That is an unusually concrete and useful theoretical explanation of why exploration level cannot be taken arbitrarily small or large.

---

## 7. Implementation decisions these papers still leave open

Even after reading all of these papers, there are still several choices your repo must make.

### 7.1 Discretization convention
The papers are continuous-time in derivation but discretized in implementation. You still need to decide:
- Euler vs exact GBM step for simulation,
- whether rewards are computed as integrals or simple step approximations,
- whether action is left-constant on each interval.

### 7.2 Data convention
You need to define clearly:
- action in **dollar allocation** vs **portfolio weights**,
- discounted wealth vs nominal wealth,
- whether returns are fed directly or reconstructed from prices.

These papers often work in discounted wealth with dollar allocations.

### 7.3 Matrix positivity
For covariance parameters:
- never update a covariance matrix naively without enforcing PSD/PD structure,
- use precision, Cholesky, spectral clipping, or projection.

### 7.4 Execution vs learning policy
You should make the distinction explicit in code and logs:
- `behavior_policy`,
- `execution_policy`.

### 7.5 Constraints
The papers include:
- leverage cap,
- exclusion of risk-free asset,
- rebalancing frequency.

These should be modular post-processing / constraint layers, not hard-coded into the actor itself.

### 7.6 Multi-asset scaling
The simple one-asset EMV paper is much cleaner than the multi-asset CTRL papers. If your first implementation is multi-asset, expect:
- covariance-conditioning issues,
- larger variance in actor updates,
- more delicate covariance parameterization.

---

## 8. Most actionable takeaways for your repo

If I were converting these papers into a modular research repo, I would follow these rules:

1. **Keep Gaussian actor and quadratic critic as first-class baseline objects.**  
2. **Store behavior and execution policies separately.**  
3. **Update covariance through a constrained parameterization, not raw entries.**  
4. **Treat the $w$-update as a slower outer loop.**  
5. **Make rebalancing frequency independent of learning frequency.**  
6. **Implement leverage and risky-only projection as optional wrappers.**  
7. **Support mini-batch and history-dependent traces from the start.**  
8. **Name hyperparameters carefully to avoid notation collisions.**  
9. **Use jump-diffusion as an environment module, not necessarily a new algorithm module.**  
10. **Document clearly which parts are theorem-backed baseline CTRL and which are empirical practical modifications.**

---

## 9. Main ambiguities / cautions to carry into implementation

### Wang–Zhou (2020)
- Very elegant, but narrow and lightly specified for software engineering.
- Multi-asset extension is not written as a full algorithm.
- Hyperparameter choices are mostly left to the implementer.

### Huang–Jia–Zhou (2022)
- Practical and useful, but more like an implementation paper than a theorem paper.
- Constraint handling is partly heuristic and benchmark-driven.

### Huang–Jia–Zhou (2025)
- Baseline algorithm is theorem-backed.
- Modified online algorithm is more practical, but some choices are empirical rather than fully pinned down by theory.
- The E-companion is essential; the main paper alone is not enough for implementation.

### Gao–Li–Zhou (2025)
- Great portability result for q-learning algorithms.
- But one should not overgeneralize the Gaussian/quadratic invariance beyond the MV/LQ setting.

### Dai et al.
- Extremely useful for thinking about exploration and randomization.
- But the objective is utility maximization, not mean-variance, so only some of its implementation ideas transfer directly.

---

## 10. Bottom line

The main missing implementation lesson across these papers is this:

**theory determines the right structure, but practical performance depends heavily on update scheduling, covariance parametrization, constraint handling, batch construction, and the separation between stochastic learning and deterministic execution.**

That is where your repo design will matter most.

