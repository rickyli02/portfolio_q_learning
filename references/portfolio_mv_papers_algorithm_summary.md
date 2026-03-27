# Detailed Algorithm Summaries for Three Continuous-Time Mean–Variance Portfolio Papers

This note by ChatGPT summarizes the main algorithms, equations, and implementation-relevant details from the following three papers:

1. **Zhou and Li (2000)** — *Continuous-Time Mean-Variance Portfolio Selection: A Stochastic LQ Framework*
2. **Wang and Zhou (2019/2020)** — *Continuous-Time Mean–Variance Portfolio Selection: A Reinforcement Learning Framework*
3. **Huang, Jia, and Zhou (2025)** — *Mean–Variance Portfolio Selection by Continuous-Time Reinforcement Learning: Algorithms, Regret Analysis, and Empirical Study* Link: https://arxiv.org/abs/2412.16175

The focus here is on **algorithmic structure**, **key mathematical expressions**, and **implementation-relevant caveats**. I do **not** reproduce proofs unless they directly affect implementation.

---

## 0. High-level relationship among the three papers

These three papers form a fairly natural progression.

- **Zhou–Li (2000)** gives the **classical model-based continuous-time mean–variance solution**. The main contribution is an **embedding into a stochastic linear-quadratic (LQ) control problem**, leading to an explicit optimal policy and efficient frontier.
- **Wang–Zhou (2019/2020)** keeps the same mean–variance objective, but reframes it as an **entropy-regularized exploratory control problem** suitable for reinforcement learning. The paper derives the **optimal exploratory Gaussian policy**, proves a **policy improvement theorem**, and proposes the **EMV algorithm**.
- **Huang–Jia–Zhou (2025)** develops a more modern **continuous-time actor–critic / martingale / stochastic-approximation RL framework**, proves **convergence and regret** for a Black–Scholes benchmark, and proposes a more practical **CTRL algorithm** with an online implementation.

A useful conceptual slogan is:

- **2000 paper**: solve the model if the coefficients are known.
- **2019 paper**: add principled exploration and learn an optimal Gaussian exploratory policy.
- **2025 paper**: learn the policy directly from data using martingale conditions, with convergence/regret guarantees in a benchmark setting.

---

# 1. Zhou and Li (2000): Stochastic LQ framework

## 1.1 Core problem

The paper studies a market with one bond and $m$ risky assets. The bond evolves as

$$
dP_0(t)=r(t)P_0(t)\,dt,
$$

and the risky assets satisfy

$$
dP_i(t)=P_i(t)\left(b_i(t)\,dt+\sum_{j=1}^m \sigma_{ij}(t)\,dW^j(t)\right), \qquad i=1,\dots,m.
$$

The investor wealth $x(t)$ evolves under self-financing trading as

$$
dx(t)=\left(r(t)x(t)+\sum_{i=1}^m [b_i(t)-r(t)]u_i(t)\right)dt
+\sum_{j=1}^m\sum_{i=1}^m \sigma_{ij}(t)u_i(t)\,dW^j(t),
$$

where $u_i(t)$ is the **dollar amount invested in risky asset $i$**.

The mean–variance objective is the bi-criterion problem:

$$
\min \big(-\mathbb E[x(T)],\ \mathrm{Var}(x(T))\big).
$$

The efficient frontier is obtained by solving, for $\mu>0$,

$$
\min_{u(\cdot)} -\mathbb E[x(T)] + \mu\,\mathrm{Var}(x(T)].
$$

---

## 1.2 Main difficulty

The term $(\mathbb E[x(T)])^2$ inside the variance makes the objective **non-separable** for dynamic programming. This is the central obstacle.

The paper's key idea is to **embed** the mean–variance problem into an auxiliary stochastic LQ problem.

---

## 1.3 Auxiliary problem and embedding

Define the auxiliary problem $A(\mu,\lambda)$:

$$
\min_{u(\cdot)} \mathbb E\{\mu x(T)^2 - \lambda x(T)\}.
$$

The important theorem is that any optimizer of the original weighted mean–variance problem is also an optimizer of this auxiliary problem for a suitable $\lambda$.

This is the main algorithmic reduction:

### Classical implementation recipe
1. Choose the risk-aversion / tradeoff parameter $\mu>0$.
2. Solve the auxiliary LQ problem $A(\mu,\lambda)$.
3. Determine the correct $\lambda$ (equivalently $\gamma=\lambda/(2\mu)$) so that the solution corresponds to the desired mean–variance tradeoff.
4. Recover the efficient policy and efficient frontier.

---

## 1.4 Reduction to a general stochastic LQ problem

Set

$$
\gamma = \frac{\lambda}{2\mu}, \qquad y(t)=x(t)-\gamma.
$$

Then the auxiliary problem becomes

$$
\min \mathbb E\left[\frac{\mu}{2} y(T)^2\right]
$$

subject to a linear controlled SDE

$$
dy(t)=\{A(t)y(t)+B(t)u(t)+f(t)\}dt+\sum_{j=1}^m D_j(t)u(t)\,dW^j(t),
$$

with

$$
A(t)=r(t), \qquad B(t)=(b_1(t)-r(t),\dots,b_m(t)-r(t)),
$$

$$
f(t)=\gamma r(t), \qquad D_j(t)=(\sigma_{1j}(t),\dots,\sigma_{mj}(t)).
$$

This is a stochastic LQ control problem, except that the running control penalty $R$ is zero, so the problem is **singular**.

---

## 1.5 General LQ solution template

For the general LQ problem, the paper introduces the Riccati equation

$$
\dot P(t)
=
- P(t)A(t)-A(t)^\top P(t)-Q(t)
+P(t)B(t)\Big(R(t)+\sum_{j=1}^m D_j(t)^\top P(t)D_j(t)\Big)^{-1}B(t)^\top P(t),
$$

with terminal condition

$$
P(T)=H,
$$

and positivity condition

$$
K(t):=R(t)+\sum_{j=1}^m D_j(t)^\top P(t)D_j(t) > 0.
$$

Then define $g(t)$ from

$$
\dot g(t)= -A(t)^\top g(t)
+P(t)B(t)K(t)^{-1}B(t)^\top g(t)-P(t)f(t),
\qquad g(T)=0.
$$

The optimal feedback control is

$$
u^*(t,x)= -K(t)^{-1}B(t)^\top (P(t)x+g(t)).
$$

This is the core general-purpose LQ solution used by the portfolio problem.

---

## 1.6 Specialization to the portfolio problem

Define

$$
\rho(t)=B(t)[\sigma(t)\sigma(t)^\top]^{-1}B(t)^\top.
$$

Then the scalar Riccati equation becomes

$$
\dot P(t) = (\rho(t)-2r(t))P(t), \qquad P(T)=\mu,
$$

so

$$
P(t)=\mu \exp\left(-\int_t^T (\rho(s)-2r(s))\,ds\right).
$$

The auxiliary optimal control becomes

$$
\bar u(t,x)
=
[\sigma(t)\sigma(t)^\top]^{-1}B(t)^\top
\left(\gamma e^{-\int_t^T r(s)\,ds}-x\right).
$$

This is the explicit model-based optimal feedback allocation.

---

## 1.7 Efficient frontier

The paper derives closed forms for $\mathbb E[x(T)]$ and $\mathbb E[x(T)^2]$, and then obtains the efficient frontier:

$$
\mathrm{Var}(\bar x(T))
=
\frac{e^{-\int_0^T \rho(t)\,dt}}
{1-e^{-\int_0^T \rho(t)\,dt}}
\left(\mathbb E[\bar x(T)]-x_0 e^{\int_0^T r(t)\,dt}\right)^2.
$$

This gives the exact continuous-time mean–variance frontier.

---

## 1.8 What is algorithmic here?

Although this is not an RL paper, it still contains a very clear computational pipeline:

### Practical solution pipeline
1. Specify $r(t)$, $b(t)$, $\sigma(t)$.
2. Compute $\rho(t)=B(\sigma\sigma^\top)^{-1}B^\top$.
3. Solve for $P(t)$.
4. Solve for $g(t)$ or directly substitute the specialized formula.
5. Compute the feedback control $\bar u(t,x)$.
6. Back out the efficient frontier and portfolio corresponding to a chosen target return / tradeoff.

---

## 1.9 Important details emphasized by the paper

- The mean–variance problem is **not** a standard dynamic programming problem because of $(\mathbb E[x(T)])^2$.
- The **embedding into an auxiliary LQ problem** is the key structural move.
- The resulting LQ problem is **singular**, with $R=0$, but still solvable because the diffusion term contributes positively through $\sum D_j^\top P D_j$.
- The framework is intended to be broader than this one portfolio problem and can handle more complicated settings.

---

## 1.10 Implementation caveats and ambiguities

1. **This is fully model-based.** You need $r(t)$, $b(t)$, and $\sigma(t)$.
2. **No estimation procedure is given.** The paper solves the control problem assuming the coefficients are known.
3. **No transaction costs or trading frictions.**
4. **Continuous-time exact solution**, but actual implementation needs discretization.
5. The paper focuses on the exact optimal control and efficient frontier, not on numerical robustness under parameter error. That becomes a major issue in later RL papers.

---

# 2. Wang and Zhou (2019/2020): Exploratory MV via entropy-regularized RL

This paper is the bridge between the classical continuous-time MV problem and a reinforcement-learning formulation.

## 2.1 Classical setup recalled

The paper simplifies to **one risky asset + one riskless asset**. The risky asset price follows

$$
dS_t = S_t(\mu\,dt+\sigma\,dW_t),
$$

and the discounted wealth under control $u_t$ evolves as

$$
dx_t^u = \sigma u_t(\rho\,dt+dW_t), \qquad \rho=\frac{\mu-r}{\sigma}.
$$

The classical pre-committed mean–variance problem is

$$
\min_u \mathrm{Var}(x_T^u)
\quad\text{s.t.}\quad
\mathbb E[x_T^u]=z,
$$

or, with Lagrange multiplier $w$,

$$
\min_u \mathbb E[(x_T^u-w)^2]-(w-z)^2.
$$

The classical optimal policy is

$$
u^*(t,x;w) = -\frac{\rho}{\sigma}(x-w),
$$

with optimal value

$$
V^{cl}(t,x;w) = (x-w)^2e^{-\rho^2(T-t)}-(w-z)^2.
$$

---

## 2.2 Exploratory control and relaxed dynamics

Instead of using a deterministic action $u_t$, the paper uses a **distributional control** $\pi_t(u)$, i.e. a density over actions.

Define mean and variance of the exploratory action:

$$
\mu_t = \int_{\mathbb R} u\,\pi_t(u)\,du,
\qquad
\sigma_t^2 = \int_{\mathbb R} u^2\pi_t(u)\,du - \mu_t^2.
$$

Then the relaxed / exploratory wealth dynamics are

$$
dX_t^\pi
=
\rho\sigma \mu_t\,dt
+
\sigma\sqrt{\mu_t^2+\sigma_t^2}\,dW_t.
$$

The differential-entropy penalty is

$$
H(\pi)= -\int_0^T \int_{\mathbb R}\pi_t(u)\ln \pi_t(u)\,du\,dt.
$$

The exploratory mean–variance objective is

$$
\min_{\pi}
\mathbb E\left[
(X_T^\pi-w)^2
+
\lambda\int_0^T\int_{\mathbb R}\pi_t(u)\ln\pi_t(u)\,du\,dt
\right]
-(w-z)^2,
$$

where $\lambda>0$ is the exploration weight / temperature.

---

## 2.3 HJB equation and exact optimal exploratory policy

The HJB equation becomes

$$
v_t(t,x;w)
+
\min_{\pi\in\mathcal P(\mathbb R)}
\int_{\mathbb R}
\left(
\frac12 \sigma^2u^2 v_{xx}(t,x;w)
+\rho\sigma u\,v_x(t,x;w)
+\lambda\ln\pi(u)
\right)\pi(u)\,du
=0,
$$

with terminal condition

$$
v(T,x;w)=(x-w)^2-(w-z)^2.
$$

Solving the minimization over $\pi$ yields a **Gaussian optimal feedback policy**:

$$
\pi^*(u;t,x,w)
=
\mathcal N\left(
-\frac{\rho}{\sigma}\frac{v_x(t,x;w)}{v_{xx}(t,x;w)},
\ \frac{\lambda}{\sigma^2 v_{xx}(t,x;w)}
\right).
$$

After solving the HJB, the paper gets the explicit optimal value function

$$
V(t,x;w)
=
(x-w)^2 e^{-\rho^2(T-t)}
+\frac{\lambda\rho^2}{4}(T^2-t^2)
-\frac{\lambda}{2}\left(\rho^2T-\ln(\sigma^2\pi\lambda)\right)(T-t)
-(w-z)^2,
$$

and therefore the explicit optimal exploratory policy

$$
\pi^*(u;t,x,w)
=
\mathcal N\left(
-\frac{\rho}{\sigma}(x-w),
\ \frac{\lambda}{2\sigma^2}e^{\rho^2(T-t)}
\right).
$$

### Important interpretation
- **Mean** of Gaussian = exploitation term.
- **Variance** of Gaussian = exploration term.
- Exploration variance is **time-decaying** as $t\to T$, i.e. annealing is endogenous.

This is one of the most important conceptual contributions of the paper.

---

## 2.4 Optimal wealth process under exploration

Under $\pi^*$, the wealth satisfies

$$
dX_t^*
=
-\rho^2(X_t^*-w)\,dt
+
\sqrt{
\rho^2(X_t^*-w)^2+\frac{\lambda}{2}e^{\rho^2(T-t)}
}\,dW_t,
\qquad
X_0^*=x_0.
$$

The Lagrange multiplier is still

$$
w=
\frac{ze^{\rho^2T}-x_0}{e^{\rho^2T}-1}.
$$

The paper stresses that the classical and exploratory problems have the **same $w$**.

---

## 2.5 Solvability equivalence and exploration cost

The paper proves a solvability equivalence between the classical and exploratory problems: solving either one gives the other.

As $\lambda\to 0$,

- the exploratory Gaussian policy converges weakly to the deterministic classical optimizer,
- the exploratory value function converges to the classical value function.

The exploration cost is explicitly

$$
C^{u^*,\pi^*}(0,x_0;w)=\frac{\lambda T}{2}.
$$

So the cost of exploration is linear in both the exploration weight $\lambda$ and horizon $T$.

---

## 2.6 Policy Improvement Theorem (PIT)

For any admissible feedback policy $\pi$ with smooth value function $V^\pi$, define the improved policy

$$
\tilde\pi(u;t,x,w)
=
\mathcal N\left(
-\frac{\rho}{\sigma}\frac{V^\pi_x(t,x;w)}{V^\pi_{xx}(t,x;w)},
\ \frac{\lambda}{\sigma^2V^\pi_{xx}(t,x;w)}
\right).
$$

Then

$$
V^{\tilde\pi}(t,x;w)\le V^\pi(t,x;w).
$$

This is the theoretical basis for iterative policy improvement.

---

## 2.7 Finite-step convergence for a good Gaussian initialization

If the initial policy is chosen as a Gaussian of the form

$$
\pi_0(u;t,x,w)=\mathcal N(u\mid a(x-w), c_1e^{c_2(T-t)}),
$$

then the iterated policy-improvement sequence converges to $\pi^*$, and in the idealized exact setting the paper shows convergence happens after **two iterations**.

This is mostly a theoretical insight, not a practical literal two-step algorithm, because in practice value functions are approximated.

---

## 2.8 EMV algorithm: implementable RL procedure

The **EMV algorithm** has three components:

1. **Policy evaluation**
2. **Policy improvement**
3. **Stochastic approximation update for the Lagrange multiplier $w$**

### (a) Policy evaluation

Bellman consistency for a fixed policy $\pi$ implies

$$
V^\pi(t,x)
=
\mathbb E\left[
V^\pi(s,X_s)
+\lambda\int_t^s\int_{\mathbb R}\pi_v(u)\ln\pi_v(u)\,du\,dv
\mid X_t=x
\right].
$$

Taking $s\to t$ leads to the continuous-time Bellman / TD error

$$
\delta_t
=
\dot V_t^\pi + \lambda\int_{\mathbb R}\pi_t(u)\ln\pi_t(u)\,du.
$$

The empirical policy-evaluation objective over simulated data $D=\{(t_i,x_i)\}$ is

$$
C(\theta,\phi)
=
\frac12\sum_{(t_i,x_i)\in D}
\left(
\dot V^\theta(t_i,x_i)
+\lambda\int_{\mathbb R}\pi_{t_i}^\phi(u)\ln\pi_{t_i}^\phi(u)\,du
\right)^2 \Delta t.
$$

### (b) Parametrization

The paper does **not** use deep neural networks. It uses explicit parametric forms suggested by theory.

#### Policy parametrization
The Gaussian policy entropy is parameterized as

$$
H(\pi_t^\phi)=\phi_1+\phi_2(T-t),
\qquad \phi=(\phi_1,\phi_2)^\top.
$$

#### Value-function parametrization
$$
V^\theta(t,x)
=
(x-w)^2 e^{-\theta_3(T-t)} + \theta_2 t^2 + \theta_1 t + \theta_0,
\qquad \theta=(\theta_0,\theta_1,\theta_2,\theta_3)^\top.
$$

Using the Gaussian policy structure, the parameters satisfy

$$
\sigma^2=\lambda\pi e^{1-2\phi_1},
\qquad
\theta_3=2\phi_2=\rho^2.
$$

Then the policy becomes

$$
\pi(u;t,x,w)
=
\mathcal N\left(
-\sqrt{\frac{2\phi_2}{\lambda\pi}}\,e^{(2\phi_1-1)/2}(x-w),
\ \frac{1}{2\pi}e^{2\phi_2(T-t)+2\phi_1-1}
\right).
$$

The objective simplifies to

$$
C(\theta,\phi)
=
\frac12
\sum_{(t_i,x_i)\in D}
\big(\dot V^\theta(t_i,x_i)-\lambda(\phi_1+\phi_2(T-t_i))\big)^2 \Delta t.
$$

### (c) Gradients for SGD

The paper explicitly derives

$$
\frac{\partial C}{\partial \theta_1}
=
\sum_{(t_i,x_i)\in D}
\big(\dot V^\theta(t_i,x_i)-\lambda(\phi_1+\phi_2(T-t_i))\big)\Delta t,
$$

$$
\frac{\partial C}{\partial \theta_2}
=
\sum_{(t_i,x_i)\in D}
\big(\dot V^\theta(t_i,x_i)-\lambda(\phi_1+\phi_2(T-t_i))\big)(t_{i+1}^2-t_i^2),
$$

$$
\frac{\partial C}{\partial \phi_1}
=
-\lambda
\sum_{(t_i,x_i)\in D}
\big(\dot V^\theta(t_i,x_i)-\lambda(\phi_1+\phi_2(T-t_i))\big)\Delta t,
$$

and a longer expression for $\partial C/\partial \phi_2$ involving the time derivative of the exponential term in $V^\theta$. In code, this is just another explicitly computable scalar gradient.

Then:

- update $\theta_1,\theta_2$ by SGD,
- set
  $$
  \theta_3=2\phi_2,
  $$
- enforce terminal condition
  $$
  \theta_0 = -\theta_2T^2-\theta_1T-(w-z)^2.
  $$

### (d) Lagrange multiplier update

The paper uses stochastic approximation:

$$
w_{n+1}=w_n-\alpha_n(X_T-z).
$$

In practice it suggests replacing $X_T$ by a recent sample average:

$$
w \leftarrow w - \alpha\left(\frac1N\sum_{j} x_T^j - z\right).
$$

This is explicitly described as **self-correcting**.

---

## 2.9 EMV pseudocode structure

At a high level:

1. Initialize $\theta,\phi,w$.
2. Simulate a trajectory under current Gaussian policy.
3. Build data $D=\{(t_i,x_i)\}$.
4. Update $\theta$ using gradients of $C(\theta,\phi)$.
5. Enforce $\theta_0,\theta_3$ structural relations.
6. Update $\phi$ using gradients of $C(\theta,\phi)$.
7. Reconstruct Gaussian policy $\pi^\phi$.
8. Every $N$ episodes, update $w$ from terminal-wealth error.

---

## 2.10 Important details emphasized by the paper

- The optimal exploratory policy is **Gaussian with time-decaying variance**.
- Exploration and exploitation separate cleanly:
  - policy mean = exploitation,
  - policy variance = exploration.
- The PIT reduces general policy improvement to a Gaussian family.
- The algorithm uses **simple parametric forms**, not neural networks.
- The method avoids estimating $\mu,\sigma$ explicitly.

---

## 2.11 Implementation caveats and ambiguities

1. **Single risky asset exposition.** Extending to many assets is conceptually straightforward but not spelled out algorithmically in detail.
2. **The EMV algorithm is structurally clear, but hyperparameter guidance is thin.**
   - learning-rate schedules,
   - stabilization tricks,
   - batching choices
   are not deeply specified.
3. **Value-function parameter tying is partly hand-imposed.**
   - $\theta_3=2\phi_2$,
   - $\theta_0$ from terminal condition.
   This is elegant, but it means the parametrization is not fully free.
4. **The paper assumes positive Sharpe ratio** in the policy rewrite around Eq. (46). The negative-Sharpe case is said to be analogous, but is not detailed.
5. The practical Bellman-error objective depends on discretization $\Delta t$, but the discretization error itself is not the focus here.

---

# 3. Huang, Jia, and Zhou (2025): Model-free continuous-time RL with convergence/regret

This paper is more ambitious and modern. It uses the continuous-time RL theory from Jia and Zhou to derive a data-driven actor–critic method from **martingale conditions**, and then proves convergence/regret in the Black–Scholes benchmark.

## 3.1 General market setting

There are $d+1$ assets:

- one risk-free asset $S^0(t)$,
- $d$ risky assets $S^1(t),\dots,S^d(t)$,
- and $m$ observable factors $F(t)\in\mathbb R^m$.

The discounted wealth process under portfolio $u(t)\in\mathbb R^d$ is

$$
dx^u(t)
=
\sum_{i=1}^d u_i(t)\frac{dS_i(t)}{S_i(t)}
-
e_d^\top u(t)\frac{dS_0(t)}{S_0(t)}.
$$

The key structural assumption is only that $(S(t),F(t))$ are **Itô diffusions**. Their coefficients are unknown and are **not estimated**.

---

## 3.2 RL formulation with stochastic policies

A stochastic policy maps state to a distribution:

$$
\pi:\ (t,x,F)\mapsto \pi(\cdot\mid t,x,F)\in \mathcal P(\mathbb R^d).
$$

At time $t$,

$$
u^\pi(t)\sim \pi(\cdot\mid t, x^{u^\pi}(t), F(t)).
$$

The entropy-regularized objective is

$$
\mathbb E\left[
(x^{u^\pi}(T)-w)^2
+
\gamma\int_0^T
\log \pi(u^\pi(t)\mid t,x^{u^\pi}(t),F(t))\,dt
\right]
-(w-z)^2.
$$

Compared with the 2019 paper:
- same broad idea of entropy-regularized exploratory control,
- but the algorithmic machinery is now based on **martingale PE + policy gradient**, not HJB + explicit PIT + Bellman-error fitting.

---

## 3.3 Policy evaluation via martingale conditions

For a given policy $\pi$, define the value function

$$
J(t,x,F;\pi;w)
=
\mathbb E\left[
(x^{u^\pi}(T)-w)^2
+
\gamma\int_t^T
\log \pi(u^\pi(s)\mid s,x^{u^\pi}(s),F(s))\,ds
\;\middle|\;
x^{u^\pi}(t)=x,\ F(t)=F
\right]
-(w-z)^2.
$$

The key theoretical statement is:

$$
J(t,x_t,F_t;\pi;w)
+
\gamma\int_0^t \log \pi(u^\pi(s)\mid s,x_s,F_s)\,ds
$$

is a martingale.

Using test functions $I(t)$, the paper imposes the moment condition

$$
\mathbb E\left[
\int_0^T
I(t)
\left(
dJ(t,x_t,F_t;w;\theta)
+
\gamma \log \pi(u^\pi(t)\mid t,x_t,F_t)\,dt
\right)
\right]
=0.
$$

This is the policy-evaluation equation.

### Important conceptual point
The paper explicitly rejects the old MSTDE-style continuous-time PE objective used in earlier work. It argues the correct basis is **martingality**, not minimizing expected quadratic variation of a martingale.

---

## 3.4 Policy gradient

The policy-gradient identity is written as

$$
\frac{\partial}{\partial \phi}J(0,x_0,F_0;\pi_\phi;w)
=
\mathbb E\left[
\int_0^T
\left(
\frac{\partial}{\partial \phi}\log \pi(u^{\pi_\phi}(t)\mid t,x_t,F_t;w;\phi)
+H(t)
\right)
\left(
dJ(t,x_t,F_t;w;\theta)
+\gamma \log \pi(\cdot)\,dt
\right)
\right].
$$

This yields a model-free actor update once the critic $J(\cdot;\theta)$ is approximated.

---

## 3.5 Joint moment system for actor, critic, and multiplier

The coupled system is:

$$
\mathbb E\left[
\int_0^T
I(t)\{dJ+\gamma\log\pi\,dt\}
\right]=0,
$$

$$
\mathbb E\left[
\int_0^T
\left(
\frac{\partial}{\partial \phi}\log\pi + H(t)
\right)
\{dJ+\gamma\log\pi\,dt\}
\right]=0,
$$

$$
\mathbb E[x^{u^{\pi_\phi}}(T)-z]=0.
$$

This is the conceptual backbone of the whole algorithm.

---

## 3.6 Black–Scholes benchmark: specific parametrizations

For the frictionless multi-stock Black–Scholes case without factors, the paper chooses structured critic/actor classes.

### Critic
$$
J(t,x;w;\theta)
=
(x-w)^2 e^{-\theta_3(T-t)}
+\theta_2(t^2-T^2)
+\theta_1(t-T)
-(w-z)^2.
$$

### Stochastic actor
$$
\pi(\cdot\mid t,x;w;\phi)
=
\mathcal N\big(
-\phi_1(x-w),\ \phi_2 e^{\phi_3(T-t)}
\big),
$$

where:
- $\phi_1\in\mathbb R^d$,
- $\phi_2\in S_{++}^d$,
- $\phi_3$ is treated as fixed,
- they often set $\phi_3=\theta_3$.

This is a multivariate Gaussian analogue of the 2019 exploratory policy.

---

## 3.7 Reparameterization trick for covariance update

A nontrivial technical move is to optimize with respect to $\phi_2^{-1}$, not $\phi_2$. The policy-gradient equation is rewritten in terms of

$$
\frac{\partial}{\partial \phi_2^{-1}}\log \pi(\cdot).
$$

This is explicitly said to be instrumental for the convergence proof.

---

## 3.8 Baseline CTRL algorithm

The paper then defines a stochastic-approximation algorithm.

At iteration $n$, generate a trajectory under current policy $\pi(\cdot\mid t,x;w_n;\phi_n)$, and update:

### Critic update
$$
\theta_{n+1}
\leftarrow
\Pi_{K_{\theta,n}}
\left(
\theta_n
+
a_n
\int_0^T
\frac{\partial J}{\partial \theta}(t,x_n(t);w_n;\theta_n)
\big[dJ+\gamma \log \pi\,dt\big]
\right).
$$

### Actor mean update
$$
\phi_{1,n+1}
\leftarrow
\Pi_{K_{1,n}}
\big(\phi_{1,n}-a_n Z_{1,n}(T)\big).
$$

### Actor covariance update
$$
\phi_{2,n+1}
\leftarrow
\Pi_{K_{2,n}}
\big(\phi_{2,n}+a_n Z_{2,n}(T)\big).
$$

### Lagrange multiplier update
$$
w_{n+1}
\leftarrow
\Pi_{K_{w,n}}
\big(w_n-a_{w,n}(x_n(T)-z)\big).
$$

Here $Z_{1,n}(T)$ and $Z_{2,n}(T)$ are the actor-gradient integrals defined from the policy-gradient equations.

The projection sets expand over time, so the algorithm remains model-free while controlling instability.

---

## 3.9 Theorem 1: exploration–exploitation tradeoff in update noise

One of the nicest insights in the paper is the analysis of the update direction $Z_{1,n}(T)$.

Its conditional mean is

$$
\mathbb E[Z_{1,n}(T)\mid \theta_n,\phi_n,w_n]
=
- R(\phi_{1,n},\phi_{2,n},w_n)\,(\mu-r-\Sigma \phi_{1,n}),
$$

and its conditional variance is bounded by a U-shaped expression in the exploration covariance $\phi_{2,n}$.

### Interpretation
- Too little exploration: policy becomes nearly deterministic, weak policy-improvement signal.
- Too much exploration: update noise becomes too large.
- Hence there is a **U-shaped exploration/exploitation tradeoff**.

This is one of the main implementation-level messages of the paper.

---

## 3.10 Convergence theorem

Under Black–Scholes assumptions and step-size/projection conditions, the paper proves:

$$
\phi_{1,n}\to \phi_1^*=(\sigma\sigma^\top)^{-1}(\mu-r),
$$

$$
\phi_{2,n}\to \phi_2^*=\frac{\gamma}{2}(\sigma\sigma^\top)^{-1},
$$

$$
w_n\to w^*
=
\frac{
z e^{(\mu-r)^\top(\sigma\sigma^\top)^{-1}(\mu-r)T}-x_0
}{
e^{(\mu-r)^\top(\sigma\sigma^\top)^{-1}(\mu-r)T}-1
}.
$$

It also proves the rate

$$
\mathbb E\|\phi_{1,n+1}-\phi_1^*\|^2
\le
C\frac{(\log n)^p\sqrt{\log\log n}}{n}
$$

(up to the exact formatting used in the paper). The point is that the rate is nearly $1/n$, up to logarithmic factors.

---

## 3.11 Deterministic execution theorem

A very important practical observation:

If two stochastic Gaussian policies have the same mean but different covariance, the one with smaller covariance has the same expected terminal wealth but smaller terminal variance. Therefore **deterministic execution dominates stochastic execution** once training is done.

So the paper defines the greedy deterministic policy

$$
u(t,x;w;\phi) = -\phi_1(x-w),
$$

and evaluates performance by the Sharpe ratio

$$
SR(\phi_1)
=
\frac{\mathbb E[x^u(T)/x^u(0)]-1}
{\sqrt{\mathrm{Var}(x^u(T)/x^u(0))}}.
$$

This is specific to the **small-investor frictionless setting**, where counterfactual portfolio consequences are effectively observable “on paper.”

---

## 3.12 Regret theorem

The paper proves a cumulative regret bound in Sharpe ratio:

$$
\mathbb E\left[\sum_{n=1}^N (SR(\phi_1^*)-SR(\phi_{1,n}))\right]
\le
C + C\sqrt{N(\log N)^p \log\log N}.
$$

So regret is sublinear in $N$, meaning average regret vanishes.

This is one of the headline contributions.

---

## 3.13 Modified online CTRL algorithm

The practical version goes beyond the baseline theorem-backed algorithm.

### Modifications
1. **Online incremental updates**
   - update parameters at every timestep instead of only at episode end.
2. **Offline pre-training**
   - initialize from historical data before online deployment.
3. **Mini-batching**
   - lower variance of gradient estimates.
4. **History-dependent test functions**
   - TD($\lambda$)-style weighted traces:
   $$
   I(t)=\int_0^t \lambda^{t-s}\frac{\partial J}{\partial \theta}(s,x(s);w;\theta)\,ds,
   $$
   $$
   H(t)=\int_0^t \lambda^{t-s}\frac{\partial}{\partial \phi}\log\pi(u(s)\mid s,x(s);w;\phi)\,ds.
   $$
5. **Risk-free asset exclusion for experiments**
   - project the unconstrained portfolio onto risky assets only:
   $$
   \hat u(t)=\frac{u(t)}{\sum_{i=1}^d u_i(t)}x(t).
   $$
6. **Rebalancing frequency**
   - e.g. monthly execution with more frequent parameter updates.
7. **Off-policy learning**
   - deterministic greedy target policy for trading,
   - stochastic Gaussian behavior policy for learning.

### Online one-step gradient increments
The modified online algorithm defines one-step quantities

$$
G_{\theta,k}
=
\frac{\partial J}{\partial \theta}(t_k,x(t_k);w;\theta_k)
\left[
J(t_{k+1},x(t_{k+1});w;\theta_k)-J(t_k,x(t_k);w;\theta_k)
+\gamma \hat p(t_k,\phi_k)\Delta t
\right],
$$

and analogous expressions $G_{\phi_1,k}$, $G_{\phi_2^{-1},k}$, then averages over mini-batches and updates online.

The paper gives explicit pseudocode for this modified algorithm in the E-companion.

---

## 3.14 Important details emphasized by the paper

- The algorithm is **model-free**, but not model-free in the vacuous sense: it still assumes a diffusion/Markov structure.
- The policy-evaluation foundation is **martingality**, not MSTDE.
- The Black–Scholes benchmark permits actual convergence/regret analysis.
- Exploration is useful for training but, in the small-investor frictionless setting, **deterministic execution is preferable at deployment**.
- Empirically, the method strongly outperforms model-based continuous-time MV and compares well against many practical baselines.

---

## 3.15 Implementation caveats and ambiguities

1. **Main-paper vs E-companion split**
   - Many implementation formulas live in the E-companion, not the main paper.
   - For coding, you need both.
2. **The online modified algorithm is practical, but the strongest theorem is for the baseline algorithm**, not for the full production-style modified one.
3. **Deterministic execution result is setting-specific**
   - small investor,
   - no market impact,
   - frictionless model.
   It should not be transplanted blindly to large-investor or impact settings.
4. **Discretization is deferred to the final stage**
   - analytically elegant,
   - but real code still depends on $\Delta t$, trace decay choices, rebalancing schedule, and projection implementation.
5. **Sign conventions require care**
   - especially around $\phi_1$, covariance inverse updates, and the relation between actor mean and greedy deterministic execution.
6. The convergence theorem is for a **Black–Scholes, no-factor benchmark**. General factor-driven diffusion settings are algorithmically covered, but not with the same full regret guarantee.

---

# 4. Comparison of the three papers from an implementation perspective

## 4.1 If you want the classical analytic benchmark
Use **Zhou–Li (2000)**.

It gives:
- exact optimal portfolio under known coefficients,
- exact efficient frontier,
- an LQ control backbone.

This is the right benchmark for validating simulators and checking whether a learning algorithm recovers the oracle solution.

---

## 4.2 If you want a minimal exploratory RL formulation with explicit Gaussian structure
Use **Wang–Zhou (2019/2020)**.

It gives:
- explicit entropy-regularized exploratory problem,
- explicit Gaussian optimal exploratory policy,
- PIT,
- a compact handcrafted EMV algorithm.

It is cleaner and simpler than the 2025 framework, but also narrower and less modern.

---

## 4.3 If you want the most realistic research starting point
Use **Huang–Jia–Zhou (2025)**.

It gives:
- general factor-diffusion setup,
- proper martingale policy evaluation,
- actor–critic updates,
- convergence/regret in Black–Scholes,
- online practical modifications.

This is the strongest paper algorithmically, but also the most demanding to implement faithfully.

---

# 5. Concrete ambiguities / places to be careful in code

## 5.1 Zhou–Li (2000)
- No estimation layer is specified.
- No discrete-time implementation details.
- The paper is exact-control theory, not a learning algorithm.

## 5.2 Wang–Zhou (2019/2020)
- Hyperparameter schedules are not deeply specified.
- Multi-asset implementation is not developed in the same detail as the single-asset exposition.
- The EMV algorithm mixes theory-driven parameter constraints with SGD updates; this should be implemented carefully.
- The derivation around the sign / positivity of the Sharpe ratio in the policy rewrite should be checked if you want to cover negative-risk-premium cases.

## 5.3 Huang–Jia–Zhou (2025)
- Several critical implementation formulas are in the E-companion.
- The baseline algorithm is the theorem-backed one; the modified online algorithm is more practical but theoretically less fully pinned down.
- The covariance update through $\phi_2^{-1}$ is easy to mishandle numerically.
- Projection operators onto expanding bounded sets are essential to the proof and may matter in stable implementation.
- There is a subtle but important distinction between:
  - stochastic behavior policy for learning,
  - deterministic target policy for execution.

---

# 6. Suggested use of the three papers in a repo

If the goal is a modular research repo, the cleanest architecture is:

### Layer A — analytic oracle
Implement the **Zhou–Li (2000)** solution first as a benchmark.

### Layer B — theorem-aligned CTRL baseline
Implement the **Huang–Jia–Zhou (2025)** baseline CTRL next:
- Black–Scholes synthetic benchmark first,
- stochastic actor / deterministic execution split,
- martingale critic moment conditions,
- covariance-inverse-safe update path,
- outer-loop $w$ update.

Use **Wang–Zhou (2019/2020)** as a derivational reference for the exploratory Gaussian structure and the role of the outer-loop $w$ update, but do **not** make EMV a required implementation layer in the repo roadmap.

### Layer C — practical constrained / online CTRL
Add the **Huang–Jia–Zhou (2022)** and **2025 E-companion** practical improvements only after the baseline version is stable:
- leverage control,
- risky-only projection when benchmark matching requires it,
- separate rebalance and learning schedules,
- TD($\lambda$)-style traces,
- mini-batching,
- modified online updates.

### Layer D — jump-diffusion portability
Only after the above layers are stable, reuse the framework for jump-diffusion settings.

---

# 7. Bottom line

- **Zhou–Li (2000)** gives the exact **model-based oracle**.
- **Wang–Zhou (2019/2020)** gives the first clean **exploratory Gaussian RL formulation** and the **EMV algorithm**.
- **Huang–Jia–Zhou (2025)** gives the strongest modern **martingale actor–critic / stochastic approximation / regret** framework and the more practical **CTRL algorithm**.

If you are implementing these papers, the most important structural distinctions are:

1. **Estimate-then-optimize** (classical) vs **learn-policy-directly** (RL).
2. **Bellman-error fitting with handcrafted Gaussian/value parametrization** (2019) vs **martingale moment conditions + actor–critic** (2025).
3. **Stochastic policy for learning** vs **deterministic policy for deployment** (2025, in the small-investor setting).

---

## Final note on ambiguities

A few implementation details are underspecified across the papers, especially:
- learning-rate schedules,
- numerical stabilization for covariance updates,
- discretization conventions,
- multi-asset extensions of the simpler 2019 EMV algorithm,
- and how to organize offline pre-training versus online updates in production code.

So for a real implementation, these papers provide the **mathematical blueprint**, but not a complete plug-and-play software specification. Those ambiguities are manageable, but they should be resolved explicitly in your repo design docs before coding.
