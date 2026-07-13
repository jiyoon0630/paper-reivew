---
layout: paper
lang: en
ref: pld-self-improving-vla
title: "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL (PLD)"
date: 2025-10-30
venue: "ICLR 2026 · arXiv:2511.00091"
tags: [VLA, Reinforcement-Learning, Robot-Foundation-Model, Residual-RL, Paper-Review]
authors: "Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi \"Jim\" Fan, Guanya Shi, Yuke Zhu"
affiliations: "NVIDIA (GEAR) · Carnegie Mellon University (LeCAR) · UC Berkeley · UT Austin"
summary: "A self-improving loop that auto-generates deployment-aligned data by wrapping a frozen VLA in residual RL — no human teleoperation."
paper_url: "https://arxiv.org/abs/2511.00091"
---

> **Core claim** — What a VLA needs in order to improve is not more human demonstrations, but **data collected from the states the policy actually visits**. And that data can be generated automatically, without humans, by wrapping a frozen VLA in residual RL.

---

## Introduction

The standard recipe for VLA (Vision-Language-Action) models was carried over wholesale from LLMs: large-scale pretraining on mixed web + robot data, then SFT on a small amount of high-quality teleoperation demonstrations for the target robot and task. π₀, OpenVLA, Octo, and GR00T all sit inside this frame.

But this recipe has a crack that opened up as it crossed from language to robots. PLD points at that crack precisely, and fills it with RL. This post follows the paper's argument while building up the necessary concepts along the way, so that even a reader with a shallow RL background can see the **reason** behind each design decision.

---

## 1. The Problem — Why Human Demonstrations Fall Short

### 1.1 Two distinct defects

The teleoperation data used for SFT suffers from two problems of different natures, layered on top of each other.

**⓵ Scalability** — Robot demonstrations can only be obtained by a human teleoperating the robot. There is no route to acquiring them at scale the way an LLM scrapes internet text.

**⓶ Distributional decoupling** — this one is more fundamental. Human-generated demonstration data is collected **independently of the policy that will be deployed**. A human bakes into the demos only the failure modes they imagined, but the failure states the deployed VLA actually falls into appear nowhere in those demos.

Drawing the second problem as a state distribution:

```
Overlaying the two distributions over the state space S:

  p_human(s)   ....#####.....................     <- states covered by human demos
               -------------------------------> S

  d^pi_b(s)    ....#####..........#####......     <- states the VLA actually visits at deployment
               -------------------------------> S
                                  ^^^^^
                                  coverage gap
                                  = failure region absent from human data
                                    but the policy actually falls into
```

Here $d^{\pi_b}(s)$ is the **state visitation distribution** of the base policy $\pi_b$ — the frequency of landing in state $s$ when $\pi_b$ is deployed. SFT minimizes loss over $p_{\text{human}}$, but evaluation and deployment happen over $d^{\pi_b}$. This mismatch is the classic affliction of imitation learning, **covariate shift**, and it is why a model does well on the tasks it learned yet has no guarantee of transferring to new situations.

### 1.2 The question the paper poses

> With minimal human effort, can a VLA improve itself **using only data that RL curates on its own**? And can that data match or surpass SFT on human-expert demonstrations, both in-distribution and out-of-distribution?

The natural first answer is "train a per-task expert with RL and harvest data from it." But naively bolting RL onto a VLA runs into two walls. What these two walls are, and how PLD breaks through them, is the narrative of the entire paper.

**⛔ Wall 1 — sparse reward.** The reward for a language-conditioned manipulation task is a single success/failure bit. RL becomes unstable and sample-inefficient.

**⛔ Wall 2 — the cost of direct RL on a giant policy.** Directly training an expressive action head such as flow-matching by Q-value maximization is very hard and memory-hungry. By the paper's numbers, OpenVLA-OFT uses up to ~62.5 GB per GPU at batch size 8 when trained on LIBERO. Moreover, an expert obtained this independently drifts in distribution from the generalist, and once converged its behavior is monotonous, so as SFT data it lacks coverage.

---

## 2. Background — The Minimal RL Needed to Understand the Two Walls

You have to know what the two walls really are to see why PLD is shaped the way it is. This section builds up that minimum. If you're comfortable with RL, feel free to skip to §3.

### 2.1 Setup: a goal-conditioned MDP

We view continuous control as an MDP.

$$\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ \rho,\ \rho_0,\ r,\ \gamma)$$

| Symbol | Name | Meaning for a robot |
|---|---|---|
| $\mathcal{S}$ | state space | the space formed by robot proprioception + RGB observations |
| $\mathcal{A}$ | action space | 7-DoF (6-DoF delta pose + 1-DoF gripper) |
| $\rho(s'\mid s,a)$ | transition dynamics | probability of reaching $s'$ after action $a$ in state $s$ = the laws of physics |
| $\rho_0$ | initial state distribution | the distribution episode-start states $s_0$ are drawn from |
| $r$ | reward function | scalar reward |
| $\gamma\in(0,1]$ | discount factor | discount on future reward |

A task is given as a language goal $g\in\mathcal{G}$ ($g\sim p(g)$). Reward and policy become conditioned on the goal.

$$r:\ \mathcal{S}\times\mathcal{A}\times\mathcal{G}\to\mathbb{R},\qquad \pi:\ \mathcal{S}\times\mathcal{G}\to\Delta(\mathcal{A})$$

Here $\Delta(\mathcal{A})$ is the **set of probability distributions over the action space**. That is, the policy does not emit a single action but a distribution $\pi(a\mid s,g)$ over actions. The objective is the expected discounted cumulative reward.

$$J(\pi)\ =\ \mathbb{E}_{g\sim p(g)}\ \mathbb{E}_{s_0\sim\rho_0,\ a_t\sim\pi(\cdot\mid s_t,g),\ s_{t+1}\sim\rho(\cdot\mid s_t,a_t)}\left[\sum_{t=0}^{\infty}\gamma^{t}\, r(s_t,a_t,g)\right]$$

- Outer expectation: average over goals (multi-task)
- Inner expectation: average over trajectories generated by the policy and the dynamics
- $\gamma^t$: down-weights the further future

### 2.2 What Wall 1 really is: sparse binary reward

The paper's reward is:

$$r(s,a,g)\ =\ \mathbf{1}\big[\,d(\phi(s),\,g)\ \le\ \varepsilon\,\big]$$

- $\phi(s)$ — a representation extracting only the goal-relevant part of the state (e.g., an object's position)
- $d(\cdot,\cdot)$ — a distance function
- $\varepsilon$ — a tolerance
- $\mathbf{1}[\cdot]$ — the indicator function (1 if the condition holds, else 0)

1 if close enough to the goal, 0 otherwise. **It carries no information about the process.** Until the object is placed within $\pm\varepsilon$ of the target, the reward stays 0.

A natural objection arises here. Doesn't LLM RL also use sparse reward in the end? GRPO runs on an outcome reward that only looks at final correctness. That works fine — so why is sparse reward a wall for robots?

> ### 💡 Sparse reward is only a wall for robots — because of "how often the reward fires"
>
> Sparse reward isn't bad in itself. What decides success or failure is **how often the current policy stumbles onto reward 1 through its own exploration**, i.e., $\Pr[r=1 \mid \pi_{\text{current}}]$. If this probability collapses to 0, the expected gradient is also 0 and learning can't even begin.
>
> **LLM reasoning** — the pretraining prior is so strong that plain sampling already hits the answer often (base success 30–70%). The sparse reward fires every batch, and GRPO becomes an **exploitation-centric** problem that reweights already-working success samples.
>
> **Robot manipulation** — on a new task the base success rate can collapse to 0. And in a continuous, high-dimensional action space the set of trajectories reaching success is effectively measure-zero, so random exploration has almost no chance of hitting the reward. It becomes a **needle-in-a-haystack exploration** problem.
>
> They differ along four axes.
>
> | Axis | LLM reasoning | Robot manipulation |
> |---|---|---|
> | strength of the prior | strong — exploration is effectively in-distribution | weak on new tasks — exploration is OOD |
> | search-space structure | discrete tokens, many correct paths | continuous & high-dim, success manifold is measure-zero |
> | verification cost | checking against the answer = automatic & free | needs a success predicate / reward classifier |
> | reset cost | a new prompt = free, massively parallel | physical reset required, real-time cost per rollout |
>
> This table becomes the lens for reading PLD's design later. What PLD does is not to make the reward dense, but to **shift the robot's learning setup toward the LLM's conditions** (we confirm this in §3.2).

### 2.3 Q-values and actor-critic

The fundamental dilemma of RL: to fix the policy you need to know which action is good, but the goodness of an action depends on all the future reward that follows it. This is divided across two networks.

**Critic** — learns the action-value function $Q^\pi(s,a)$.

$$Q^\pi(s,a)\ =\ \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\ \Big|\ s_t=s,\ a_t=a\right]$$

"The expected discounted cumulative reward from taking action $a$ now in state $s$ and following $\pi$ thereafter." This function satisfies a recursion in itself (the Bellman equation), so we train it by using that identity as the regression target (TD-learning).

$$Q^\pi(s_t,a_t)\ \leftarrow\ \underbrace{r(s_t,a_t)}_{\text{immediate reward}}\ +\ \gamma\,\mathbb{E}_{s_{t+1}\sim\rho}\big[\underbrace{Q^\pi_{\text{target}}(s_{t+1},a_{t+1})}_{\text{bootstrapped future value}}\big]$$

**Actor** — the policy $\pi_\theta$, trained to output actions the critic scores highly.

$$\max_\theta\ J(\theta)\ =\ \mathbb{E}_{s}\big[\,Q(s,\ \pi_\theta(s))\,\big]$$

How this objective is optimized is the key to understanding Wall 2. The problem is that the parameters $\theta$ **do not appear directly** inside $Q$. $\theta$ influences $Q$ **only through** the action $a=\pi_\theta(s)$. So we split it with the chain rule.

$$\nabla_\theta J\ =\ \mathbb{E}_s\Big[\ \underbrace{\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)}}_{\textbf{A: the piece the critic gives}}\ \cdot\ \underbrace{\nabla_\theta \pi_\theta(s)}_{\textbf{B: the piece the actor gives}}\ \Big]$$

- **A** $=\nabla_a Q$ — "which direction to push the action to raise Q." The **improvement direction in action space** that the critic reports. Since the critic's parameters are frozen during the actor update, this is treated as a constant vector at this step.
- **B** $=\nabla_\theta\pi_\theta$ — "which direction the action moves when you push the parameters." The actor's own sensitivity (its Jacobian).
- **A·B** — their product is the final **parameter update direction**.

Intuitively, the actor cannot pick the action directly. The only handle in its hand is $\theta$, and the action is its output. So the goal "produce a good action" has to be translated into "push $\theta$ in this direction," and the chain rule does the interpreting. It's also important that A is a **local direction**, not the destination ($\arg\max_a Q$). Because the destination is unknown, we repeat small gradient-ascent pushes.

Comparing with supervised learning sharpens the picture. The mechanism (backprop, chain rule) is exactly the same. What differs is the **source of the signal**. Supervised learning shrinks the distance to a fixed ground-truth label $y^*$ ($\nabla_a L = 2(a-y^*)$), whereas actor-critic has no ground-truth label, so **the critic takes its place — and that critic is itself learned and moving**. Indeed, if the critic had the parabolic form $Q(s,a) = -\lVert a - a^*\rVert^2$, then $\nabla_a Q = -2(a-a^*)$, the same expression as the supervised gradient up to sign.

### 2.4 What Wall 2 really is: the requirement of differentiable action generation

The decisive premise of the derivation above is that **B must be computable** — i.e., $a=\pi_\theta(s)$ must be smoothly differentiable with respect to $\theta$. This is where the type of action head splits things.

**✅ Gaussian policy** — solved in one shot by reparameterization.

$$a\ =\ \mu_\theta(s)\ +\ \sigma_\theta(s)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)$$

Pulling $\epsilon$ outside makes $a$ a **single smooth function** of $\theta$, and $\nabla_\theta\pi$ comes out immediately. SAC/DDPG/TD3 run as-is.

**❌ Flow-matching / diffusion head** — the action is the product of a multi-step generative process.

$$z_0\ \xrightarrow{\ \text{step 1}\ }\ z_1\ \xrightarrow{\ \text{step 2}\ }\ \cdots\ \xrightarrow{\ \text{step N}\ }\ a$$

To get $\nabla_\theta\pi$ you must backprop through this entire N-step chain (BPTT through an ODE solver or the denoising chain). Memory explodes and it is numerically unstable. On top of that, these heads are trained with a score-matching/flow-matching objective, not likelihood maximization, so there is no clean $\log\pi$ or reparameterized sample to plug straight into a policy gradient. This is the "policy-agnostic RL" difficulty the paper cites.

**This is the exact content of Wall 2.** The more modern the VLA (like π₀), the more it uses an expressive flow head — and it is precisely that expressiveness that makes direct RL impossible.

But another objection arises. LLM RL doesn't suffer this problem — GRPO happily runs RL on giant models. What's different?

> ### 💡 There are two branches of policy gradient, and LLMs and robots ride different ones
>
> What we derived in §2.3 is in fact only **one branch** of policy gradient.
>
> **Branch A — Pathwise / DPG**
>
> $$\nabla_\theta J\ =\ \mathbb{E}_s\big[\nabla_a Q(s,a)\cdot\nabla_\theta\pi_\theta(s)\big]$$
>
> **Backprops through** Q and the action generation. Low-variance and off-policy-friendly, but **differentiable action generation is mandatory**. Representatives: DDPG, TD3, SAC. **PLD belongs here.**
>
> **Branch B — Score-function / REINFORCE**
>
> $$\nabla_\theta J\ =\ \mathbb{E}\big[\nabla_\theta \log\pi_\theta(a\mid s)\cdot \hat{A}\big]$$
>
> Does not differentiate Q. It treats the advantage $\hat A$ as a **scalar weight (a constant)** and differentiates only $\log\pi_\theta$, "pushing up the log-probability of good actions." Since only $\log\pi$ is needed, there is no need to backprop through action generation. In exchange it is high-variance and usually on-policy. Representatives: REINFORCE, PPO, **GRPO**.
>
> This distinction, meshed with the table in §2.2, explains everything.
>
> - **LLMs ride Branch B.** Softmax gives $\log\pi$ for free, and no action backprop is needed at all. Branch B's weaknesses — high variance and being on-policy (throwing samples away) — aren't a problem for LLMs, because resets are free and massively parallel sampling is available. **So LLMs need not care about flow heads or anything of the sort.**
> - **Robots must ride Branch A.** Real rollouts are expensive so data can't be thrown away; reusing a replay buffer (off-policy) is a survival condition. But Branch A demands differentiable action generation. **This is exactly where the flow head gets stuck.**
>
> In short, Wall 2 is the structural dilemma: "robots must ride Branch A because samples are expensive, but Branch A is incompatible with the flow head." PLD's solution lies in **going around** this dilemma rather than through it.

### 2.5 One more pass on off-policy

Let's make the reason Branch A is mandatory explicit. RL algorithms split by "whose data can you learn from."

| | On-policy | Off-policy |
|---|---|---|
| training data | only what this policy just produced | data any policy produced in the past is OK too |
| replay buffer | not reusable (discarded when the policy changes) | reusable |
| sample efficiency | low | high |
| representatives | PPO, REINFORCE | DQN, DDPG, TD3, SAC |

Off-policy is possible because learning centers on the **Bellman equation (Q)** rather than the policy. Bellman doesn't care which policy the data came from.

In PLD off-policy is not a choice but a **necessity**. As we'll see, PLD fills the buffer with **success trajectories produced by the base policy $\pi_b$**, while the thing being trained is the **residual policy $\pi_\delta$**. The policy that made the data differs from the policy being trained — by definition this only works if it's off-policy.

### 2.6 The SFT loss (used in the Distill stage)

Finally, let's lay out the VLA's SFT loss. A VLA policy takes an observation $o_t$ and goal $g$ and emits an action.

$$a_t\ =\ D_\phi\big(h_\theta(o_t,\ g)\big)$$

- $h_\theta$ — the vision-language backbone
- $D_\phi$ — the action head

The SFT loss differs by head type. This is what later grounds PLD's claim to be "architecture-agnostic."

**Autoregressive tokens** (OpenVLA) — NLL over the action-token sequence.

$$\mathcal{L}_{\text{AR}}(\theta)\ =\ -\,\mathbb{E}_{k\sim[K]}\big[\log p_\theta(u_k\mid u_{<k},\ x)\big],\qquad x=(o_t,\ g_t)$$

- $u_k$ — the $k$-th action token, $K$ — number of tokens

**Diffusion** (Octo, Diffusion Policy) — score-matching MSE.

$$\mathcal{L}_{\text{diff}}(\theta)\ =\ \mathbb{E}_{t,\,\epsilon,\,(x,a)}\big[\ \lVert \epsilon\ -\ \epsilon_\theta(a_t^{(\text{noisy})},\ x,\ t)\rVert_2^2\ \big]$$

- $\epsilon$ — the injected noise, $\epsilon_\theta$ — the noise-prediction network, $t$ — the diffusion timestep

**Flow-matching** (π₀) — an $L_2$ loss on the velocity field that transports the prior onto the action distribution.

---

## 3. Method — PLD's Three Stages

Now that the two walls are clear, we can read why PLD's solution is shaped the way it is.

The core strategy is **decoupling**. It does not RL the giant policy as a whole. It **freezes** the base VLA, trains only a lightweight **residual policy** on top with RL to obtain a task expert, harvests data from that expert, and **distills it back into the base with standard SFT**.

```
  STAGE 1: LEARN            STAGE 2: PROBE             STAGE 3: DISTILL
  ------------------        --------------------       -------------------
  frozen base pi_b          base pi_b walks T_base     distill tau_demo
     +  residual pi_d          steps (probe)              back into base
     -> abar = a_b + a_d    -> residual expert            via standard SFT
                               takes over and          -> improved generalist
  off-policy RL yields         shows recovery
  an expert that               (takeover)              experts discarded;
  surpasses the base                                   only generalist ships
                            data production,
                            not training
```

### 3.1 The residual policy — the structure that bypasses Wall 2

Keep the base policy $\pi_b$ frozen and train only a **small Gaussian policy conditioned on the base action**, $\pi_\delta$, on top. The action that goes out into the environment is the sum of the two.

$$\bar{a}\ =\ a_b\ +\ a_\delta,\qquad a_b\sim\pi_b(\cdot\mid s),\qquad a_\delta\sim\pi_\delta(\cdot\mid s,\ a_b)$$

- $a_b$ — the action from the frozen base VLA (**not differentiated**; used only as a conditioning input)
- $a_\delta$ — the learned residual correction, $a_\delta\in[-\xi,\ \xi]$
- $\xi\in[0,1]$ — the cap on residual magnitude. Controlled by a scheduler, kept small early so the policy doesn't diverge sharply from the base.

The critic for the combined policy $\bar\pi$ is trained with TD-learning.

$$Q^{\bar\pi}(s_t,\ \bar a_t)\ \leftarrow\ r(s,a)\ +\ \gamma\,\mathbb{E}_{s_{t+1}\sim\rho(\cdot\mid s_t,\bar a_t)}\big[\,Q^{\bar\pi}_{\text{target}}(s_{t+1},\ \bar a_{t+1})\,\big],\qquad \bar a = a_b + a_\delta$$

**Let's see how this structure bypasses Wall 2.** In §2.4 the problem was the flow head's $\nabla_\theta\pi$. PLD doesn't differentiate the flow head at all — it freezes it and uses it only as the conditioning input $a_b$. The only thing RL attaches to is the Gaussian residual, and a Gaussian differentiates in one shot via reparameterization. That is, it **uses Branch A (pathwise) but confines action generation to the differentiable part**.

As a result, in the paper's words, "the residual Gaussian policy can be trained with **any off-the-shelf off-policy RL algorithm**." Off-the-shelf here means "a stock product, unmodified" — you can attach SAC or TD3 with no customization, in direct contrast to direct RL on the flow head, which would require building special machinery to punch through multi-step sampling.

At the same time this structure solves **exploration initialization**. The base policy, even if it doesn't generalize perfectly, still makes a "plausible attempt." Since the residual explores around it (within $[-\xi,\xi]$), exploration happens not at random but **in the neighborhood of meaningful actions**. This is the first device for breaking Wall 1.

### 3.2 Three devices for sample efficiency — breaking Wall 1

The residual structure alone isn't enough. To learn sample-efficiently under sparse reward, three more things are needed.

**⓵ RLPD-style symmetric replay — always plant success in the batch**

Maintain two buffers.

$$\mathcal{B}_{\text{offline}} = \{\tau_1,\ \tau_2,\ \dots\}\quad(\text{only } \pi_b \text{ success trajectories}),\qquad \mathcal{B}_{\text{online}}\quad(\text{newly collected online experience})$$

And draw the minibatch **in equal numbers** from the two buffers. As the paper notes, filling the offline buffer with success trajectories only acts as **importance sampling that preserves only successful attempts**, and symmetric sampling guarantees the value function is always trained **on high-value state-action pairs**. That is, every batch the critic sees contains **half reward-1 samples**.

**⓶ Warm-up — ground the critic on sane experience first**

Early in training the critic is immature and the residual is effectively random. If you unleash the residual right away, the $a_\delta$ in $\bar a = a_b + a_\delta$ is noise, so it visits nonsensical states and the critic learns from noise and diverges. So for the first while, **collect data with $\pi_b$ alone** (the WSRL approach) to fill the buffer and critic with sensible experience. This stabilizes off-policy learning and mitigates forgetting.

**⓷ Cal-QL — initialize the critic to conservative but "calibrated" values**

Stage 1 is essentially an **offline → online transition** (pretrain the critic on base success trajectories, then the residual starts online exploration). This transition point is notoriously unstable.

> ### 💡 Why Cal-QL rather than plain CQL
>
> **Root of the problem — overestimation in offline RL.** Q-learning takes a max/expectation over actions in the bootstrap. For out-of-distribution (OOD) actions Q is extrapolated, and because the max operation **picks out the overestimated one**, the error skews optimistic. The policy chases these ghostly-high Q values and collapses. In PLD the residual proposes combined actions $\bar a = a_b + a_\delta$ absent from the base data, so it is exposed to exactly this problem.
>
> **CQL's prescription.** Push down the Q of OOD actions the policy proposes, and push up the Q of actions present in the data. As a result the learned Q becomes a **conservative lower bound** on the true value, so the policy can't chase ghosts.
>
> **CQL's side effect.** The conservatism overshoots and the **scale of Q itself goes off**. Entering online fine-tuning in this state, the online updates first have to undo (unlearn) these over-suppressed Q values, so early performance drops sharply. This is the famous offline→online "initial dip."
>
> **Cal-QL's prescription — calibration.** Keep the conservatism but **lay down a floor.** Sandwich the learned Q so that it is a lower bound on the true value while also being an upper bound on some reference (behavior) policy's value.
>
> $$V^{\text{ref}}(s)\ \le\ Q_{\text{learned}}(s,a)\ \le\ Q^{\pi}_{\text{true}}(s,a)$$
>
> That is, suppress the Q of OOD actions but **never below the reference policy's value.** It stays conservative yet keeps a "reasonable scale," and as a result the catastrophic dip at the online transition disappears. The implementation is essentially a one-line change on top of CQL.
>
> Because PLD's Stage 1 is exactly offline→online, this device is needed. Indeed, it is thanks to this that the **temporary early-training performance drop** the paper reports (a trace of early exploration where the residual diverges from the base and visits suboptimal states) stays a shallow, recoverable dip rather than a catastrophe.

**And what it deliberately does not do — a behavior constraint**

Many offline / offline-to-online RL methods put a **behavior constraint** on the policy loss to tie the learned policy near the data-collection policy (the BC penalty in TD3+BC, the implicit constraint in AWAC, etc.) — another safeguard against overestimation. **PLD deliberately does not impose this.** The reason is clear — tying the policy near the base makes the expert **inherit the base's ceiling** as-is. PLD's goal is for the expert to **surpass** the base, so it unclips this leash.

So where does stability come from? This is the elegant point of the design. PLD **removes stabilization from the policy loss and distributes it across three other places.**

| Stabilization responsibility | Handled by |
|---|---|
| Structural side | $\xi$-bounded residual — the action is structurally tied near the base (not a loss penalty) |
| Critic side | Cal-QL — handles overestimation on the critic side |
| Data side | symmetric replay — grounds the critic on high-value states |

The policy is free to surpass the base while the system as a whole stays stable.

Now we're ready to return to the perspective foreshadowed in §2.2.

> ### 📌 PLD's core insight — don't fix the reward; move the robot into the LLM's conditions
>
> In §2.2 we distilled the conditions under which sparse reward works for LLMs into three. Seen through that lens, PLD's Stage 1 transplants one of those conditions each onto the robot.
>
> | Condition under which sparse reward fires for LLMs | The device by which PLD reproduces it for robots |
> |---|---|
> | **ⓐ A strong prior produces success samples often** | **policy prior warm-start** — the frozen base secures a non-zero success rate, and the residual explores only within $[-\xi,\xi]$. Same setup as GRPO sampling from an already-capable LLM |
> | **ⓑ Success samples always exist in the batch** | **RLPD symmetric replay** — prefill the offline buffer with the base's success trajectories and sample fifty-fifty. LLMs get success mixed into group samples naturally; robots **guarantee it artificially via the buffer** |
> | **ⓒ Verification is free and resets are free** | robots don't have this luxury → compensate with **Cal-QL** instead. It's scaffolding **needed only for robots** to keep off-policy Q from diverging when success is sparse and costly (unnecessary for GRPO, which has no critic) |
>
> **In one sentence** — PLD doesn't make the reward dense via reward shaping. Instead it **reconstructs the LLM's success conditions on the robot**: "a capable prior + a state where that prior's successes are always in the batch." Then the very sparse reward that stalled naive robot RL **fires**, just as the outcome reward fires in GRPO. On top of this, the Gaussian residual of §3.1 makes "those successes learnable by an off-the-shelf off-policy algorithm," completing the puzzle.

With this combination the expert reaches over 99% success per task (over 95% across the 120+ reported tasks).

### 3.3 Probe — collecting data aligned with the deployment distribution

Now that we have the expert, it's time to harvest data. But there's a counterintuitive trap here.

**Pure RL-expert data is too optimal.** It is consistent, hesitation-free, and finishes smoothly in a short horizon. And **that is exactly the problem.** This narrow, unimodal distribution underrepresents OOD states and failure states. No matter how much such data you add, performance doesn't rise; instead the generalist overfits and its robustness and generalization suffer.

> **"more optimal data" ≠ "better SFT data"**

A student who only watches smooth, competent demonstrations never learns **how to get out when things slip**. The coverage gap of §1 recurs verbatim.

**The fix is to put the base back on stage.** Let the base walk the front of the trajectory to enter the deployment distribution, and from the back have the expert take over and demonstrate recovery. This is the **hybrid rollout**, and the front segment is called **base policy probing**.

```
  t=0                        t=T_base                          t=T
   |<------ base pi_b ------->|<------ residual expert ------->|
   |         (probe)          |          (takeover)            |
   |                          |                                |
   | states : s_1 ... s_{t-1} | states : s_t, s_{t+1}, ...     |
   | action : a_b             | action : abar = a_b + a_delta  |
   |                          |                                |
   +-- prefix ----------------+-- suffix ----------------------+

  T_base ~ Uniform[0, alpha * T]
```

- **prefix** — the segment entering the states the base actually visits at deployment (probing)
- **suffix** — the segment where the expert takes over from that point and demonstrates recovery
- **alpha** — the cap ratio on probing length. The hyperparameter that controls data diversity

The collected demonstration trajectory is:

$$\tau_{\text{demo}}\ =\ \underbrace{\big\{(s_1,\ a_{b,1}),\ \dots,\ (s_{t-1},\ a_{b,t-1})\big\}}_{\textbf{prefix — deployment-distribution states the base visited}}\ \cup\ \underbrace{\big\{(s_t,\ a_{b,t}+\bar a_t),\ \dots\big\}}_{\textbf{suffix — the expert's recovery actions}}$$

- $s_1,\dots,s_{t-1}$ — the states the base actually visited during probing. **The distribution the VLA actually falls into at deployment**
- $t = 1 + T_{\text{base}}$ — the takeover moment. The larger $T_{\text{base}}$, the deeper the base drifts
- the recorded action in the suffix — the **executed combined action** after takeover. It captures the expert's behavior recovering from potentially suboptimal regions

> ### ⚠️ Fact-check — a notational inconsistency in the $\tau_{\text{demo}}$ formula
>
> In §3.1 $\bar a = a_b + a_\delta$ was defined as the **combined action**, yet the $\tau_{\text{demo}}$ formula above writes "$a_{b,t} + \bar a_t$." Read literally, the base action is double-counted. The intended meaning in context is "**record the executed combined action (base + residual)**," where $\bar a_t$ should be read as loose notation for the residual component. A minor notational inconsistency in the paper.

**There's a subtle but important asymmetry in the probing segment.** The probing steps are used **only for state initialization** and are **not added to the replay buffer** — because the base's probing actions are suboptimal and must not become RL's learning target. On the other hand, **the SFT data $\tau_{\text{demo}}$ includes both the base prefix and the expert suffix.** The point is to have the generalist learn to behave like the base in normal states and to recover like the expert when it has drifted.

Two questions naturally arise here. One is "isn't this just DAgger?" and the other is "then when does the expert learn failure recovery?"

> ### 💡 This is the autonomous version of robot-gated DAgger
>
> The DAgger family treats exactly the same disease (covariate shift) with the same skeleton: the student enters its own distribution, an intervener demonstrates recovery at that point, and the student is retrained on that data.
>
> | Method | Who decides intervention timing | Expert |
> |---|---|---|
> | DAgger | no gating — roll out the whole student trajectory, then label every state | human/oracle |
> | HG-DAgger | a **human** judges risk and intervenes | human |
> | robot-gated DAgger | an **algorithm** estimates uncertainty/risk and requests intervention | human |
> | **PLD** | a **random schedule** $T_{\text{base}}\sim[0,\alpha T]$ | a **learned RL residual** |
>
> PLD's probe→takeover has the same structure as robot-gated DAgger, but three things differ decisively.
>
> - **The expert is a learned RL residual, not a human.** The biggest cost of the DAgger family — "keeping a human in the loop" — disappears.
> - **The gating is a random schedule, not uncertainty.** Robot-gated DAgger intervenes *when it's risky*, but PLD takes over **unconditionally** after a random $T_{\text{base}}$ — because the goal is not risk avoidance but **securing diversity**.
> - **It's not online iterative aggregation; the data is gathered and SFT'd once, offline.**
>
> So it's accurate to read PLD as **"the autonomous version of robot-gated DAgger with the human removed and the gating randomized."**

> ### 💡 When does the expert learn "how to recover from failure" — separating training from demonstration
>
> The scene where the expert "demonstrates" recovery in Stage 2 and the scene where the expert "learns" recovery look identical but are different events. **The only difference is whether parameters are updated.**
>
> **The expert's training (finishes in Stage 1).** RL only gets good at states it **practiced from as starting points**. So for the expert to take over from arbitrary base-visited states, you have to change the episode start during training. Instead of the default initial distribution $\rho_0$,
>
> $$s_0\ \sim\ p_0^{\pi_b}\quad(\text{the distribution of states reached after the base walks random steps})$$
>
> start from there. Only this way does the expert repeatedly practice recovery from such states and become robust.
>
> **Let's get one phrasing exactly right.** $p_0^{\pi_b}$ is **not** a "failure-state distribution." It is the **entire state distribution the base actually visits** — normal trajectories, mild drift, and near-severe-failure all mixed together. Near-failure is only an **important subset** of it. And this is exactly what fills the coverage gap of §1. What PLD does is not "cherry-picking failures" but **aligning the data with the deployment distribution**; failure recovery is the most valuable byproduct of that alignment.
>
> **Data generation (Stage 2).** Here the expert is **already done training**. The recovery "demonstration" is not parameter-updating learning but the act of **producing data** for the generalist.
>
> **Final robustness (Stage 3).** The one that actually becomes robust to failure is the **generalist** that learns that data via SFT.
>
> A driving analogy. The instructor putting the car halfway into a ditch and saying "get it out" = **setting the start state**. Me repeatedly getting out and improving = **training (Stage 1)**. After becoming skilled, filming the ditch-escape to make a manual = **data generation (Stage 2)**. The beginner who learns from that manual = **the generalist (Stage 3)**.

**$\alpha$ is the knob that controls diversity.** Since probing length is drawn as $T_{\text{base}}\sim\text{Uniform}[0,\ \alpha T]$, a larger $\alpha$ means the base walks longer and drifts deeper, the **detour** correcting it grows longer, and trajectory diversity increases. The paper's sweep over $\alpha\in\{0.0,\ 0.2,\ 0.4,\ 0.6,\ 0.8\}$ yields an inverted U.

```
  performance
   ^
   |                    .----*----.                 <- plateau at alpha = 0.6
   |              .----'           '--.
   |        .----'                     '--.         <- too much -> decline
   |  .----'
   +----+-------+-------+-------+-------+------> alpha
      0.0     0.2     0.4     0.6     0.8

      too little diversity <-------------> departs from base distribution
```

Moderate probing gives coverage of recovery scenarios, but too much strays too far from the base distribution and does harm instead.

### 3.4 Distill — folding it back with standard SFT

Distill the collected $\tau_{\text{demo}}$ as-is with the **standard SFT loss** matching the base VLA's head (one of §2.6's AR NLL / diffusion MSE / flow-matching $L_2$). This folds the experts of many tasks into a single generalist; the experts were scaffolding for data production, so they are discarded and only the one generalist is deployed zero-shot.

A terminological clarification is needed here. **This "distill" is not knowledge distillation where a student matches a teacher's logits.** It is standard BC SFT of the generalist on the **trajectories the expert generated** (hard action labels) — i.e., **policy distillation through generated data.**

> ### ⚠️ Fact-check — there is no "Section 3.3" in the paper
>
> The method section ends Overview → 3.1 → 3.2. The Distill stage has **no method subsection of its own.** The technical content (the SFT loss) is already defined up front in the Preliminaries, and "what it does" appears only briefly in the Method Overview.
>
> **And this is not an accident — it is the thesis itself.** The very fact that Stage 3 is "just standard SFT," with no new methodology to explain, is PLD's **plug-and-play** claim. All the RL complexity is isolated in the to-be-discarded expert (Stage 1), and only clean SFT data is handed to the generalist. So you can plug the data into any SFT pipeline of any VLA.

> ### 📌 The improved generalist surpasses "the average expert"
>
> A notable observation in the paper: the generalist distilled from them is better than the individual per-task experts. There are two reasons.
>
> - **Aggregation** — data from many task experts is gathered into one model.
> - **Preserved generalization (less forgetting)** — because PLD data sits near the base distribution, the base's generalization ability is lost less (§4).
>
> This synergy answers "why folding many experts into one isn't a loss." And feeding this output (the improved generalist) back in as the Stage 1 base gives a **self-improving flywheel.**
>
> $$\text{generalist}\ \to\ \text{residual expert}\ \to\ \tau_{\text{demo}}\ \to\ \text{better generalist}\ \to\ \cdots$$

---

## 4. Why It Works

The paper explains why PLD data beats both human demonstrations and pure RL data along two axes.

**⓵ Less forgetting.** PLD data clusters near the base's rollouts (since the prefix comes from the base distribution). The paper connects this to an observation from LLM fine-tuning — from the viewpoint of "RL's Razor" (Shenfeld et al., 2025), that **KL-divergence is the indicator of forgetting**, data that stays near the base keeps the policy's post-fine-tuning KL shift small, and thus loses less of the base's generalization ability.

**⓶ Coverage.** By capturing diverse recovery behaviors and representing failure states, it raises robustness in sequential decision-making (consistent with the observation of Kelly et al., 2019, HG-DAgger).

These two axes are observed exactly in the real world. In the Franka cube pick-up experiment, policies trained on RLPD data or human data repeatedly failed by pushing the cube into the upper-left corner and jamming the gripper. The PLD-trained policy, by contrast, recovered by repositioning the cube before grasping. Analyzing the distributions, **that corner state never appeared in the human demonstrations nor in the pure RL rollouts; only PLD's probing captured that case in the data.** This is the most direct evidence that the self-improving data flywheel actually works.

---

## 5. Experiments

### 5.1 Sample efficiency of expert training

Against WSRL (offline initialization only) and RLPD (no base-policy guidance) as baselines, a wide margin on 8 tasks of LIBERO-90. Per task: 50 success trajectories of pretraining data, Cal-QL critic initialization, 250k steps of online interaction, 3-seed 95% CI. Reached over 95% across the 120+ reported tasks.

### 5.2 In-distribution performance

With **zero** additional human demonstrations, a consistent absolute improvement over SFT on human demonstrations.

**LIBERO** (evaluated over 50 episodes per task)

| | Spatial | Object | Goal | **Avg** |
|---|---|---|---|---|
| π₀ baseline | 95.2 | 97.6 | 87.4 | **93.4** |
| π₀ + PLD | 97.7 | 98.5 | 95.3 | **97.2** (+3.8) |
| OpenVLA baseline | 92.9 | 99.1 | 83.25 | **91.8** |
| OpenVLA + PLD | 99.5 | 99.1 | 98.9 | **99.2** (+7.4) |

**SimplerEnv** (Octo-based)

| | Eggplant | Carrot | Open Drawer | Coke Can | **Avg** |
|---|---|---|---|---|---|
| Octo-SFT | 65.5 | 43.3 | 92.5 | 85.7 | **71.8** |
| + ours | 97.8 | 93.9 | 99.3 | 95.5 | **96.6** |
| Δ | +32.3 | **+50.6** | +6.8 | +9.8 | **+24.9** |

> ### ⚠️ Fact-check — "over 50% gains in SimplerEnv" is not the benchmark average
>
> The abstract and intro state "over 50%" improvement on SimplerEnv, but the table shows the **average absolute improvement is +24.9 pp**. The "50%" refers to the maximum, the **+50.6 pp of a single task (Carrot Pick)**. Even computed as relative improvement, the average is 71.8 → 96.6, about +34.7%, short of 50%.
>
> The LIBERO "99%" claim, by contrast, matches the table exactly (OpenVLA + PLD, Avg 99.2). The two should be distinguished when cited.

### 5.3 Generalization

- **Unseen task (zero-shot)** — training on only 10% of LIBERO-90's tasks still yields 24.4% SR on unseen tasks. By contrast, base-policy rollout (0-1 REINFORCE-style self-bootstrap) is weak even in-distribution and fails to generalize. Human data is similar zero-shot but lags in-distribution.
- **Out-of-domain (few-shot)** — increasing PLD data from 50 → 500 trajectories monotonically raises target performance.
- **Long-horizon** — on LIBERO-10, better than self-bootstrap but **still short of SFT on human demonstrations.** A limitation the paper states explicitly.

### 5.4 Real world

**Franka 7-DoF** (pick-and-place, peg insertion) — a hard setting with no restriction on task randomization. Both the PLD and RLPD experts reached 100% success within 2 hours with no human intervention, and each auto-collected 200 success demonstrations to re-SFT π₀. Over 30 random trials:

| | cube pick-up | peg insertion |
|---|---|---|
| π₀ + PLD | **30 / 30** | 30 / 30 |
| π₀ + RLPD | 16 / 30 | 30 / 30 |
| π₀ + Human | 10 / 30 | 30 / 30 |

**YAM dual-arm 6-DoF** (industrial GPU insertion) — decomposed into 4 stages: insert the GPU into slot 1 → move to slot 3 → insert → pull out and return to the table, with a reward classifier orchestrating the state machine. After up to 8 hours of training per subtask, distilled into a single BC policy. It ran the full loop **continuously for at least an hour with no human intervention or resets**, recovering from failures on its own to keep the flywheel going.

---

## 6. Positioning — Among Neighboring Work

PLD sits at the intersection of three lineages.

**⓵ Robot foundation models / VLA post-training** — RT-1/2, OpenVLA, Octo, GR00T, the π-series. The data scarcity and coverage limits of the "large-scale pretraining + small-demo SFT" standard are PLD's starting point.

**⓶ Sample-efficient RL based on data/policy priors** — on the data-prior side there are RLPD, Cal-QL, WSRL; on the policy-prior side there are **ResiP** (training a residual with PPO) and **EXPO** (co-training an off-policy residual with the base). These two are PLD's closest cousins. PLD's distinction is that it **warm-starts a non-zero success rate from a suboptimal base while requiring no oracle demonstrations or human intervention at all.**

**⓷ RL post-training for VLAs** — VLA-RL, on-policy interaction post-training, etc. These either involve heavy human intervention, collect data independently of the generalist's behavior, or sacrifice generalization to single-task optimization. PLD targets all three at once.

The most interesting contrast is with a cousin paper released around the same time.

> ### 🔗 π*₀.₆ / RECAP (Physical Intelligence, 2025-11) — the opposite design choices
>
> A self-improving VLA from the same period targeting the same problem (imitation learning's error accumulation and its inability to recover from its own mistakes). The RECAP method comprises three stages — demonstration → real-time correction by a human teleoperator → autonomous practice — and its core mechanism is **advantage conditioning.**
>
> Placed side by side with PLD, it picks the opposite side on every axis.
>
> | | **PLD** | **π*₀.₆ / RECAP** |
> |---|---|---|
> | Human intervention | **none** (an autonomous expert demonstrates recovery) | **yes** (real-time teleoperation correction) |
> | RL style | residual pathwise RL + SFT distillation | advantage-conditioning (no policy gradient) |
> | Critic | **off-policy Q** (Cal-QL) | **on-policy V** (distributional MC) |
> | Failure data | converted into recovery trajectories and distilled | absorbed with a low-advantage label |
>
> On the "human intervention vs autonomy" axis set up in the §3.3 DAgger comparison, **PLD is the autonomous version with the human removed, and RECAP is the side that keeps teleoperation correction.** Reading the two papers together brings the current landscape of self-improving VLAs into focus.

It's interesting that the way advantage is handled splits three ways here. In §2.4 we saw two branches of policy gradient, but RECAP is in fact a **third path** belonging to neither.

> ### 💡 How advantage is computed vs how it's used — GRPO, RECAP, PLD compared
>
> All three share the skeleton "$A$ = (estimate of this action's goodness) − (baseline)." What differs is **where the baseline comes from.**
>
> **GRPO — no value network at all.** Sample $G$ outputs for one prompt and normalize within the group.
>
> $$\hat A_i\ =\ \frac{r_i\ -\ \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$
>
> The baseline is not a learned $V$ but the **average reward of samples drawn from the same prompt.**
>
> **RECAP — learns only $V$.** With no separate $Q$ network, it estimates the $Q$-equivalent term via n-step reward + $V$ bootstrap.
>
> $$A^{\pi_{\text{ref}}}(o_t,\ a_t)\ =\ \mathbb{E}\Big[\underbrace{\textstyle\sum_{t'=t}^{t+N-1} r_{t'}\ +\ V^{\pi_{\text{ref}}}(o_{t+N})}_{n\text{-step estimate of }Q}\Big]\ -\ \underbrace{V^{\pi_{\text{ref}}}(o_t)}_{\text{baseline}}$$
>
> $V$ is represented distributionally (a distribution over discrete value bins) and regressed onto empirical returns — a **Monte Carlo, on-policy** estimate. The paper itself admits this is less optimal than an off-policy Q estimator but is simpler and more reliable.
>
> **PLD — learns an off-policy $Q$ (Cal-QL).** Exactly the opposite choice from RECAP.
>
> **And the usage differs across the three, too.**
>
> | | Computation (baseline) | Usage |
> |---|---|---|
> | GRPO | group-sample mean | **gradient weight** $\nabla\log\pi\cdot\hat A$ |
> | RECAP | learned $V$ (on-policy MC) | **conditioning input** — binarize $A$ by a threshold, feed it as a condition of $\pi(a\mid o,\ I)$, and request "high advantage" at inference |
> | PLD | off-policy $Q$ (Cal-QL) | **trains a pathwise actor via the critic** |
>
> RECAP's approach doesn't use policy gradient at all. It's closer to learning actions wholesale together with a scorecard, then at test time conditioning on "give me a full-marks one" (the upside-down RL / decision-transformer family). So it can absorb failure data too, with a "low score" label, rather than discarding it.

---

## 7. Limitations

A combined list of what the paper states itself and what's worth flagging additionally on a read.

**Limitations the paper acknowledges**

- **Long-horizon compositional ability** — on long tasks like LIBERO-10, still short of SFT on human demonstrations.
- **Manual structuring of the YAM experiment** — the task is hand-decomposed into 4 stages and a separate reward classifier is trained for each. Less "fully autonomous" than **autonomy on top of a well-crafted state machine.**

**Additional points to flag**

- **Reward engineering is the real bottleneck** — though called a sparse binary reward, the success predicate and reward classifier of $\mathbf{1}[d(\phi(s),g)\le\varepsilon]$ must be built per task. This can become the real cost when scaling up the number of real-world tasks.
- **Hyperparameter sensitivity** — the schedule of the residual scale $\xi$ and the probing ratio $\alpha$ (plateau at 0.6) may be task-specific. This somewhat weakens the plug-and-play claim.
- **Number of real-world tasks** — 2 on Franka, 1 on YAM. The simulation (LIBERO, SimplerEnv) results are large, but the real-world generalization claim should be read cautiously.
- **Interpreting the abstract's numbers** — see the fact-check in §5.2.

---

## 8. Closing — What This Paper Suggests

PLD's real contribution is not a particular RL algorithm. It is having demonstrated, in the real world, **a loop that auto-generates deployment-aligned data without human teleoperation.** If the bottleneck of VLA scaling is data rather than the model, then the flywheel that makes that data is itself the asset.

And this paper reads especially well for someone with an LLM/diffusion background.

- **"frozen backbone + trainable residual"** is the RL counterpart of the PEFT/LoRA idea. It trains only a small, tractable module without shaking the giant model as a whole.
- **"RL forgets less than SFT (KL is the indicator of forgetting)"** connects directly to the LLM alignment literature.
- **the choice to bypass direct RL on the flow head with a Gaussian residual** is a familiar trade-off for anyone who knows the RL-optimization difficulty of diffusion policies.

Above all, the insight distilled in §3.2 — **instead of fixing the reward, move the robot's learning conditions toward the LLM's** — looks like a general principle applicable to every attempt to transplant LLM experience into physical AI.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **residual policy** | a small correction delta added to the frozen base action. $\bar a = a_b + a_\delta$, $a_\delta\in[-\xi,\xi]$ |
| **base policy probing** | rolling the base for random steps to enter deployment-distribution states; sets the takeover start point |
| **hybrid rollout** | a trajectory made of probe (base) + takeover (expert); how $\tau_{\text{demo}}$ is generated |
| **RLPD symmetric replay** | equal-count sampling from the offline (base-success) and online buffers to ground the critic on high-value states |
| **Cal-QL** | a Q initialization that is conservative but calibrated not to fall below the reference policy's value; stabilizes the offline→online transition |
| **pathwise vs score-function** | the two branches of policy gradient. PLD is pathwise (off-policy), GRPO is score-function (on-policy) |
| **data flywheel** | the self-improving cycle (improved generalist) → (residual expert) → (data) → (better generalist) |

**Original paper** — [arXiv:2511.00091](https://arxiv.org/abs/2511.00091) · **Project page** — wenlixiao.com/self-improve-VLA-PLD
