---
layout: paper
lang: en
ref: hirobot-hierarchical-vla
title: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
date: 2025-02-26
venue: "ICML 2025 · arXiv:2502.19417"
tags: [VLA, Hierarchical-Policy, Robot-Foundation-Model, Synthetic-Data, Instruction-Following, Paper-Review]
authors: "Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn"
affiliations: "Physical Intelligence · Stanford University · UC Berkeley"
summary: "The bottleneck is the head, not the hands — freeze the real action labels and reverse-synthesize the prompts that would have asked for them to train a hierarchical VLA's high level."
paper_url: "https://arxiv.org/abs/2502.19417"
---

> **Core claim** — The bottleneck of robot foundation models is now the head, not the hands. And that head is filled not by prompting a bigger frontier model, but by **training on domain data**. The data needed for that training doesn't exist in the world, but it can be made — by holding the action labels of real demonstrations fixed and **synthesizing, in reverse, the sentences that would have asked for those actions**.

---

## Introduction

Language-conditioned VLAs (Vision-Language-Action) already work well. The dexterity shown by π₀, OpenVLA, RT-2, and Octo would have been hard to imagine a few years ago.

But the commands these models receive mostly look like this.

> "Put the cup on the plate."

That is not how people actually talk to robots. This post follows a paper that tackles that gap head-on, while building up the necessary concepts along the way so that even a reader with a shallow robotics background can see the **reason** behind each design decision.

---

## 1. The Problem — Human Speech Is Not an Atomic Command

### 1.1 What real prompts look like

Consider the example the paper gives in its introduction.

> "Could you make a vegetarian sandwich? Hold the tomato. And if there's ham or roast beef, make a separate one for my friend with that."

Four things overlap inside this single sentence.

**⓵ Multi-step planning** — two sandwiches, each with a different spec
**⓶ Constraint** — exclude tomato
**⓷ Conditional branching** — ham or roast beef, whichever is available
**⓸ Implicit reasoning** — it doesn't tell you what "vegetarian" excludes

On top of that, the human interjects mid-execution: "that's not trash," "stop, don't add more," "put in a KitKat too."

One decisive point here. **"That's not trash" cannot be interpreted from language alone.** What that sentence refers to depends on what the gripper is holding right now. It's not a problem of language understanding but of **grounding**.

### 1.2 The spectrum of the prompt slot

Introducing the paper's notation up front makes what follows easier. Write the language prompt the policy receives as $\ell$. $\ell$ is not a fixed form but a **variable slot**, and the values that can go into it form a wide spectrum.

```
  the spectrum of l -- all go into the same variable slot

  [SIMPLE]     "make a sandwich"                   <- present in demo data
               "put away the dishes"

  [COMPLEX]    "vegetarian, no pickles"            <- absent
               "only trash, not dishes"
               "bus all the yellowish things"

  [INTERJECT]  "that's not trash"                  <- absent
               "leave the rest"                       (arrives mid-execution)
               "I also want a KitKat"
```

What existing VLAs handle is the top row. What the paper targets is the bottom two layers.

There's an easy misconception here. It's tempting to lump the bottom two layers together as "a feature for robots that converse with people," but reading it that way narrows the scope on your own.

> ### 💡 A complex prompt is not a conversation
>
> Re-splitting the three layers by **whether a human is involved in real time** makes the boundary shift.
>
> | Layer | Example | Human involved in real time? |
> |---|---|---|
> | SIMPLE | "make a sandwich" | ✗ |
> | **COMPLEX** | "vegetarian, no pickles" | **✗** |
> | INTERJECT | "that's not trash" | ✓ |
>
> **A complex prompt is a single order slip.** It can be given once at deployment and be done. The human says nothing afterward.
>
> $$\text{complex}\ \ne\ \text{interaction},\qquad \text{complex}\ =\ \text{composition · constraint · negation · indirect instruction}$$
>
> So even a robot that doesn't converse with people may need this ability. With 6 ingredients there are $2^6=64$ combinations, and you can't collect a demonstration for each. A **negative instruction** like "don't touch the seats where guests are eating" leaves no trace in the trajectory and so can't be learned from demonstrations. Every situation where **the job is fixed but the spec changes** falls here.
>
> There's a single criterion — **does the task spec change after deployment.** If not, this paper's machinery is unnecessary, and indeed the paper says so: if you only want to command a task the low-level policy learned directly, you can just set $\hat{\ell}_t = \ell_t$ and proceed as in prior work.

### 1.3 The two existing lineages the paper lays out

In related work the paper divides VLM-based robot control into two categories and spells out the deficiency of each.

| Lineage | Representatives | Good at | Deficiency the paper notes |
|---|---|---|---|
| **⓵ Directly finetune a VLM for control** (flat VLA) | RT-2, OpenVLA, π₀, RDT-1B, TinyVLA, FAST | excellent physical dexterity, generalization | **trained for relatively simple commands** |
| **⓶ Off-the-shelf VLM/LLM + predefined skills** | SayCan, Code as Policies, VoxPoser, MOKA, PIVOT, OK-Robot | parsing complex commands, using visual context | **limited physical dexterity**, limited real-time language interaction with humans |

The direction may feel counterintuitive. **The side where reasoning is alive is ⓶ — the off-the-shelf, un-finetuned VLM.** The reason matters.

Finetuning a VLM for control means **swapping the output channel from language to action**. That model no longer unfolds its thinking in language, and since its training data is robot demonstrations, the language distribution narrows to the level of "pick up one slice of bread." That is, finetuning closes **the channel for using the head** as the price of gaining hands. Conversely, the off-the-shelf VLM keeps its language output so reasoning stays alive, but it has no path out to action and must depend on a predefined skill API — and there its dexterity is capped.

Even within lineage ⓶ generations diverge. The paper makes a more fundamental point about the early generation (Huang et al. 2022, SayCan) — these systems, which combine a language model with learned or hand-crafted skills, are **limited in their ability to reflect complex context such as image observations into the reasoning process.** It means planning happens only in the text world and has no eyes.

| | Head (reasoning) | Hands (dexterity) | Eyes (observation at the planning stage) |
|---|---|---|---|
| ⓵ flat VLA | ✗ | ✓ | ✓ (for action) |
| ⓶-early (SayCan) | ✓ | ✗ | **✗** |
| ⓶-late (VoxPoser, MOKA) | ✓ | ✗ | ✓ (but no real-time interaction) |

### 1.4 The three walls to clear

This contrast is the paper's starting point. From here on, read every section as a process of dissecting or breaking through these three.

> **⛔ Wall 1 — the trade-off between reasoning and dexterity.** Bolt a skill API onto an off-the-shelf VLM and reasoning works but the hands are clumsy; grow a single VLA and the hands are good but it can't read complex prompts.
>
> **⛔ Wall 2 — grounding.** Feedback like "that's not trash" can only be interpreted by looking at the current observation. A text-only planner cannot do it in principle.
>
> **⛔ Wall 3 — the data gap.** Even if you want to train a high-level reasoner, there's no training data. Robot demonstrations carry only atomic labels; there are no paired examples with prompts like "I'm allergic to pickles."

**So the question the paper poses is this.**

> If we make both layers VLMs and connect them with **free-form natural language rather than a fixed API**, can we handle open-ended prompts and real-time intervention without losing dexterity? And where do we get the data to train that high level?

---

## 2. Background — The Bare Minimum

### 2.1 Action chunks and observations

The policy takes an observation $\mathbf{o}_t$ and emits an **action chunk**.

$$\mathbf{A}_t = [\mathbf{a}_t,\ \mathbf{a}_{t+1},\ \dots,\ \mathbf{a}_{t+H-1}],\qquad \mathbf{o}_t = [\mathbf{I}_t^1,\dots,\mathbf{I}_t^n,\ \ell_t,\ \mathbf{q}_t]$$

| Symbol | Meaning |
|---|---|
| $\mathbf{A}_t$ | the bundle predicting the next $H$ actions at once (action chunk) |
| $H$ | chunk length |
| $\mathbf{I}_t^1,\dots,\mathbf{I}_t^n$ | $n$ camera images (base + wrist) |
| $\mathbf{q}_t$ | configuration — joint angles and gripper position |
| $\ell_t$ | language prompt |

The distribution the policy expresses is $p(\mathbf{A}_t \mid \mathbf{o}_t)$. The reason it emits chunks rather than one step at a time is to keep consistency across actions in high-frequency control and to cut the number of inference calls.

### 2.2 From VLM to VLA

A VLM is a conditional distribution that takes an image-language prefix and emits a language suffix.

$$p(\ell' \mid \mathbf{I},\ \ell)$$

- $\ell$ — prefix (the prompt given together with the image)
- $\ell'$ — suffix (the model's output text)

A standard VLA finetunes this VLM so that the action $\mathbf{A}_t$ is represented as tokens inside the suffix $\ell'$, usually by discretizing and tokenizing the action.

Hi Robot's low level uses **π₀**. π₀ additionally handles multiple images and the continuous state $\mathbf{q}_t$, and modifies the VLM to emit a continuous action-chunk distribution via **flow matching**. Its configuration is a PaliGemma-3B backbone plus a 300M **action expert**, 3.3B total.

From the number 300M alone it sounds like an output head tacked onto the back of the backbone. But reading it that way makes the arithmetic later not add up.

> ### 💡 The action expert stands beside the backbone, not behind it
>
> The π₀ paper **analogizes this design to an MoE with two mixture elements** — the first for image/text input, the second for robotics input/output. And the action expert uses a **full bidirectional attention mask so that all action tokens attend to one another**. A concept absent from a head.
>
> ```
>   LAYER i   (repeated in every layer)
>   =========================================================
>   img/text tokens                      state/action tokens
>        |                                       |
>   [ VLM expert  ]                       [ action expert ]
>   [   (wide)    ]                       [   (narrow)    ]
>        |                                       |
>        +------------> JOINT ATTENTION <--------+
>        |          (action tokens read
>        |           the prefix K,V)
> ```
>
> **Same depth, just narrower width.** The π₀.₅ appendix gives the exact dimensions — the backbone is width 2048 / depth 18 / mlp 16,384, and the action expert is **width 1024 / depth 18 / mlp 4,096**. Parameters scale roughly with the square of width, so you get 300M, one-tenth of 3B.
>
> This structure is what makes real-time control possible. At inference the prefix's attention key/value is cached and only the action tokens are recomputed at each integration step, so the cost splits like this.
>
> $$\underbrace{3\text{B} \times 1}_{\text{prefix encoding}}\ +\ \underbrace{0.3\text{B} \times 10}_{\text{10 flow-integration steps}}$$
>
> Had the expert been the same width as the backbone, 50 Hz would have been impossible. The small expert is not a performance tweak but a **structural requirement for real-time control**.

Remember one fact that runs through this entire paper. **Hi Robot's high level and low level are nearly identical architectures starting from the same backbone (PaliGemma-3B); the only difference is that the low level has a flow-matching action expert attached.** This fact is used decisively again in §6.

### 2.3 The System 1 / System 2 frame

The paper uses Kahneman's dual-process theory as a frame.

```
  SYSTEM 2  (deliberative)          SYSTEM 1  (automatic)
  ------------------------          ---------------------
  high-level VLM policy             low-level VLA policy
  parses complex prompt             executes atomic command
  reads user feedback               produces action chunks
  low frequency  (~1 Hz)            high frequency
  output: natural language          output: continuous actions
```

- The upper layer judges "what to do now" using semantic and visual knowledge from web-scale pretraining
- The lower layer only needs to understand simple commands like "pick up one slice of bread," but in exchange is physically precise

The analogy is clean. Just how well it matches the actual implementation is revisited in §7.

---

## 3. Method — Hi Robot

### 3.1 Hierarchical reasoning — the structure that breaks Wall 1

The core is to **decompose $p(\mathbf{A}_t \mid \mathbf{o}_t)$ into two reasoning processes mediated by natural language.**

$$p^{\text{hi}}(\hat{\ell}_t \mid \mathbf{I}_t^1,\dots,\mathbf{I}_t^n,\ \ell_t),\qquad p^{\text{lo}}(\mathbf{A}_t \mid \mathbf{I}_t^1,\dots,\mathbf{I}_t^n,\ \hat{\ell}_t,\ \mathbf{q}_t)$$

| Symbol | Meaning |
|---|---|
| $\ell_t$ | the user's **open-ended prompt** (complex, includes constraints, changeable mid-way) |
| $\hat{\ell}_t$ | the **intermediate language command** the high level generates — an atomic command the low level can understand |
| $u_t$ | the **robot's utterance** the high level optionally emits alongside (confirmation/clarification) |

The low-level policy has exactly the same form as a standard VLA, except that **the input language has changed from the user prompt $\ell_t$ to the high-level output $\hat{\ell}_t$.** In the paper's words, the high-level process essentially "talks to" the low-level process.

```
  user prompt l_t   +   images I_t (base + wrist)
       |
       v
  +-------------------------------+
  |  HIGH-LEVEL  p^hi             |   PaliGemma-3B finetuned
  |  rerun every 1 s, OR on       |   low frequency
  |  any new user utterance       |
  +-------------------------------+
       |
       |   output = [ u_t ] + l_hat_t
       |
       +----> u_t ----> TTS ----> speaker
       |
       v   l_hat_t only  ("pick up the roast beef")
  +-------------------------------+
  |  LOW-LEVEL  p^lo  (pi_0)      |   PaliGemma-3B
  |  + flow-matching expert       |   high frequency
  +-------------------------------+
       |
       v
    A_t = [a_t, ..., a_{t+H-1}]
```

- The utterance $u_t$ is played to the user via TTS and **removed before being passed to the low level** — the low level receives only the command
- The two layers run at different rates. The high level is slow but semantically heavy; the low level is fast but simple.

**See how this design breaks Wall 1.** Lineage ⓶ (the SayCan family) keeps the planner as an off-the-shelf model and fixes skills as a predefined set. Hi Robot makes **both layers learned VLMs**, and the interface between them is **free-form natural language rather than function calls**. Dexterity comes straight from π₀, and reasoning comes from the finetuned high-level VLM.

But a structure where a big model plans and a small executor carries out isn't unfamiliar. It looks identical on the surface to tool-calling in LLM agents. What's different?

> ### 💡 The price of an interface that is a sentence, not JSON
>
> | Axis | LLM agent (tool-calling) | Hi Robot |
> |---|---|---|
> | Interface | fixed-schema function call (JSON) | **free-form natural-language string** |
> | Executor | deterministic code/API | **a learned VLA** — generalizes over command phrasing |
> | Planner's input | text (+ tool return values) | **image observations** — the current scene on every call |
> | Planner's training | none (prompting) | **finetuned on domain data** |
>
> The real benefit of free-form natural language is that **you don't have to enumerate the skill set**. Since the low level is already a language-conditioned VLA, it handles, to some degree, phrasing combinations it never saw in training.
>
> In exchange there's a price — **there is no contract.** The high level doesn't know exactly what the low level can do. A function signature tells the planner whether a call is possible, but a string carries no such guarantee. The paper admits this as a limitation itself, and we revisit it in §7.

### 3.2 Handling user intervention — breaking Wall 2

The user can intervene at any time during execution to give extra information and feedback or to change the task entirely. When an intervention comes in, the high-level reasoning is **immediately rerun** to recompute $\hat{\ell}_t$.

The re-invocation policy itself is surprisingly simple. The high-level reasoning is rerun **when 1 second has elapsed, or when a new interaction with the user occurs.** The paper notes that a smarter scheme detecting command completion to infer the next command is also possible, while stating that this simple scheme worked well.

**The answer to Wall 2 is here.** The high level's response is contextual because it looks not only at the prompt $\ell_t$ but also at **the current image observation**, so it can correctly ground feedback like "that's not trash" — something impossible for a language-only system.

In other words, what breaks Wall 2 is not a separate technique but **the very choice to make the high level a VLM rather than an LLM.** Recalling that the early works of lineage ⓶ got stuck exactly here, this simple choice is the paper's real fork in the road.

### 3.3 Data generation — breaking Wall 3

Now the remaining problem. How do you train $p^{\text{hi}}$?

The paper diagnoses this itself in the introduction — it expects that robot demonstrations annotated only with atomic commands won't be enough to train a high-level model to follow complex, open-ended prompts, and therefore **representative examples of following complex prompts are needed.**

But why is that data missing? Looking at how robot demonstration data is made gives the answer.

**The standard production method is "shoot first, label after."**

| Output | Who | When |
|---|---|---|
| $\ell$ (simple) — "make a sandwich" | the collection lead | **before the demo**, when deciding the task |
| $\mathbf{A}_t$ (action trajectory) | the teleoperator | during the demo |
| $\hat{\ell}$ (atomic skill) — "pick up one piece of lettuce" | annotation | **after the demo**, watching the video |

Hi Robot follows this method too. Episodes carrying a coarse language annotation of the overall goal are split into short skill units, usually **1–3 seconds** long. On top of that, basic movement primitives like "move the right arm left" are **heuristically extracted from the raw robot actions.** The result is $\mathcal{D}_{labeled}$.

$$\mathcal{D}_{labeled} = \{(\hat{\ell}_t,\ \mathbf{I}_t^1,\dots,\mathbf{I}_t^n)\}$$

**Here the identity of Wall 3 becomes clear.** $\hat{\ell}$ is fully obtained this way. What's missing are the complex values of $\ell$. The annotator is someone who **describes what happened**, and "who might have said what" is not an object of description — because on the demo site no one said "I'm allergic to pickles."

**The fix is to reverse the direction of generation.**

```
  FORWARD  (deployment)              REVERSE  (data generation)
  ----------------------             --------------------------
  l_t       ---> [p^hi] ---> l_hat   l_hat     ---> [p^gen] ---> l_t , u_t
  user prompt         atomic cmd     atomic cmd           user prompt
  GIVEN               PREDICT        GIVEN (real label)    IMAGINE
```

Give a large VLM $p^{\text{gen}}$ the visual context and the skill label, and have it **imagine a plausible interaction that, in a real user exchange, would have led to that $\hat{\ell}_t$.**

$$p^{\text{gen}}\big(\ell_t,\ u_t \mid \mathbf{I}_t^1,\dots,\mathbf{I}_t^n,\ \hat{\ell}_0,\dots,\hat{\ell}_{t-1},\ \hat{\ell}_t,\ \mathcal{P}\big)$$

| Symbol | Meaning |
|---|---|
| $\hat{\ell}_0,\dots,\hat{\ell}_{t-1}$ | prior skill history of the same episode. For keeping multi-step tasks consistent |
| $\hat{\ell}_t$ | the true skill label at the current step. **the anchor of generation** |
| $\mathcal{P}$ | a designed prompt holding the task description and scenario taxonomy |
| $\ell_t,\ u_t$ | the generated output — user prompt and robot utterance |

Appendix A.1 states that scenarios are typed for generation diversity.

| Scenario type | Content | Which layer of §1.2 |
|---|---|---|
| **negative task** | the user instructs what not to do | COMPLEX |
| **specific constraint** | a specific constraint like a dietary preference | COMPLEX |
| **situated correction** | adjusting a prior command according to progress | **INTERJECT** |

The third corresponds to intervention. **Conditioning on the history $\hat{\ell}_0,\dots,\hat{\ell}_{t-1}$ is what makes intervention generation possible.** Demonstrations naturally contain **behavior-switch points** where a person changes their mind or drops and re-grasps, and the VLM assigns, after the fact, an explanation of the form "it's because I heard this" to those points.

**Training is standard.** The high level is trained with the cross-entropy of next-token prediction over $\mathcal{D}_{syn}\cup\mathcal{D}_{labeled}$, and the low level with the flow-matching objective over $\mathcal{D}_{labeled}\cup\mathcal{D}_{demo}$.

But "make training data with a VLM" is usually something to be wary of, because the generative model's errors become label noise directly. Why doesn't that risk bite here?

> ### 💡 What's synthesized is the prompt, not the label
>
> In this structure, **what's synthesized is the input (the user prompt), and the output (the skill label) is real, coming from actual teleoperation data.** That is, however much imagination $p^{\text{gen}}$ exercises, that imagination is tied to **an action already confirmed to be physically executable.** In the paper's words, synthetic yet situated examples.
>
> On the LLM side this structure isn't unfamiliar.
>
> | | Already present (real) | Generated (synthetic) |
> |---|---|---|
> | Instruction backtranslation | the response (a web document) | the instruction |
> | Hindsight relabeling | the trajectory | the goal |
> | **Hi Robot** | **observation + skill label** | **user prompt + robot utterance** |
>
> In all three, **the side expensive to verify is kept real and the cheap side is synthesized.** In robotics the expensive thing is the label "this action is correct in this observation," and the cheap thing is a sentence that would have requested that action.
>
> And $\mathcal{D}_{syn}$ **adds not a single grain of robot data.** Zero images, zero actions, zero skill labels. The only thing that grows is $\ell$.
>
> $$\text{1 demonstration}\ \longrightarrow\ N\text{ prompts}\qquad(\text{marginal cost} \approx \text{VLM inference})$$
>
> It's **relabeling**, not collection. Teleoperation time doesn't grow by even a second.

### 3.4 Implementation

Both the low level and the high level start from the same base VLM, **PaliGemma-3B**. The low level is the π₀ VLA; the high level is a separate model finetuned on the image-language tuples of §3.3.

```
  Weight accounting of the Hi Robot system
  ---------------------------------------------
  p^hi  :  PaliGemma-3B (finetuned for text)      ~3B
  p^lo  :  PaliGemma-3B + action expert 300M      ~3.3B
                                                  -----
  two independent checkpoints                     ~6.3B
```

The inference hardware is **one or two consumer RTX 4090 GPUs**, speech input is Whisper large-v2, and output is a TTS API. The robots are a single-arm UR5e (7-DoF), a bimanual ARX (14-DoF), and a mobile ARX (based on Mobile ALOHA, 16-DoF actions).

---

## 4. Why It Works

The paper doesn't devote a separate section to it, but three mechanisms can be read from its experimental discussion and design.

**⓵ Periodic re-injection of constraints.** The classic failure of a flat policy is that when the prompt changes mid-way it **reverts to default behavior** — picking up every visible object, or putting nearly every ingredient into the sandwich. In the hierarchy, the original prompt and the current image are re-injected at every high-level step, so constraints aren't diluted.

A familiar phenomenon in LLMs: as a long generation proceeds, system-prompt constraints grow progressively fainter — the hierarchy can be seen as a device that forces those constraints to be **repeatedly re-evaluated within a short context.**

**⓶ Compositional generalization from the language interface.** Since the low level is already language-conditioned, when the high level calls the skills present in the training data **in new orders and combinations**, that becomes a new task as-is.

**⓷ An interpretable intermediate representation.** Because $\hat{\ell}_t$ is a human-readable sentence, you can diagnose whether a failure is due to reasoning or execution, separately. Indeed the metric design of §5 depends on this property.

---

## 5. Experiments

### 5.1 Three domains

| Domain | Robot | Physical difficulty | Linguistic difficulty |
|---|---|---|---|
| Table bussing | single-arm UR5e | grasping plates by the edge, separating objects, pouring trash onto a plate | "only trash," "only the yellowish things" — judging a reusable plastic cup as dishware and a paper cup as trash |
| Sandwich making | bimanual ARX | carefully picking deformable, delicate ingredients and placing them precisely | "vegetarian, pickle allergy," "stop, don't add more" |
| Grocery shopping | mobile bimanual ARX | mobile manipulation, varied shapes | "something sweet," "something to drink," "add a KitKat too" |

### 5.2 Baselines — they map exactly onto the three walls

| Baseline | What it removes | Maps to |
|---|---|---|
| **Expert human high level** (oracle) | replaces the high level with a human → measures the low level's ceiling | (upper-bound baseline) |
| **GPT-4o high-level** | keeps the hierarchy, high level is an un-finetuned off-the-shelf large VLM | **Wall 2 — grounding** |
| **Flat VLA** (π₀) | no hierarchy and no synthetic data | Wall 1 + Wall 3 |
| **Flat VLA + synthetic data** | given synthetic data but no hierarchy | **Wall 1 — isolates the effect of the hierarchy** |
| **Hi Robot w/o synthetic data** | has the hierarchy but no synthetic data | **Wall 3 — isolates the effect of synthetic data** |

The GPT-4o baseline was tuned for fairness. To match the robot's affordances, the **most common skill labels from the human-annotated dataset were ranked** and included in the prompt, and GPT-4o was made to choose from among them.

The paper positions this baseline as **similar to an advanced version of SayCan.** You have to know what SayCan is to see exactly what this baseline tests.

> ### 💡 Remove the affordance function from SayCan and you get the GPT-4o baseline
>
> SayCan in one line: "the LLM knows what is useful, the value function knows what is possible now. Multiply the two."
>
> $$k^\star = \arg\max_{k\in\mathcal{K}}\ \underbrace{p_{\text{LLM}}(k \mid \ell,\ k_{1:t-1})}_{\text{Say — usefulness}}\ \cdot\ \underbrace{V^{k}(s_t)}_{\text{Can — feasibility}}$$
>
> | Symbol | Meaning |
> |---|---|
> | $\mathcal{K}$ | a pre-enumerated skill library |
> | $k_{1:t-1}$ | history of skills executed so far (accumulated in the prompt as text) |
> | $p_{\text{LLM}}$ | the log-likelihood the LLM assigns to each skill sentence. **It does not look at images** |
> | $V^k(s_t)$ | affordance — the success probability of that skill in the current state. Learned by RL |
>
> Mapping onto Hi Robot, three things changed.
>
> - **The affordance function $V^k$ is gone.** The high level looks at images directly and internalizes grounding through finetuning. But without $V^k$ there's **no way to know whether the low level succeeded**
> - **The discrete argmax became free-form string generation**, making skill enumeration unnecessary
> - **The planner is learned, not prompted**
>
> The GPT-4o baseline is exactly "the version of SayCan with the LLM upgraded to a VLM and the skill list put in the prompt." **The defeat of that baseline is precisely the paper's argument that "the planner must be learned."**

### 5.3 Evaluation metrics

Measured by human evaluators blind to the method, with **20 trials per method/task.**

- **Instruction Accuracy (IA)** — the fraction of high-level predicted commands consistent with **both** the user command and the current observation
- **Task Progress (TP)** — the fraction of objects successfully placed in the correct position or configuration

### 5.4 Results

**⓵ Open-ended instruction following.** Substantially higher IA and TP than GPT-4o and the flat baseline on every task. GPT-4o's failure pattern is specific — **once physical interaction begins it frequently loses context**, issuing nonsensical commands ("pick up bermuda triangle") or labeling everything as "plate" or "spoon," collapsing long-horizon planning.

**⓶ Situated reasoning and feedback adaptation.** When the request changes mid-way, Hi Robot updates the command. GPT-4o **fails to maintain a consistent internal state**, telling the gripper to pick up a new object when it's already holding one, or switching tasks prematurely; the flat baseline doesn't respond to real-time feedback at all.

**⓷ Effective across all three platforms.**

**⓸ The oracle comparison exposes the bottleneck.** When a human gives the high-level instructions, the low-level policy performs **almost perfectly**, showing that **failures stem from reasoning rather than actuation.** This result underpins the paper's entire thesis — π₀'s hands are already enough; what was lacking was the head.

### 5.5 Ablations — the contributions separate cleanly

**(A) Synthetic data.** Without it, the system ignores clarifications like "this isn't trash" or adds a forbidden ingredient (pickles). What's interesting is the **character** of the failure. Per the Figure 6 description, a high-level policy without synthetic data is **well aligned with the image observation but ignores user constraints.**

$$\underbrace{\text{visual grounding}}_{\text{from }\mathcal{D}_{labeled}}\ +\ \underbrace{\text{constraint compliance}}_{\text{from }\mathcal{D}_{syn}}$$

In other words, what $\mathcal{D}_{syn}$ buys is not "the ability to chat" but **the ability to obey constraints.** This is where the "complex ≠ interaction" point from §1.2 is confirmed experimentally.

**(B) The hierarchy.** A flat policy trained on the same synthetic data reverts to clearing everything or fails to handle partial instructions ("only the yellowish things").

### 5.6 Before citing numbers

> ### ⚠️ This paper has no numeric tables at all
>
> Quantitative results are presented **only as bar charts (Figures 5, 7, 8)**, with no numeric table anywhere in the text. The only concrete figure in the text is a single sentence in the **caption** of Figure 5 — "Hi Robot achieves on average over 40% higher instruction accuracy than GPT-4o" — and even that can't be pinned down from the caption alone as absolute percentage points versus relative improvement.
>
> To turn it into numbers you have to read the original figures directly, and it's safest to flag such read-off values as approximate.

> ### ⚠️ IA is scored differently for different methods
>
> IA looks at the high level's **language output.** But the flat baseline has no language output. The paper handles it thus — for the flat baseline, which has no interpretable language prediction, scoring is based on **the evaluator's interpretation of the intent of the policy's behavior.**
>
> That is, Hi Robot is scored by "the sentence it emitted" and the flat baseline by "the intent a human back-inferred from watching the behavior." **A comparison that puts two different kinds of measurement on the same axis.** On top of that there are 20 trials per method/task with no confidence intervals reported. TP is the object-placement fraction and thus far more objective, so it's safest not to read the two metrics with equal weight.

> ### ⚠️ The high-level policy was trained separately for each task
>
> It appears at the end of Appendix A.2 — in this work, for clarity and benchmarking convenience, **a separate $\mathcal{D}_{syn}$ was generated per task and a separate high-level policy trained**, though the architecture readily applies to a unified multi-task formulation.
>
> Reading only the main text, it's easy to read as if one high-level policy spans the three domains, but in reality there are **three, one per domain.** Multi-task unification is an unproven prospect. **What "open-ended" points to is prompt openness within one domain**, not cross-domain generalization.

---

## 6. Positioning

It's the intersection of three lineages.

**⓵ Direct VLA training** — RT-2, OpenVLA, π₀, RDT-1B, CogACT, FAST, ECoT. Hi Robot **adopts a product of this lineage (π₀) directly as its low-level executor.** A consumption relationship, not competition.

**⓶ Off-the-shelf VLM + predefined skills** — SayCan, Code as Policies, VoxPoser, MOKA, PIVOT, OK-Robot, BUMBLE. The GPT-4o baseline enters the experiments as the representative of this lineage (§5.2).

**⓷ Interaction learning from language feedback** — the closest cousins. The paper spells out the differences directly.

| Cousin | Their approach | The difference the paper states |
|---|---|---|
| **OLAF** (Liu et al. 2023) | uses an LLM to correct robot trajectories | Hi Robot does observation-based situated correction, real-time response, complex prompts |
| **YAY Robot** (Shi et al. 2024) | capable of real-time situated correction | **limited to a single prompt**, and to corrections present in human-written data |
| **RACER** (Dai et al. 2024) | capable of situated correction | uses a **physics simulator** to construct recovery behaviors. Hi Robot uses only real demonstrations with no deliberate perturbation |

**The closest cousin is YAY Robot** (same first author), and the ablation "Hi Robot without synthetic data" is explicitly identified as **an advanced VLM-based version of YAY Robot.** So it's not far off to read this paper's net contribution as the delta of §5.5 ablation (A) — **the improvement due to synthetic data.**

The paper acknowledges this exactly — individual components such as the low-level VLA policy were already addressed in prior work, and **what's new is the combination of these components and the synthetic-data-generation method.**

> ### 📌 The distinction to remember from this paper
>
> **That the high-level planner is "trained."** Finetuning a 3B backbone on domain data beats attaching a bigger frontier VLM via prompt engineering. And the reason is pinned down as **grounding** — once physical interaction begins, off-the-shelf models fail to track scene state.
>
> **What made that training possible is reverse prompt generation.** Labels real, prompts synthetic. A structure that keeps the expensive side real.

### 6.1 Two months later, the same team folds this design back up

The fact I asked you to remember in §2.2 — that **the two layers are nearly identical architectures and the only difference is the flow-matching head** — the paper itself raises in its Discussion. And that paragraph becomes the spec sheet for a follow-up paper.

> ### 🔗 π₀.₅ (2025-04) is the paper that solves the homework Hi Robot left behind
>
> Hi Robot's Discussion writes — **the role separation at the model level is not essential to this design**, and a natural next step for future work is **to combine the two systems into a single model and place the System 1 / System 2 distinction purely at inference time.**
>
> Two months later, the same team's π₀.₅ §IV-A is exactly that spec.
>
> $$\pi_\theta(\mathbf{a}_{t:t+H},\hat{\ell}\mid\mathbf{o}_t,\ell) = \underbrace{\pi_\theta(\mathbf{a}_{t:t+H}\mid\mathbf{o}_t,\hat{\ell})}_{\text{low level}}\ \underbrace{\pi_\theta(\hat{\ell}\mid\mathbf{o}_t,\ell)}_{\text{high level}}$$
>
> **Both distributions are expressed by the same model.** That the low level depends only on $\hat{\ell}$ and not $\ell$ is also structurally the same choice as Hi Robot stripping off $u_t$ and passing only $\hat{\ell}_t$.
>
> What's interesting is that π₀.₅ **cites Hi Robot as an instance of "the two-model approach we aim to move beyond"** — taking the form of a break rather than a succession. Yet it was Hi Robot itself that requested that break.
>
> Succession and abandonment, laid out:
>
> | Element of Hi Robot | In π₀.₅ |
> |---|---|
> | hierarchical reasoning, the $\hat{\ell}$ concept, the free-form natural-language interface | ✅ inherited |
> | the GPT-4 high-level baseline recipe (common-label list as prompt) | ✅ **almost verbatim** |
> | the human-HL oracle condition | ✅ inherited (but in π₀.₅ the model surpasses the oracle) |
> | two separate checkpoints | ❌ abandoned |
> | synthetic prompts $\mathcal{D}_{syn}$ | ❌ not adopted (replaced by manual annotation + verbal instruction) |
> | real-time intervention + robot utterance $u_t$ | ❌ not adopted |
>
> And the arrow comes back around. In its limitations section π₀.₅ admits it **handles relatively simple prompts**, and writes that more complex preferences and instructions could be incorporated by having human labelers or **synthetically** produce more elaborate annotations. It effectively points at its own gap and offers Hi Robot's method as the solution.
>
> $$\underbrace{\pi_{0.5}}_{\text{to work anywhere}}\ +\ \underbrace{\text{Hi Robot}}_{\text{to say anything}}$$
>
> It's more accurate to see the two approaches not as competition but as **an unfinished convergence.**

---

## 7. Limitations

**What the paper acknowledges**

- High-level training **depends on prompt engineering** — to produce synthetic examples that elicit the desired behavior
- **The two layers don't know each other's capabilities except through training examples.** The "no contract" price foreshadowed in §3.1 is collected here. Coupling the high level so it perceives the low level's success is future work
- High-level invocation is on a **fixed schedule.** Asynchronous, adaptive multi-layer reasoning is offered as a future direction

**Additional points to flag**

- **The actual reasoning is shallow relative to the name "System 2."** Contrasting the §2.3 analogy with the implementation, $p^{\text{hi}}$ is a **single forward pass** from (image, prompt) → (utterance, atomic command). No explicit multi-step reasoning, no plan backtracking, no search. It's closer to a reactive mapping that looks at the scene each second and emits the next single skill.
- **The high level's state representation is images only.** The definition $p^{\text{hi}}(\hat{\ell}_t\mid \mathbf{I}_t,\ell_t)$ does not condition on prior skill history. History conditioning exists **only in the data generator $p^{\text{gen}}$** (§3.3). The paper criticizes GPT-4o for "failing to maintain a consistent internal state," yet Hi Robot's high level also formally has no memory, with the current scene standing in for state. The same vulnerability can appear on tasks that require unobservable progress state (what's already between the slices of bread).
- **"open-ended" is a property of the input side, not the output side.** The set of synthesizable prompts is ultimately confined to the following set.

$$\{\text{generatable } \ell\}\ \subseteq\ \{\ell \mid \text{answerable with skills in }\mathcal{D}_{labeled}\}$$

  What expands is **the diversity of ways to call**, not the list of what can be called. A new physical skill still requires a new demonstration.
- **Absence of a failure-recovery loop.** The high level emits commands open-loop, and the low level's failure is only indirectly detected when it leaves a trace in the next image. A silent failure (grasp-then-drop) can be mistaken for completion. The slot where SayCan's $V^k$ sat, seen in §5.2, is left empty.
- **Evaluation scale and metrics.** 3 domains, 20 trials per method/task, no confidence intervals, and IA is scored differently across methods (§5.6).

---

## 8. Closing — What This Paper Suggests

Reduced to one sentence:

> **The bottleneck of robot foundation models is now the head, not the hands, and that head is filled by training on domain data, not by prompting an off-the-shelf frontier model.**

The oracle experiment (when a human stands in for the high level, the low level is nearly perfect) is direct evidence of this proposition.

There are three points that read especially well for someone with an LLM/agentic background.

**⓵ Natural language as interface design.** Unlike tool-calling agents that use a JSON schema as the interface, here a free-form sentence is the interface. It's a choice that holds only when the executor is a learned language-conditioned model, and it's exempt from skill enumeration at the cost of having no contract. The **"loose interface vs. specified interface"** trade-off is reproduced verbatim.

**⓶ Expensive labels real, cheap conditions synthetic.** The robotics version of instruction backtranslation. In physical AI a human demonstration is the most expensive asset, and **retroactively attaching new conditioning sentences to that asset to recycle it** looks like a general recipe portable to other tasks. If you can pull many prompts from one demonstration, data efficiency multiplies.

**⓷ The hierarchy as a constraint-maintenance problem.** The way a flat policy forgets constraints on a long task and reverts to default behavior is the same face as the LLM problem of instructions being diluted in a long context. From this view, one can read **the reason the hierarchy wins as context management rather than architectural expressiveness.**

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **action chunk $\mathbf{A}_t$** | the bundle predicting the next $H$ actions at once |
| **$\hat{\ell}_t$ (intermediate language command)** | the atomic command the high level generates and passes to the low level. The interface between the two layers |
| **$u_t$ (robot utterance)** | the optional confirmation/clarification the high level emits. Played via TTS and not passed to the low level |
| **$\mathcal{D}_{labeled}$** | (skill, image) pairs from splitting demonstrations into 1–3 s skills and labeling them |
| **$\mathcal{D}_{syn}$** | (user prompt, robot utterance) that $p^{\text{gen}}$ reverse-generates from skill labels |
| **situated correction** | a correction that can only be interpreted from progress state and observation |
| **action expert** | π₀'s robotics-token-only weights. Shares attention with the backbone but is narrower in width |
| **Instruction Accuracy (IA)** | the fraction of high-level commands consistent with both user intent and the current observation |
| **Task Progress (TP)** | the fraction of objects placed in the correct position/configuration |

**Original paper** — [arXiv:2502.19417](https://arxiv.org/abs/2502.19417) (ICML 2025) · **Project page** — [pi.website/research/hirobot](https://www.pi.website/research/hirobot)
