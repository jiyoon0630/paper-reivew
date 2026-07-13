---
layout: paper
lang: ko
ref: pld-self-improving-vla
title: "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL (PLD)"
date: 2025-10-30
venue: "ICLR 2026 · arXiv:2511.00091"
tags: [VLA, Reinforcement-Learning, Robot-Foundation-Model, Residual-RL, Paper-Review]
authors: "Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi \"Jim\" Fan, Guanya Shi, Yuke Zhu"
affiliations: "NVIDIA (GEAR) · Carnegie Mellon University (LeCAR) · UC Berkeley · UT Austin"
summary: "사람 시연 없이, 얼린 VLA에 잔차 RL을 씌워 배포 분포에 정렬된 데이터를 자동 생성하는 self-improving 루프."
paper_url: "https://arxiv.org/abs/2511.00091"
---

> **핵심 주장** — VLA를 개선하는 데 필요한 것은 더 많은 사람 시연이 아니라, **정책이 실제로 방문하는 상태에서 수집된 데이터**다. 그리고 그 데이터는 사람 없이, 얼린 VLA에 잔차 RL을 씌워 자동으로 만들 수 있다.

---

## 들어가며

VLA(Vision-Language-Action) 모델의 표준 레시피는 LLM에서 그대로 건너왔다. 웹·로봇 혼합 데이터로 대규모 사전학습을 하고, 목표 로봇과 태스크에 대해 소량의 고품질 teleoperation 시연으로 SFT를 한다. π₀, OpenVLA, Octo, GR00T가 모두 이 틀 안에 있다.

그런데 이 레시피에는 언어에서 로봇으로 넘어오며 생긴 균열이 있다. PLD는 그 균열을 정확히 지목하고, RL로 메운다. 이 글은 논문의 논지를 따라가되, 강화학습 배경이 얕은 독자도 각 설계 결정의 **이유**를 납득할 수 있도록 필요한 개념을 그때그때 쌓아가며 진행한다.

---

## 1. 문제 — 사람 시연은 왜 부족한가

### 1.1 두 가지 결함

SFT용 teleoperation 데이터에는 성격이 다른 두 문제가 겹쳐 있다.

**⓵ 확장성** — 로봇 시연은 사람이 직접 조종해야 얻어진다. LLM이 인터넷 텍스트를 긁어모으듯 대량 확보하는 길이 없다.

**⓶ 분포 분리(decoupling)** — 이쪽이 더 본질적이다. 사람이 만든 시연 데이터는 **배포될 정책과 무관하게** 수집된다. 사람은 자기가 상상한 실패 모드만 시연에 담지만, 정작 배포된 VLA가 빠지는 실패 상태는 그 시연 어디에도 없다.

두 번째 문제를 상태 분포로 그리면 이렇다.

```
상태 공간 S 위에서 두 분포를 겹쳐 보면:

  p_human(s)   ....#####.....................     <- 사람 시연이 덮는 상태
               -------------------------------> S

  d^pi_b(s)    ....#####..........#####......     <- VLA가 배포 시 실제 방문하는 상태
               -------------------------------> S
                                  ^^^^^
                                  coverage gap
                                  = 사람 데이터에 없는데
                                    정책은 실제로 빠지는 실패 영역
```

여기서 $d^{\pi_b}(s)$는 base 정책 $\pi_b$의 **상태 방문 분포(state visitation distribution)** 로, $\pi_b$를 배포했을 때 상태 $s$에 머무는 빈도를 뜻한다. SFT는 $p_{\text{human}}$ 위에서 손실을 최소화하지만, 평가와 배포는 $d^{\pi_b}$ 위에서 일어난다. 이 불일치가 모방학습의 고전적 병폐인 **공변량 이동(covariate shift)** 이며, 학습한 태스크는 잘하는데 새 상황으로의 전이가 보장되지 않는 이유다.

### 1.2 그래서 논문이 던지는 질문

> 사람 노력을 최소화한 채, **RL이 스스로 큐레이션한 데이터만으로** VLA가 자기 자신을 개선할 수 있는가? 그 데이터가 사람 전문가 시연 SFT를 in-distribution과 out-of-distribution 양쪽에서 따라잡거나 넘어설 수 있는가?

자연스러운 첫 대답은 "RL로 태스크별 전문가를 학습시켜 데이터를 뽑자"이다. 그런데 VLA에 RL을 순진하게 붙이면 두 벽에 부딪힌다. 이 두 벽이 무엇이고 PLD가 어떻게 뚫는지가 이 논문 전체의 서사다.

**⛔ 벽 1 — 희소 보상** 언어 조건부 조작 태스크의 보상은 성공/실패 이진값뿐이다. RL이 불안정하고 표본 비효율적이 된다.

**⛔ 벽 2 — 거대 정책 직접 RL의 비용** flow-matching 같은 표현력 높은 action head를 Q값 최대화로 직접 학습하는 것은 매우 어렵고 메모리를 크게 먹는다. 논문이 드는 수치로, OpenVLA-OFT는 LIBERO 학습 시 batch size 8에서 GPU당 최대 약 62.5GB를 쓴다. 게다가 이렇게 독립적으로 얻은 전문가는 generalist와 분포가 어긋나고, 수렴 후엔 행동이 단조로워 SFT용 데이터로서 커버리지가 부족하다.

---

## 2. 배경 — 두 벽을 정확히 이해하기 위한 최소한의 RL

두 벽의 정체를 알아야 PLD의 설계가 왜 그 모양인지 보인다. 이 절은 그 최소한을 쌓는다. RL에 익숙하면 §3으로 건너뛰어도 좋다.

### 2.1 문제 설정: Goal-Conditioned MDP

연속 제어를 MDP로 본다.

$$\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ \rho,\ \rho_0,\ r,\ \gamma)$$

| 기호 | 이름 | 로봇에서의 의미 |
|---|---|---|
| $\mathcal{S}$ | 상태공간 | 로봇 proprioception + RGB 관측이 만드는 공간 |
| $\mathcal{A}$ | 행동공간 | 7-DoF (6-DoF delta pose + 1-DoF 그리퍼) |
| $\rho(s'\mid s,a)$ | 전이 동역학 | 상태 $s$에서 행동 $a$ 후 $s'$가 될 확률 = 물리 법칙 |
| $\rho_0$ | 초기 상태 분포 | episode 시작 상태 $s_0$가 뽑히는 분포 |
| $r$ | 보상 함수 | 스칼라 보상 |
| $\gamma\in(0,1]$ | 감가율 | 미래 보상의 할인 계수 |

태스크는 언어 목표 $g\in\mathcal{G}$로 주어진다($g\sim p(g)$). 보상과 정책이 목표에 조건부가 된다.

$$r:\ \mathcal{S}\times\mathcal{A}\times\mathcal{G}\to\mathbb{R},\qquad \pi:\ \mathcal{S}\times\mathcal{G}\to\Delta(\mathcal{A})$$

여기서 $\Delta(\mathcal{A})$는 **행동공간 위의 확률분포 집합**이다. 즉 정책은 행동 하나를 뱉는 게 아니라 행동에 대한 확률분포 $\pi(a\mid s,g)$를 낸다. 목적함수는 감가 누적 보상의 기대값이다.

$$J(\pi)\ =\ \mathbb{E}_{g\sim p(g)}\ \mathbb{E}_{s_0\sim\rho_0,\ a_t\sim\pi(\cdot\mid s_t,g),\ s_{t+1}\sim\rho(\cdot\mid s_t,a_t)}\left[\sum_{t=0}^{\infty}\gamma^{t}\, r(s_t,a_t,g)\right]$$

- 바깥 기대값: 여러 목표에 대한 평균 (멀티태스크)
- 안쪽 기대값: 정책과 동역학이 만드는 궤적에 대한 평균
- $\gamma^t$: 먼 미래일수록 가중치를 줄임

### 2.2 벽 1의 정체: 희소 이진 보상

이 논문의 보상은 다음과 같다.

$$r(s,a,g)\ =\ \mathbf{1}\big[\,d(\phi(s),\,g)\ \le\ \varepsilon\,\big]$$

- $\phi(s)$ — 상태에서 목표와 관련된 부분만 뽑은 표현 (예: 물체의 위치)
- $d(\cdot,\cdot)$ — 거리 함수
- $\varepsilon$ — 허용 오차
- $\mathbf{1}[\cdot]$ — 지시함수 (조건 참이면 1, 아니면 0)

목표에 충분히 가까우면 1, 아니면 0. **과정에 대한 정보가 전혀 없다.** 물체를 목표 위치 $\pm\varepsilon$에 놓기 전까지 보상은 계속 0이다.

그런데 여기서 자연스러운 반문이 생긴다. LLM의 RL도 결국 희소 보상을 쓰지 않나? GRPO는 최종 정답 여부만 보는 결과 보상(outcome reward)으로 돌아간다. 그게 잘 되는데 왜 로봇에서는 희소 보상이 벽이 되는가?

> ### 💡 희소 보상은 로봇에서만 벽이 된다 — 이유는 "보상이 켜지는 빈도"
>
> 희소 보상 자체가 나쁜 게 아니다. 성패를 가르는 건 **현재 정책이 자기 탐색만으로 보상 1을 얼마나 자주 얻어걸리느냐**, 즉 $\Pr[r=1 \mid \pi_{\text{current}}]$ 다. 이 확률이 0에 수렴하면 gradient의 기대값도 0이 되어 학습이 시작조차 못 한다.
>
> **LLM 추론** — 사전학습 prior가 워낙 강해 그냥 샘플링만 해도 정답을 자주 맞힌다(base가 30~70% 성공). 희소 보상이 배치마다 켜지고, GRPO는 이미 되는 성공 샘플을 재가중하는 **exploitation 중심** 문제가 된다.
>
> **로봇 조작** — 새 태스크에서 base 성공률이 0에 수렴할 수 있다. 게다가 연속·고차원 액션 공간에서 성공에 이르는 궤적 집합은 사실상 measure-zero라, 무작위 탐색으로 보상을 밟을 확률이 거의 없다. **needle-in-haystack 탐색** 문제가 된다.
>
> 네 축에서 차이가 난다.
>
> | 축 | LLM 추론 | 로봇 조작 |
> |---|---|---|
> | prior의 힘 | 강함 — 탐색이 사실상 in-distribution | 새 태스크에선 약함 — 탐색이 OOD |
> | 탐색공간 구조 | 이산 토큰, 정답 경로가 여럿 | 연속 고차원, 성공 다양체가 measure-zero |
> | 검증 비용 | 정답 대조 = 자동·공짜 | success predicate·reward classifier 필요 |
> | 리셋 비용 | 새 프롬프트 = 공짜, 대량 병렬 | 물리 리셋 필요, 롤아웃당 실시간 비용 |
>
> 이 표는 뒤에서 PLD의 설계를 읽는 렌즈가 된다. PLD가 하는 일은 보상을 조밀하게 바꾸는 게 아니라, **로봇의 학습 설정을 LLM 쪽 조건으로 옮겨놓는 것**이기 때문이다(§3.2에서 확인한다).

### 2.3 Q값과 Actor-Critic

RL의 근본 딜레마는 이렇다. 정책을 고치려면 어떤 행동이 좋은지 알아야 하는데, 행동의 좋음은 그 이후의 미래 보상까지 다 봐야 안다. 이를 두 네트워크로 분업한다.

**Critic** — 행동가치 함수 $Q^\pi(s,a)$를 학습한다.

$$Q^\pi(s,a)\ =\ \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\ \Big|\ s_t=s,\ a_t=a\right]$$

"상태 $s$에서 행동 $a$를 지금 하고, 그 뒤로는 $\pi$를 따를 때 받을 감가 누적 보상의 기대값." 이 함수는 자기 자신에 대한 재귀식(Bellman 방정식)을 만족하므로, 그 등식을 회귀 목표로 삼아 학습한다(TD-learning).

$$Q^\pi(s_t,a_t)\ \leftarrow\ \underbrace{r(s_t,a_t)}_{\text{즉시 보상}}\ +\ \gamma\,\mathbb{E}_{s_{t+1}\sim\rho}\big[\underbrace{Q^\pi_{\text{target}}(s_{t+1},a_{t+1})}_{\text{미래 가치 부트스트랩}}\big]$$

**Actor** — 정책 $\pi_\theta$로, critic이 높은 점수를 주는 행동을 내도록 학습한다.

$$\max_\theta\ J(\theta)\ =\ \mathbb{E}_{s}\big[\,Q(s,\ \pi_\theta(s))\,\big]$$

이 목적을 어떻게 최적화하는지가 벽 2를 이해하는 열쇠다. 문제는 파라미터 $\theta$가 $Q$ 안에 **직접 들어있지 않다**는 것이다. $\theta$는 오직 행동 $a=\pi_\theta(s)$를 **통해서만** $Q$에 영향을 준다. 그래서 연쇄법칙으로 쪼갠다.

$$\nabla_\theta J\ =\ \mathbb{E}_s\Big[\ \underbrace{\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)}}_{\textbf{A: critic이 주는 조각}}\ \cdot\ \underbrace{\nabla_\theta \pi_\theta(s)}_{\textbf{B: actor가 주는 조각}}\ \Big]$$

- **A** $=\nabla_a Q$ — "행동을 어느 방향으로 밀면 Q가 오르나." critic이 알려주는 **행동공간에서의 개선 방향**. actor 업데이트 시 critic 파라미터는 고정되므로, 이 스텝에서는 상수 벡터로 취급된다.
- **B** $=\nabla_\theta\pi_\theta$ — "파라미터를 밀면 행동이 어느 방향으로 움직이나." actor 자신의 민감도(Jacobian).
- **A·B** — 둘의 곱이 최종 **파라미터 업데이트 방향**이다.

직관적으로, actor는 행동을 직접 고를 수 없다. 손에 쥔 손잡이는 $\theta$뿐이고 행동은 그 산출물이다. 그래서 "좋은 행동을 내라"는 목표가 "θ를 이 방향으로 밀어라"로 번역되어야 하고, 연쇄법칙이 그 통역을 맡는다. A는 **국소 방향**이지 목적지($\arg\max_a Q$)가 아니라는 점도 중요하다. 목적지를 모르기 때문에 조금씩 미는 경사상승을 반복하는 것이다.

여기서 지도학습과 비교하면 성격이 선명해진다. 메커니즘(backprop, 연쇄법칙)은 완전히 같다. 다른 것은 **신호의 출처**다. 지도학습은 고정된 정답 라벨 $y^*$까지의 거리를 줄이지만($\nabla_a L = 2(a-y^*)$), actor-critic은 정답 라벨이 없어 **critic이 그 자리를 대신하고, 그 critic 자체도 학습되며 움직인다**. 실제로 critic이 포물선 $Q(s,a) = -\lVert a - a^*\rVert^2$ 형태라면 $\nabla_a Q = -2(a-a^*)$가 되어, 지도학습의 gradient와 부호만 다른 같은 식이 된다.

### 2.4 벽 2의 정체: 미분 가능한 행동생성이라는 요구조건

위 유도의 결정적 전제는 **B가 계산 가능해야 한다**는 것이다. 즉 $a=\pi_\theta(s)$가 $\theta$에 대해 매끄럽게 미분 가능해야 한다. 여기서 action head의 종류가 갈린다.

**✅ 가우시안 정책** — reparameterization으로 단번에 해결된다.

$$a\ =\ \mu_\theta(s)\ +\ \sigma_\theta(s)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)$$

$\epsilon$을 바깥으로 빼내면 $a$는 $\theta$의 **한 번의 매끄러운 함수**가 되고, $\nabla_\theta\pi$가 즉시 나온다. SAC/DDPG/TD3가 그대로 돈다.

**❌ Flow-matching / Diffusion head** — 행동이 다단계 생성 과정의 산물이다.

$$z_0\ \xrightarrow{\ \text{step 1}\ }\ z_1\ \xrightarrow{\ \text{step 2}\ }\ \cdots\ \xrightarrow{\ \text{step N}\ }\ a$$

$\nabla_\theta\pi$를 얻으려면 이 N-스텝 체인 전체를 뚫고 backprop해야 한다(ODE solver 또는 denoising chain을 통한 BPTT). 메모리가 폭증하고 수치적으로 불안정하다. 게다가 이 head들은 score-matching/flow-matching 목적으로 학습되지 가능도 최대화로 학습되지 않아, 정책경사에 바로 꽂을 깔끔한 $\log\pi$나 재매개변수 샘플도 없다. 논문이 인용하는 "policy-agnostic RL" 난제가 이것이다.

**이것이 벽 2의 정확한 내용이다.** π₀ 같은 최신 VLA일수록 표현력 좋은 flow head를 쓰는데, 바로 그 표현력 때문에 RL을 직접 걸 수 없다.

그런데 여기서 다시 반문이 생긴다. LLM의 RL은 이런 문제를 겪지 않는다. GRPO는 거대 모델을 잘만 RL로 돌린다. 무엇이 다른가?

> ### 💡 정책경사에는 두 갈래가 있고, LLM과 로봇은 서로 다른 갈래를 탄다
>
> 우리가 §2.3에서 유도한 것은 사실 정책경사의 **한 갈래**일 뿐이다.
>
> **갈래 A — Pathwise / DPG**
>
> $$\nabla_\theta J\ =\ \mathbb{E}_s\big[\nabla_a Q(s,a)\cdot\nabla_\theta\pi_\theta(s)\big]$$
>
> Q와 행동생성을 **뚫고 backprop**한다. 저분산이고 off-policy에 친화적이지만, **미분 가능한 행동생성이 필수**다. 대표: DDPG, TD3, SAC. **PLD가 여기 속한다.**
>
> **갈래 B — Score-function / REINFORCE**
>
> $$\nabla_\theta J\ =\ \mathbb{E}\big[\nabla_\theta \log\pi_\theta(a\mid s)\cdot \hat{A}\big]$$
>
> Q를 미분하지 않는다. advantage $\hat A$를 **스칼라 가중치(상수)** 로 두고, $\log\pi_\theta$만 미분해 "잘한 행동의 로그확률을 밀어 올린다." $\log\pi$만 있으면 되므로 행동생성을 backprop할 필요가 없다. 대신 고분산이고 보통 on-policy다. 대표: REINFORCE, PPO, **GRPO**.
>
> 이 구분이 §2.2의 표와 맞물려 모든 것을 설명한다.
>
> - **LLM은 갈래 B를 탄다.** softmax라 $\log\pi$가 공짜로 나오고, 행동 backprop이 아예 필요 없다. 갈래 B의 약점인 고분산·on-policy(샘플을 버림)는 LLM에서 문제가 안 된다 — 리셋이 공짜고 대량 병렬 샘플링이 되니까. **그래서 LLM은 flow head든 뭐든 신경 쓸 필요가 없다.**
> - **로봇은 갈래 A를 타야 한다.** 실물 롤아웃이 비싸 데이터를 버릴 수 없으니 replay buffer를 재사용하는 off-policy가 생존 조건이다. 그런데 갈래 A는 미분 가능한 행동생성을 요구한다. **바로 여기서 flow head가 걸린다.**
>
> 요약하면, 벽 2는 "로봇은 샘플이 비싸서 갈래 A를 타야 하는데, 갈래 A는 flow head와 상극"이라는 구조적 딜레마다. PLD의 해법은 이 딜레마를 정면돌파하지 않고 **우회**하는 데 있다.

### 2.5 Off-policy를 한 번 더 짚고 넘어가기

갈래 A가 필수인 이유를 명확히 해두자. RL 알고리즘은 "어떤 정책이 만든 데이터로 학습할 수 있나"로 나뉜다.

| | On-policy | Off-policy |
|---|---|---|
| 학습 데이터 | 지금 그 정책이 방금 만든 것만 | 아무 정책이 과거에 만든 것도 OK |
| replay buffer | 재사용 불가 (정책 바뀌면 폐기) | 재사용 가능 |
| 표본 효율 | 낮음 | 높음 |
| 대표 | PPO, REINFORCE | DQN, DDPG, TD3, SAC |

off-policy가 가능한 이유는 정책이 아니라 **Bellman 방정식(Q)** 을 중심으로 학습하기 때문이다. Bellman은 그 데이터가 어떤 정책에서 나왔는지 따지지 않는다.

PLD에서 off-policy는 선택이 아니라 **필수**다. 뒤에서 보겠지만 PLD는 **base 정책 $\pi_b$가 만든 성공 궤적**으로 버퍼를 채우고, 학습 대상은 **잔차 정책 $\pi_\delta$** 다. 데이터를 만든 정책과 학습하는 정책이 다르다 — 이건 정의상 off-policy가 아니면 성립하지 않는다.

### 2.6 SFT 손실 (Distill 단계에서 쓰인다)

마지막으로 VLA의 SFT 손실을 정리해둔다. VLA 정책은 관측 $o_t$와 목표 $g$를 받아 행동을 낸다.

$$a_t\ =\ D_\phi\big(h_\theta(o_t,\ g)\big)$$

- $h_\theta$ — vision-language 백본
- $D_\phi$ — action head

head 종류마다 SFT 손실이 다르다. 이 점이 나중에 PLD가 "architecture-agnostic"이라 주장하는 근거가 된다.

**Autoregressive 토큰** (OpenVLA) — 액션 토큰 시퀀스의 NLL.

$$\mathcal{L}_{\text{AR}}(\theta)\ =\ -\,\mathbb{E}_{k\sim[K]}\big[\log p_\theta(u_k\mid u_{<k},\ x)\big],\qquad x=(o_t,\ g_t)$$

- $u_k$ — $k$번째 액션 토큰, $K$ — 토큰 개수

**Diffusion** (Octo, Diffusion Policy) — score-matching MSE.

$$\mathcal{L}_{\text{diff}}(\theta)\ =\ \mathbb{E}_{t,\,\epsilon,\,(x,a)}\big[\ \lVert \epsilon\ -\ \epsilon_\theta(a_t^{(\text{noisy})},\ x,\ t)\rVert_2^2\ \big]$$

- $\epsilon$ — 주입한 노이즈, $\epsilon_\theta$ — 노이즈 예측 네트워크, $t$ — diffusion timestep

**Flow-matching** (π₀) — prior를 액션 분포로 옮기는 속도장(velocity field)에 대한 $L_2$ 손실.

---

## 3. 방법 — PLD의 세 단계

이제 두 벽의 정체가 분명해졌으니 PLD의 해법이 왜 그 모양인지 읽을 수 있다.

핵심 전략은 **decoupling**이다. 거대 정책을 통째로 RL하지 않는다. base VLA를 **얼리고**, 그 위에 가벼운 **잔차 정책**만 RL로 학습해 태스크 전문가를 얻고, 그 전문가로 데이터를 뽑아, **표준 SFT로 base에 되돌려 증류**한다.

```
  STAGE 1: LEARN            STAGE 2: PROBE             STAGE 3: DISTILL
  ------------------        --------------------       -------------------
  frozen base pi_b          base pi_b 가 T_base        tau_demo 를 표준 SFT
     +  residual pi_d          스텝 걷는다 (probe)        로 base 에 되증류
     -> abar = a_b + a_d    -> residual expert 가     -> 개선된 generalist
                               이어받아 복구 시연
  off-policy RL 로               (takeover)            전문가는 폐기,
  base 를 능가하는                                      generalist 만 배포
  전문가를 얻는다            학습이 아니라 데이터 생산
```

### 3.1 잔차 정책 — 벽 2를 우회하는 구조

base 정책 $\pi_b$를 얼린 채로 두고, 그 위에 **base 액션을 조건으로 받는 작은 가우시안 정책** $\pi_\delta$만 학습한다. 실제 환경에 나가는 행동은 둘의 합이다.

$$\bar{a}\ =\ a_b\ +\ a_\delta,\qquad a_b\sim\pi_b(\cdot\mid s),\qquad a_\delta\sim\pi_\delta(\cdot\mid s,\ a_b)$$

- $a_b$ — 얼린 base VLA가 낸 행동 (**미분하지 않는다**. 조건 입력으로만 쓴다)
- $a_\delta$ — 학습되는 잔차 교정량, $a_\delta\in[-\xi,\ \xi]$
- $\xi\in[0,1]$ — 잔차 크기 상한. 스케줄러로 조절하며, 초기에는 작게 두어 base에서 급격히 벗어나지 않게 한다

결합 정책 $\bar\pi$에 대한 critic은 TD-learning으로 학습한다.

$$Q^{\bar\pi}(s_t,\ \bar a_t)\ \leftarrow\ r(s,a)\ +\ \gamma\,\mathbb{E}_{s_{t+1}\sim\rho(\cdot\mid s_t,\bar a_t)}\big[\,Q^{\bar\pi}_{\text{target}}(s_{t+1},\ \bar a_{t+1})\,\big],\qquad \bar a = a_b + a_\delta$$

**이 구조가 벽 2를 어떻게 우회하는지** 보자. §2.4에서 문제는 flow head의 $\nabla_\theta\pi$였다. PLD는 flow head를 아예 미분하지 않는다. 얼려서 조건 입력 $a_b$로만 쓴다. RL이 붙는 대상은 가우시안 잔차뿐이고, 가우시안은 reparameterization으로 한 번에 미분된다. 즉 **갈래 A(pathwise)를 쓰되, 행동생성을 미분 가능한 부분으로 국한**시킨 것이다.

그 결과 논문의 표현대로 "잔차 가우시안 정책은 **아무 off-the-shelf off-policy RL 알고리즘으로도** 학습된다." 여기서 off-the-shelf는 "기성품을 개조 없이 그대로"라는 뜻이다 — SAC나 TD3를 커스터마이징 없이 붙일 수 있다는 말이고, 이는 다단계 샘플링을 뚫는 특수 기계장치를 새로 짜야 하는 flow head 직접 RL과 정면으로 대비된다.

동시에 이 구조는 **탐색의 초기화**도 해결한다. base 정책은 완벽히 일반화하지는 못해도 "그럴듯한 시도"는 한다. 잔차가 그 근처($[-\xi,\xi]$ 안)를 탐색하므로, 탐색이 무작위가 아니라 **유의미한 행동 근방**에서 이루어진다. 이것이 벽 1을 뚫는 첫 번째 장치가 된다.

### 3.2 표본 효율의 세 장치 — 벽 1을 뚫기

잔차 구조만으로는 부족하다. 희소 보상 아래서 표본 효율적으로 학습하려면 세 가지가 더 필요하다.

**⓵ RLPD식 대칭 replay — 배치에 성공을 항상 심는다**

두 개의 버퍼를 유지한다.

$$\mathcal{B}_{\text{offline}} = \{\tau_1,\ \tau_2,\ \dots\}\quad(\pi_b\ \text{의 성공 궤적만}),\qquad \mathcal{B}_{\text{online}}\quad(\text{새로 수집한 온라인 경험})$$

그리고 미니배치를 두 버퍼에서 **동수로** 뽑는다. 논문이 명시하듯 offline buffer를 성공 궤적으로만 채우는 것은 **성공 시도만 보존하는 importance sampling** 역할을 하고, 대칭 샘플링은 value function이 **항상 고가치 상태-행동 쌍 위에서** 학습되도록 보장한다. 즉 critic이 보는 모든 배치에 **보상 1인 샘플이 절반** 들어 있다.

**⓶ Warm-up — critic을 정상 경험에 먼저 접지시킨다**

학습 초기에 critic은 미숙하고 잔차는 사실상 랜덤이다. 이 상태로 곧장 잔차를 풀면 $\bar a = a_b + a_\delta$의 $a_\delta$가 난수라 엉뚱한 상태를 방문하고, critic이 노이즈로부터 학습해 발산한다. 그래서 처음 얼마간은 **$\pi_b$ 단독으로만 데이터를 수집**해(WSRL 방식) 버퍼와 critic을 말이 되는 경험으로 채운다. off-policy 학습을 안정화하고 망각을 완화하는 역할이다.

**⓷ Cal-QL — critic을 보수적이되 "보정된" 값으로 초기화한다**

Stage 1은 본질적으로 **offline → online 전환**이다(critic을 base 성공 궤적에 사전학습한 뒤 잔차가 online 탐색을 시작). 이 전환 지점이 악명 높게 불안정하다.

> ### 💡 왜 그냥 CQL이 아니라 Cal-QL인가
>
> **문제의 뿌리 — offline RL의 과대평가.** Q-learning은 부트스트랩에서 행동에 대해 max/기댓값을 취한다. 데이터에 없는(OOD) 행동은 Q가 외삽되는데, max 연산이 **과대추정된 놈을 골라내는** 성질 때문에 오차가 낙관 쪽으로 쏠린다. 정책은 이 유령처럼 높은 Q를 좇다가 붕괴한다. PLD에서는 잔차가 base 데이터에 없던 결합 행동 $\bar a = a_b + a_\delta$를 제안하니 정확히 이 문제에 노출된다.
>
> **CQL의 처방.** 정책이 뽑은 OOD 행동의 Q는 눌러 내리고, 데이터에 있는 행동의 Q는 올린다. 결과적으로 학습된 Q가 참값의 **보수적 하한**이 되어 정책이 유령을 못 좇는다.
>
> **CQL의 부작용.** 보수성이 지나쳐 Q의 **스케일 자체가 어긋난다**. 이 상태로 online 미세조정에 들어가면, online 업데이트가 먼저 이 과도하게 눌린 Q를 되돌리느라(unlearning) 초반 성능이 급락한다. offline→online의 유명한 "초기 dip"이다.
>
> **Cal-QL의 처방 — calibration.** 보수성은 유지하되 **기준선을 깔아준다.** 학습된 Q가 참 가치의 하한이면서 동시에 어떤 기준(behavior) 정책 가치의 상한이 되도록 샌드위치한다.
>
> $$V^{\text{ref}}(s)\ \le\ Q_{\text{learned}}(s,a)\ \le\ Q^{\pi}_{\text{true}}(s,a)$$
>
> 즉 OOD 행동의 Q를 누르되 **기준 정책의 가치 아래로는 내리지 않는다.** 여전히 보수적이면서도 "합리적 스케일"을 유지하고, 그 결과 online 전환의 파국적 dip이 사라진다. 구현은 CQL 위에 사실상 한 줄 변경이다.
>
> PLD의 Stage 1이 정확히 offline→online이므로 이 장치가 필요하다. 실제로 논문이 보고하는 **학습 초기의 일시적 성능 하락**(잔차가 base에서 발산하며 suboptimal 상태를 방문하는 탐색 초기의 흔적)이 파국이 아니라 회복 가능한 얕은 dip에 머무는 것이 이 덕분이다.

**그리고 일부러 하지 않는 것 — behavior constraint**

많은 offline/offline-to-online RL은 정책 손실에 **행동 제약**을 걸어 학습 정책을 데이터 수집 정책 근처에 묶는다(TD3+BC의 BC 페널티, AWAC의 암묵 제약 등). 과대평가를 막는 또 하나의 안전장치다. **PLD는 이것을 일부러 걸지 않는다.** 이유는 명확하다 — 정책을 base 근처에 묶으면 전문가가 **base의 천장을 그대로 상속**받는다. PLD의 목표는 전문가가 base를 **능가**하는 것이므로 이 목줄을 푼다.

그러면 안정성은 어디서 오는가? 여기가 설계의 우아한 지점이다. PLD는 안정화를 **정책 손실에서 빼서 다른 세 곳으로 분산**시켰다.

| 안정화 책임 | 담당 |
|---|---|
| 구조 측 | $\xi$-제한 잔차 — 행동이 구조적으로 base 근처에 묶임 (손실 페널티가 아님) |
| Critic 측 | Cal-QL — 과대평가를 critic 쪽에서 처리 |
| 데이터 측 | 대칭 replay — critic을 고가치 상태에 접지 |

정책은 자유롭게 base를 넘어설 수 있으면서도 시스템 전체는 안정하다.

이제 §2.2에서 예고한 관점으로 돌아갈 준비가 됐다.

> ### 📌 PLD의 핵심 통찰 — 보상을 고치지 않고, 로봇을 LLM 조건으로 옮긴다
>
> §2.2에서 LLM의 희소 보상이 작동하는 조건을 셋으로 정리했다. PLD의 Stage 1을 그 렌즈로 다시 보면, 세 장치가 각각 그 조건 하나씩을 로봇에 이식하고 있다.
>
> | LLM에서 희소 보상이 켜지는 조건 | PLD가 로봇에서 재현하는 장치 |
> |---|---|
> | **ⓐ 강한 prior가 성공 샘플을 자주 낸다** | **policy prior warm-start** — 얼린 base가 non-zero 성공률을 확보하고, 잔차는 $[-\xi,\xi]$ 안에서만 탐색한다. GRPO가 이미 유능한 LLM에서 샘플을 뽑는 구도와 같다 |
> | **ⓑ 배치에 성공 샘플이 항상 존재한다** | **RLPD 대칭 replay** — offline 버퍼를 base의 성공 궤적으로 미리 채우고 반반 샘플링한다. LLM은 그룹 샘플에 성공이 자연히 섞이지만, 로봇은 이를 **버퍼로 인공 보장**한다 |
> | **ⓒ 검증이 공짜고 리셋이 무료다** | 로봇에는 이 사치가 없다 → **Cal-QL**로 대체 보강한다. 성공이 희소하고 값비쌀 때 off-policy Q가 발산하지 않도록 하는, **로봇에만 필요한 비계**다 (critic이 없는 GRPO에는 애초에 불필요하다) |
>
> **한 문장으로** — PLD는 reward shaping으로 보상을 조밀하게 만들지 않는다. 대신 "유능한 prior + 그 prior의 성공이 항상 배치에 들어 있는 상태"라는 **LLM의 성공 조건 자체를 로봇에 재구성**한다. 그러면 순진한 로봇 RL을 멈춰 세우던 바로 그 희소 보상이, GRPO에서 결과 보상이 켜지듯 **켜진다.** 여기에 §3.1의 가우시안 잔차가 "그 성공들을 기성 off-policy 알고리즘으로 학습 가능하게" 만들어 퍼즐을 완성한다.

이 조합으로 전문가는 태스크당 99% 이상 성공률에 도달한다(보고된 120개 이상의 태스크에서 95% 이상).

### 3.3 Probe — 배포 분포에 정렬된 데이터 수집

전문가를 얻었으니 데이터를 뽑을 차례다. 그런데 여기에 반직관적인 함정이 있다.

**순수 RL 전문가 데이터는 지나치게 최적이다.** 일관되고, 망설임이 없고, 짧은 horizon으로 매끄럽게 끝낸다. 그리고 **바로 그것이 문제다.** 이 단봉(unimodal)의 좁은 분포는 OOD 상태와 실패 상태를 과소표현한다. 그런 데이터를 아무리 늘려도 성능이 오르지 않고, 오히려 generalist가 과적합해 강건성과 일반화를 해친다.

> **"더 최적인 데이터" ≠ "더 좋은 SFT 데이터"**

매끈하게 잘하는 시범만 본 학생은 **삐끗했을 때 어떻게 빠져나오는지**를 배우지 못한다. §1의 coverage gap이 그대로 재현된다.

**해법은 base를 다시 무대에 올리는 것이다.** 궤적의 앞부분은 base가 걷게 두어 배포 분포로 진입시키고, 뒷부분부터 전문가가 이어받아 복구를 시연한다. 이것이 **hybrid rollout**이며, 앞 구간을 **base policy probing**이라 부른다.

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

- **prefix** — base가 배포 시 실제로 방문하는 상태로 진입하는 구간 (probing)
- **suffix** — 전문가가 그 지점에서 이어받아 복구를 시연하는 구간
- **alpha** — probing 길이의 상한 비율. 데이터 다양성을 조절하는 하이퍼파라미터

수집되는 시연 궤적은 다음과 같다.

$$\tau_{\text{demo}}\ =\ \underbrace{\big\{(s_1,\ a_{b,1}),\ \dots,\ (s_{t-1},\ a_{b,t-1})\big\}}_{\textbf{prefix — base가 방문한 배포 분포 상태}}\ \cup\ \underbrace{\big\{(s_t,\ a_{b,t}+\bar a_t),\ \dots\big\}}_{\textbf{suffix — 전문가의 복구 행동}}$$

- $s_1,\dots,s_{t-1}$ — base가 probing으로 실제 방문한 상태들. **배포 시 VLA가 실제로 빠지는 분포**
- $t = 1 + T_{\text{base}}$ — takeover 시점. $T_{\text{base}}$가 클수록 base가 더 깊이 드리프트한다
- suffix의 기록 액션 — takeover 이후 **실행된 결합 액션**. 잠재적으로 suboptimal한 영역에서 회복하는 전문가의 행동을 담는다

> ### ⚠️ 팩트체크 — $\tau_{\text{demo}}$ 수식의 표기 비일관
>
> §3.1에서 $\bar a = a_b + a_\delta$를 **결합 액션**으로 정의했는데, 위 $\tau_{\text{demo}}$ 식은 "$a_{b,t} + \bar a_t$"로 쓴다. 문자 그대로 읽으면 base 액션이 이중 계산된다. 문맥상 의도는 "**실행된 결합 액션(base + 잔차)을 기록한다**"이며, 여기서 $\bar a_t$는 잔차 성분을 가리키는 느슨한 표기로 읽어야 한다. 논문의 사소한 표기 비일관성이다.

**probing 구간에는 미묘하지만 중요한 비대칭이 있다.** probing 스텝은 **상태 초기화 용도로만** 쓰이고 **replay buffer에는 추가되지 않는다.** base의 probing 액션은 suboptimal하므로 RL의 학습 타깃이 되어서는 안 되기 때문이다. 반면 **SFT 데이터 $\tau_{\text{demo}}$에는 base prefix와 전문가 suffix가 모두 포함된다.** generalist가 정상 상태에서는 base처럼 행동하고, 드리프트했을 때는 전문가처럼 복구하도록 배우게 하려는 것이다.

여기서 자연스럽게 떠오르는 질문이 둘 있다. 하나는 "이거 DAgger 아닌가?"이고, 다른 하나는 "그럼 전문가는 언제 실패 복구를 배우는가?"이다.

> ### 💡 이것은 robot-gated DAgger의 자율판이다
>
> DAgger 계열은 정확히 같은 병(공변량 이동)을 같은 뼈대로 치료한다. 학생이 자기 분포로 진입하고, 개입자가 그 지점에서 복구를 시연하고, 그 데이터로 학생을 다시 학습시킨다.
>
> | 방법 | 개입 타이밍을 정하는 주체 | 전문가 |
> |---|---|---|
> | DAgger | 게이팅 없음 — 학생 전 구간 롤아웃 후 전 상태에 라벨 | 사람/oracle |
> | HG-DAgger | **사람**이 위험을 판단해 개입 | 사람 |
> | robot-gated DAgger | **알고리즘**이 불확실성·위험을 추정해 개입 요청 | 사람 |
> | **PLD** | **랜덤 스케줄** $T_{\text{base}}\sim[0,\alpha T]$ | **학습된 RL 잔차** |
>
> PLD의 probe→takeover는 robot-gated DAgger와 구조가 같지만, 세 가지가 결정적으로 다르다.
>
> - **전문가가 사람이 아니라 학습된 RL 잔차다.** DAgger 계열의 최대 비용인 "사람을 루프에 계속 붙여두기"가 사라진다.
> - **게이팅이 불확실성이 아니라 랜덤 스케줄이다.** robot-gated DAgger는 *위험할 때* 개입하지만, PLD는 랜덤 $T_{\text{base}}$ 이후 **무조건** 이어받는다. 목적이 위험 회피가 아니라 **다양성 확보**이기 때문이다.
> - **온라인 반복 집계가 아니라, 모아서 한 번 offline SFT한다.**
>
> 그래서 PLD는 **"robot-gated DAgger에서 사람을 제거하고 게이팅을 랜덤화한 자율판"** 으로 읽으면 정확하다.

> ### 💡 전문가는 언제 "실패에서 복구하는 법"을 배우는가 — 학습과 시연의 분리
>
> Stage 2에서 전문가가 복구를 "시연"하는 장면과, 전문가가 복구를 "학습"하는 장면은 그림이 똑같아 보이지만 다른 사건이다. **차이는 파라미터가 갱신되느냐 아니냐뿐이다.**
>
> **전문가의 학습 (Stage 1에서 끝난다).** RL은 **출발점으로 삼아 연습한 상태**에서만 잘하게 된다. 그래서 전문가가 임의의 base-방문 상태에서 이어받으려면, 학습할 때 episode의 시작을 바꿔야 한다. 기본 초기 분포 $\rho_0$ 대신,
>
> $$s_0\ \sim\ p_0^{\pi_b}\quad(\text{base가 랜덤 스텝 걸은 뒤 도달한 상태의 분포})$$
>
> 에서 출발시킨다. 이렇게 해야 전문가가 그런 상태들에서 복구를 반복 연습하고 강건해진다.
>
> **여기서 한 가지 표현을 정확히 해두자.** $p_0^{\pi_b}$는 "실패 상태 분포"가 **아니다.** base가 실제로 방문하는 **상태 분포 전반**이다 — 정상 궤도, 가벼운 드리프트, 심한 실패 근처가 모두 섞여 있다. 실패 근처는 그중 **중요한 부분집합**일 뿐이다. 그리고 이것이 정확히 §1의 coverage gap을 메우는 일이다. PLD가 하는 일은 "실패만 골라 넣기"가 아니라 **데이터를 배포 분포에 정렬시키기**이고, 실패 복구는 그 정렬의 가장 값진 부산물이다.
>
> **데이터 생성 (Stage 2).** 여기서 전문가는 **이미 학습을 마친 상태**다. 복구 "시연"은 파라미터를 갱신하는 학습이 아니라, generalist를 위한 **데이터를 생산하는 행위**다.
>
> **최종 강건성 (Stage 3).** 실제로 실패에 강건해지는 주체는 그 데이터를 SFT로 배우는 **generalist**다.
>
> 운전으로 비유하면 이렇다. 강사가 차를 도랑에 반쯤 넣어두고 "빼봐"라고 시키는 것 = **시작 상태 세팅**. 내가 반복해서 빠져나오며 실력이 느는 것 = **학습(Stage 1)**. 숙련된 뒤 도랑 탈출 장면을 촬영해 교본을 만드는 것 = **데이터 생성(Stage 2)**. 그 교본으로 배우는 초보 = **generalist(Stage 3)**.

**$\alpha$는 다양성을 조절하는 손잡이다.** probing 길이를 $T_{\text{base}}\sim\text{Uniform}[0,\ \alpha T]$로 뽑으므로, $\alpha$가 클수록 base가 오래 걸어 더 깊이 드리프트하고, 그것을 교정하는 **우회(detour)** 가 길어지며, 궤적의 다양성이 증가한다. 논문이 $\alpha\in\{0.0,\ 0.2,\ 0.4,\ 0.6,\ 0.8\}$로 실험한 결과는 역U자다.

```
  성능
   ^
   |                    .----*----.                 <- alpha = 0.6 에서 plateau
   |              .----'           '--.
   |        .----'                     '--.         <- 과하면 하락
   |  .----'
   +----+-------+-------+-------+-------+------> alpha
      0.0     0.2     0.4     0.6     0.8

      다양성 부족 <---------------------> base 분포에서 이탈
```

적당한 probing은 복구 시나리오의 커버리지를 주지만, 과하면 base 분포에서 너무 멀어져 오히려 해가 된다.

### 3.4 Distill — 표준 SFT로 되돌리기

수집한 $\tau_{\text{demo}}$를 base VLA의 head에 맞는 **표준 SFT 손실**(§2.6의 AR NLL / diffusion MSE / flow-matching $L_2$ 중 하나)로 그대로 증류한다. 여러 태스크의 전문가를 하나의 generalist로 접어 넣는 과정이며, 전문가들은 데이터 생산용 비계였으므로 폐기하고 generalist 하나만 zero-shot으로 배포한다.

여기서 용어를 정확히 해둘 필요가 있다. **이 "distill"은 teacher의 logit을 student가 맞추는 지식 증류가 아니다.** 전문가가 **생성한 궤적**(hard action label) 위에서 generalist를 표준 BC로 SFT하는 것, 즉 **생성 데이터를 통한 정책 증류**다.

> ### ⚠️ 팩트체크 — 논문에는 "Section 3.3"이 없다
>
> 방법론 섹션은 Overview → 3.1 → 3.2로 끝난다. Distill 단계에는 **별도의 방법론 절이 배정되어 있지 않다.** 기술적 내용(SFT 손실)은 이미 Preliminaries에 앞당겨 정의되어 있고, "무엇을 하는가"는 Method Overview에만 짧게 나온다.
>
> **그리고 이것은 우연이 아니라 논지 그 자체다.** Stage 3가 "그냥 표준 SFT"라서 새로 설명할 방법론이 없다는 사실이야말로 PLD의 **plug-and-play** 주장이다. 모든 RL 복잡성은 어차피 버려질 전문가(Stage 1)에 격리되고, generalist에게는 깨끗한 SFT 데이터만 넘어간다. 그래서 어떤 VLA의 어떤 SFT 파이프라인에도 데이터만 꽂으면 된다.

> ### 📌 개선된 generalist가 "평균적인 전문가"를 능가한다
>
> 논문의 주목할 만한 관찰이다. 개별 태스크 전문가보다 그것들을 증류한 generalist가 더 낫다. 이유는 두 가지다.
>
> - **집계(aggregation)** — 여러 태스크 전문가의 데이터가 한 모델에 모인다.
> - **일반화 유지(less forgetting)** — PLD 데이터가 base 분포 근처에 있어 base의 일반화 능력을 덜 잃는다(§4).
>
> 이 시너지가 "왜 여러 전문가를 하나로 접어도 손해가 아닌가"에 대한 답이다. 그리고 이 출력(개선된 generalist)을 다시 Stage 1의 base로 넣으면 **self-improving flywheel**이 된다.
>
> $$\text{generalist}\ \to\ \text{잔차 전문가}\ \to\ \tau_{\text{demo}}\ \to\ \text{더 나은 generalist}\ \to\ \cdots$$

---

## 4. 왜 작동하는가

PLD 데이터가 사람 시연과 순수 RL 데이터를 모두 이기는 이유를 논문은 두 축으로 설명한다.

**⓵ 덜 잊는다 (less forgetting).** PLD 데이터는 base 시행 근처에 군집한다(prefix가 base 분포에서 나오므로). 논문은 이를 LLM fine-tuning의 관찰과 연결한다 — **KL-divergence가 망각의 지표**라는 "RL's Razor"(Shenfeld et al., 2025)의 관점에서, base 근처에 머무는 데이터는 fine-tuning 후 정책의 KL 이동을 작게 유지하고, 따라서 base의 일반화 능력을 덜 잃는다.

**⓶ 커버리지 (coverage).** 다양한 복구 행동을 담아 실패 상태를 대표함으로써, 순차 의사결정에서의 강건성을 높인다(Kelly et al., 2019, HG-DAgger의 관찰과 일치).

이 두 축이 실환경에서 정확히 관찰된다. Franka cube pick-up 실험에서 RLPD 데이터나 사람 데이터로 학습한 정책은 큐브를 왼쪽 위 모서리로 밀어 그리퍼가 끼이는 실패를 반복했다. 반면 PLD로 학습한 정책은 잡기 전에 큐브를 재배치해 복구했다. 분포를 분석해보니 **사람 시연에도, 순수 RL 롤아웃에도 그 모서리 상태는 등장한 적이 없었고, PLD의 probing만이 그 케이스를 데이터에 담았다.** self-improving data flywheel이 실제로 작동한다는 가장 직접적인 증거다.

---

## 5. 실험

### 5.1 전문가 학습의 표본 효율

WSRL(offline 초기화만 사용)과 RLPD(base 정책 유도 없음)를 비교군으로, LIBERO-90의 8개 태스크에서 큰 폭의 우위. 태스크당 사전학습 데이터는 성공 궤적 50개, critic은 Cal-QL 초기화, 온라인 상호작용 250k step, 3 seed 95% CI. 보고된 120개 이상의 태스크에서 95% 이상 도달.

### 5.2 In-distribution 성능

추가 사람 시연 **0개**로, 사람 시연 SFT 대비 일관된 절대 향상.

**LIBERO** (태스크당 50 episode 평가)

| | Spatial | Object | Goal | **Avg** |
|---|---|---|---|---|
| π₀ baseline | 95.2 | 97.6 | 87.4 | **93.4** |
| π₀ + PLD | 97.7 | 98.5 | 95.3 | **97.2** (+3.8) |
| OpenVLA baseline | 92.9 | 99.1 | 83.25 | **91.8** |
| OpenVLA + PLD | 99.5 | 99.1 | 98.9 | **99.2** (+7.4) |

**SimplerEnv** (Octo 기반)

| | Eggplant | Carrot | Open Drawer | Coke Can | **Avg** |
|---|---|---|---|---|---|
| Octo-SFT | 65.5 | 43.3 | 92.5 | 85.7 | **71.8** |
| + ours | 97.8 | 93.9 | 99.3 | 95.5 | **96.6** |
| Δ | +32.3 | **+50.6** | +6.8 | +9.8 | **+24.9** |

> ### ⚠️ 팩트체크 — "over 50% gains in SimplerEnv"는 벤치마크 평균이 아니다
>
> abstract와 intro는 SimplerEnv에서 "50% 이상"의 향상을 말하지만, 표를 보면 **평균 절대 향상은 +24.9%p**다. "50%"는 **단일 태스크(Carrot Pick)의 +50.6%p**라는 최댓값을 가리킨다. 상대 향상으로 계산해도 평균은 71.8 → 96.6, 약 +34.7%로 50%에 미치지 못한다.
>
> 반면 LIBERO의 "99%" 주장은 표(OpenVLA + PLD, Avg 99.2)와 정확히 일치한다. 인용할 때 이 둘을 구분할 필요가 있다.

### 5.3 일반화

- **Unseen task (zero-shot)** — LIBERO-90의 태스크 중 10%만 학습해도 unseen 태스크에서 24.4% SR. 반면 base-policy rollout(0-1 REINFORCE 방식의 self-bootstrap)은 in-distribution에서도 부진하고 일반화에 실패한다. 사람 데이터는 zero-shot은 비슷하나 in-distribution에서 뒤처진다.
- **Out-of-domain (few-shot)** — PLD 데이터를 50 → 500 궤적으로 늘리면 target 성능이 단조 증가.
- **Long-horizon** — LIBERO-10 평가에서 self-bootstrap보다는 낫지만 **사람 시연 SFT에는 아직 못 미친다.** 논문이 명시한 한계다.

### 5.4 실환경

**Franka 7-DoF** (pick-and-place, peg insertion) — 태스크 랜덤화에 제한을 두지 않은 어려운 설정. PLD와 RLPD 전문가 모두 사람 개입 없이 2시간 내 100% 성공에 도달했고, 각각 200개의 성공 시연을 자동 수집해 π₀를 재SFT했다. 30회 랜덤 시행 결과:

| | cube pick-up | peg insertion |
|---|---|---|
| π₀ + PLD | **30 / 30** | 30 / 30 |
| π₀ + RLPD | 16 / 30 | 30 / 30 |
| π₀ + Human | 10 / 30 | 30 / 30 |

**YAM 양팔 6-DoF** (산업용 GPU 삽입) — GPU를 슬롯1에 삽입 → 슬롯3으로 이동 → 삽입 → 뽑아서 테이블 복귀의 4단계로 분해하고, reward classifier가 상태기계를 조율. 서브태스크당 최대 8시간 학습 후 단일 BC 정책으로 증류. **사람 개입이나 리셋 없이 최소 1시간 연속**으로 전체 루프를 수행하며, 실패해도 스스로 회복해 flywheel을 유지했다.

---

## 6. 위치잡기 — 이웃 연구들 속에서

PLD는 세 계보의 교차점에 있다.

**⓵ 로봇 파운데이션 모델 / VLA post-training** — RT-1/2, OpenVLA, Octo, GR00T, π-series. "대규모 사전학습 + 소량 시연 SFT"라는 표준이 가진 데이터 희소성과 커버리지 한계가 PLD의 출발점이다.

**⓶ 데이터·정책 prior 기반 표본효율 RL** — 데이터 prior 쪽에 RLPD, Cal-QL, WSRL이 있고, 정책 prior 쪽에 잔차를 PPO로 학습하는 **ResiP**와 off-policy 잔차를 base와 공동학습하는 **EXPO**가 있다. 이 둘이 PLD의 가장 가까운 사촌이다. PLD의 차별점은 **suboptimal base로 non-zero 성공률을 warm-start하되, oracle 시연이나 사람 개입을 전혀 요구하지 않는다**는 점이다.

**⓷ VLA를 위한 RL post-training** — VLA-RL, on-policy 상호작용 post-training 등. 이들은 사람 개입이 많거나, generalist의 행동과 무관하게 데이터를 모으거나, 단일 태스크 최적화로 일반화를 희생한다. PLD는 이 셋을 동시에 겨냥한다.

가장 흥미로운 대비는 같은 시기에 나온 사촌 논문과의 비교다.

> ### 🔗 π*₀.₆ / RECAP (Physical Intelligence, 2025-11) — 정반대의 설계 선택
>
> 같은 시기, 같은 문제(모방학습의 오차 누적과 자기 실수로부터의 복구 불가)를 겨냥한 self-improving VLA다. 방법론 RECAP은 시연 → 사람 원격조종자의 실시간 교정 → 자율 practice의 3단계로 구성되며, 핵심 메커니즘은 **advantage conditioning**이다.
>
> PLD와 나란히 놓으면 축마다 반대편을 택한다.
>
> | | **PLD** | **π*₀.₆ / RECAP** |
> |---|---|---|
> | 사람 개입 | **없음** (자율 전문가가 복구 시연) | **있음** (원격조종 실시간 교정) |
> | RL 방식 | 잔차 pathwise RL + SFT 증류 | advantage-conditioning (정책경사 없음) |
> | Critic | **off-policy Q** (Cal-QL) | **on-policy V** (distributional MC) |
> | 실패 데이터 | 복구 궤적으로 변환해 증류 | 낮은 advantage 라벨로 흡수 |
>
> §3.3의 DAgger 비교에서 세운 "사람 개입 vs 자율" 축에서, **PLD는 사람을 제거한 자율판이고 RECAP은 원격조종 교정을 유지하는 쪽**이다. 두 논문을 함께 읽으면 self-improving VLA의 현재 지형이 선명해진다.

여기서 advantage를 다루는 방식이 세 갈래로 갈린다는 점이 흥미롭다. §2.4에서 정책경사의 두 갈래를 봤는데, RECAP은 사실 그 둘 어디에도 속하지 않는 **세 번째 길**이다.

> ### 💡 advantage를 계산하는 법 vs 사용하는 법 — GRPO, RECAP, PLD 비교
>
> 셋 다 "$A$ = (이 행동의 좋음 추정) − (기준선)"이라는 골격은 같다. 다른 것은 **기준선을 어디서 얻느냐**다.
>
> **GRPO — value network가 아예 없다.** 한 프롬프트에 대해 $G$개 출력을 샘플하고 그룹 내에서 정규화한다.
>
> $$\hat A_i\ =\ \frac{r_i\ -\ \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$
>
> 기준선은 학습된 $V$가 아니라 **같은 프롬프트에서 뽑은 샘플들의 평균 보상**이다.
>
> **RECAP — $V$만 학습한다.** 별도의 $Q$ 네트워크 없이, $Q$에 해당하는 항을 n-step 보상 + $V$ 부트스트랩으로 추정한다.
>
> $$A^{\pi_{\text{ref}}}(o_t,\ a_t)\ =\ \mathbb{E}\Big[\underbrace{\textstyle\sum_{t'=t}^{t+N-1} r_{t'}\ +\ V^{\pi_{\text{ref}}}(o_{t+N})}_{Q\text{의 n-step 추정}}\Big]\ -\ \underbrace{V^{\pi_{\text{ref}}}(o_t)}_{\text{기준선}}$$
>
> $V$는 distributional(이산 value bin 위의 분포)로 표현되고 경험적 return에 회귀시킨다 — **Monte Carlo, on-policy** 추정이다. 논문 스스로 이것이 off-policy Q 추정기보다 덜 최적이지만 단순하고 신뢰할 만하다고 인정한다.
>
> **PLD — off-policy $Q$(Cal-QL)를 학습한다.** RECAP과 정확히 반대편의 선택이다.
>
> **그리고 사용법도 셋이 다르다.**
>
> | | 계산 (기준선) | 사용 |
> |---|---|---|
> | GRPO | 그룹 샘플 평균 | **그래디언트 가중치** $\nabla\log\pi\cdot\hat A$ |
> | RECAP | 학습된 $V$ (on-policy MC) | **조건 입력** — $A$를 임계값으로 이진화해 $\pi(a\mid o,\ I)$의 조건으로 넣고, 추론 시 "높은 advantage"를 요청 |
> | PLD | off-policy $Q$ (Cal-QL) | **critic으로 pathwise actor 학습** |
>
> RECAP의 방식은 정책경사를 아예 쓰지 않는다. 행동을 점수표와 함께 통째로 학습해두고, 시험 때 "만점짜리를 내놓아라"라고 조건을 거는 것에 가깝다(upside-down RL / decision transformer 계열). 그래서 실패 데이터도 버리지 않고 "낮은 점수" 라벨과 함께 흡수할 수 있다.

---

## 7. 한계

논문이 스스로 밝힌 것과, 읽으며 추가로 짚어둘 만한 것을 함께 정리한다.

**논문이 인정한 한계**

- **Long-horizon 조합 능력** — LIBERO-10 같은 장기 태스크에서 여전히 사람 시연 SFT에 못 미친다.
- **YAM 실험의 수동 구조화** — 태스크를 4단계로 손수 분해하고 각 단계마다 reward classifier를 따로 학습해야 한다. "완전 자율"이라기보다 **잘 짜인 상태기계 위에서의 자율**에 가깝다.

**추가로 짚을 지점**

- **보상 엔지니어링이 실질적 병목** — 희소 이진 보상이라고 하지만, $\mathbf{1}[d(\phi(s),g)\le\varepsilon]$의 success predicate와 reward classifier를 태스크마다 구성해야 한다. 실환경 태스크 수를 늘릴 때 이것이 진짜 비용이 될 수 있다.
- **하이퍼파라미터 민감도** — 잔차 스케일 $\xi$의 스케줄과 probing 비율 $\alpha$(0.6에서 plateau)는 태스크 특이적일 가능성이 있다. 이는 plug-and-play 주장을 다소 약화시킨다.
- **실환경 태스크의 수** — Franka 2종, YAM 1종이다. 시뮬레이션(LIBERO, SimplerEnv) 성과는 크지만, 실환경 일반화 주장은 신중히 읽어야 한다.
- **abstract 수치의 해석** — §5.2의 팩트체크 참조.

---

## 8. 마치며 — 이 논문이 시사하는 것

PLD의 진짜 기여는 특정 RL 알고리즘이 아니다. **사람 teleoperation 없이 배포에 정렬된 데이터를 자동 생성하는 루프**를 실환경에서 돌려 보였다는 점이다. VLA 스케일링의 병목이 모델이 아니라 데이터라면, 그 데이터를 만드는 flywheel 자체가 자산이 된다.

그리고 이 논문은 LLM·diffusion 배경을 가진 사람에게 유난히 잘 읽힌다.

- **"얼린 백본 + 학습 가능한 잔차"** 는 PEFT/LoRA 발상의 RL판이다. 거대 모델을 통째로 흔들지 않고 작고 다루기 쉬운 모듈만 학습시킨다.
- **"RL이 SFT보다 덜 잊는다(KL이 망각의 지표)"** 는 LLM 정렬 문헌과 직결된다.
- **flow head를 직접 RL하기 어려워 가우시안 잔차로 우회한 선택**은, diffusion policy의 RL 최적화 난제를 아는 사람에게는 익숙한 트레이드오프다.

무엇보다, §3.2에서 정리한 통찰 — **보상을 고치는 대신 로봇의 학습 조건을 LLM 쪽으로 옮긴다** — 는 physical AI에 LLM의 경험을 이식하려는 모든 시도에 적용 가능한 일반 원리로 보인다.

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **잔차 정책 (residual policy)** | 얼린 base 액션에 더하는 작은 교정 델타. $\bar a = a_b + a_\delta$, $a_\delta\in[-\xi,\xi]$ |
| **base policy probing** | base를 랜덤 스텝 굴려 배포 분포 상태로 진입시키는 것. takeover 시작점 세팅 |
| **hybrid rollout** | probe(base) + takeover(전문가)로 구성된 궤적. $\tau_{\text{demo}}$의 생성 방식 |
| **RLPD 대칭 replay** | offline(base 성공) 버퍼와 online 버퍼에서 동수 샘플링해 critic을 고가치 상태에 접지 |
| **Cal-QL** | 보수적이되 기준 정책 가치 아래로는 내리지 않도록 보정한 Q 초기화. offline→online 전환 안정화 |
| **pathwise vs score-function** | 정책경사의 두 갈래. PLD는 pathwise(off-policy), GRPO는 score-function(on-policy) |
| **data flywheel** | (개선된 generalist) → (잔차 전문가) → (데이터) → (더 나은 generalist)의 자기개선 순환 |

**원문** — [arXiv:2511.00091](https://arxiv.org/abs/2511.00091) · **프로젝트 페이지** — wenlixiao.com/self-improve-VLA-PLD
