---
title: "GRPO Only Learns When the Rewards Disagree"
tab_title: GRPO
code: https://github.com/SadeekFarhan21/projects/tree/main/math-rlvr
date: 2026-09-13 13:43:19
description: "An RLVR pipeline on Qwen2.5-0.5B-Instruct, published before the training run. GRPO learns from the spread of rewards rather than their level, and four bugs never raised an exception."
tags:
  - reinforcement-learning
  - llm
  - grpo
---

This RLVR pipeline produced a diagnosis and four post-mortems. The diagnosis is that the learning signal in GRPO<sup>[[1]](#ref-1)</sup> is the within-group *spread* of rewards rather than their level, which makes "my dataset is too easy" and "my dataset is too hard" indistinguishable from outside the reward function. The post-mortems cover four engineering failures (a token-id collision, non-terminating rollouts, generation running in train mode, and a silently frozen adapter), none of which raised an exception, and all of which produced a training run that looked healthy.

We also measured hardware instead of assuming it. For this workload a rented Modal L4 ran about 5% faster per step than an M4 Pro laptop, because per-step time is dominated by serial decode overhead rather than arithmetic. That is one paired observation, not a benchmark.

*Reading note.* The argument is the variance diagnostic and why the four bugs share a shape.

## Why It Matters

Reinforcement learning from verifiable rewards<sup>[[2]](#ref-2)</sup> is the cleanest setup in post-training. There is no reward model to train, no human preference data, and no judge model to be gamed. A math problem has a right answer, a program checks it, and the check *is* the reward. It is the recipe behind the reasoning-model results of the last two years<sup>[[3]](#ref-3)</sup>, and at 500M it fits on one GPU.

That cleanliness is also what makes the failure modes legible. When the only moving parts are sample, score, and update, anything that goes wrong is in one of three places, which is why this project is worth writing up even without a training curve.

## Technical Details

The setup is GRPO via TRL 0.16<sup>[[4]](#ref-4)</sup> on **Qwen2.5-0.5B-Instruct**<sup>[[5]](#ref-5)</sup>, LoRA<sup>[[6]](#ref-6)</sup> ($r = 32$, $\alpha = 64$, dropout 0.05, all projection modules) over a frozen base, `math_verify`<sup>[[7]](#ref-7)</sup> as the reward, 8 generations per prompt, learning rate $10^{-6}$, $\beta = 0.04$, temperature 0.9, completions capped at 640 tokens.

<figure class="excal" data-diagram="rlvr-pipeline"><a href="/img/diagrams/rlvr-pipeline.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/rlvr-pipeline.webp" alt="GRPO pipeline on one Modal L4 GPU with no vLLM: prompts from the data.py loaders and a frozen Qwen2.5-0.5B-Instruct with LoRA (r=32, alpha=64) feed a rollout of 8 completions per prompt; correctness_reward (last boxed answer checked by math_verify, which prints reward mean and std) and format_reward feed a group advantage that is zero when std is 0, then a GRPO update of the LoRA weights only (lr 1e-6, KL beta 0.04) loops back to the policy, save_model writes the adapter to the volume, a dashed arrow reloads it with is_trainable=True, and step4_eval.py compares base against adapted." width="2400" height="2474" loading="lazy" decoding="async"></a></figure>

### GRPO, and Where the Signal Comes From

For each prompt, sample a group of $k$ completions and score them all. The advantage of completion $i$ is its reward minus the group mean, normalized by the group's spread.

$$
A_i = \frac{r_i - \mu}{\sigma}, \qquad
\mu = \frac{1}{k}\sum_{j=1}^{k} r_j, \qquad
\sigma = \sqrt{\frac{1}{k}\sum_{j=1}^{k}(r_j - \mu)^2}
$$

The policy is then pushed toward the above-average members of its own group. There is no value network and no critic to train. The group average *is* the baseline. That is the entire simplification, and at this scale it is the reason to use GRPO rather than PPO<sup>[[8]](#ref-8)</sup>.

Read the numerator once more, because the dominant failure mode falls directly out of it. If every completion in a group receives the same reward, then $r_i = \mu$ for all $i$, every advantage is zero, and the gradient contribution of that prompt is exactly zero. You spend a GPU-minute generating 8 completions and learn nothing from them.

The condition for learning is therefore $\sigma > 0$ within the group, and that is a property of the *difficulty distribution of the prompts*, not of the model's competence in any absolute sense.

<figure data-figure="grpo-advantage"></figure>

Zero within-group variance arrives from both directions, and the two are indistinguishable downstream.

| Training set | What happens | Reward std | Learning signal |
| --- | --- | --- | --- |
| GSM8K<sup>[[9]](#ref-9)</sup> | The Instruct model aces the problem 8/8 | $\approx 0$ | None |
| AIME | 90 olympiad problems, 0/8 on nearly all | $\approx 0$ | None |
| MATH<sup>[[10]](#ref-10)</sup> levels 3–5 | Sometimes right, sometimes wrong | $> 0$ | Yes |

"Too easy" and "too hard" produce the *identical* symptom. Flat reward, no movement, and a run that otherwise looks completely healthy. Wall-clock burns, the loss curve is plausible, and nothing improves. Any diagnosis that only asks whether reward is increasing cannot separate them, and the two have opposite fixes.

### The Verifier Is Part of the Objective

The other half of RLVR is deciding whether a completion's answer matches the gold answer. Completions are prompted to end in `\boxed{...}`, so extraction is a regex on the last match. Comparison is where the subtlety lives.

**String Equality vs. Symbolic Comparison**

| Model's boxed answer | Gold | String equality | `math_verify` |
| --- | --- | --- | --- |
| `72` | `72` | correct | correct |
| `0.5` | `1/2` | wrong | correct |
| `\frac{1}{2}` | `1/2` | wrong | correct |
| `40` | `72` | wrong | wrong |
| *(no `\boxed{}`)* | `72` | none | wrong |

Rows two and three are the reason this matters. A naive string check produces a **biased** reward. It systematically punishes correct answers for being written in a different form, and therefore trains the model toward the dataset's formatting conventions rather than toward being right. Parsing both sides symbolically and checking mathematical equality is the only version that rewards the intended thing.

Stated generally, the verifier *is* the objective. Every systematic error in it becomes a systematic pressure on the policy.

## Implementation

### The Diagnostic

So the reward function prints its own variance on every batch.

```python
n = len(rewards)
mean = sum(rewards) / n if n else 0.0
var = sum((r - mean) ** 2 for r in rewards) / n if n else 0.0
print(f"[reward] n={n} mean={mean:.3f} std={var ** 0.5:.3f} "
      f"(std~0 => no learning signal)")
```

Ten lines of arithmetic that convert a silent failure into a visible one. It is the single most useful thing in the pipeline. The general form is that **the learning signal in policy-gradient RL comes from the spread of outcomes, not their level.** That makes curriculum design a precondition here.

### Training Configuration

| Parameter | Value |
| --- | --- |
| Base model | Qwen2.5-0.5B-Instruct |
| Adapter | LoRA, $r = 32$, $\alpha = 64$, dropout 0.05, all projection modules |
| Generations per prompt ($k$) | 8 |
| Learning rate | $10^{-6}$ |
| KL coefficient ($\beta$) | 0.04 |
| Sampling temperature | 0.9 |
| Max completion length | 640 tokens |
| Correctness reward | 1.0 if `math_verify` accepts, else 0.0 |
| Format shaping | 0.2 for exactly one well-formed `\boxed{}` |
| Planned horizon | 300 steps (not launched) |

The format shaping term exists so that early in training there is a gradient toward parseable output before there is any gradient toward correctness. It is deliberately small relative to the correctness reward, so it cannot dominate once completions parse.

## Problems

None of these raised an exception. All four produced a training run that appeared to work. They share one shape. **Two components with defensible independent defaults must agree, and don't.** That is the most common category of bug we hit in this project.

<figure class="excal" data-diagram="rlvr-silent-failure-map"><a href="/img/diagrams/rlvr-silent-failure-map.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/rlvr-silent-failure-map.webp" alt="Diagram of one GRPO training step from tokenizer through rollout, completion mask, reward and LoRA update, with four numbered red badges marking the seams where the eos/pad mismatch, missing stop criterion, no-KV-cache generate, and frozen adapter each failed silently." width="2400" height="2488" loading="lazy" decoding="async"></a></figure>

### 1. The EOS/Pad Collision

Instruct models set `eos_token` to `<|im_end|>`. HuggingFace Transformers<sup>[[11]](#ref-11)</sup> pads stopped rollouts with `pad_token`. TRL masks each completion at the first `eos`. **If those two tokens disagree, the completion mask is wrong, and nothing tells you, because a wrong mask is still a valid mask.**

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tokenizer.eos_token = tokenizer.pad_token  # both <|endoftext|>
```

One line, several hours of diagnosis.

### 2. Rollouts That Never Terminate

We started on the Qwen2.5-Math-1.5B<sup>[[12]](#ref-12)</sup> *base* model. Its chat tokens are untrained, so it never emits `<|im_end|>` at all, so every rollout ran to the completion-length cap. At 8 generations per prompt and 640 tokens each, **nearly all of the compute was spent generating text after the answer had already been given.**

Two fixes followed: switching to the 500M Instruct model, which also roughly halved per-step time, and adding a stopping criterion that halts a rollout as soon as it contains a closed `\boxed{}`.

```python
class _StopAfterBoxed(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_len:], skip_special_tokens=True
        )
        return torch.tensor([BOXED_RE.search(t) is not None for t in texts],
                            device=input_ids.device)
```

The prompt itself mentions `\boxed{}`, so the criterion has to skip the prompt prefix, the kind of detail that turns a clean idea into a debugging session.

### 3. Generation Running in Train Mode

TRL 0.16 calls `generate()` without switching the model to eval. With gradient checkpointing enabled, that forces `use_cache=False`. **No KV cache means decoding is quadratic in sequence length rather than linear, on the single most expensive part of the step.**

```python
was_training = model.training
model.eval()
try:
    return orig_generate(input_ids, *args, **kwargs)
finally:
    if was_training:
        model.train()
```

A large speedup for a small patch, and again not something that surfaces as an error. It surfaces as "training is slower than I expected," which is indistinguishable from "training is this slow."

### 4. The Adapter That Loaded Frozen

If an adapter already exists at the output directory, the trainer loads it through PEFT<sup>[[13]](#ref-13)</sup> with `is_trainable=True` and continues, rather than starting a fresh LoRA.

```python
model = PeftModel.from_pretrained(base, args.output_dir, is_trainable=True)
```

**Omitting `is_trainable` loads the adapter frozen.** Training then updates nothing, reports no error, and produces a complete run with a plausible log.

A fifth issue is unresolved rather than fixed. TRL 0.16's vLLM<sup>[[14]](#ref-14)</sup> integration is server-based and wants a second GPU, and single-GPU colocation requires TRL 0.18 or later. The runs therefore use HuggingFace `generate` on one L4 and accept the throughput. That decision is what the next section measures.

## Experiments

### The Hardware Measurement

| Setup | Time per step |
| --- | --- |
| M4 Pro laptop (MPS) | ~77 s |
| Modal L4 (CUDA, bf16) | ~73 s |

The rented datacenter GPU was roughly 5% faster than the laptop. At 300 steps that is about 6 to 6.5 hours either way.

This is a single paired observation (one configuration, one run on each device, no repetitions), so it carries no interval and should not be read as a benchmark. What it does support is a directional claim with a mechanism behind it. **Per-step time is dominated by generation overhead in HuggingFace `generate`, not by matrix multiplication.** Sampling 8 completions of up to 640 tokens is a long serial sequence of small, latency-bound decode steps, and a wider GPU does not shorten a serial loop.

The consequences follow from the mechanism rather than from the 5%. Larger GPUs cost proportionally more per hour for roughly the same wall-clock, and a T4 is a false economy because it lacks bf16 and loses more time than it saves in price. The real fix is batched inference through vLLM, continuous batching<sup>[[15]](#ref-15)</sup> and paged attention<sup>[[14]](#ref-14)</sup>, which is blocked on the TRL version above.

## Results

### What Exists

**We have no before/after RL numbers.** The pipeline runs end to end, the 300-step training run has not been launched, and we are publishing before it deliberately. Nothing below is contingent on what the final $\text{pass@}1$ turns out to be.

| Component | State |
| --- | --- |
| Data loaders (GSM8K, MATH, AIME → prompt/answer pairs) | Done |
| Verifier and reward-variance diagnostic | Done |
| GRPO training with LoRA, adapter-resume across runs | Done |
| $\text{pass@}1$ / $\text{pass@}k$ evaluation, base and adapted | Done |
| Modal deployment, persistent volume, wandb | Done |
| 300-step training run | Not launched |
| Before/after results | None |

### Conclusions

Had we reported only that the pipeline was built, this post would have described a working RLVR stack. The honest version is that the stack is built and untested against its own objective, and that everything of value so far came out of it refusing to work.

- **Reward variance is the metric to watch, not reward mean.** Printing $\sigma$ every batch is ten lines and converts the most likely failure from invisible to obvious.
- **Too-easy and too-hard are the same observation from outside.** Any diagnosis that checks only whether reward is increasing cannot separate them, and they have opposite fixes.
- **The verifier is the objective.** A string-equality check is a biased reward function, not an approximate one.
- **Every failure here was silent.** EOS/pad mismatch, non-terminating rollouts, generation in train mode, a frozen adapter. Four bugs, zero exceptions.
- **Measure before renting.** For this workload the GPU bought about 5%, on one paired observation, because the bottleneck is serial decode rather than arithmetic.
- **The difficulty band is unmeasured.** Levels 3–5 produce nonzero variance; we have not swept difficulty against $\text{pass@}8$, so the curriculum is a working choice rather than a tuned one.

The open question is whether 300 steps on a 500M model moves $\text{pass@}1$ measurably at all. The frame we will judge it against is the $\text{pass@}k \gg \text{pass@}1$ gap. A model that solves a problem 1 time in 8 already contains the capability, and RL's job is to shift probability mass onto reasoning it can already occasionally produce rather than to teach it something new<sup>[[16]](#ref-16)</sup>. If the gap does not narrow, the report will say that it did not.

Find the code at [github.com/SadeekFarhan21/projects/math-rlvr](https://github.com/SadeekFarhan21/projects/tree/main/math-rlvr).

## What I Would Change

The claim above is that MATH levels 3–5 produce nonzero within-group variance and the alternatives did not. That is an observation from the diagnostic, not a characterization of the difficulty band.

We did not measure where the band actually sits for this model. The principled version sweeps difficulty against measured $\text{pass@}8$<sup>[[17]](#ref-17)</sup> and selects problems whose success probability is near $0.5$, which is where the expected within-group spread is largest. For a binary reward with success probability $p$, the group variance is maximized at $p = 0.5$, since

$$
\sigma^2 = p(1-p).
$$

That sweep has not been run, so "levels 3–5" is a working choice supported by a binary observation, not a tuned curriculum. It is the first thing we would run given more time, and it is cheap relative to training.

## References

1. <span id="ref-1"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint, 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
2. <span id="ref-2"></span>Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al. *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. COLM, 2025. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
3. <span id="ref-3"></span>DeepSeek-AI, Daya Guo, Dejian Yang, et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. Nature, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
4. <span id="ref-4"></span>Leandro von Werra, Younes Belkada, Lewis Tunstall, et al. *TRL: Transformers Reinforcement Learning*. Hugging Face, 2020. [link](https://github.com/huggingface/trl)
5. <span id="ref-5"></span>Qwen: An Yang, Baosong Yang, Beichen Zhang, et al. *Qwen2.5 Technical Report*. arXiv preprint, 2024. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
6. <span id="ref-6"></span>Edward J. Hu, Yelong Shen, Phillip Wallis, et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR, 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
7. <span id="ref-7"></span>Hynek Kydlíček. *Math-Verify: Math Verification Library*. Hugging Face, 2025. [link](https://github.com/huggingface/Math-Verify)
8. <span id="ref-8"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, et al. *Proximal Policy Optimization Algorithms*. arXiv preprint, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
9. <span id="ref-9"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. *Training Verifiers to Solve Math Word Problems*. arXiv preprint, 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
10. <span id="ref-10"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, et al. *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS Datasets and Benchmarks, 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874)
11. <span id="ref-11"></span>Thomas Wolf, Lysandre Debut, Victor Sanh, et al. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP System Demonstrations, 2020. [doi:10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
12. <span id="ref-12"></span>An Yang, Beichen Zhang, Binyuan Hui, et al. *Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*. arXiv preprint, 2024. [arXiv:2409.12122](https://arxiv.org/abs/2409.12122)
13. <span id="ref-13"></span>Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, et al. *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods*. Hugging Face, 2022. [link](https://github.com/huggingface/peft)
14. <span id="ref-14"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP, 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
15. <span id="ref-15"></span>Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI, 2022. [link](https://www.usenix.org/conference/osdi22/presentation/yu)
16. <span id="ref-16"></span>Yang Yue, Zhiqi Chen, Rui Lu, et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* NeurIPS, 2025. [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)
17. <span id="ref-17"></span>Mark Chen, Jerry Tworek, Heewoo Jun, et al. *Evaluating Large Language Models Trained on Code*. arXiv preprint, 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
