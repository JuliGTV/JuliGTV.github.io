---
layout: default
title: "Inference Time Scaling Algorithm"
permalink: /writeups/scaling/
---


# rEVOLVE

### A simple inference time scaling algorithm with SOTA results



This write up presents a naive inference time algorithm for using LLMs to answer hard optimisation problems. The algorithm achieves a SOTA score on the *26 circle packing problem* (one of the problems used to prove the merits of AlphaEvolve).



Github repo: https://github.com/JuliGTV/rEVOLVE




>[!info] Note on dates
>I am writing this on 2025-09-27 but the work was done in June. The SOTA score was achieved on 2024-06-22. While I am not sure I can prove that date, I believe the Github Verified commit with hash `f325d9d210c65e810c471e7c7b12edc062422e34` proves that it was done before 11th July.
>(If not you will have to take my word for it since it's not very important!)



#### Similar works

This project was originally intended to be a reverse engineering of Google Deepmind's [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), and the existing open source implementation [OpenEvolve](https://github.com/codelion/openevolve).

The project ultimately ended up much more similar to what Sakana would later call [Adaptive-branching Monte-Carlo tree search](https://sakana.ai/ab-mcts/). 

Sakana have since released the much more efficient [ShinkaEvolve](https://sakana.ai/shinka-evolve/) which achieves a circle packing score virtually identical to mine (and uses about a 50th of the compute). 

There has been much work on various more primitive scaling algorithms such as repeated sampling and iterative generation.



## The algorithm

Inspired by the amazing results achieved by AlphaEvolve, I decided to see if I could create a similar algorithm. After spending some time trying to figure out what an evolutionary algorithm *"inspired by a combination of the MAP elites algorithm and island-based population models"* should look like I decided instead to simplify things. I boiled the approach down to what seemed to me the key principles:

1. Generate lots of solutions to the problem
2. Iterate by asking the LLM to improve the best ones
3. Balance exploration and exploitation

On this basis I came up an algorithm summarised here:

1. Before starting the user specifies the problem. This involves writing the prompt describing the problem to the LLM (including specifying the solution format e.g. *python function with the following name and type signature...*), a python function for evaluating solutions which returns a score, and a starting solution which can just be a dummy solution with the right format.
2. The starting solution is scored and put into a collection that stores solutions and scores. Then the following is repeated iteratively up to some chosen limit or until a target score is reached.
	1. Pick a past solution from the collection (I will discuss how this choice is made afterwards).
	2. Construct a prompt that specifies the problem then shows the past solution and its score and instructs the model to make improvements.
	3. Use this prompt to generate a new solution. Score it and add it to the collection.

So far this is incredibly general and could more or less be used to describe any of the notable scaling algorithms such as AlphaEvolve, ABMCTS, or my own. Most of the difference comes down to the implementation of steps 2.1 and 2.2.

For picking the next solution to improve I came up with three primitive strategies:

Greedy: Always expand the best solution so far
Random: Pick a solution uniformly at random
Weighted: Pick a solution at random using the scores as weights (subject to some appropriate normalisations)

I then constructed a hybrid strategy which consisted of picking one of these three at random, weighted with parameters chosen when initialising the search.

For 2.2 I simply used added the code and score of the previous solution to a prompt with a description of the problem, and instructions to improve the given solution. (more sophisticated approaches may show the full evolutionary history of solutions or show multiple performant solutions for combination).


Pseudocode:

```
Inputs:
  problem_spec, start_code
  scorer(code) -> numeric score   # fast, deterministic
  LLM(prompt) -> code
  p_elite, p_uniform in [0,1], with p_elite + p_uniform ≤ 1
  gamma ≥ 0, ε > 0
  K  # number of iterations/evals

Init:
  s0 = scorer(start_code)
  P  = [(start_code, s0)]
  best = (start_code, s0)

Loop (t = 1..K):
  # selection
  u = rand()
  if u < p_elite:
      parent = argmax(P, key=score)
  elif u < p_elite + p_uniform:
      parent = uniform_choice(P)
  else:
      weights = [max(ε, s)^gamma for (_, s) in P]
      parent = weighted_choice(P, weights)

  # improvement
  prompt = f"""
  Problem:
  {problem_spec}

  Current solution (score={parent.score}):
  {parent.code}

  Improve it. Keep determinism and runtime constraints.
  Output code only.
  """
  child_code  = LLM(prompt)
  child_score = scorer(child_code)

  P.append((child_code, child_score))
  if child_score > best.score:
      best = (child_code, child_score)

Return:
  best.code, best.score, P

```



## Results

This section covers the results I achieved with the algorithm described above. In particular I focus on the 26 circle packing problem, which asks the algorithm to pack 26 circles into a unit square optimally such as to maximise the sum of their radii, by writing a python function that will itself output the circles' centres and radii.

I chose this problem primarily because it was simply one of the most intelligible of the problems that AlphaEvolve had beaten the SOTA for. Others such as "Third autocorrelation inequality" are far more mathematically involved, and unlike the AlphaEvolve team I don't have Terrence Tao advising me on how to formulate my prompts!



#### Scaling is all you need

After exploring various permutations of model choice, prompt, and hyperparameters without much success, I noticed two things:
1. More powerful models (e.g. `gpt-4.1`) and reasoning models were not substantially better than lesser ones especially factoring for time and cost, although the weakest models (e.g. `gpt-4o-mini`) could not make progress at all.
2. While the rate of progress would rapidly diminish, new best solutions were still appearing near the ends of my runs (doing 100s of generations).

On this basis I decided to find the cheapest model that could still make progress, and then substantially scale up the size of my runs. I found that `Deepseek-V3` offered the best value at only $0.07/M input tokens and $1.10/M output tokens (and half price during Chinese off hours).

I modified my code to generate multiple new solutions in parallel and then launched a new run with 10,000 solutions generated. This run not only beat AlphaEvolve's score on the problem, but also beat the new SOTA that had since been set by FICO Xpress Solver (see table below) and as far as I can tell remains the state of the art to this day.


| Algorithm                                                                                                                         | Creator                  | Date        | Score              |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------- | ------------------ |
| [rEVOLVE](https://github.com/JuliGTV/rEVOLVE)                                                                                     | Me                       | 2025-06-22  | 2.6359828880000005 |
| [ShinkaEvolve](https://sakana.ai/shinka-evolve/)                                                                                  | Sakana                   | 2025-09-25  | 2.6359828390115476 |
| [OpenEvolve](https://github.com/codelion/openevolve/issues/156)                                                                   | Yiping Wang / OpenEvolve | 2025-07-21  | 2.635977           |
| [FICO Xpress Solver](https://www.fico.com/blogs/best-global-optimization-solver)                                                  | FICO                     | 2025-06-13  | 2.63591551         |
| [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | DeepMind                 | 2025-05-14  | 2.6358627564136983 |
| Pre-AlphaEvolve SOTA (baseline cited in paper & notebook)                                                                         | —                        | ≤2025-05-14 | 2.634              |




#### A note on numerical accuracy

The scorer I used to score circle packings was the implementation from the Openevolve library. This scorer allows for $1 \times 10^{-6}$ of numerical slack, so the solution created by rEVOLVE had circles that very slightly overlapped. To address this it sufficed to subtract $7 \times 10^{-9}$ from each radius diminishing the score from $2.6359830853311843$ to the reported $2.6359828880000005$.

Sakana mention in their paper also doing something similar reducing their score from $2.635983099011548$ to $2. 6359828390115476$. It may be that some of the other implementations in the table (especially the OpenEvolve one itself) still use the slacker criteria.


#### Limitations

- I struggled to generalise this result to other problems. I suspect a lot of this is just a question of prompt engineering, and iteration (so I was bottlenecked by compute budget and mathematical expertise)
- ShinkaEvolve's score (which is equal to mine up to ) was achieved with 150 LLM generations compared to my 10,000. Considering the scores themselves are virtually identical I think it would be fairer to call them the SOTA.

## Conclusion

This style of black box inference time scaling offers a powerful paradigm for using LLMs to make progress on genuinely hard problems. For such a naive implementation to achieve such a remarkable result is sign that there is likely much more yet to come from this area of research. 







