---
layout: page
title: "Inference Time Scaling Algorithm"
permalink: /writeups/scaling/
---

# rEVOLVE

### A simple inference time scaling algorithm with SOTA results

This write up presents a naive inference time algorithm for using LLMs to answer hard optimisation problems. The algorithm achieves a SOTA score on the *26 circle packing problem* (one of the problems used to prove the merits of AlphaEvolve).








>[!info] Note on dates
>I am writing this on 2025-09-27 but the work was done in June. The SOTA score was achieved on 2025-06-22. While I am not sure I can prove that date, I believe the Github Verified commit with hash `f325d9d210c65e810c471e7c7b12edc062422e34` proves that it was done before 11th July.
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




















2025-06-22_00-13-25_circle_pac
