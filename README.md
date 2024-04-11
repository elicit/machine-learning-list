# Elicit Machine Learning Reading List

## Purpose

The purpose of this curriculum is to help new [Elicit](https://elicit.com/) employees learn background in machine learning, with a focus on language models. I’ve tried to strike a balance between papers that are relevant for deploying ML in production and techniques that matter for longer-term scalability.

If you don’t work at Elicit yet - we’re [hiring ML and software engineers](https://elicit.com/careers).

## How to read

Recommended reading order:

1. Read “Tier 1” for all topics
2. Read “Tier 2” for all topics
3. Etc

✨ Added after 2024/4/1

## Table of contents

- [Fundamentals](#fundamentals)
  * [Introduction to machine learning](#introduction-to-machine-learning)
  * [Transformers](#transformers)
  * [Key foundation model architectures](#key-foundation-model-architectures)
  * [Training and finetuning](#training-and-finetuning)
- [Reasoning and runtime strategies](#reasoning-and-runtime-strategies)
  * [In-context reasoning](#in-context-reasoning)
  * [Task decomposition](#task-decomposition)
  * [Debate](#debate)
  * [Tool use and scaffolding](#tool-use-and-scaffolding)
  * [Honesty, factuality, and epistemics](#honesty-factuality-and-epistemics)
- [Applications](#applications)
  * [Science](#science)
  * [Forecasting](#forecasting)
  * [Search and ranking](#search-and-ranking)
- [ML in practice](#ml-in-practice)
  * [Production deployment](#production-deployment)
  * [Benchmarks](#benchmarks)
  * [Datasets](#datasets)
- [Advanced topics](#advanced-topics)
  * [World models and causality](#world-models-and-causality)
  * [Planning](#planning)
  * [Uncertainty, calibration, and active learning](#uncertainty-calibration-and-active-learning)
  * [Interpretability and model editing](#interpretability-and-model-editing)
  * [Reinforcement learning](#reinforcement-learning)
- [The big picture](#the-big-picture)
  * [AI scaling](#ai-scaling)
  * [AI safety](#ai-safety)
  * [Economic and social impacts](#economic-and-social-impacts)
  * [Philosophy](#philosophy)
- [Maintainer](#maintainer)

## Fundamentals

### Introduction to machine learning

**Tier 1**

- [A short introduction to machine learning](https://www.alignmentforum.org/posts/qE73pqxAZmeACsAdF/a-short-introduction-to-machine-learning)
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk&t=0s)
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)

**Tier 2**

- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- [An introduction to deep reinforcement learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)

**Tier 3**

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)

### Transformers

**Tier 1**

- ✨ [But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M)
- ✨ [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [the transformer ... “explained”?](https://nostalgebraist.tumblr.com/post/185326092369/the-transformer-explained)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)

**Tier 2**

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [A Practical Survey on Faster and Lighter Transformers](https://arxiv.org/abs/2103.14636)

**Tier 3**

- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks](https://arxiv.org/abs/2311.12997)
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)

</details>

### Key foundation model architectures

**Tier 1**

- [Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe) (GPT-2)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)

**Tier 2**

- ✨ [LLaMA: Open and Efficient Foundation Language Models](http://arxiv.org/abs/2302.13971) (LLaMA)
- ✨ [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v1) (Mamba)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (T5)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (OpenAI Codex)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI Instruct)

**Tier 3**

- ✨ [Mistral 7B](http://arxiv.org/abs/2310.06825) (Mistral)
- ✨ [Mixtral of Experts](http://arxiv.org/abs/2401.04088) (Mixtral)
- ✨ [Gemini: A Family of Highly Capable Multimodal Models](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) (Gemini)
- ✨ [Textbooks Are All You Need II: phi-1.5 technical report](http://arxiv.org/abs/2309.05463) (phi 1.5)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Flan)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Consistency Models](http://arxiv.org/abs/2303.01469)
- ✨ [Model Card and Evaluations for Claude Models](https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226/Model-Card-Claude-2.pdf) (Claude 2)
- ✨ [OLMo: Accelerating the Science of Language Models](http://arxiv.org/abs/2402.00838)
- ✨ [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403) (Palm 2)
- ✨ [Visual Instruction Tuning](http://arxiv.org/abs/2304.08485) (LLaVA)
- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Google Instruct)
- [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085)
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) (Google Dialog)
- [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2112.11446) (Meta GPT-3)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (PaLM)
- [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) (Google Codex)
- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) (Gopher)
- [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) (Minerva)
- [UL2: Unifying Language Learning Paradigms](http://aima.cs.berkeley.edu/) (UL2)

</details>

### Training and finetuning

**Tier 2**

- ✨ [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Learning to summarise with human feedback](https://arxiv.org/abs/2009.01325)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

**Tier 3**

- ✨ [Pretraining Language Models with Human Preferences](http://arxiv.org/abs/2302.08582)
- ✨ [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](http://arxiv.org/abs/2312.09390)
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638v1)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Unsupervised Neural Machine Translation with Generative Language Models Only](https://arxiv.org/abs/2110.05448)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](http://arxiv.org/abs/2312.06585)
- ✨ [Improving Code Generation by Training with Natural Language Feedback](http://arxiv.org/abs/2303.16749)
- ✨ [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668v1)
- ✨ [LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206)
- ✨ [Learning to Compress Prompts with Gist Tokens](http://arxiv.org/abs/2304.08467)
- ✨ [Lost in the Middle: How Language Models Use Long Contexts](http://arxiv.org/abs/2307.03172)
- ✨ [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314)
- ✨ [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](http://arxiv.org/abs/2403.09629)
- ✨ [Reinforced Self-Training (ReST) for Language Modeling](http://arxiv.org/abs/2308.08998)
- ✨ [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)
- ✨ [Tell, don't show: Declarative facts influence how LLMs generalize](http://arxiv.org/abs/2312.07779)
- ✨ [Textbooks Are All You Need](http://arxiv.org/abs/2306.11644)
- ✨ [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](http://arxiv.org/abs/2305.07759)
- ✨ [Training Language Models with Language Feedback at Scale](http://arxiv.org/abs/2303.16755)
- ✨ [Turing Complete Transformers: Two Transformers Are More Powerful Than One](https://openreview.net/forum?id=MGWsPGogLH)
- [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626)
- [Data Distributional Properties Drive Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2205.05055)
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
- [ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- [ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning](https://arxiv.org/abs/2111.10952)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://arxiv.org/abs/2106.02584)
- [True Few-Shot Learning with Prompts -- A Real-World Perspective](https://arxiv.org/abs/2111.13440)

</details>

## Reasoning and runtime strategies

### In-context reasoning

**Tier 2**

- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (Let's think step by step)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

**Tier 3**

- ✨ [Chain-of-Thought Reasoning Without Prompting](http://arxiv.org/abs/2402.10200)
- ✨ [Why think step-by-step? Reasoning emerges from the locality of experience](http://arxiv.org/abs/2304.03843)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Baldur: Whole-Proof Generation and Repair with Large Language Models](https://arxiv.org/abs/2303.04910v1)
- ✨ [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](http://arxiv.org/abs/2403.05518)
- ✨ [Certified Reasoning with Language Models](http://arxiv.org/abs/2306.04031)
- ✨ [Hypothesis Search: Inductive Reasoning with Language Models](http://arxiv.org/abs/2309.05660)
- ✨ [LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations](http://arxiv.org/abs/2305.18354)
- ✨ [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798v1)
- ✨ [Stream of Search (SoS): Learning to Search in Language](http://arxiv.org/abs/2404.03683)
- ✨ [Training Chain-of-Thought via Latent-Variable Inference](http://arxiv.org/abs/2312.02179)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
- [Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right](https://arxiv.org/abs/2104.08315)

</details>

### Task decomposition

**Tier 1**

- [Supervise Process, not Outcomes](https://ought.org/updates/2022-04-06-process)
- [Supervising strong learners by amplifying weak experts](https://arxiv.org/abs/1810.08575)

**Tier 2**

- ✨ [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](http://arxiv.org/abs/2305.10601)
- [Factored cognition](https://ought.org/research/factored-cognition)
- [Iterated Distillation and Amplification](https://ai-alignment.com/iterated-distillation-and-amplification-157debfd1616)
- [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862)
- [Solving math word problems with process-based and outcome-based feedback](https://arxiv.org/abs/2211.14275)

**Tier 3**

- [Evaluating Arguments One Step at a Time](https://ought.org/updates/2020-01-11-arguments)
- [Faithful Reasoning Using Large Language Models](https://arxiv.org/abs/2208.14271)
- [Humans consulting HCH](https://ai-alignment.com/humans-consulting-hch-f893f6051455)
- [Iterated Decomposition: Improving Science Q&A by Supervising Reasoning Processes](https://arxiv.org/abs/2301.01751)
- [Language Model Cascades](https://arxiv.org/abs/2207.10342)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Decontextualization: Making Sentences Stand-Alone](https://doi.org/10.1162/tacl_a_00377)
- ✨ [Factored Cognition Primer](https://primer.ought.org)
- ✨ [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](http://arxiv.org/abs/2308.09687)
- ✨ [Parsel: A Unified Natural Language Framework for Algorithmic Reasoning](http://arxiv.org/abs/2212.10561)
- [AI Chains: Transparent and Controllable Human-AI Interaction by Chaining Large Language Model Prompts](https://arxiv.org/abs/2110.01691)
- [Challenging BIG-Bench tasks and whether chain-of-thought can solve them](https://arxiv.org/abs/2210.09261)
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.11822)
- [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)
- [Measuring and narrowing the compositionality gap in language models](https://arxiv.org/abs/2210.03350)
- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning](https://arxiv.org/abs/2205.10625)
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114)
- [Summ^N: A Multi-Stage Summarization Framework for Long Input Dialogues and Documents](https://arxiv.org/abs/2110.10150)
- [Thinksum: probabilistic reasoning over sets using large language models](https://arxiv.org/abs/2210.01293)

</details>

### Debate

**Tier 2**

- [AI safety via debate](https://openai.com/blog/debate/)

**Tier 3**

- ✨ [Debate Helps Supervise Unreliable Experts](https://twitter.com/joshua_clymer/status/1724851456967417872)
- [Two-Turn Debate Doesn’t Help Humans Answer Hard Reading Comprehension Questions](https://arxiv.org/abs/2210.10860)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Scalable AI Safety via Doubly-Efficient Debate](http://arxiv.org/abs/2311.14125)
- ✨ [Improving Factuality and Reasoning in Language Models through Multiagent Debate](http://arxiv.org/abs/2305.14325)

</details>

### Tool use and scaffolding

**Tier 2**

- ✨ [Measuring the impact of post-training enhancements](https://metr.github.io/autonomy-evals-guide/elicitation-gap/)
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)

**Tier 3**

- ✨ [AI capabilities can be significantly improved without expensive retraining](http://arxiv.org/abs/2312.07413)
- ✨ [Automated Statistical Model Discovery with Language Models](http://arxiv.org/abs/2402.17879)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](http://arxiv.org/abs/2310.03714)
- ✨ [Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution](http://arxiv.org/abs/2309.16797)
- ✨ [Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation](https://arxiv.org/abs/2310.02304v1)
- ✨ [Voyager: An Open-Ended Embodied Agent with Large Language Models](http://arxiv.org/abs/2305.16291)
- [ReGAL: Refactoring Programs to Discover Generalizable Abstractions](http://arxiv.org/abs/2401.16467)

</details>

### Honesty, factuality, and epistemics

**Tier 2**

- ✨ [Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2206.05802v2)

**Tier 3**

- ✨ [What Evidence Do Language Models Find Convincing?](http://arxiv.org/abs/2402.11782)
- ✨ [How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions](https://arxiv.org/abs/2309.15840)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](http://arxiv.org/abs/2305.04388)
- ✨ [Long-form factuality in large language models](http://arxiv.org/abs/2403.18802)

</details>

## Applications

### Science

**Tier 3**

- ✨ [Can large language models provide useful feedback on research papers? A large-scale empirical analysis](http://arxiv.org/abs/2310.01783)
- ✨ [Large Language Models Encode Clinical Knowledge](http://arxiv.org/abs/2212.13138)
- ✨ [The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4](http://arxiv.org/abs/2311.07361)
- [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://arxiv.org/abs/2105.03011)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](http://arxiv.org/abs/2311.16452)
- ✨ [Nougat: Neural Optical Understanding for Academic Documents](http://arxiv.org/abs/2308.13418)
- ✨ [Scim: Intelligent Skimming Support for Scientific Papers](http://arxiv.org/abs/2205.04561)
- ✨ [SynerGPT: In-Context Learning for Personalized Drug Synergy Prediction and Drug Design](https://www.biorxiv.org/content/10.1101/2023.07.06.547759v1)
- ✨ [Towards Accurate Differential Diagnosis with Large Language Models](http://arxiv.org/abs/2312.00164)
- ✨ [Towards a Benchmark for Scientific Understanding in Humans and Machines](http://arxiv.org/abs/2304.10327)
- [A Search Engine for Discovery of Scientific Challenges and Directions](https://arxiv.org/abs/2108.13751)
- [A full systematic review was completed in 2 weeks using automation tools: a case study](https://pubmed.ncbi.nlm.nih.gov/32004673/)
- [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974)
- [Multi-XScience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles](https://arxiv.org/abs/2010.14235)
- [PEER: A Collaborative Language Model](https://arxiv.org/abs/2208.11663)
- [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146)
- [SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts](https://arxiv.org/abs/2104.08809)
- [SciTail: A Textual Entailment Dataset from Science Question Answering](http://ai2-website.s3.amazonaws.com/team/ashishs/scitail-aaai2018.pdf)

</details>

### Forecasting

**Tier 3**

- ✨ [AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy](https://arxiv.org/abs/2402.07862v1)
- ✨ [Approaching Human-Level Forecasting with Language Models](http://arxiv.org/abs/2402.18563)
- ✨ [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- [Forecasting Future World Events with Neural Networks](https://arxiv.org/abs/2206.15474)

### Search and ranking

**Tier 2**

- [Learning Dense Representations of Phrases at Scale](https://arxiv.org/abs/2012.12624)
- [Text and Code Embeddings by Contrastive Pre-Training](https://arxiv.org/abs/2201.10005) (OpenAI embeddings)

**Tier 3**

- ✨ [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](http://arxiv.org/abs/2306.17563)
- [Not All Vector Databases Are Made Equal](https://dmitry-kan.medium.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Task-aware Retrieval with Instructions](https://arxiv.org/abs/2211.09260)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!](http://arxiv.org/abs/2312.02724)
- ✨ [Some Common Mistakes In IR Evaluation, And How They Can Be Avoided](https://dl.acm.org/doi/10.1145/3190580.3190586)
- [Boosting Search Engines with Interactive Agents](https://arxiv.org/abs/2109.00527)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking](https://arxiv.org/abs/2212.01340)
- [UnifiedQA: Crossing Format Boundaries With a Single QA System](https://arxiv.org/abs/2005.00700)

</details>


## ML in practice

### Production deployment

**Tier 1**

- [Machine Learning in Python: Main developments and technology trends in data science, machine learning, and AI](https://arxiv.org/abs/2002.04803v2)
- [Machine Learning: The High Interest Credit Card of Technical Debt](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

**Tier 2**

- ✨ [Designing Data-Intensive Applications](https://dataintensive.net/)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)

### Benchmarks

**Tier 2**

- ✨ [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](http://arxiv.org/abs/2311.12022)
- ✨ [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770v1)
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)

**Tier 3**

- [FLEX: Unifying Evaluation for Few-Shot NLP](https://arxiv.org/abs/2107.07170)
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2107.07170) (HELM)
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [RAFT: A Real-World Few-Shot Text Classification Benchmark](https://arxiv.org/abs/2109.14076)
- [True Few-Shot Learning with Language Models](https://arxiv.org/abs/2105.11447)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [GAIA: a benchmark for General AI Assistants](http://arxiv.org/abs/2311.12983)
- [ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers](https://arxiv.org/abs/2110.06884)
- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
- [QuALITY: Question Answering with Long Input Texts, Yes!](https://arxiv.org/abs/2112.08608)
- [SCROLLS: Standardized CompaRison Over Long Language Sequences](https://arxiv.org/abs/2201.03533)
- [What Will it Take to Fix Benchmarking in Natural Language Understanding?](https://arxiv.org/abs/2104.02145)

</details>

### Datasets

**Tier 2**

- [Common Crawl](https://arxiv.org/abs/2105.02732)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)

**Tier 3**

- [Dialog Inpainting: Turning Documents into Dialogs](https://arxiv.org/abs/2205.09073)
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)
- [Microsoft Academic Graph](https://internal-journal.frontiersin.org/articles/10.3389/fdata.2019.00045/full)
- [TLDR9+: A Large Scale Resource for Extreme Summarization of Social Media Posts](https://arxiv.org/abs/2110.01159)

## Advanced topics

### World models and causality

**Tier 3**

- ✨ [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](http://arxiv.org/abs/2210.13382)
- ✨ [From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought](http://arxiv.org/abs/2306.12672)
- [Language Models Represent Space and Time](http://arxiv.org/abs/2310.02207)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Amortizing intractable inference in large language models](http://arxiv.org/abs/2310.04363)
- ✨ [CLADDER: Assessing Causal Reasoning in Language Models](http://zhijing-jin.com/files/papers/CLadder_2023.pdf)
- ✨ [Causal Bayesian Optimization](https://proceedings.mlr.press/v108/aglietti20a.html)
- ✨ [Causal Reasoning and Large Language Models: Opening a New Frontier for Causality](http://arxiv.org/abs/2305.00050)
- ✨ [Generative Agents: Interactive Simulacra of Human Behavior](http://arxiv.org/abs/2304.03442)
- ✨ [Passive learning of active causal strategies in agents and language models](http://arxiv.org/abs/2305.16183)

</details>

### Planning

- ✨ [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](http://arxiv.org/abs/2402.14083)
- ✨ [Cognitive Architectures for Language Agents](http://arxiv.org/abs/2309.02427)

### Uncertainty, calibration, and active learning

**Tier 2**

- ✨ [Experts Don't Cheat: Learning What You Don't Know By Predicting Pairs](http://arxiv.org/abs/2402.08733)
- [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476)
- [Plex: Towards Reliability using Pretrained Large Model Extensions](https://arxiv.org/abs/2207.07411)

**Tier 3**

- ✨ [Active Preference Inference using Language Models and Probabilistic Reasoning](http://arxiv.org/abs/2312.12009)
- ✨ [Eliciting Human Preferences with Language Models](http://arxiv.org/abs/2310.11589)
- [Active Learning by Acquiring Contrastive Examples](https://arxiv.org/abs/2109.03764)
- [Describing Differences between Text Distributions with Natural Language](https://arxiv.org/abs/2201.12323)
- [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning](http://arxiv.org/abs/2402.06025)
- ✨ [STaR-GATE: Teaching Language Models to Ask Clarifying Questions](http://arxiv.org/abs/2403.19154)
- [Active Testing: Sample-Efficient Model Evaluation](https://arxiv.org/abs/2103.05331)
- [Uncertainty Estimation for Language Reward Models](https://arxiv.org/abs/2203.07472)

</details>

### Interpretability and model editing

**Tier 2**

- [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827v1)

**Tier 3**

- ✨ [Interpretability at Scale: Identifying Causal Mechanisms in Alpaca](http://arxiv.org/abs/2305.08809)
- ✨ [Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks](http://arxiv.org/abs/2311.12786)
- ✨ [Representation Engineering: A Top-Down Approach to AI Transparency](http://arxiv.org/abs/2310.01405)
- ✨ [Studying Large Language Model Generalization with Influence Functions](http://arxiv.org/abs/2308.03296)
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Codebook Features: Sparse and Discrete Interpretability for Neural Networks](http://arxiv.org/abs/2310.17230)
- ✨ [Eliciting Latent Predictions from Transformers with the Tuned Lens](http://arxiv.org/abs/2303.08112)
- ✨ [How do Language Models Bind Entities in Context?](http://arxiv.org/abs/2310.17191)
- ✨ [Opening the AI black box: program synthesis via mechanistic interpretability](https://arxiv.org/abs/2402.05110v1)
- ✨ [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](http://arxiv.org/abs/2403.19647)
- ✨ [Uncovering mesa-optimization algorithms in Transformers](http://arxiv.org/abs/2309.05858)
- [Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)
- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)

</details>

### Reinforcement learning

**Tier 2**

- ✨ [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](http://arxiv.org/abs/2305.18290)
- ✨ [Reflexion: Language Agents with Verbal Reinforcement Learning](http://arxiv.org/abs/2303.11366)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (AlphaZero)
- [MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

**Tier 3**

- ✨ [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](http://arxiv.org/abs/2307.15217)
- [AlphaStar: mastering the real-time strategy game StarCraft II](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210) (EfficientZero)
- [Mastering Stratego, the classic game of imperfect information](https://www.science.org/doi/10.1126/science.add4679) (DeepNash)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning](http://arxiv.org/abs/2308.03526)
- ✨ [Bayesian Reinforcement Learning with Limited Cognitive Load](http://arxiv.org/abs/2305.03263)
- ✨ [Contrastive Prefence Learning: Learning from Human Feedback without RL](http://arxiv.org/abs/2310.13639)
- ✨ [Grandmaster-Level Chess Without Search](http://arxiv.org/abs/2402.04494)
- [A data-driven approach for learning to control computers](https://arxiv.org/abs/2202.08137)
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/abs/2111.09259)
- [Player of Games](https://arxiv.org/abs/2112.03178)
- [Retrieval-Augmented Reinforcement Learning](https://arxiv.org/abs/2202.08417)

</details>

## The big picture

### AI scaling

**Tier 1**

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Takeoff speeds](https://sideways-view.com/2018/02/24/takeoff-speeds/)
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

**Tier 2**

- [AI and compute](https://openai.com/blog/ai-and-compute/)
- [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)

**Tier 3**

- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- [Transcending Scaling Laws with 0.1% Extra Compute](https://arxiv.org/abs/2210.11399) (U-PaLM)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](http://arxiv.org/abs/2404.05405)
- ✨ [Power Law Trends in Speedrunning and Machine Learning](http://arxiv.org/abs/2304.10004)
- ✨ [Scaling laws for single-agent reinforcement learning](http://arxiv.org/abs/2301.13442)
- [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486)
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113)

</details>

### AI safety

**Tier 1**

- [Three impacts of machine intelligence](https://www.effectivealtruism.org/articles/three-impacts-of-machine-intelligence-paul-christiano/)
- [What failure looks like](https://www.alignmentforum.org/posts/HBxe6wdjxK239zajf/what-failure-looks-like)
- [Without specific countermeasures, the easiest path to transformative AI likely leads to AI takeover](https://www.lesswrong.com/posts/pRkFkzwKZ2zfa3R6H/without-specific-countermeasures-the-easiest-path-to)

**Tier 2**

- ✨ [An Overview of Catastrophic AI Risks](http://arxiv.org/abs/2306.12001)
- [Clarifying “What failure looks like” (part 1)](https://www.lesswrong.com/posts/v6Q7T335KCMxujhZu/clarifying-what-failure-looks-like-part-1)
- [Deep RL from human preferences](https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/)
- [The alignment problem from a deep learning perspective](https://arxiv.org/abs/2209.00626)

**Tier 3**

- ✨ [Scheming AIs: Will AIs fake alignment during training in order to get power?](http://arxiv.org/abs/2311.08379)
- [Measuring Progress on Scalable Oversight for Large Language Models](https://arxiv.org/abs/2211.03540)
- [Risks from Learned Optimization in Advanced Machine Learning Systems](https://arxiv.org/abs/1906.01820)
- [Scalable agent alignment via reward modelling](https://arxiv.org/abs/1811.07871)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [AI Deception: A Survey of Examples, Risks, and Potential Solutions](http://arxiv.org/abs/2308.14752)
- ✨ [Benchmarks for Detecting Measurement Tampering](http://arxiv.org/abs/2308.15605)
- ✨ [Chess as a Testing Grounds for the Oracle Approach to AI Safety](http://arxiv.org/abs/2010.02911)
- ✨ [Close the Gates to an Inhuman Future: How and why we should choose to not develop superhuman general-purpose artificial intelligence](https://papers.ssrn.com/abstract=4608505)
- ✨ [Model evaluation for extreme risks](http://arxiv.org/abs/2305.15324)
- ✨ [Responsible Reporting for Frontier AI Development](http://arxiv.org/abs/2404.02675)
- ✨ [Safety Cases: How to Justify the Safety of Advanced AI Systems](http://arxiv.org/abs/2403.10462)
- ✨ [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](http://arxiv.org/abs/2401.05566)
- ✨ [Technical Report: Large Language Models can Strategically Deceive their Users when Put Under Pressure](http://arxiv.org/abs/2311.07590)
- ✨ [Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game](http://arxiv.org/abs/2311.01011)
- ✨ [Tools for Verifying Neural Models' Training Data](http://arxiv.org/abs/2307.00682)
- ✨ [Towards a Cautious Scientist AI with Convergent Safety Bounds](https://yoshuabengio.org/2024/02/26/towards-a-cautious-scientist-ai-with-convergent-safety-bounds/)
- [Alignment of Language Agents](https://arxiv.org/abs/2103.14659)
- [Eliciting Latent Knowledge](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit?usp=sharing)
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)
- [Red Teaming Language Models with Language Models](https://storage.googleapis.com/deepmind-media/Red%20Teaming/Red%20Teaming.pdf)
- [Unsolved Problems in ML Safety](https://arxiv.org/abs/2109.13916)

</details>

### Economic and social impacts

**Tier 3**

- ✨ [Explosive growth from AI automation: A review of the arguments](http://arxiv.org/abs/2309.11690)
- ✨ [Language Models Can Reduce Asymmetry in Information Markets](http://arxiv.org/abs/2403.14443)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero](http://arxiv.org/abs/2310.16410)
- ✨ [Foundation Models and Fair Use](https://arxiv.org/abs/2303.15715v1)
- ✨ [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](http://arxiv.org/abs/2303.10130)
- ✨ [Levels of AGI: Operationalizing Progress on the Path to AGI](http://arxiv.org/abs/2311.02462)
- ✨ [Opportunities and Risks of LLMs for Scalable Deliberation with Polis](http://arxiv.org/abs/2306.11932)
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)

</details>

### Philosophy

**Tier 2**

- [Meaning without reference in large language models](https://arxiv.org/abs/2208.02957)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Consciousness in Artificial Intelligence: Insights from the Science of Consciousness](http://arxiv.org/abs/2308.08708)
- ✨ [Philosophers Ought to Develop, Theorize About, and Use Philosophically Relevant AI](https://philarchive.org/archive/CLAPOT-16)
- ✨ [Towards Evaluating AI Systems for Moral Status Using Self-Reports](http://arxiv.org/abs/2311.08576)

</details>

## Maintainer

[andreas@elicit.com](mailto:andreas@elicit.com)

