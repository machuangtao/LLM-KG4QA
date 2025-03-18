# LLM-KG4QA: Large Language Models and Knowledge Graphs for Question Answering

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![](https://img.shields.io/github/last-commit/machuangtao/LLM-KG4QA?color=blue) ![Stars](https://img.shields.io/github/stars/machuangtao/LLM-KG4QA?color=blue)  ![Forks](https://img.shields.io/github/forks/machuangtao/LLM-KG4QA?color=blue&label=Fork)

<!-- ## 🔔 News
- **`2025-02`** Our [tutorial](https://machuangtao.github.io/LLM-KG4QA/tutorial-edbt25) was accepted to be presented at **EDBT/ICDT 2025 joint conference.**
- **`2024-12`** We create this repository to maintain a paper list on **Large Language Models and Knowledge Graphs for Question Answering.** -->

## Content
- [LLM and KGs for QA](#1-llms-and-kgs-for-qa)
  - [KGs as Background Knowledge](#kgs-as-background-knowledge)
  - [KGs as Reasoning Guideline](#kgs-as-reasoning-guideline)
  - [KGs as Refiner and Filter](#kgs-as-refiner-and-filter)
  <!--- - [Hybrid Methods](#hybrid-methods) -->
- [Complex QA](#2-complex-qa)
  - [Explainable QA](#explainable-qa)
  - [Multi-modal QA](#multi-modal-qa)
  - [Multi-document QA](#multi-document-qa)
  - [Multi-Hop QA](#multi-hop-qa)
  - [Multi-run and Conversational QA](#multi-run-and-conversational-qa)
- [Advanced Topics](#3-advanced-topics)
  - [Optimization](#optimization)
  - [Data Management](#data-management)
- [Benchmark and Applications](#4-benchmark-and-applications)
  - [Benchmark Dataset](#benchmark-dataset)
  - [Industrial and Scientific Applications](#industrial-and-scientific-applications)
  - [Demo](#demo)
- [Related Survey](#5-related-survey)
---
## 1. LLMs and KGs for QA

### KGs as Background Knowledge
#### Pre-training and Fine-tuning

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Deep Bidirectional Language-Knowledge Graph pretraining   | NeurIPS | 2022|  Pre-training    | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/f224f056694bcfe465c5d84579785761-Paper-Conference.pdf)                             
| 2 | GreaseLM: Graph REASoning Enhanced Language Models    | ICLR    | 2022 |  Pre-training     | [Link](https://openreview.net/forum?id=41e9o6cQPj) 
| 3 | InfuserKI: Enhancing Large Language Models with Knowledge Graphs via Infuser-Guided Knowledge Integration  | LLM+KG@VLDB   | 2024 | Pre-training     | [Link](https://arxiv.org/abs/2402.11441)
| 4 | KaLM: Knowledge-aligned Autoregressive Language Modeling via Dual-view Knowledge Graph Contrastive Learning  | arXiv    | 2024 | Pre-training     | [Link](https://arxiv.org/abs/2412.04948)
| 5 | KnowLA: Enhancing Parameter-efficient Finetuning with Knowledgeable Adaptation  | NAACL   | 2024 |  Fine-Tuning     | [Link](https://aclanthology.org/2024.naacl-long.396/)
| 6 | KG-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning  | ACL Findlings  | 2024 | Fine-Tuning     | [Link](https://aclanthology.org/2024.findings-acl.229/) 
| 7 | A GAIL Fine-Tuned LLM Enhanced Framework for Low-Resource Knowledge Graph Question Answering  | CIKM    | 2024 |  Fine-Tuning    | [Link](https://dl.acm.org/doi/10.1145/3627673.3679753)
| 8 | Prompting Large Language Models with Knowledge Graphs for Question Answering Involving Long-tail Facts  | arXiv       | 2024 |  KG-Augmented Prompting    | [Link](https://arxiv.org/abs/2405.06524)
| 9 | KnowGPT: Knowledge Graph based Prompting for Large Language Models  | arXiv     | 2024 |  KG-Augmented Prompting    | [Link](https://arxiv.org/abs/2312.06185) 
| 10 | Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering  | NLRSE          | 2023 |  KG-Augmented Prompting    | [Link](https://aclanthology.org/2023.nlrse-1.7/)
| 11 | Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering | IJCKG    | 2023 |  KG-Augmented Prompting    | [Link](https://ijckg2023.knowledge-graph.jp/pages/proc/paper_30.pdf/)


#### RAG (Retrieval Augmented Generation)

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation| arXiv          | 2024 | RAG   | [Link](https://arxiv.org/abs/2402.05128)
| 2 | Retrieval-enhanced Knowledge Editing in Language Models for Multi-Hop Question Answering | CIKM   | 2024 |  RAG   | [Link](https://dl.acm.org/doi/abs/10.1145/3627673.3679722) 
| 3 | Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation | arXiv   | 2024 |  RAG   | [Link](https://arxiv.org/abs/2406.18676) 
| 4 | RAG-based Question Answering over Heterogeneous Data and Text | arXiv   | 2024 |  RAG   | [Link](https://arxiv.org/abs/2412.07420) 
| 5 | Awakening Augmented Generation: Learning to Awaken Internal Knowledge of Large Language Models for Question Answering | COLING   | 2025 |  RAG   | [Link](https://aclanthology.org/2025.coling-main.89/) 
| 6 | SAGE: A Framework of Precise Retrieval for RAG | arXiv   | 2025 |  RAG   | [Link](https://arxiv.org/abs/2503.01713) 
| 7 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization  | arXiv  | 2024 | Graph RAG| [Link](https://arxiv.org/abs/2404.16130)
| 8 | LightRAG: Simple and Fast Retrieval-Augmented Generatio  | arXiv   | 2024 |  Graph RAG | [Link](https://arxiv.org/abs/2410.05779)
| 9 | GRAG: Graph Retrieval-Augmented Generation | arXiv   | 2024 |  Graph RAG    | [Link](https://arxiv.org/abs/2405.16506) 
| 10 | HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases | arXiv   | 2024 |  Graph RAG    | [Link](https://arxiv.org/abs/2412.16311) 
| 11 | CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs | arXiv   | 2025 |  Graph RAG    | [Link](https://arxiv.org/abs/2501.15067) 
| 12 | MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation | arXiv  | 2025 | Graph RAG | [Link](https://arxiv.org/abs/2501.06713)
| 13 | GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation | arXiv  | 2025 | Graph RAG | [Link](https://arxiv.org/abs/2502.01113)
| 14 | MSG-LLM: A Multi-scale Interactive Framework for Graph-enhanced Large Language Models | COLING  | 2025 | Graph RAG | [Link](https://aclanthology.org/2025.coling-main.648/)
| 15 | PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths | arXiv  | 2025 | Graph RAG | [Link](https://arxiv.org/abs/2502.14902/)
| 16 | In-depth Analysis of Graph-based RAG in a Unified Framework | arXiv  | 2025 | Graph RAG | [Link](https://www.arxiv.org/abs/2503.04338/)
| 17 | KG-RAG: Bridging the Gap Between Knowledge and Creativity | arXiv   | 2024 |  KG RAG   | [Link](https://arxiv.org/abs/2405.12035)
| 18 | Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering| SIGIR         | 2024 | KG RAG    | [Link](https://dl.acm.org/doi/10.1145/3626772.3661370)
| 19 | REnhancing Large Language Models with Knowledge Graphs for Robust Question Answering | ICPADS   | 2024 |  KG RAG    | [Link](https://doi.ieeecomputersociety.org/10.1109/ICPADS63350.2024.00042) 
| 20 | FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs | arXiv   | 2025 |  KG RAG    | [Link](https://arxiv.org/abs/2501.09957) 
| 21 | SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation | arXiv   | 2025 |  KG RAG    | [Link](https://arxiv.org/abs/2412.15272) 
| 22 | RGR-KBQA: Generating Logical Forms for Question Answering Using Knowledge-Graph-Enhanced Large Language Model | COLING   | 2025 |  KG RAG    | [Link](https://aclanthology.org/2025.coling-main.205) 
| 23 | Knowledge Graph-Guided Retrieval Augmented Generation | arXiv   | 2025 |  KG RAG    | [Link](https://arxiv.org/abs/2502.06864) 
| 24 | Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation | ICLR | 2025 | KG RAG  | [Link](https://openreview.net/forum?id=JvkuZZ04O7) 
| 25 | Empowering LLMs by hybrid retrieval-augmented generation for domain-centric Q&A in smart manufacturing | Advanced Engineering Informatics | 2025| Hybrid RAG  | [Link](https://doi.org/10.1016/j.aei.2025.103212) 
| 26 | Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Spatial Reasoning Questions | arXiv | 2025 | Spatial RAG  | [Link](https://arxiv.org/abs/2502.18470) 

### KGs as Reasoning Guideline

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answerings   | ACL | 2022 |  Offline KG Guidelines    | [Link](https://aclanthology.org/2022.acl-long.396/)
| 2 | keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM   | arXiv  | 2023 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2401.00426)
| 3 | Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph   | arXiv  | 2024 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2406.01145) 
| 4 | Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models   | arXiv  | 2024 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2410.13080)
| 5 | Reasoning with Trees: Faithful Question Answering over Knowledge Graph   | COLING  | 2025 |  Offline KG Guidelines    | [Link](https://aclanthology.org/2025.coling-main.211/)
| 6 | Empowering Language Models with Knowledge Graph Reasoning for Open-Domain Question Answering   | EMNLP | 2022 |  Online KG Guildlines    | [Link](https://aclanthology.org/2022.emnlp-main.650/)
| 7 | Knowledge-Enhanced Iterative Instruction Generation and Reasoning for Knowledge Base Question Answering   | NLPCC  | 2022 |  Online KG Guildlines    | [Link](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_34)
| 8 | Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs   | arXiv  | 2023 |  Online KG Guildlines    | [Link](https://arxiv.org/abs/2312.11282)
| 9 | Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph   | ICLR   | 2024 |  Online KG Guildlines    | [Link](https://openreview.net/forum?id=nnVO1PvbTv)
| 10 | Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation   | ICLR  | 2024 |  Online KG Guildlines    | [Link](https://openreview.net/forum?id=oFBu7qaZpS)
| 11 | KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation   | arXiv  | 2024 |  Online KG Guildlines    | [Link](https://arxiv.org/abs/2412.20995)
| 12 | Retrieval and Reasoning on KGs: Integrate Knowledge Graphs into Large Language Models for Complex Question Answering   | EMNLP  | 2024 |  Online KG Guildlines    | [Link](https://aclanthology.org/2024.findings-emnlp.446)
| 13 | KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Grap   | arXiv  | 2024 |  Agent-based KG Guildlines    | [Link](https://arxiv.org/abs/2402.11163) 
| 14 | ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs   | ACL Findings  | 2024 |  Agent-based KG Guildlines    | [Link](https://aclanthology.org/2024.findings-acl.442/)                                                         |
| 15 | A Collaborative Reasoning Framework Powered by Reinforcement Learning and Large Language Models for Complex Questions Answering over Knowledge Graph | COLING| 2025 |  Collaborative Reasoning   | [Link](https://aclanthology.org/2025.coling-main.712/)  

### KGs as Refiner and Filter

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Answer Candidate Type Selection: Text-To-Text Language Model for Closed Book Question Answering Meets Knowledge Graphs   | KONVENS                                                                                           | 2023 |  KG-Driven Filtering and Validation    | [Link](https://aclanthology.org/2023.konvens-main.16/)
| 2 | KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques      | BioNLP Workshop                                               | 2024 |KG-Driven Filtering and Validation | [Link](https://aclanthology.org/2024.bionlp-1.13/) 
| 3 | Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting      | AAAI                                                | 2024 |KG-Driven Filtering and Validation | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29770/31326)
| 4 | Evidence-Focused Fact Summarization for Knowledge-Augmented Zero-Shot Question Answering      | ariXv                                               | 2024 |KG-Augmented Output Refinement | [Link](https://arxiv.org/abs/2403.02966)
| 5 | Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models  | ACL                                  | 2024 |KG-Augmented Output Refinement | [Link](https://aclanthology.org/2024.acl-long.569/)
| 6 | Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs  | arXiv                                  | 2024 |KG-Augmented Output Refinement | [Link](https://arxiv.org/abs/2406.14282)
| 7 | Optimizing Knowledge Integration in Retrieval-Augmented Generation with Self-Selection  | arXiv                                  | 2025 |RAG-based Answers Selection | [Link](https://arxiv.org/abs/2502.06148)

## 2. Complex QA

### Explainable QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Reasoning over Hierarchical Question Decomposition Tree for Explainable Question Answering   | ACL        | 2024 |  -    | [Link](https://aclanthology.org/2023.acl-long.814/)
| 2 | Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks   | SIGIR               | 2023 |  -    | [Link](https://dl.acm.org/doi/10.1145/3539618.3591682)
| 3 | Retrieval In Decoder benefits generative models for explainable complex question answering   | Neural Networks       | 2025 |  -    | [Link](https://doi.org/10.1016/j.neunet.2024.106833)

### Multi-Modal QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Lako: Knowledge-driven visual question answering via late knowledge-to text injection   | IJCKG  | 2022 |  VQA   | [Link](https://dl.acm.org/doi/10.1145/3579051.3579053)
| 2 | Modality-Aware Integration with Large Language Models for Knowledge-Based Visual Question Answering   | ACL   | 2024 |  VQA    | [Link](https://aclanthology.org/2024.acl-long.132/)
| 3 | Knowledge-Enhanced Visual Question Answering with Multi-modal Joint Guidance   |JCKG   | 2024 |  VQA    | [Link](https://dl.acm.org/doi/10.1145/3579051.3579073)
| 4 | ReasVQA: Advancing VideoQA with Imperfect Reasoning Process   |arXiv   | 2025 |  VQA    | [Link](https://arxiv.org/abs/2501.13536)
| 5 | Fine-grained knowledge fusion for retrieval-augmented medical visual question answering  | Information Fusion   | 2025 |  VQA    | [Link](https://doi.org/10.1016/j.inffus.2025.103059)
| 6 | RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering   | arXiv   | 2025 | Multi-Modal QA | [Link](https://arxiv.org/abs/2501.13297)
| 7 | MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering   | arXiv   | 2024 | Multi-Modal QA | [Link](https://arxiv.org/abs/2408.08521)


### Multi-Document QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Knowledge Graph Prompting for Multi-Document Question Answering   | AAAI  | 2024 |  Multi-doc QA | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29889)
| 2 | CuriousLLM: Elevating Multi-Document QA with Reasoning-Infused Knowledge Graph Prompting   | arXiv  | 2024 |  Multi-doc QA | [Link](https://arxiv.org/abs/2404.09077)
| 3 | VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation   | arXiv  | 2024 |  Multi-doc QA | [Link](https://arxiv.org/abs/2412.10704)

### Multi-Hop QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | GraphLLM: A General Framework for Multi-hop Question Answering over Knowledge Graphs Using Large Language Models   | NLPCC         | 2024 |  Multi-Hop QA    | [Link](https://link.springer.com/chapter/10.1007/978-981-97-9431-7_11)
| 2 | LLM-KGMQA: Large Language Model-Augmented Multi-Hop Question-Answering System based on Knowledge Graph in Medical Field   | KBS         | 2024 |  Multi-Hop QA    | [Link](https://doi.org/10.21203/rs.3.rs-4721418/v1)
| 3 | PokeMQA: Programmable knowledge editing for Multi-hop Question Answering | ACL   | 2024 |  Multi-Hop QA    | [Link](https://aclanthology.org/2024.acl-long.438/)
| 4 | HOLMES: Hyper-Relational Knowledge Graphs for Multi-hop Question Answering using LLMs | ACL   | 2024 |  Multi-Hop QA    | [Link](https://aclanthology.org/2024.acl-long.717/)
| 5 | LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments | EMNLP  | 2024 |  Multi-Hop QA    | [Link](https://aclanthology.org/2024.findings-emnlp.844/)
| 6 | SG-RAG: Multi-Hop Question Answering With Large Language Models Through Knowledge Graphs | ICNLSP  | 2024 |  Multi-Hop QA    | [Link](https://aclanthology.org/2024.icnlsp-1.45/)
| 7 | From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM   | COLING         | 2025 |  Multi-Hop QA   | [Link](https://aclanthology.org/2025.coling-main.55/)
| 8 | Multi-Hop Question Answering with LLMs & Knowledge Graphs   | Blog           | 2023 |  Multi-Hop QA    | [Link](https://www.wisecube.ai/blog-2/multi-hop-question-answering-with-llms-knowledge-graphs/)
| 9 | Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering   | arXiv           | 2025 |  Multi-Hop QA    | [Link](https://arxiv.org/abs/2502.14245)

### Multi-run and Conversational QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks   | SIGIR        | 2023 |  Conversational QA    | [Link](https://dl.acm.org/doi/10.1145/3539618.3591682)
| 2 | Conversational Question Answering with Language Models Generated Reformulations over Knowledge Graph   | ACL Findings           | 2024 |  Conversational QA    | [Link](https://aclanthology.org/2024.findings-acl.48/)  
| 3 | LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments   | EMNLP          | 2024 | Multi-Hop QA   | [Link](https://aclanthology.org/2024.findings-emnlp.844/)
| 4 | Learning When to Retrieve, What to Rewrite, and How to Respond in Conversational QA   | EMNLP | 2024 |   Conversational QA     | [Link](https://aclanthology.org/2024.findings-emnlp.622)
| 5 | ConvKGYarn: Spinning Configurable and Scalable Conversational Knowledge Graph QA Datasets with Large Language Models  | EMNLP | 2024 |   Conversational QA     | [Link](https://aclanthology.org/2024.emnlp-industry.89)
| 6 | Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs  | SIGMOD | 2025 |   Dialogue      | [Link](https://arxiv.org/abs/2501.09928)

## 3. Advanced Topics

### Optimization

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Empowering Large Language Models to Set up a Knowledge Retrieval Indexer via Self-Learning   | arXiv | 2023 |  Index-based Optimization| [Link](https://arxiv.org/abs/2405.16933)
| 2 | Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs   | ICLR   | 2024 |Index-based Optimization   | [Link](https://openreview.net/forum?id=6LKmaC4cO0/) 
| 3 | KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models | arXiv   | 2024 |Index-based Optimization   | [Link](https://arxiv.org/abs/2412.05547) 
| 4 | Prompting Is Programming: A Query Language for Large Language Models   | PLDL    | 2023 |Prompting-based Optimization   | [Link](https://dl.acm.org/doi/10.1145/3591300/) 
| 5 | LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs   | ACL Findings   | 2024 |Prompting-based Optimization   | [Link](https://aclanthology.org/2024.findings-acl.224/) 
| 6 | LightRAG: Simple and Fast Retrieval-Augmented Generation   | arXiv  | 2024 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2410.05779)
| 7 | Clue-Guided Path Exploration: Optimizing Knowledge Graph Retrieval with Large Language Models to Address the Information Black Box Challenge   | arXiv   | 2024 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2401.13444) 
| 8 | Optimizing open-domain question answering with graph-based retrieval augmented generation   | arXiv   | 2025 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2503.02922) 
| 9 | Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation   | WWW   | 2025 | Graph retrieval-based optimization   | [Link](https://openreview.net/forum?id=2ZaqnRIUCV) 
| 10 | Optimizing Knowledge Integration in Retrieval-Augmented Generation with Self-Selection   | arXiv   | 2025 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2502.06148) 
| 11 | Systematic Knowledge Injection into Large Language Models via Diverse Augmentation for Domain-Specific RAG   | arXiv   | 2025 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2502.08356) 
| 12 | KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques   | BioNLP Workshop   | 2024 | Ranking-based optimization | [Link](https://aclanthology.org/2024.bionlp-1.13/) 
| 13 | KS-LLM: Knowledge Selection of Large Language Models with Evidence Document for Question Answering   | arXiv  | 2024 | Ranking-based optimization   | [Link](https://arxiv.org/abs/2404.15660) 
| 14 | RAG-based Question Answering over Heterogeneous Data and Text   | arXiv  | 2024 | Ranking-based optimization   | [Link](https://arxiv.org/abs/2412.07420)
| 15 | Cost-efficient Knowledge-based Question Answering with Large Language Models   | arXiv  | 2024 | Cost-based optimization   | [Link](https://arxiv.org/abs/2405.17337)    
| 16 | KGLens: Towards Efficient and Effective Knowledge Probing of Large Language Models with Knowledge Graphs   | arXiv   | 2024 | Cost-based optimization   | [Link](https://arxiv.org/abs/2312.11539) 
| 17 | Knowledge Graph-Enhanced Large Language Models via Path Selection   | ACL Findings  | 2024 | Path-based optimization   | [Link](https://aclanthology.org/2024.findings-acl.376)
| 18 | LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration   | arXiv  | 2024 | Path-based optimization   | [Link](https://arxiv.org/abs/2411.05844)
| 19 | Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models   | arXiv    | 2024 | Query-based optimization   | [Link](https://arxiv.org/abs/2411.07820)
| 20 | A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models   | arXiv    | 2024 | MapReduce-based optimization   | [Link](https://arxiv.org/abs/2412.15271)
| 21 | PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning   | arXiv    | 2025 | Knowledge conflicts mitigation   | [Link](https://arxiv.org/abs/2502.15543)


### Data Management

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Triple Augmented Generative Language Models for SPARQL Query Generation from Natural Language Questions   | arXiv | 2024 |  NL2GQL    | [Link](https://dl.acm.org/doi/10.1145/3673791.3698426)
| 2 | R3-NL2GQL: A Model Coordination and Knowledge Graph Alignment Approach for NL2GQL   | ACL Findings      | 2024 |  NL2GQL    | [Link](https://aclanthology.org/2024.findings-emnlp.800/)
| 3 | Aligning Large Language Models to a Domain-specific Graph Database for NL2GQL   | CIKM  | 2024 |  NL2GQL    | [Link](https://dl.acm.org/doi/10.1145/3627673.3679713)
| 4 | UniOQA: A Unified Framework for Knowledge Graph Question Answering with Large Language Models  | arXiv  | 2024 |  NL2GQL    | [Link](https://arxiv.org/abs/2406.02110)
| 5 | NAT-NL2GQL: A Novel Multi-Agent Framework for Translating Natural Language to Graph Query Language  | arXiv  | 2024 |  NL2GQL    | [Link](https://arxiv.org/abs/2412.10434)
| 6 | CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era  | arXiv  | 2024 |  NL2GQL    | [Link](https://arxiv.org/abs/2412.18702)
| 7 | SpCQL: A Semantic Parsing Dataset for Converting Natural Language into Cypher  | CIKM  | 2022 |  NL2GQL    | [Link](https://dl.acm.org/doi/10.1145/3511808.3557703)
| 8 | Robust Text-to-Cypher Using Combination of BERT, GraphSAGE, and Transformer (CoBGT) Model  |Applied Sciences  | 2024 |  NL2GQL    | [Link](https://doi.org/10.3390/app14177881)
| 9 | Real-Time Text-to-Cypher Query Generation with Large Language Models for Graph Databases  |  Future Internet  | 2024 |  NL2GQL    | [Link](https://doi.org/10.3390/fi16120438)
| 10 | LLM4QA: Leveraging Large Language Model for Efficient Knowledge Graph Reasoning with SPARQL Query  |  JAIT  | 2024 |  NL2GQL    | [Link](https://doi.org/10.12720/jait.15.10.1157-1162)
| 11 | Text to Graph Query Using Filter Condition Attributes  |  LSGDA@VLDB  | 2024 |  NL2GQL    | [Link](https://vldb.org/workshops/2024/proceedings/LSGDA/LSGDA24.09.pdf)
| 12 | Text-to-CQL Based on Large Language Model and Graph Pattern Enhancement  |  PRML  | 2024 |  NL2GQL    | [Link](https://ieeexplore.ieee.org/document/10779814)
| 13 | Demystifying Natural Language to Cypher Conversion with OpenAI, Neo4j, LangChain, and LangSmith  |  Blog  | 2024 |  NL2GQL    | [Link](https://medium.com/@muthoju.pavan/demystifying-natural-language-to-cypher-conversion-with-openai-neo4j-langchain-and-langsmith-2dbecb1e2ce9/)
| 14 | Text2Cypher, the beginning of the Graph + LLM stack  |  Blog  | 2023 |  NL2GQL    | [Link](https://siwei.io/en/llm-text-to-nebulagraph-query/)
| 15 | Text2Cypher - Natural Language Queries  |  Blog  | 2023 |  NL2GQL    | [Link](https://neo4j.com/labs/neodash/2.4/user-guide/extensions/natural-language-queries/)
| 16 | LLaSA: Large Language and Structured Data Assistant   | arXiv | 2024 | Structured Data Assistant | [Link](https://arxiv.org/abs/2411.14460)     
| 17 | GraphRAG and role of Graph Databases in Advancing AI   | IJRCAIT | 2024 | Graph DB | [Link](https://doi.org/10.5281/zenodo.13908615) 
| 18 | TigerVector: Supporting Vector Search in Graph Databases for Advanced RAGs | arXiv | 2025 | Graph DB | [Link](https://arxiv.org/abs/2501.11216) 
| 19 | Increasing Accuracy of LLM-powered Question Answering on SQL databases: Knowledge Graphs to the Rescue  | Data Engineering Bulletin | 2024 | RDB QA | [Link](http://sites.computer.org/debull/A24dec/p109.pdf)  
| 20 | Symphony: Towards Trustworthy Question Answering and Verification using RAG over Multimodal Data Lakes  | Data Engineering Bulletin | 2024 | RDB QA | [Link](http://sites.computer.org/debull/A24dec/p135.pdf)  
| 21 | Increasing the LLM Accuracy for Question Answering: Ontologies to the Rescue!  | arXiv | 2024 | RDB QA | [Link](https://arxiv.org/abs/2405.11706)    


## 4. Benchmark and Applications

### Benchmark Dataset

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year |          Dataset            | Category                    | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | The Value of Semantic Parse Labeling for Knowledge Base Question Answering   | ACL    |  2016 |  [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) | KBQA and KGQA| [Link](https://aclanthology.org/P16-2033/)
| 2 | Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs   | arXiv  | 2024 | CAQA  |  KBQA and KGQA| [Link](https://arxiv.org/abs/2401.14640/)
| 3 | G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering   | NeurIPS  | 2024 | [GraphQA](https://github.com/XiaoxinHe/G-Retriever)  |  KBQA and KGQA| [Link](https://openreview.net/forum?id=MPJ3oXtTZl)
| 4 | Automatic Question-Answer Generation for Long-Tail Knowledge  | KnowledgeNL@KDD  | 2023 | Long-tail QA  |  KBQA and KGQA| [Link](https://knowledge-nlp.github.io/kdd2023/papers/Kumar5.pdf)
| 5 | BioASQ-QA: A manually curated corpus for Biomedical Question Answering  | Scientific Data |2023 | [BioASQ-QA](https://zenodo.org/records/7655130)|   KBQA and KGQA| [Link](https://pubmed.ncbi.nlm.nih.gov/36973320/)
| 6 | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering  | EMNLP | 2018 | [HotpotQA](https://github.com/hotpotqa/hotpot)|   KBQA and KGQA| [Link](https://aclanthology.org/D18-1259)
| 7 | CR-LT-KGQA: A Knowledge Graph Question Answering Dataset Requiring Commonsense Reasoning and Long-Tail Knowledge  | arXiv | 2024 | [CR-LT-KGQA](https://github.com/D3Mlab/cr-lt-kgqa)|   KBQA and KGQA| [Link](https://arxiv.org/abs/2403.01395)
| 8 | CPAT-Questions: A Self-Updating Benchmark for Present-Anchored Temporal Question-Answering  | ACL Findings  | 2024 | [TemporalQA](https://github.com/D3Mlab/cr-lt-kgqa) |  KBQA and KGQA| [Link](https://arxiv.org/abs/2403.01395)
| 9 | SituatedQA: Incorporating Extra-Linguistic Contexts into QA | EMNLP  | 2024 | [SituatedQA](https://github.com/mikejqzhang/SituatedQA)|  Open-retrieval QA | [Link](https://aclanthology.org/2021.emnlp-main.586/)
| 10 | CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge | NAACL  | 2024 | [CommonsenseQA](https://github.com/jonathanherzig/commonsenseqa)|  Multiple-choice QA| [Link](https://aclanthology.org/N19-1421)
| 11 | FanOutQA: A Multi-Hop, Multi-Document Question Answering Benchmark for Large Language Models | ACL  | 2024 | [FanOutQA](https://github.com/zhudotexe/fanoutqa)|  Multi-hop QA| [Link](https://aclanthology.org/2024.acl-short.2)
| 12 | MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge | arXiv  | 2024 | [MINTQA](https://github.com/probe2/multi-hop/)|  Multi-hop QA| [Link](https://arxiv.org/abs/2412.17032)
| 13 | What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams | Applied Sciences   | 2021 | [MedQA](https://github.com/jind11/MedQA)|  Multiple-choice QA | [Link](https://www.mdpi.com/2076-3417/11/14/6421)
| 14 | PAT-Questions: A Self-Updating Benchmark for Present-Anchored Temporal Question-Answering | ACL Findings  | 2024 | [PAQA](https://github.com/jannatmeem95/PAT-Questions)|  Temporal QA| [Link](https://aclanthology.org/2024.findings-acl.777)
| 15 | MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models | ACL Findings  | 2023 | [MenatQA](https://github.com/weiyifan1023/MenatQA)|  Temporal QA| [Link](https://aclanthology.org/2023.findings-emnlp.100)
| 16 | TempTabQA: Temporal Question Answering for Semi-Structured Tables| EMNLP | 2023 | [TempTabQA](https://github.com/temptabqa/temptabqa)|  Temporal QA| [Link](https://aclanthology.org/2023.emnlp-main.149)
| 17 | Complex Temporal Question Answering on Knowledge Graphs| CIKM | 2021 | [EXAQT](https://exaqt.mpi-inf.mpg.de/)|  Temporal QA| [Link](https://dl.acm.org/doi/10.1145/3459637.3482416)
| 18 | Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA| EMNLP | 2024 | [Loong](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Loong)|  Multi-doc QA| [Link](https://aclanthology.org/2024.emnlp-main.322)
| 19 | MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation | arXiv  | 2025 | [MRAMG](https://huggingface.co/MRAMG)|  Multi-modal QA| [Link](https://arxiv.org/abs/2502.04176)
| 20 | OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems | EMNLP  | 2024 | [OMG-QA](https://github.com/linyongnan/OMG-QA)|  Multi-modal QA| [Link](https://aclanthology.org/2024.emnlp-industry.75)
| 21 | M3SciQA: A Multi-Modal Multi-Document Scientific QA Benchmark for Evaluating Foundation Models | ACL Findings  | 2024 | [M3SciQA](https://github.com/yale-nlp/M3SciQA)|  Multi-modal QA| [Link](https://aclanthology.org/2024.findings-emnlp.904)
| 22 | A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases  | GRADES-NDA  | 2024 | [ChatData](https://github.com/datadotworld/cwd-benchmark-data) |  LLM and KGs for QA| [Link](https://dl.acm.org/doi/10.1145/3661304.3661901)
| 23 | XplainLLM: A Knowledge-Augmented Dataset for Reliable Grounded Explanations in LLMs  | EMNLP  | 2024 | [XplainLLM](https://github.com/chen-zichen/XplainLLM_dataset) |  LLM and KGs for QA| [Link](https://arxiv.org/abs/2311.08614)
| 24 | Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering  | SEMANTICS   | 2023 | [LLM-KG-Bench](https://github.com/AKSW/LLM-KG-Bench)|  LLM and KGs for QA| [Link](https://ceur-ws.org/Vol-3526/paper-04.pdf)
| 25 | Docugami Knowledge Graph Retrieval Augmented Generation (KG-RAG) Datasets | -  | 2023 | [KG-RAG](https://github.com/docugami/KG-RAG-datasets)|  LLM and KGs for QA| -
| 26 | How Credible Is an Answer From Retrieval-Augmented LLMs? Investigation and Evaluation With Multi-Hop QA  | ACL ARR   | 2024 |- |  LLM and KGs for QA| [Link](https://openreview.net/forum?id=YsmnPHBbx1f)
| 27 | Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering  | arXiv  | 2024 |[OKGQA](https://anonymous.4open.science/r/OKGQA-CBB0) |  LLM and KGs for QA| [Link](https://arxiv.org/abs/2410.08085)
| 28 | MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation | arXiv  | 2025 | [LiHua-World](https://github.com/HKUDS/MiniRAG/tree/main/dataset/LiHua-World)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2501.06713)
| 29 | Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering   | arXiv | 2025 | [OKGQA](https://anonymous.4open.science/r/OKGQA-CBB0) | LLM and KGs for QA| [Link](https://arxiv.org/abs/2410.08085)

### Industrial and Scientific Applications

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year |          Github            | Category                    | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation  | arXiv   | 2024 | [KAG](https://github.com/OpenSPG/KAG)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2409.13731)
| 2 | Fact Finder -- Enhancing Domain Expertise of Large Language Models by Incorporating Knowledge Graphs  | arXiv   | 2024 | [Fact Finder](https://github.com/chrschy/fact-finder/)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2408.03010)
| 3 | Leveraging Large Language Models and Knowledge Graphs for Advanced Biomedical Question Answering Systems  | CSA 2024   | 2024 | [Cypher Translator](https://github.com/phdResearcherDz/CypherTranslator/)|  LLM and KGs for QA| [Link](https://link.springer.com/chapter/10.1007/978-3-031-71848-9_31)
| 4 | A Prompt Engineering Approach and a Knowledge Graph based Framework for Tackling Legal Implications of Large Language Model Answers  | arXiv   | 2024 | - |  LLM and KGs for QA| [Link](https://link.springer.com/chapter/10.1007/978-3-031-71848-9_31)
| 5 | Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education  | arXiv   | 2024 | - |  LLM and KGs for QA| [Link](https://arxiv.org/abs/2412.14191)
| 6 |Knowledge Graphs as a source of trust for LLM-powered enterprise question answering  | Journal of Web Semantics  | 2025 | - |  LLM and KGs for QA| [Link](https://doi.org/10.1016/j.websem.2024.100858)
| 7 |MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot |WWW| 2025 | [MedRAG](https://github.com/SNOWTEAM2023/MedRAG)|  LLM and KGs for QA| [Link](https://openreview.net/pdf/7d3d9ad2d616ceae8c5b77eb94019086b980ceda.pdf)
| 8 |EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents |arXiv| 2025 | - | LLM and KGs for QA| [Link](https://arxiv.org/abs/2501.13746)
| 9 |Nanjing Yunjin intelligent question-answering system based on knowledge graphs and retrieval augmented generation technology |Heritage Science| 2025 | - | LLM and KGs for QA| [Link](https://www.nature.com/articles/s40494-024-01231-3)

### Demo

| NO | Name |  Description |Source |Github   | 
|----|------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|------|
| 1 | GraphRAG-QA  | An industrial demo of GraphRAG integrating several query engine for augmenting QA, NLP2Cypher-based KG query engine, vector RAG query engine, and Graph vector RAG query engine.  |NebulaGraph| [GraphRAG-QA](https://github.com/wey-gu/demo-kg-build)
| 2 | Neo4jRAG-QA  | This sample application demonstrates how to implement a Large Language Model (LLM) and Retrieval Augmented Generation (RAG) system with a Neo4j Graph Database.  | Neo4j Graph | [Neo4j Graph RAG](https://github.com/neo4j-examples/rag-demo)
| 3 | BioGraphRAG  | This a platform to integrate biomedical knowledge graphs stored in NebulaGraph with LLMs via GraphRAG architecture.  |  | [BioGraphRAG](https://github.com/devingupta1/BioGraphRAG)
| 4 | kotaemon  |An open-source clean & customizable RAG UI for chatting with your documents. Built with both end users and developers in mind. | Cinnamon AI | [kotaemon](https://github.com/Cinnamon/kotaemon)
| 5 | PIKE-RAG  |A secIalized KnowledgE and Rationale Augmented Generation, which focuses on extracting, understanding, and applying domain-specific knowledge to gradually guide LLMs toward accurate responses. | Microsoft | [PIKE-RAG](https://github.com/microsoft/PIKE-RAG)

## 5. Related Survey

| NO | Title |  Venue | Year | Paper Link   | 
|----|------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|------|
| 1 | Unifying Large Language Models and Knowledge Graphs: A Roadmap   | TKDE    |  2024 |  [Link](https://doi.org/10.1109/TKDE.2024.3352100)
| 2 | Graph Retrieval-Augmented Generation: A Survey   | arXiv    |  2024 |  [Link](https://arxiv.org/abs/2408.08921)
| 3 | Retrieval-Augmented Generation with Graphs (GraphRAG)   | arXiv    |  2024 |  [Link](https://arxiv.org/abs/2501.00309)
| 4 | Multilingual Question Answering Systems for Knowledge Graphs—A Survey   | Semantic Web   |  2024 |  [Link](https://www.semantic-web-journal.net/content/multilingual-question-answering-systems-knowledge-graphs%E2%80%94-survey)
| 5 | Temporal Knowledge Graph Question Answering: A Survey   | arXiv    |  2024 |  [Link](https://arxiv.org/abs/2406.14191)
| 6 | Knowledge Graph and Large Language Model Co-learning via Structure-oriented Retrieval Augmented Generation  | Data Engineering Bulletin  |  2024 |  [Link](http://sites.computer.org/debull/A24dec/p9.pdf)
| 7 | Research Trends for the Interplay between Large Language Models and Knowledge Graphs  | LLM+KG@VLDB2024|  2024 |  [Link](https://vldb.org/workshops/2024/proceedings/LLM+KG/LLM+KG-9.pdf)
| 8 | Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective  | arXiv|  2024 |  [Link](https://arxiv.org/abs/2412.10390)
| 9 | Large Language Models, Knowledge Graphs and Search Engines: A Crossroads for Answering Users' Questions   | arXiv    |  2025 |  [Link](https://arxiv.org/abs/2501.06699)
| 10 | Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspective   | Journal of Web Semantics  |  2025 |  [Link](https://doi.org/10.1016/j.websem.2024.100844)
| 11 | A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models   | arXiv    |  2025 |  [Link](https://arxiv.org/abs/2501.13958)
| 12 | Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG   | arXiv   |  2025 |  [Link](https://arxiv.org/abs/2501.09136)
| 13 | A survey on augmenting knowledge graphs (KGs) with large language models (LLMs): models, evaluation metrics, benchmarks, and challenges  |Discover Artificial Intelligence   |  2024 |  [Link](https://link.springer.com/article/10.1007/s44163-024-00175-8)
