# LLM-KG4QA: Unifying Large Language Models and Knowledge Graphs (LLM-KG) for Question Answering (QA)
## Content
- [Unifying LLM with KGs for QA](#1unifying-llms-and-kgs-for-qa)
  - [KGs as Background Knowledg](#KGs-as-background-knowledge)
  - [KGs as Reasoning Guideline](#KGs-as-reasoning-guideline)
  - [KGs as Refiner and Filter](#KGs-as-refiner-and-filter)
  <!--- - [Hybrid Methods](#hybrid-methods) -->
- [Advanced Topics ](#2advanced-topics)
  - [Explainable QA](#explainable-qa)
  - [Visual QA](#Visual-qa)
  - [Conversational QA](#Conversational-qa)
  - [Optimization](#optimization)
- [Benchmark and Applications](#3benchmark-and-applications)
  - [Benchmark Dataset](#benchmark-dataset)
  - [Industrial Applications](#industrial-applications)
  - [Demo](#demo)

---
## 1. Unifying LLMs and KGs for QA

### KGs as Background Knowledge
| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Deep Bidirectional Language-Knowledge Graph pretraining   | NeurIPS                                                                                            | 2022|  Knowledge Integration and Fusion    | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/f224f056694bcfe465c5d84579785761-Paper-Conference.pdf)                                            |
| 2 | GreaseLM: Graph REASoning Enhanced Language Models    | ICLR                                                                                            | 2022 |  Knowledge Integration and Fusion    | [Link](https://openreview.net/forum?id=41e9o6cQPj) 
| 3 | InfuserKI: Enhancing Large Language Models with Knowledge Graphs via Infuser-Guided Knowledge Integration  | LLM+KG@VLDB                                                                                            | 2024 |  Knowledge Integration and Fusion    | [Link](https://arxiv.org/abs/2402.11441)
| 4 | KnowLA: Enhancing Parameter-efficient Finetuning with Knowledgeable Adaptation  | NAACL                                                                                        | 2024 |  Knowledge Integration and Fusion    | [Link](https://aclanthology.org/2024.naacl-long.396/)
| 5 | KG-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning  | ACL Findlings                                                                                        | 2024 |  Knowledge Integration and Fusion    | [Link](https://aclanthology.org/2024.findings-acl.229/) 
| 6 | A GAIL Fine-Tuned LLM Enhanced Framework for Low-Resource Knowledge Graph Question Answering  | CIKM                                                                                         | 2024 |  Knowledge Integration and Fusion    | [Link](https://dl.acm.org/doi/10.1145/3627673.3679753)
| 7 | Prompting Large Language Models with Knowledge Graphs for Question Answering Involving Long-tail Facts  | arXiv                                                                                         | 2024 |  KG-Augmented Prompting    | [Link](https://arxiv.org/abs/2405.06524)
| 8 | KnowGPT: Knowledge Graph based Prompting for Large Language Models  | arXiv                                                                                         | 2024 |  KG-Augmented Prompting    | [Link](https://arxiv.org/abs/2312.06185) 
| 9 | Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering  | NLRSE                                                                                         | 2023 |  KG-Augmented Prompting    | [Link](https://aclanthology.org/2023.nlrse-1.7/)
| 10 | Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering | IJCKG                                                                                         | 2023 |  KG-Augmented Prompting    | [Link](https://ijckg2023.knowledge-graph.jp/pages/proc/paper_30.pdf/)
| 11 | Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation| arXiv                                                                                         | 2024 |  Retrieval Augmented Generation    | [Link](https://arxiv.org/abs/2402.05128)
| 12 | GRAG: Graph Retrieval-Augmented Generation | arXiv                                                                                         | 2024 |  Retrieval Augmented Generation    | [Link](https://arxiv.org/abs/2405.16506) 
| 13 | KG-RAG: Bridging the Gap Between Knowledge and Creativity | arXiv                                                                                         | 2024 |  Retrieval Augmented Generation    | [Link](https://arxiv.org/abs/2405.12035)
| 14 | Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering| SIGIR                                                                                       | 2024 |  Retrieval Augmented Generation    | [Link](https://dl.acm.org/doi/10.1145/3626772.3661370)
| 15 | Retrieval-enhanced Knowledge Editing in Language Models for Multi-Hop Question Answering | CIKM                                                                                         | 2024 |  Retrieval Augmented Generation    | [Link](https://dl.acm.org/doi/abs/10.1145/3627673.3679722) 


### KGs as Reasoning Guideline

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answerings   | ACL                                                                                           | 2022 |  Offline KG Guidelines    | [Link](https://aclanthology.org/2022.acl-long.396/)
| 2 | keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM   | arXiv                                                                                           | 2023 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2401.00426)
| 3 | Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph   | arXiv                                                                                           | 2024 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2406.01145) 
| 4 | Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models   | arXiv                                                                                           | 2024 |  Offline KG Guidelines    | [Link](https://arxiv.org/abs/2410.13080)
| 5 | Empowering Language Models with Knowledge Graph Reasoning for Open-Domain Question Answering   | EMNLP                                                                                           | 2022 |  Online KG Guildlines    | [Link](https://aclanthology.org/2022.emnlp-main.650/)
| 6 | Knowledge-Enhanced Iterative Instruction Generation and Reasoning for Knowledge Base Question Answering   | NLPCC                                                                                           | 2022 |  Online KG Guildlines    | [Link](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_34)
| 7 | Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs   | arXiv                                                                                           | 2023 |  Online KG Guildlines    | [Link](https://arxiv.org/abs/2312.11282)
| 8 | Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph   | ICLR                                                                                           | 2024 |  Online KG Guildlines    | [Link](https://openreview.net/forum?id=nnVO1PvbTv)
| 9 | Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation   | ICLR                                                                                           | 2024 |  Online KG Guildlines    | [Link](https://openreview.net/forum?id=oFBu7qaZpS)
| 10 | KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Grap   | arXiv                                                                                           | 2024 |  Agent-based KG Guildlines    | [Link](https://arxiv.org/abs/2402.11163) 
| 11 | ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs   | ACL Findings                                                                                           | 2024 |  Agent-based KG Guildlines    | [Link](https://aclanthology.org/2024.findings-acl.442/)                                                         |


### KGs as Refiner and Filter

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Answer Candidate Type Selection: Text-To-Text Language Model for Closed Book Question Answering Meets Knowledge Graphs   | KONVENS                                                                                           | 2023 |  KG-Driven Filtering and Validation    | [Link](https://aclanthology.org/2023.konvens-main.16/)
| 2 | Answer Candidate Type Selection: Text-To-Text Language Model for Closed Book Question Answering Meets Knowledge Graphs   | KONVENS                                                                                           | 2023 |  KG-Driven Filtering and Validation    | [Link](https://aclanthology.org/2023.konvens-main.16/)  
| 3 | KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques      | BioNLP Workshop                                               | 2024 |KG-Driven Filtering and Validation | [Link](https://aclanthology.org/2024.bionlp-1.13/) 
| 4 | Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting      | AAAI                                                | 2024 |KG-Driven Filtering and Validation | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29770/31326)
| 5 | Evidence-Focused Fact Summarization for Knowledge-Augmented Zero-Shot Question Answering      | ariXv                                               | 2024 |KG-Augmented Output Refinement | [Link](https://arxiv.org/abs/2403.02966)
| 6 | Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models  | ACL                                  | 2024 |KG-Augmented Output Refinement | [Link](https://aclanthology.org/2024.acl-long.569/)
| 7 | Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs  | arXiv                                  | 2024 |KG-Augmented Output Refinement | [Link](https://arxiv.org/abs/2406.14282)

## 2. Advanced Topics

### Explainable QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Retrieval In Decoder benefits generative models for explainable complex question answering   | Neural Networks                                                                                           | 2024 |  -    | [Link](https://doi.org/10.1016/j.neunet.2024.106833)
| 2 | Reasoning over Hierarchical Question Decomposition Tree for Explainable Question Answering   | ACL                                                                                           | 2024 |  -    | [Link](https://aclanthology.org/2023.acl-long.814/)


### Visual QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       |Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Lako: Knowledge-driven visual question answering via late knowledge-to text injection   | IJCKG                                                                                           | 2022 |  -    | [Link](https://dl.acm.org/doi/10.1145/3579051.3579053)
| 2 | Modality-Aware Integration with Large Language Models for Knowledge-Based Visual Question Answering   | ACL                                                                                           | 2024 |  -    | [Link](https://aclanthology.org/2024.acl-long.132/)
| 3 | Knowledge-Enhanced Visual Question Answering with Multi-modal Joint Guidance   | IJCKG                                                                                           | 2024 |  -    | [Link](https://dl.acm.org/doi/10.1145/3579051.3579073/)

### Conversational QA

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Explainable Conversational Question Answering over Heterogeneous Sources via Iterative Graph Neural Networks   | SIGIR                                                                                           | 2023 |  -    | [Link](https://dl.acm.org/doi/10.1145/3539618.3591682)
| 2 | Conversational Question Answering with Language Models Generated Reformulations over Knowledge Graph   | ACL Findlings                                                                                 | 2024 |  -    | [Link](https://aclanthology.org/2024.findings-acl.48/)           

### Optimization

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year | Category                                       | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Empowering Large Language Models to Set up a Knowledge Retrieval Indexer via Self-Learning   | arXiv                                                                                           | 2023 |  Index-based Optimization| [Link](https://arxiv.org/abs/2405.16933)
| 2 | Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs   | ICLR                                                                                 | 2024 |Index-based Optimization   | [Link](https://openreview.net/forum?id=6LKmaC4cO0/) 
| 3 | Prompting Is Programming: A Query Language for Large Language Models   | PLDL                                                                                 | 2023 |Prompting-based Optimization   | [Link](https://dl.acm.org/doi/10.1145/3591300/) 
| 4 | LightRAG: Simple and Fast Retrieval-Augmented Generation   | arXiv                                                                                 | 2024 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2410.05779)
| 5 | Clue-Guided Path Exploration: Optimizing Knowledge Graph Retrieval with Large Language Models to Address the Information Black Box Challenge   | arXiv                                       | 2024 | Graph retrieval-based optimization   | [Link](https://arxiv.org/abs/2401.13444) 
| 6 | KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques      | BioNLP Workshop                                               | 2024 | Ranking-based optimization | [Link](https://aclanthology.org/2024.bionlp-1.13/) 
| 7 | KS-LLM: Knowledge Selection of Large Language Models with Evidence Document for Question Answering   | arXiv                                       | 2024 | Ranking-based optimization   | [Link](https://arxiv.org/abs/2404.15660) 
| 8 | Cost-efficient Knowledge-based Question Answering with Large Language Models   | arXiv                                       | 2024 | Cost-based optimization   | [Link](https://arxiv.org/abs/2405.17337)    
| 9 | KGLens: Towards Efficient and Effective Knowledge Probing of Large Language Models with Knowledge Graphs   | arXiv                                       | 2024 | Cost-based optimization   | [Link](https://arxiv.org/abs/2312.11539)        

## 3. Benchmark and Applications

### Benchmark Dataset

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year |          Dataset            | Category                    | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | The Value of Semantic Parse Labeling for Knowledge Base Question Answering   | ACL    |  2016 |  WebQSP | KBQA and KGQA| [Link](https://aclanthology.org/P16-2033/)
| 2 | Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs   | arXiv  | 2024 | CAQA  |  KBQA and KGQA| [Link](https://arxiv.org/abs/2401.14640/)
| 3 | G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering   | arXiv  | 2024 | GraphQA  |  KBQA and KGQA| [Link](https://arxiv.org/abs/2402.07630/)
| 4 | Automatic Question-Answer Generation for Long-Tail Knowledge  | KnowledgeNL@KDD  | 2023 | Long-tail QA  |  KBQA and KGQA| [Link](https://knowledge-nlp.github.io/kdd2023/papers/Kumar5.pdf)
| 5 | CPAT-Questions: A Self-Updating Benchmark for Present-Anchored Temporal Question-Answering  | ACL Findings  | 2024 | [TemporalQA](https://github.com/D3Mlab/cr-lt-kgqa) |  KBQA and KGQA| [Link](https://arxiv.org/abs/2403.01395)
| 6 | A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases  | GRADES-NDA  | 2024 | EnterpriseQA |  LLM and KGs for QA| [Link](https://dl.acm.org/doi/10.1145/3661304.3661901)
| 7 | XplainLLM: A Knowledge-Augmented Dataset for Reliable Grounded Explanations in LLMs  | EMNLP  | 2024 | XplainLLM |  LLM and KGs for QA| [Link](https://arxiv.org/abs/2311.08614)
| 8 | Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering  | SEMANTICS   | 2023 | [LLM-KG-Bench](https://github.com/AKSW/LLM-KG-Bench)|  LLM and KGs for QA| [Link](https://ceur-ws.org/Vol-3526/paper-04.pdf)

### Industrial Applications

| NO | Title                                                                                                        | Venue                                                                                                                                                    | Year |          Github            | Category                    | Paper Link                                                                                           |
|----|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation  | arXiv   | 2024 | [KAG](https://github.com/OpenSPG/KAG)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2409.13731)
| 2 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization  | arXiv  | 2024 | [GraphRAG](https://github.com/microsoft/graphrag)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2404.16130)
| 3 | LightRAG: Simple and Fast Retrieval-Augmented Generatio  | arXiv   | 2024 | [LightRAG](https://github.com/HKUDS/LightRAG)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2410.05779)
| 4 | Fact Finder -- Enhancing Domain Expertise of Large Language Models by Incorporating Knowledge Graphs  | arXiv   | 2024 | [Fact Finder](https://github.com/chrschy/fact-finder/)|  LLM and KGs for QA| [Link](https://arxiv.org/abs/2408.03010)


### Demo

| NO | Name |  Description |Source |Github   | 
|----|------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|------|
| 1 | GraphRAG-QA  | An industrial demo of GraphRAG integrating several query engine for augmenting QA, NLP2Cypher-based KG query engine, vector RAG query engine, and Graph vector RAG query engine.  |NebulaGraph| [GraphRAG-QA](https://github.com/wey-gu/demo-kg-build)
| 2 | Neo4jRAG-QA  | This sample application demonstrates how to implement a Large Language Model (LLM) and Retrieval Augmented Generation (RAG) system with a Neo4j Graph Database.  | Neo4j Graph | [Neo4jRAG-QA](https://github.com/neo4j-examples/rag-demo)
| 3 | BioGraphRAG  | This a platform to integrate biomedical knowledge graphs stored in NebulaGraph with LLMs via GraphRAG architecture.  | NebulaGraph | [Neo4jRAG-QA](https://github.com/devingupta1/BioGraphRAG)
