# Model Families for Jira Issue Similarity and Duplicate Detection

## Overview

This report outlines the main model families and architectural approaches that should be compared when designing a "Jira issue intelligence" system for similarity search and duplicate detection on issue trackers such as Jira.
It focuses on approaches that have been studied for duplicate bug report detection and semantic similarity in software repositories and related domains, and highlights how they trade off accuracy, latency, robustness, and implementation complexity.[^1][^2][^3]
The goal is to give a short-list of approach categories you should explicitly benchmark against each other when building tools for industrial use.

## Problem Decomposition

The problem naturally splits into two tasks:

- **Retrieval / similarity search**: Given a new issue, retrieve the most relevant historical issues (candidate set) from a large corpus.
- **Duplicate classification / ranking**: Given a query issue and its candidate neighbors, decide whether each pair is a true duplicate and rank candidates accordingly.

Many systems unify these into a single similarity scoring step, but from an architectural point of view it is useful to separate **retrieval** (fast, approximate, high recall) from **classification** (slower, more precise, higher precision)."[^2][^3]

The model families below can be used at one or both stages.

## High-Level Approach Families to Compare

At a high level, the approaches to compare fall into the following families:

1. **Sparse lexical retrieval models** (TF–IDF, BM25 and variants)
2. **Classical supervised ML using engineered features** (e.g., SVMs, Random Forests on lexical and metadata features)
3. **Dense semantic embedding models** (Siamese/bi-encoder architectures like SBERT, domain-tuned sentence transformers)
4. **Hybrid sparse–dense retrieval** (BM25 plus dense embeddings with score fusion)
5. **Deep neural architectures for pairwise duplicate classification** (Siamese transformers, hybrid retrieval+classifier pipelines)
6. **LLM-based and RAG-style approaches** (using large language models for embeddings, re-ranking, or classification)

Each family corresponds to a different set of assumptions and operational trade-offs and has been evaluated in the context of bug-report deduplication or semantically similar text retrieval.[^4][^3][^1][^2]

The following sections describe these families in more detail and explain why they are worth including in a comparative study.

## 1. Sparse Lexical Retrieval (TF–IDF, BM25)

Sparse lexical methods treat issues as bags of words and match based on overlapping terms weighted by term frequency and inverse document frequency.
In practice, BM25 is the standard baseline for information-retrieval style search and remains very competitive for bug-report deduplication when the task is framed as ranking similar reports rather than binary classification.[^5][^3]

Key points:

- **Models**: TF–IDF cosine similarity, BM25 (and minor variants like BM25L, BM25+).
- **Pros**: Extremely fast, easy to implement, works well when summaries/descriptions share specific tokens (component names, error codes, stack traces, ticket IDs), incremental updates are cheap.[^6][^5]
- **Cons**: Fails on vocabulary mismatch (synonyms, paraphrases), struggles with very short or noisy descriptions, and does not exploit structure in the tickets.[^5]

Because several empirical studies find that deep models do not consistently dominate IR baselines in ranking settings, sparse retrieval is a mandatory baseline in any comparison.[^3]

## 2. Classical Supervised ML with Engineered Features

Before deep transformers, duplicate bug detection often used classical ML with hand-crafted features.
Features include lexical similarity scores (TF–IDF, BM25), overlap in components, products, versions, reporters, and basic temporal information.[^7][^3]

Key points:

- **Models**: Logistic regression, SVMs, Random Forests, Gradient Boosted Trees trained on pairwise feature vectors.
- **Features**: Cosine similarity, Jaccard similarity, token-level edit distances, shared categorical fields (component, product, labels), time difference, etc.[^3][^7]
- **Pros**: Interpretable feature contributions, easy to incorporate non-text fields, relatively light-weight.
- **Cons**: Requires labeled duplicate/non-duplicate pairs; may underperform deep models when strong semantic paraphrasing is common.

This family is still relevant for industrial systems because it integrates structured Jira fields naturally and offers controllable decision thresholds.

## 3. Dense Semantic Embedding Models (Sentence Transformers, Siamese Encoders)

Dense retrieval models map issues into continuous vector embeddings such that semantically similar issues are close in vector space.
Siamese architectures like Sentence-BERT (SBERT) are widely used for duplicate bug detection and duplicate question detection on Q&A sites.[^4][^2]

Key points:

- **Models**: 
  - Off-the-shelf sentence transformers (e.g., SBERT, MPNet-based variants).
  - Domain-tuned sentence transformers trained or fine-tuned on bug-report or Jira-like corpora.[^2][^4]
  - Siamese networks that jointly encode textual and structured features.
- **Use cases**:
  - Direct KNN search in embedding space for similarity search.
  - Candidate generation (ANN search with FAISS or similar).
- **Pros**: Handles paraphrases and vocabulary mismatch; supports multilingual and cross-project generalization; works well as the core of a semantic "Similar Issues" feature.[^8][^4][^2]
- **Cons**: Requires embedding computation and vector index maintenance; may miss exact-identifier queries (product codes, stack traces) unless combined with sparse methods.[^5]

In recent work on duplicate bug reports, SBERT and RoBERTa-based encoders form the state-of-the-art for retrieval and classification respectively, particularly when used in hybrid pipelines.[^2]

## 4. Hybrid Sparse–Dense Retrieval

Hybrid retrieval combines sparse and dense signals, typically by using BM25 and a dense encoder in parallel and then fusing their scores.
Multiple studies and practitioner reports find that hybrid search improves recall and robustness over either method alone, which is especially important for production systems where both exact identifier matches and semantic paraphrases must be captured.[^9][^6][^5]

Key points:

- **Fusion strategies**: Reciprocal Rank Fusion (RRF), weighted linear score combination, cascade where dense reranks BM25 candidates.[^6][^5]
- **Pros**: 
  - Dense embeddings recover semantically similar tickets with low lexical overlap.
  - BM25 ensures exact token matches (error codes, project names) are not dropped.
  - Hybrid systems usually achieve 15–30 percent relative recall improvements with modest additional complexity.[^6]
- **Cons**: Two indices to maintain; requires calibration of score fusion and ranking.

Given the complementary failure modes of BM25 and dense retrieval, hybrid sparse–dense search is a key model family to include for realistic Jira issue intelligence.[^5][^6]

## 5. Deep Neural Architectures for Pairwise Duplicate Classification

Beyond retrieval, many works treat duplicate detection as a supervised classification problem: given a pair of reports, predict whether they describe the same underlying bug.
Modern approaches use transformers for text and often fuse structured metadata.

Key points:

- **Siamese and cross-encoders**: 
  - SiameseQAT-style models, where two bug reports are encoded with shared BERT encoders and then compared via similarity or a small MLP using both text and metadata features.[^7][^2]
  - Cross-encoder transformers that jointly attend over concatenated report pairs for maximum accuracy (at higher cost).
- **Hybrid retrieval + classification**: Retrieval model narrows candidates; classification model makes final duplicate decision, balancing efficiency and accuracy.[^2]
- **Reported performance**: Transformer-based systems can significantly improve F1 and MAP versus classical baselines, but must be designed with runtime constraints in mind for large bug repositories.[^10][^2]

When designing an industrial system, these models are typically used as a second-stage ranker/classifier operating on a small candidate set returned by BM25 and/or dense retrieval.

## 6. LLM-Based and RAG-Style Approaches

Large language models (LLMs) offer two capabilities that are relevant for Jira issue intelligence:

- High-quality text embeddings for retrieval and clustering.
- Direct reasoning or classification over retrieved tickets with natural-language prompts.

Recent work on duplicate bug detection has started to compare classic encoders against embeddings derived from large language models and to propose amalgamated approaches that use both classical and LLM-based embeddings.[^1]
Other domain-specific duplication studies show that project-specific embedding models fine-tuned with LLM-generated augmentation can substantially improve semantic duplicate detection in specialized domains.[^11]

Key points:

- **Embeddings from LLMs**: Use modern LLM-based embedding APIs or open-source models to build vector indices for Jira issues; compare against SBERT-style models.
- **Prompted LLM classification**: Given a new issue and its top-k neighbors, prompt an LLM to decide whether each neighbor is a duplicate, possibly using chain-of-thought or few-shot prompts.[^12]
- **RAG-style workflows**: Retrieve top similar issues via hybrid search; pass them with the new issue into an LLM to explain similarity, justify duplicate decisions, or generate summaries to assist human triage.[^13][^12]

A comparative study should include at least:

- Classical sentence-transformer embeddings versus LLM-based embeddings for retrieval.[^1]
- LLM-in-the-loop classification versus lightweight learned classifiers for the duplicate decision.

## 7. Graph- and Metadata-Aware Models

Some approaches enrich textual similarity with structural or metadata information, such as component hierarchies, project relationships, and temporal patterns.
In related domains (e.g., power-service tickets), multimodal models combine text embeddings with geospatial and categorical encodings and then use transformer or MLP layers over the fused representation.[^12][^13]

For Jira, relevant signals include:

- Project and component fields, labels, versions, environments.
- Reporter/assignee, team, service, or subsystem identifiers.
- Temporal proximity (issues reported within a short time window).

Models:

- Embedding concatenation: concatenate text embeddings with one-hot or learned embeddings for metadata, and train a classifier or interaction model.[^12]
- Graph-based methods: construct graphs of issues and components and learn similarity that respects graph structure (less common in current bug-deduplication literature but promising by analogy with code-similarity work).[^14]

These approaches should be considered as variants layered on top of the core sparse/dense/LLM families rather than entirely separate baselines.

## Recommended Comparison Matrix

The following table summarizes the most important approach families and how they map to the retrieval and classification stages.

| Family | Example Models / Tools | Used for Retrieval | Used for Classification | Key Pros | Key Cons |
|--------|------------------------|--------------------|-------------------------|----------|----------|
| Sparse lexical IR | TF–IDF, BM25, BM25+ | Yes | Indirect (score threshold) | Simple, fast, strong baseline, great for identifiers | Misses paraphrases and synonyms, limited semantics[^5][^6][^3] |
| Classical ML with features | SVM, RF over lexical + metadata features | Candidate scoring | Yes | Integrates structured Jira fields, interpretable feature weights | Requires labeled pairs, limited semantic generalization[^3][^7] |
| Dense semantic encoders | SBERT, MPNet-based sentence transformers | Yes (ANN KNN) | Sometimes (pairwise similarity) | Good at semantic similarity and paraphrases; strong retrieval quality | Needs embedding infra, can miss exact codes without hybridization[^4][^2][^8] |
| Hybrid sparse–dense search | BM25 + dense embeddings with RRF | Yes | Indirect (ranking) | Best recall across lexical and semantic matches; robust in production | More complex indexing, score fusion tuning required[^5][^6][^9] |
| Deep duplicate classifiers | SiameseQAT, RoBERTa-based pair classifiers | No | Yes | Highest precision when used as second stage; can use text + metadata | Higher latency; needs labeled duplicates and non-duplicates[^2][^7] |
| LLM-based approaches | LLM embeddings, LLM re-rankers, RAG flows | Yes (embeddings) | Yes (prompted decisions) | Strong semantic reasoning, adaptable prompts, good for explanations | Costly, latency-sensitive, behavior can drift; needs guardrails[^1][^12][^11] |

These are the primary families that should be directly compared in an empirical study for Jira issue similarity and duplicate detection.

## Evaluation and Generalization Considerations

When comparing models, it is important to evaluate them not only on a single static dataset but across multiple realistic conditions.
Recent studies emphasize that deep models may only modestly improve over IR methods in some ranking setups and that hybrid approaches often offer the best practical trade-off.[^3][^2]

Recommended dimensions:

- **Top-k retrieval metrics**: Recall@k, Mean Average Precision (MAP), Mean Reciprocal Rank (MRR) for retrieving known duplicates.[^1][^3][^2]
- **Binary classification metrics**: Precision, recall, F1, ROC-AUC when making explicit duplicate/non-duplicate decisions.[^7][^3]
- **Latency and resource usage**: Indexing time, query latency under realistic load, memory footprint.
- **Robustness and domain shift**: Performance across different projects, products, and time periods (e.g., training on historical public datasets, testing on new internal projects).

Including these dimensions ensures that the models you pick will generalize and remain useful as project structures, teams, and reporting styles evolve.

## Practical Shortlist of Approaches to Benchmark

Concretely, for a Jira issue intelligence system, the following variants form a strong comparative suite:

1. **BM25-only retrieval baseline** with simple similarity scoring.
2. **BM25 + classical classifier** using lexical similarity and Jira metadata features.
3. **Sentence-transformer dense retrieval** (e.g., SBERT or MPNet-based) with FAISS ANN index, used alone.
4. **Hybrid BM25 + dense retrieval** with score fusion (RRF or weighted combination).
5. **Hybrid retrieval + transformer-based classifier** (SiameseQAT or RoBERTa variant) for final duplicate decisions.
6. **LLM-embedding-based dense retrieval** compared against sentence-transformer embeddings.
7. **RAG-style flow** where hybrid retrieval feeds an LLM that explains similarity and optionally labels duplicates.

Together, these models and architectural choices span the key trade-offs between performance, interpretability, latency, cost, and robustness observed in the literature on duplicate bug detection and semantic similarity for issue trackers.[^8][^4][^3][^1][^2]

---

## References

1. [Amalgamation of Classical and Large Language Models for Duplicate Bug Detection: A Comparative Study](https://www.techscience.com/cmc/v83n1/60067) - : Duplicate bug reporting is a critical problem in the software repositories’ mining area. Duplicate...

2. [Balancing Efficiency and Accuracy in Duplicate Bug Report Detection](https://arxiv.org/html/2404.14877v1) - [5] proposed a SiameseQAT approach, using BERT and MLP to concatenate structured and unstructured fe...

3. [Does Deep Learning improve the performance of duplicate bug ...](https://www.sciencedirect.com/science/article/abs/pii/S016412122300002X) - In this paper, we investigate whether well-known DL-based methods outperform classic information ret...

4. [[PDF] Duplicate Bug Report Detection by Using Sentence Embedding and ...](https://ceur-ws.org/Vol-3655/ISE2023_07_Lee_Duplicate_Bug.pdf) - SBERT uses the siamese network and the triplet network to do this with a fine tuned model of Bert, m...

5. [BM25 vs Dense Retrieval: When to Use Each - System Overflow](https://www.systemoverflow.com/learn/search-ranking/ranking-algorithms/bm25-vs-dense-retrieval-when-to-use-each)

6. [Dense vs Sparse Retrieval: Mastering FAISS, BM25, and Hybrid ...](https://dev.to/qvfagundes/dense-vs-sparse-retrieval-mastering-faiss-bm25-and-hybrid-search-4kb1) - BM25: The Industry Standard. BM25 improves on TF-IDF with document length normalization and saturati...

7. [A Semantic Context-Based Duplicate Bug Report Detection Using ...](https://ieeexplore.ieee.org/iel7/6287639/9312710/09380447.pdf) - Figure 4 presents a high-level representation of SiameseQAT, a deep Siamese neural network for repre...

8. [How we built this - Jira Similar Issues linking them all together](https://community.atlassian.com/forums/Atlassian-AI-Rovo-articles/Recording-Posted-How-we-built-this-Jira-Similar-Issues-linking/ba-p/2989889) - The Similar Issues feature focuses on finding and surfacing relevant context for Jira issues. It ide...

9. [BM25 vs Dense Retrieval for RAG: What Actually Breaks in Production](https://ranjankumar.in/bm25-vs-dense-retrieval-for-rag-engineers) - This is simplified, but it captures the key insight: BM25 makes sure you don't drop critical facts. ...

10. [Journal of Computational Analysis and Applications                                                              VOL. 29, NO. 4, 2021](https://eudoxuspress.com/index.php/pub/article/download/2952/2080/5696)

11. [LLM-Based Semantic Detection of Duplicate Power Projects](https://ieeexplore.ieee.org/document/11467073/) - To address the insufficiency of traditional text duplication detection methods in semantic-level rec...

12. [A Study on Association Analysis of Electric Power Service Requests Based on Multimodal Semantic Similarity and Prompt Engineering for Large Language Models](https://ieeexplore.ieee.org/document/11290796/) - We propose an association-analysis method for electric-power service tickets that combines multimoda...

13. [Ai agent to compare jira ticket info : r/Rag - Reddit](https://www.reddit.com/r/Rag/comments/1p0z9nq/ai_agent_to_compare_jira_ticket_info/) - If you want to rapidly dedupe by both geographic and semantic similarity you just need to concat the...

14. [Fus: Combining Semantic and Structural Graph Information for Binary Code Similarity Detection](https://www.mdpi.com/2079-9292/14/19/3781) - Binary code similarity detection (BCSD) plays an important role in software security. Recent deep le...

