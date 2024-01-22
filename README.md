# Self-BioRAG: Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models

This is a repository for [Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models]()
 by Minbyul Jeong, Jiwoong Sohn, Mujeen Sung, and Jaewoo Kang.

[7B Model]() | [13B Model]() | [Paper]() | [Training data]() | [Summary]()

**Self-BioRAG** is a framework reliable for biomedical and clinical text instructions that specializes in self-reflection to retrieve, criticize, and generate explanations while preserving generation quality and reasoning ability.

The retrieval-augmented generation (RAG) framework performs document searches unconditionally, regardless of whether a query requires document retrieval. However, **Self-BioRAG** framework use domain-specific instruction-tuned language model to predict whether retrieval is necessary for given query. If the query doesn't require any retrieval of knowledge (factual contents), it directly predicts the answer. If the query necessitates retrieval knowledge, **Self-BioRAG** utilizes the domain-specific retriever (MedCPT, in our case) to retrieve relevant documents. After retriever the top-k evidence, the language model selects the most pertinent evidence for the query. Ultimately, our language model is employed to select the best evidence and generate the answer based on the selected evidence and encoded knowledge.

![](figures/intro_figure.png)

## Updates

## Content
1. [Installation](#installation)
2. [Quick Usage](#quick-usage)
3. [Retriever](#retriever)
4. [Critic LM](#critic-lm)
5. [Generator LM](#generator-lm)
6. [Inference](#inference)
7. [FAQ](#faq)
8. [Citation](#citation)
9. [Contact Information](#contact-information)

## Installation
installation



## Quick Usage
quick usage

![](figures/example_figure.png)

## Retriever
retriever

## Critic LM
critic lm

## Generator LM
generator lm

## Inference
inference 

## FAQ
FAQ

## Citation
Citation

## Contact Information
For help or issues using **Self-BioRAG**, please submit a GitHub issue. Please contact Minbyul Jeong (`minbyuljeong (at) korea.ac.kr`) for communication related to **Self-BioRAG**.



