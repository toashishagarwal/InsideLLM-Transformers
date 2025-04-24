# InsideLLM-Transformers
How do LLMs work internally? <br>
The transformers are at the heart of an LLM architecture. The fig below shows a model Tranformer architecture.
Each token in the input query - "The next day is bright" goes through these transformer layers to predict the next token, say "and".

![Alt Text](img/TransformerArchitecture.png)

It starts with tokenising the input & creating a vector representation of each token. This vector representation is a long numeric interpretation often referred as embedding. In addition to the vector representation, we also need the positional information of each token in the input query. This is important because word order is critical for understanding language - "dog bites man" means something very different from "man bites dog". This is where the positional embedding comes into picture. **Positional Emdedding** allows transformers to understand the order of tokens in language. Once we have both these vectors, they are passed through the transformer block and starts with Multi Head Attention (MHA)

To understand MHA, we need to first understand Self Attention.

This repo contains the fundamentals of LLM transformers starting from Self Attention & then gradually progressing to MHA and Causal Attention.

## Self Attention
Self-attention is a fundamental mechanism in modern language models that allows each token in a sequence to attend to (or "focus on") all other tokens in the same sequence. Here's how it works:

### Basic Self-Attention
1) Query, Key, Value Projections:
For each token in the input sequence, we create three vectors:

* Query (Q): Represents what the token is "looking for"
* Key (K): Represents what the token "contains" or "offers"
* Value (V): Contains the actual information to be aggregated <br>

2) Attention Score Calculation:
For each position, we calculate how much focus (i.e attention) to put on each other position.
This is done via **dot product** between the Query of the current token and Keys of all tokens. Dot product is one of the means to calculate similarity between 2 tokens. Scores are scaled by dividing by √d (where d is embedding dimension) to prevent exploding gradients

3) Softmax & Weighting:
Scores are passed through softmax to get weights between 0 and 1 that sum to 1. This is done so that it becomes easier to understand the percentage contribution of each token to attention. These weights are applied to the Value vectors. The weighted sum becomes the output for the current position

## Causal Attention
Coming Soon...

## Multi Head Attention
Coming Soon...

## Key Value cache
Coming Soon...

## Multi Head Latent Attention
Coming Soon...
