# Comprehensive Analysis of Modern LLM Architectures: DeepSeek R1 and Qwen

## Table of Contents
1. [DeepSeek R1 Architecture](#deepseek-r1-architecture)
   - [Inside One Transformer Block](#inside-one-transformer-block)
   - [Overall Architecture](#overall-deepseek-r1-architecture)
   - [Putting It All Together](#putting-it-all-together)
2. [Qwen Architecture](#qwen-architecture)
   - [Overview and Significance](#overview)
   - [Core Components](#core-components)
   - [Implementation Details](#implementation-details)

# DeepSeek R1 Architecture

## Inside One Transformer Block (Mixture of Experts)

Each Transformer block in the Mixture of Experts arrangement contains:

1. RMSNorm
2. Attention (Multi-Head Latent Attention, MLA)
3. RMSNorm
4. Feed-Forward Network (FFN) or MoE feed-forward (DeepSeekMoE)

### 1.1 RMSNorm Layer

**Purpose**
- Normalizes hidden states using Root Mean Square (RMS) normalization
- Maintains numerical stability and helps with training dynamics

**RMSNorm Formula**
For a hidden vector $\mathbf{h} \in \mathbb{R}^D$, RMSNorm is:

$$
\mathrm{RMSNorm}(\mathbf{h}) = \frac{\mathbf{h}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} h_i^2 + \epsilon}} \odot \boldsymbol{\gamma}
$$

where:
- $D$ is the dimension of $\mathbf{h}$
- $\epsilon$ is a small constant
- $\odot$ denotes element-wise multiplication
- $\boldsymbol{\gamma} \in \mathbb{R}^D$ is a trainable scale parameter

**Mini-Example**
For $D=4$ and hidden state:
$$
\mathbf{h} = [2,\,-2,\,1,\,1]
$$

Then:
$$
\frac{1}{D}\sum_{i=1}^{4}h_i^2 = \frac{1}{4}(2^2 + (-2)^2 + 1^2 + 1^2) = \frac{1}{4}(4 + 4 + 1 + 1) = 2.5
$$

$$
\sqrt{2.5 + \epsilon} \approx \sqrt{2.5} = 1.58
$$

If $\boldsymbol{\gamma} = [1,\,1,\,1,\,1]$:
$$
\mathrm{RMSNorm}(\mathbf{h}) = \frac{[\,2,\,-2,\,1,\,1\,]}{1.58} \approx [\,1.27,\,-1.27,\,0.63,\,0.63\,]
$$

### 1.2 Multi-Head Latent Attention (MLA)

**Purpose**
- Enables token representation attention via multiple heads
- Uses query ($\mathbf{Q}$), key ($\mathbf{K}$), and value ($\mathbf{V}$) projections
- Incorporates latent vectors for specialized transformations

**Mathematics**
Standard multi-head attention projections:

$$
\mathbf{q}_t = W^Q\,\mathbf{h}_t,\quad
\mathbf{k}_t = W^K\,\mathbf{h}_t,\quad
\mathbf{v}_t = W^V\,\mathbf{h}_t
$$

Single head attention:
$$
\mathrm{Attn}(\mathbf{q}_t, \{\mathbf{k}_j\}, \{\mathbf{v}_j\}) = \sum_{j=1}^{T} \mathrm{softmax}\!\left(\frac{\mathbf{q}_t \cdot \mathbf{k}_j}{\sqrt{d_k}}\right)\,\mathbf{v}_j
$$

Multi-head concatenation:
$$
\mathbf{o}_t = \bigl[\mathrm{Attn}^1; \mathrm{Attn}^2; \dots ; \mathrm{Attn}^H \bigr]\,W^O
$$

### 1.3 Residual Connection

After attention:
$$
\mathbf{h}_t^{(\text{after attn})} = \mathbf{h}_t + \mathbf{u}_t
$$

### 1.4 Feed-Forward Network (FFN) or Mixture-of-Experts (MoE)

**Conventional FFN**:
$$
\mathbf{z}_t = \sigma(W_1\,\mathbf{h}_t + \mathbf{b}_1),\quad
\mathbf{f}_t = W_2\,\mathbf{z}_t + \mathbf{b}_2
$$

**MoE FFN**:
$$
\mathbf{f}_t = \sum_{r\in \text{Top-}k} g_{t,r}\,\mathrm{FFN}_r(\mathbf{h}_t)
$$

## Overall DeepSeek R1 Architecture

The architecture follows:

1. Input Data → Router Network
2. Router selects MoE experts
3. Active experts process tokens
4. Expert Aggregator merges outputs
5. Multi-Head Latent Attention with KV cache
6. Attention Processing → final output

### 2.1 Router Network

Router computation:
$$
\mathbf{r}_t = W^{(\text{router})}\,\mathbf{h}_t + \mathbf{b}^{(\text{router})} \in\mathbb{R}^{N_r}
$$

### 2.2 Mixture of Experts (MoE)

Inside MoE block:
$$
\mathbf{f}_t = \sum_{r\in \text{Top-}k} g_{t,r}\,\mathrm{FFN}_r(\mathbf{h}_t)
$$

### 2.3 KV Cache

For new token position $t$:
$$
\mathbf{k}_t = W^K \,\mathbf{h}_t,\quad
\mathbf{v}_t = W^V \,\mathbf{h}_t
$$

### 2.4 Latent Vector Compression

Compression projections:
$$
\tilde{\mathbf{k}}_t = W^{(\text{compress})}_k \,\mathbf{k}_t,\quad
\tilde{\mathbf{v}}_t = W^{(\text{compress})}_v \,\mathbf{v}_t
$$

## Putting It All Together

Single Transformer block flow:
$$
\mathbf{h}_t \to \mathrm{RMSNorm} \to \mathrm{MultiHeadLatentAttention} \to \mathbf{h}_t + \dots \to \mathrm{RMSNorm} \to \mathrm{MoE\;FFN} \to \mathbf{h}_t + \dots = \mathbf{h}_t'
$$

# Qwen Architecture

## Overview

Qwen is a state-of-the-art LLM series by Alibaba featuring:
- Multilingual support
- Tool usage capabilities
- Code interpretation
- Mathematical reasoning
- Parameter sizes: 1.8B, 7B, 14B

## Core Components

### 1. Embedding Layer

**Mathematics**:
For token ID $i$:
$$
\text{Input: Token ID } i \quad \Rightarrow \quad \text{Output: } E[i] \in \mathbb{R}^d
$$

### 2. Rotary Positional Embedding (RoPE)

For position $m$, angle $\theta_j = 10000^{-2j/d}$:
$$
q_m' = q_m \odot e^{im\theta}, \quad k_n' = k_n \odot e^{in\theta}
$$

Dot product:
$$
q_m' \cdot k_n' = \text{Re}\left[\sum_{j} q_{m,j} k_{n,j} e^{i(m-n)\theta_j}\right]
$$

### 3. Attention Mechanism

#### LogN-Scaling:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \cdot \log N\right)V
$$

#### Window Attention:
$$
\text{Attention}(Q_i, K_{i-w:i+w}, V_{i-w:i+w})
$$

### 4. Feed-Forward Network (SwiGLU)

$$
\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV)
$$

### 5. RMSNorm

$$
x' = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}
$$

### 6. QKV Projections with Bias

$$
Q = xW_Q + b_Q, \quad K = xW_K + b_K, \quad V = xW_V + b_V
$$

### 7. Dynamic NTK-Aware Interpolation

RoPE base frequency adjustment:
$$
\theta_j' = \theta_j \cdot \left(1 + \gamma \log N\right)
$$

### 8. Output Projection

$$
\text{logits} = hW_{\text{out}} + b_{\text{out}}
$$

## Implementation Details

- Window sizes vary by layer (512 to 2048 tokens)
- Efficient long-context processing through NTK-aware interpolation
- Integration of RoPE, SwiGLU, and RMSNorm for optimal performance
- Bias in QKV projections for fine-grained control
