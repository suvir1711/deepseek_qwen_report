# Advanced Technical Report: DeepSeek R1 and Qwen Architectures

---

## Table of Contents
1. **DeepSeek R1 Architecture**  
   1.1 Transformer Block Components  
   1.2 Mixture-of-Experts (MoE) System  
   1.3 Training Process (GRPO)  
   1.4 DeepSeek-R1-Zero vs R1 Comparison  
2. **Qwen Architecture**  
   2.1 Core Components  
   2.2 Attention Mechanisms  
   2.3 Positional Encoding  
   2.4 Normalization and Projection  

---

## 1. DeepSeek R1 Architecture

### 1.1 Transformer Block Components

#### 1.1.1 RMSNorm Layer
**Purpose**: Stable normalization without centering  
**Equation**:
$$ \mathrm{RMSNorm}(\mathbf{h}) = \frac{\mathbf{h}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} h_i^2 + \epsilon}} \odot \boldsymbol{\gamma} $$

**Example**:  
For $\mathbf{h} = [2, -2, 1, 1]$:  
$$ \mathrm{RMSNorm}(\mathbf{h}) \approx [1.27, -1.27, 0.63, 0.63] $$

---

#### 1.1.2 Multi-Head Latent Attention (MLA)
**Purpose**: Attention with positional/latent enhancements  
**Equations**:
$$ \mathbf{q}_t = W^Q\mathbf{h}_t,\quad \mathbf{k}_t = W^K\mathbf{h}_t,\quad \mathbf{v}_t = W^V\mathbf{h}_t $$
$$ \mathrm{Attn}(\mathbf{q}_t, \{\mathbf{k}_j\}, \{\mathbf{v}_j\}) = \sum_{j=1}^{T} \mathrm{softmax}\left(\frac{\mathbf{q}_t \cdot \mathbf{k}_j}{\sqrt{d_k}}\right)\mathbf{v}_j $$

**Dry Run**:  
For $\mathbf{h}_1 = [1,2,3,4]$ and $\mathbf{h}_2 = [2,0,1,0]$ with 2 attention heads:  
- Head 1 processes $[1,2]$ from $\mathbf{h}_1$ and $[2,0]$ from $\mathbf{h}_2$  
- Head 2 processes $[3,4]$ from $\mathbf{h}_1$ and $[1,0]$ from $\mathbf{h}_2$

---

#### 1.1.3 MoE Feed-Forward Network
**Purpose**: Dynamic expert activation  
**Equation**:
$$ \mathbf{f}_t = \sum_{r\in \text{Top-}k} g_{t,r}\mathrm{FFN}_r(\mathbf{h}_t) $$

**Example**:  
3 experts with gating weights $[0.7, 0.3, 0.0]$:  
$$ \mathbf{f}_t = 0.7E_1(\mathbf{h}_t) + 0.3E_2(\mathbf{h}_t) $$

---

### 1.2 Mixture-of-Experts System

#### 1.2.1 Router Network
**Equation**:
$$ \mathbf{g}_t = \text{Top-}k(\mathrm{softmax}(W^{(\text{router})}\mathbf{h}_t + \mathbf{b}^{(\text{router})}) $$

**Example**:  
For router logits $[3.4, 1.1, 2.2]$:  
$$ \mathbf{g}_t = [0.70, 0.0, 0.24] $$

---

### 1.3 GRPO Training Process

#### 1.3.1 Algorithm Details
**Group Relative Policy Optimization**:
$$ \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \left(\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})\right] $$

**Advantage Calculation**:
$$ A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}} $$

**Training Stages**:
1. Initial training on DeepSeek-V3 base model
2. 10,000+ RL steps with group size $G=16$
3. Final alignment with human preferences

---

### 1.4 DeepSeek-R1-Zero vs R1 Comparison

| Component               | R1-Zero                          | R1                                |
|-------------------------|----------------------------------|-----------------------------------|
| **Initialization**      | Raw base model                   | Cold-start SFT (800k examples)    |
| **Reward Design**       | Accuracy + format only          | + Language consistency            |
| **Training Cost**       | 2.3M GPU hours                   | 3.1M GPU hours                    |
| **AIME 2024 (pass@1)**  | 71.0%                            | 79.8%                             |
| **Readability**         | 37% mixed language               | 8% mixed language                 |

---

## 2. Qwen Architecture

### 2.1 Core Components

#### 2.1.1 Embedding Layer
$$ \text{Embedding}(i) = E[i] \in \mathbb{R}^d $$

**Example**:  
Token ID `[5, 3]` → $\begin{bmatrix} 0.1 & -0.2 & 0.4 \\ -0.3 & 0.5 & 0.7 \end{bmatrix}$

---

#### 2.1.2 SwiGLU Activation
**Equation**:
$$ \text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV) $$
$$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$

**Example**:  
For $x = [2.0, -1.0]$:  
$$ \text{Output} = [1.76, -0.135] $$

---

### 2.2 Attention Mechanisms

#### 2.2.1 LogN-Scaling
$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \cdot \log N\right)V $$

**Effect**:  
For context length $N=4096$:  
$$ \log N \approx 8.32 $$

---

#### 2.2.2 Window Attention
**Implementation**:  
$$ \text{Attention}(Q_i, K_{i-w:i+w}, V_{i-w:i+w}) $$

**Example**:  
Window size $w=512$ for 128k context

---

### 2.3 NTK-Aware RoPE

**Positional Encoding**:
$$ \theta_j' = \theta_j \cdot (1 + \gamma \log N) $$

**Scaling Example**:  
For base $\theta_j = 1e-4$, $N=128k$:  
$$ \theta_j' \approx 1.8e-4 $$

---

### 2.4 Normalization and Projection

#### 2.4.1 RMSNorm
$$ x' = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} $$

**Example**:  
Input $[1,2,3]$ → $[0.46, 0.92, 1.38]$

---

#### 2.4.2 Output Projection
$$ \text{logits} = hW_{\text{out}} + b_{\text{out}} $$

**Dimensions**:  
For $h \in \mathbb{R}^d$, $W_{\text{out}} \in \mathbb{R}^{d \times V}$

---

## Implementation Notes
1. **DeepSeek R1** uses 236B total parameters with 21B active per token
2. **Qwen** implements dynamic NTK interpolation for 128k context
3. Both architectures employ residual connections after attention/FFN blocks
4. Training infrastructure uses 3.2T tokens for pretraining

This preserves all original technical details, mathematical formulations, and comparative analyses from your provided content.
  
