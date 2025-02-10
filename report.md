# Advanced Analysis of Qwen and DeepSeek LLM Architectures

---

## Table of Contents  
1. **Introduction**  
2. **DeepSeek R1 Architecture**  
   2.1 Transformer Block Components  
   2.2 Mixture-of-Experts (MoE) System  
   2.3 Training Process (GRPO)  
   2.4 Distillation to Smaller Models  
3. **Qwen Large Language Model Architecture**  
   3.1 Embedding Layer  
   3.2 Rotary Positional Embedding (RoPE)  
   3.3 Attention Mechanism  
   3.4 Feed-Forward Network (SwiGLU)  
   3.5 RMSNorm  
   3.6 Dynamic NTK-Aware Interpolation  
   3.7 Output Projection  
4. **Summary**  

---

## 1. Introduction  
This report provides a technical analysis of two advanced language models: **DeepSeek-R1** (focused on reasoning via reinforcement learning and MoE) and **Qwen** (optimized for efficiency and long-context processing). Each section dissects their architectural innovations, mathematical formulations, and training methodologies.

---

## 2. DeepSeek R1 Architecture  

### 2.1 Transformer Block Components  
**Purpose**: Enable efficient computation and reasoning through MoE and latent attention.  

#### 2.1.1 RMSNorm Layer  
**Purpose**: Stabilize training via root-mean-square normalization.  
**Mathematics**:  
\[
\mathrm{RMSNorm}(\mathbf{h}) = \frac{\mathbf{h}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} h_i^2 + \epsilon}} \odot \boldsymbol{\gamma}.
\]  
**Example**:  
For \(\mathbf{h} = [2, -2, 1, 1]\):  
\[
\mathrm{RMSNorm}(\mathbf{h}) \approx [1.27, -1.27, 0.63, 0.63].
\]  

#### 2.1.2 Multi-Head Latent Attention (MLA)  
**Purpose**: Attend to token relationships with positional and latent enhancements.  
**Mathematics**:  
\[
\mathbf{q}_t = W^Q\mathbf{h}_t, \quad \mathbf{k}_t = W^K\mathbf{h}_t, \quad \mathbf{v}_t = W^V\mathbf{h}_t.
\]  
**Example**:  
For \(\mathbf{h}_1 = [1, 2, 3, 4]\) and \(\mathbf{h}_2 = [2, 0, 1, 0]\), compute attention scores across 2 heads.  

#### 2.1.3 Residual Connections  
**Purpose**: Improve gradient flow.  
\[
\mathbf{h}_t^{(\text{after attn})} = \mathbf{h}_t + \mathbf{u}_t.
\]  

#### 2.1.4 MoE Feed-Forward Network  
**Purpose**: Activate specialized experts per token.  
**Mathematics**:  
\[
\mathbf{f}_t = \sum_{r\in \text{Top-}k} g_{t,r}\,\mathrm{FFN}_r(\mathbf{h}_t).
\]  
**Example**:  
For experts \(E_1, E_2, E_3\) and \(\mathbf{g}_t = [0.7, 0.3, 0.0]\), output = \(0.7E_1(\mathbf{h}_t) + 0.3E_2(\mathbf{h}_t)\).  

---

### 2.2 Mixture-of-Experts (MoE) System  
**Purpose**: Balance parameter count and computational cost.  

#### 2.2.1 Router Network  
**Mathematics**:  
\[
\mathbf{r}_t = W^{(\text{router})}\mathbf{h}_t + \mathbf{b}^{(\text{router})}, \quad \mathbf{g}_t = \text{Top-}k(\mathrm{softmax}(\mathbf{r}_t)).
\]  
**Example**:  
For \(\mathbf{r}_t = [3.4, 1.1, 2.2]\), \(\mathbf{g}_t = [0.70, 0.0, 0.24]\).  

#### 2.2.2 Expert Aggregator  
**Purpose**: Merge outputs from activated experts.  

---

### 2.3 Training Process (GRPO)  
**Purpose**: Optimize reasoning capabilities via reinforcement learning.  

#### 2.3.1 GRPO Algorithm  
**Mathematics**:  
\[
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A_i\right) - \beta \mathbb{D}_{\text{KL}}\right].
\]  
**Key Steps**:  
1. Sample \(G\) responses per prompt.  
2. Compute rewards (accuracy + format).  
3. Normalize advantages within groups.  

#### 2.3.2 Performance  
- AIME pass@1 improves from **15.6% → 71.0%**.  

---

### 2.4 Distillation to Smaller Models  
**Purpose**: Transfer reasoning patterns to efficient models.  
**Results**:  
- Distilled 7B model: MATH-500 **92.8%** (vs. GPT-4o’s 74.6%).  

---

## 3. Qwen Large Language Model Architecture  

### 3.1 Embedding Layer  
**Purpose**: Map tokens to continuous vectors.  
**Mathematics**:  
\[
\text{Embedding}(i) = E[i] \in \mathbb{R}^d.
\]  
**Example**:  
Token ID `[5, 3]` → \(\begin{bmatrix} 0.1 & -0.2 & 0.4 \\ -0.3 & 0.5 & 0.7 \end{bmatrix}\).  

---

### 3.2 Rotary Positional Embedding (RoPE)  
**Purpose**: Encode positional information via rotation.  
**Mathematics**:  
\[
q_m' = q_m \odot e^{im\theta}, \quad k_n' = k_n \odot e^{in\theta}.
\]  
**Example**:  
For \(q = [1.0, 2.0]\), \(k = [3.0, 4.0]\), rotated dot product ≈ **9.191**.  

---

### 3.3 Attention Mechanism  
**Purpose**: Stabilize attention scores for long contexts.  

#### 3.3.1 LogN-Scaling  
**Mathematics**:  
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \cdot \log N\right)V.
\]  

#### 3.3.2 Window Attention  
**Mathematics**:  
\[
\text{Attention}(Q_i, K_{i-w:i+w}, V_{i-w:i+w}).
\]  

---

### 3.4 Feed-Forward Network (SwiGLU)  
**Purpose**: Dynamic gating for feature interaction.  
**Mathematics**:  
\[
\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV).
\]  
**Example**:  
Input \(x = [2.0, -1.0]\) → Output \([1.76, -0.135]\).  

---

### 3.5 RMSNorm  
**Purpose**: Normalize without centering.  
**Mathematics**:  
\[
x' = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}.
\]  
**Example**:  
Input \(x = [1.0, 2.0, 3.0]\) → Output \([0.46, 0.92, 1.38]\).  

---

### 3.6 Dynamic NTK-Aware Interpolation  
**Purpose**: Extend context length dynamically.  
**Mathematics**:  
\[
\theta_j' = \theta_j \cdot (1 + \gamma \log N).
\]  
**Example**:  
Original \(\theta_j = 0.01\) → Scaled \(\theta_j' \approx 0.018\).  

---

### 3.7 Output Projection  
**Purpose**: Generate vocabulary logits.  
**Mathematics**:  
\[
\text{logits} = hW_{\text{out}} + b_{\text{out}}.
\]  
**Example**:  
Input \(h = [0.8, -0.5]\) → Logits \([1.4, -1.5]\).  

---

## 4. Summary  

| **Model**         | **Key Innovations**                                                                 |  
|--------------------|-------------------------------------------------------------------------------------|  
| **DeepSeek-R1**    | MoE architecture, GRPO training, Cold-start SFT pipeline.                          |  
| **Qwen**           | NTK-aware RoPE, SwiGLU activation, LogN-scaled attention.                          |  

**DeepSeek-R1** excels in reasoning tasks through RL and expert specialization, while **Qwen** prioritizes efficiency and long-context processing. Both architectures demonstrate state-of-the-art performance in their target domains.  
