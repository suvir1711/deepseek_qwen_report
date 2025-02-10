# Advanced Report on DeepSeek R1 Architecture

Below is a multi-part technical report that synthesizes the two reference diagrams and describes **DeepSeek R1** in detail. The first section focuses on the internal structure of a **single Transformer Block** as employed in the Mixture-of-Experts (MoE) mechanism (from the first image). The second section gives a **layer-by-layer breakdown** of the entire **DeepSeek R1 architecture** (as illustrated in the second image), showing how all components (Router, Experts, Multi-Head Latent Attention, etc.) come together. Numerical “dry-run” mini-examples are provided throughout so you can see how the mathematics might unfold with small input sizes.

---

## 1. Inside One Transformer Block (Mixture of Experts)

From the first figure (“Figure 2 | Illustration of the basic architecture of DeepSeek-V3…”), each Transformer block in the Mixture of Experts arrangement has the following sublayers:

1. **RMSNorm**  
2. **Attention (Multi-Head Latent Attention, MLA)**  
3. **RMSNorm**  
4. **Feed-Forward Network (FFN)** or **MoE** feed-forward (DeepSeekMoE)

All of these include residual connections around the attention and feed-forward sublayers. Below is the rationale and math for each piece.

---

### 1.1 RMSNorm Layer

**Purpose**  
- Normalizes the hidden states using Root Mean Square (RMS) normalization rather than LayerNorm.  
- Maintains numerical stability and helps with training dynamics.

**RMSNorm Formula**  
For a hidden vector \(\mathbf{h} \in \mathbb{R}^D\), RMSNorm can be written as:

\[
\mathrm{RMSNorm}(\mathbf{h}) \;=\; 
\frac{\mathbf{h}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} h_i^2 + \epsilon}} 
\;\odot\; \boldsymbol{\gamma}
\]

where:  
- \(D\) is the dimension of \(\mathbf{h}\).  
- \(\epsilon\) is a small constant to avoid division by zero.  
- \(\odot\) denotes element-wise multiplication.  
- \(\boldsymbol{\gamma} \in \mathbb{R}^D\) is a trainable scale parameter.

**Mini-Example**  
Suppose \(D=4\) and you have one token’s hidden state:

\[
\mathbf{h} = [2,\,-2,\,1,\,1].
\]

Then  
\[
\frac{1}{D}\sum_{i=1}^{4}h_i^2 
= \frac{1}{4}(2^2 + (-2)^2 + 1^2 + 1^2)
= \frac{1}{4}(4 + 4 + 1 + 1)
= 2.5.
\]

\[
\sqrt{2.5 + \epsilon} \approx \sqrt{2.5} = 1.58 
\quad (\text{assuming } \epsilon \text{ is very small}).
\]

If \(\boldsymbol{\gamma} = [1,\,1,\,1,\,1]\) (for simplicity), then

\[
\mathrm{RMSNorm}(\mathbf{h}) 
= \frac{[\,2,\,-2,\,1,\,1\,]}{1.58} 
\approx [\,1.27,\,-1.27,\,0.63,\,0.63\,].
\]

---

### 1.2 Multi-Head Latent Attention (MLA)

**Purpose**  
- Allows each token’s representation to attend to other tokens’ representations via multiple “heads.”  
- Uses query (\(\mathbf{Q}\)), key (\(\mathbf{K}\)), and value (\(\mathbf{V}\)) projections.  
- The “latent attention” extension can incorporate specialized latent vectors or gating (like RoPE, positional embeddings, or other advanced transformations).

**Mathematics**  
In a standard multi-head attention, each token embedding \(\mathbf{h}_t \in \mathbb{R}^D\) is projected into:

\[
\mathbf{q}_t = W^Q\,\mathbf{h}_t,\quad
\mathbf{k}_t = W^K\,\mathbf{h}_t,\quad
\mathbf{v}_t = W^V\,\mathbf{h}_t,
\]

where \(W^Q, W^K, W^V \in \mathbb{R}^{D \times D}\). Then attention for a single head is:

\[
\mathrm{Attn}(\mathbf{q}_t, \{\mathbf{k}_j\}, \{\mathbf{v}_j\}) 
= \sum_{j=1}^{T} \mathrm{softmax}\!\Bigl(\frac{\mathbf{q}_t \cdot \mathbf{k}_j}{\sqrt{d_k}}\Bigr)\,\mathbf{v}_j,
\]

where \(d_k = \tfrac{D}{\text{(number of heads)}}\).  
In multi-head attention, we do this for each head, then concatenate:

\[
\mathbf{o}_t = 
\bigl[\mathrm{Attn}^1; \mathrm{Attn}^2; \dots ; \mathrm{Attn}^H \bigr]\,W^O
\]
(\(H\) heads, plus a final projection \(W^O\)).

In **Multi-Head Latent Attention**, we may also incorporate:
- **RoPE** (Rotary Positional Embeddings) to transform \(\mathbf{k}_t\) and \(\mathbf{q}_t\).  
- **Latent vectors** \(\mathbf{c}_t^Q, \mathbf{c}_t^K\) (extra “latent” queries/keys) that can add further context.

The net result is an updated hidden state \(\mathbf{u}_t\), which merges standard attention with specialized latent transformations.

**Mini-Example**  
A toy multi-head attention with:

- \(D=4\).  
- 2 heads (\(H=2\)) \(\Rightarrow d_k=2\).  
- 2 tokens (\(t \in \{1, 2\}\)).

1. **Projecting inputs**  
   Let \(W^Q, W^K, W^V\) each be \(4\times4\) identity (for demonstration). For token 1, \(\mathbf{h}_1=[1,2,3,4]\). Then
   \[
   \mathbf{q}_1=[1,2,3,4],\quad \mathbf{k}_1=[1,2,3,4],\quad \mathbf{v}_1=[1,2,3,4].
   \]
   For token 2, \(\mathbf{h}_2=[2,0,1,0]\). Then
   \[
   \mathbf{q}_2=[2,0,1,0],\quad \mathbf{k}_2=[2,0,1,0],\quad \mathbf{v}_2=[2,0,1,0].
   \]

2. **Compute attention scores** (per head).  
   - Head 1 might use the first half of each vector, etc.  
   - Head 2 uses the second half.

3. **Softmax** over token dimension to gather values from each token.

Eventually, we get \(\mathbf{u}_t\) for each token after merging heads (plus any latent transformations).

---

### 1.3 Residual Connection

After the attention block, we add the original hidden state back:

\[
\mathbf{h}_t^{(\text{after attn})} = \mathbf{h}_t + \mathbf{u}_t.
\]

This helps gradient flow and stabilizes training.

---

### 1.4 Feed-Forward Network (FFN) or Mixture-of-Experts (MoE)

**Purpose**  
- Applies a position-wise nonlinear transformation to each token’s embedding.  
- In a conventional Transformer, this is typically a 2-layer MLP.  
- In **DeepSeekMoE**, it becomes a mixture-of-experts block, where the router chooses which feed-forward “expert(s)” to apply.

#### Conventional FFN

\[
\mathbf{z}_t = \sigma(W_1\,\mathbf{h}_t + \mathbf{b}_1),\quad
\mathbf{f}_t = W_2\,\mathbf{z}_t + \mathbf{b}_2,
\]

where \(\sigma\) is a nonlinear activation (e.g. GELU).

#### MoE FFN

Instead of one FFN, we have multiple “experts” \(\mathrm{FFN}_1,\mathrm{FFN}_2,\dots,\mathrm{FFN}_{N_r}\). A router produces gating weights \(\mathbf{g}_t\) to select top-\(k\) experts:

\[
\mathbf{f}_t = 
\sum_{r\in \text{Top-}k} g_{t,r}\,\mathrm{FFN}_r(\mathbf{h}_t).
\]

DeepSeek uses a large set of experts but activates only a small subset per token.

**Mini-Example**  
- 3 experts \(E_1, E_2, E_3\).  
- Gating vector \(\mathbf{g}_t = [0.7,\,0.3,\,0.0]\).  
- Output = \(0.7\,E_1(\mathbf{h}_t) + 0.3\,E_2(\mathbf{h}_t)\).

**Residual Connection**  
We again add the output of the FFN/MoE back to \(\mathbf{h}_t\):

\[
\mathbf{h}_t' 
= \mathbf{h}_t^{(\text{after attn})} + \mathbf{f}_t.
\]

---

## 2. Overall DeepSeek R1 Architecture

The second image shows a broader schematic:

1. **Input Data** flows into the **Router Network**.  
2. The router decides which experts in the **Mixture-of-Experts (MoE)** block activate.  
3. Only the active experts handle each token’s forward pass.  
4. The outputs from experts are merged in the **Expert Aggregator**.  
5. Then **Multi-Head Latent Attention** with a **KV cache** (for up to 128k tokens) plus latent compression.  
6. The resulting representations pass to **Attention Processing** and final heads for the model output.

Below is a layer-by-layer breakdown.

---

### 2.1 Router Network

**Purpose**  
- Takes token representations as input.  
- Computes a routing distribution so each token is sent to a small subset of experts.  
- Greatly reduces compute cost because only a fraction of the total parameters is active per token.

**Routing Math**  
If \(\mathbf{h}_t \in \mathbb{R}^D\), the router can compute:

\[
\mathbf{r}_t 
= W^{(\text{router})}\,\mathbf{h}_t + \mathbf{b}^{(\text{router})} 
\;\in\;\mathbb{R}^{N_r}.
\]

We convert \(\mathbf{r}_t\) into gating vector \(\mathbf{g}_t\in\mathbb{R}^{N_r}\). Often with **softmax top-\(k\)**:

1. \(\mathbf{p}_t = \mathrm{softmax}(\mathbf{r}_t)\).  
2. Sort \(\mathbf{p}_t\) to find top-\(k\).  
3. Zero out all but top-\(k\).

**Example**  
- \(N_r=3\).  
- \(\mathbf{r}_t=[3.4,\;1.1,\;2.2]\).  
- \(\mathrm{softmax}(\mathbf{r}_t)\approx[0.70,\,0.06,\,0.24]\).  
- If \(k=2\), \(\mathbf{g}_t=[0.70,\,0.0,\,0.24]\).  
- Experts 1 and 3 are active.

---

### 2.2 Mixture of Experts (MoE)

**Purpose**  
- Houses many feed-forward sub-networks (experts).  
- Each expert can specialize in different parts of representation space.  
- Only top-\(k\) experts run for each token; all are trained collectively.

Inside each “MoE block,” we have:

\[
\mathbf{f}_t 
= \sum_{r\in \text{Top-}k} g_{t,r}\,\mathrm{FFN}_r(\mathbf{h}_t).
\]

---

### 2.3 Expert Aggregator

**Purpose**  
- Collects outputs of whichever experts were active and merges them back into a single hidden vector per token.  
- May include load-balancing constraints.

After the aggregator, you have the next hidden layer representation \(\mathbf{h}_t'\).

---

### 2.4 KV Cache

**Purpose**  
- Stores key–value projections for previously processed tokens, avoiding redundant computations.  
- Enables context lengths up to 128k tokens.

**Example**  
When processing a new token at position \(t\):

\[
\mathbf{k}_t = W^K \,\mathbf{h}_t,\quad
\mathbf{v}_t = W^V \,\mathbf{h}_t.
\]
We append \(\mathbf{k}_t,\mathbf{v}_t\) to the cache for subsequent attention:

\[
\mathrm{Attn}(\mathbf{q}_t,\,[\mathbf{k}_1,\dots,\mathbf{k}_t],\,[\mathbf{v}_1,\dots,\mathbf{v}_t]).
\]

---

### 2.5 Latent Vector Compression

**Purpose**  
- Compresses or projects the keys/values into a lower-dimensional space.  
- Maintains representational capacity while controlling memory usage for very long contexts.

**Possible Math**  
\[
\tilde{\mathbf{k}}_t 
= W^{(\text{compress})}_k \,\mathbf{k}_t,\quad
\tilde{\mathbf{v}}_t
= W^{(\text{compress})}_v \,\mathbf{v}_t,
\]
reducing dimension from \(D\) to \(D'\).

---

### 2.6 Attention Processing → Model Output

After the MoE, caching, and compression steps, further multi-head attention blocks and final layers produce the **model output** (e.g. next-token logits). In the second figure:

1. Tokens go through the MoE block (router + experts).  
2. Expert aggregator merges outputs.  
3. Multi-Head Latent Attention uses the KV cache.  
4. Then final heads or further Transformer blocks produce the final layer.

A single Transformer block in **DeepSeek R1** looks like:

\[
\mathbf{h}_t
\,\xrightarrow{\text{RMSNorm}}\,
\text{(Multi-Head Latent Attention)}
\,\xrightarrow{+\mathbf{h}_t}\,
\,\xrightarrow{\text{RMSNorm}}\,
\text{(MoE FFN)}
\,\xrightarrow{+\text{previous}}\,
\mathbf{h}_t'.
\]

This repeats \(L\) times to form a deep stack.

---

## Putting It All Together

**DeepSeek R1** is a large-scale Transformer-type model that combines:

1. **Mixture of Experts** for the feed-forward portion of each Transformer block, drastically increasing total parameters while limiting active parameters per token.  
2. **Multi-Head Latent Attention** plus **KV caching** (up to 128k tokens) for extremely long sequences.  
3. **Router Network** to dynamically select experts (top-\(k\) gating).  
4. **RMSNorm** layers for stable training.  
5. **Residual connections** around both attention and feed-forward sublayers.

The overall flow (matching the second diagram) is:

1. **Input Data** → (Token Embeddings + Positional Embeddings)  
2. **Router Network** → pick top-\(k\) experts  
3. **Active Experts** (MoE) → feed-forward sub-networks  
4. **Expert Aggregator** → merges the outputs  
5. **KV Cache** → store \(\mathbf{k},\mathbf{v}\) states for long contexts  
6. **Latent Vector Compression** → keep memory usage feasible  
7. **Attention** → standard multi-head or latent extension  
8. **Attention Processing** → final or subsequent Transformer blocks  
9. **Model Output** → e.g. next-token probabilities

At each Transformer block:

\[
\mathbf{h}_t
\;\;\to\;\;\mathrm{RMSNorm}
\;\;\to\;\;\mathrm{MultiHeadLatentAttention}
\;\;\to\;\;\mathbf{h}_t + \dots
\;\;\to\;\;\mathrm{RMSNorm}
\;\;\to\;\;\mathrm{MoE\;FFN}
\;\;\to\;\;\mathbf{h}_t + \dots
\;\;=\;\;\mathbf{h}_t'
\]

---

### Summary of Key Ideas

1. **Layer-by-Layer**  
   - **RMSNorm** normalizes the hidden vectors.  
   - **Multi-Head Latent Attention** transforms them via queries/keys/values (optionally with specialized latent vectors or RoPE).  
   - **Feed-Forward (MoE)** uses a router to pick top-\(k\) experts.  
   - **Residual connections** preserve the original signal and improve training stability.

2. **Router + MoE**  
   - A router chooses which experts to run per token.  
   - This lets the total parameter count be huge, but only a fraction is active per token (saving compute).

3. **KV Cache + Latent Compression**  
   - Speeds up inference by reusing \(\mathbf{k}, \mathbf{v}\) from previous tokens.  
   - Enables extremely long context lengths (up to 128k tokens).

4. **Final Output**  
   - After multiple Transformer + MoE blocks, the model produces final representations or next-token logits.

This combination of **dynamic routing**, **expert specialization**, and **long-context attention** defines **DeepSeek R1**, enabling a large parameter model (236 B total) with only ~21 B active per token (per the second diagram).
  
