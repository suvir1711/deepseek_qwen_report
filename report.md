================== START OF TEXT FILE ==================

Below is a multi-part technical report that synthesizes the two reference diagrams and describes **DeepSeek R1** in detail. The first section focuses on the internal structure of a **single Transformer Block** as employed in the Mixture-of-Experts (MoE) mechanism (from the first image). The second section gives a **layer-by-layer breakdown** of the entire **DeepSeek R1 architecture** (as illustrated in the second image), showing how all components (Router, Experts, Multi-Head Latent Attention, etc.) come together. Numerical “dry-run” mini-examples are provided throughout so you can see how the mathematics might unfold with small input sizes.

------------------------------------------------------------
1. Inside One Transformer Block (Mixture of Experts)
------------------------------------------------------------

From the first figure (“Figure 2 | Illustration of the basic architecture of DeepSeek-V3…”), each Transformer block in the Mixture of Experts arrangement has the following sublayers:

1. RMSNorm  
2. Attention (Multi-Head Latent Attention, MLA)  
3. RMSNorm  
4. Feed-Forward Network (FFN) or MoE feed-forward (DeepSeekMoE)

All of these include residual connections around the attention and feed-forward sublayers. Below is the rationale and math for each piece.

------------------------------------------------------------
1.1 RMSNorm Layer
------------------------------------------------------------

**Purpose**  
- Normalizes the hidden states using Root Mean Square (RMS) normalization rather than LayerNorm.  
- Maintains numerical stability and helps with training dynamics.

**RMSNorm Formula**  
For a hidden vector h ∈ ℝ^D, RMSNorm can be written as:

   RMSNorm(h) = ( h / sqrt( (1/D) * ∑(hᵢ²) + ϵ ) ) ⊙ γ

where:  
- D is the dimension of h.  
- ϵ is a small constant to avoid division by zero.  
- ⊙ denotes element-wise multiplication.  
- γ ∈ ℝ^D is a trainable scale parameter.

**Mini-Example**  
Suppose D = 4 and you have one token’s hidden state:

   h = [2, -2, 1, 1].

Then

   (1/D) * ∑(hᵢ²) = (1/4) * (2² + (-2)² + 1² + 1²)
                   = (1/4) * (4 + 4 + 1 + 1)
                   = 10/4
                   = 2.5.

   sqrt(2.5 + ϵ) ≈ sqrt(2.5) = 1.58   (assuming ϵ very small).

If γ = [1, 1, 1, 1] (for simplicity in this example), then

   RMSNorm(h) = [2, -2, 1, 1] / 1.58
              ≈ [1.27, -1.27, 0.63, 0.63].

------------------------------------------------------------
1.2 Multi-Head Latent Attention (MLA)
------------------------------------------------------------

**Purpose**  
- Allows each token’s representation to attend to other tokens’ representations via multiple “heads.”  
- Uses query (Q), key (K), and value (V) projections.  
- The “latent attention” extension can incorporate specialized latent vectors or gating (like RoPE, positional embeddings, or other advanced transformations).

**Mathematics**  
In a standard multi-head attention, each token embedding hₜ ∈ ℝ^D is projected into:

   qₜ = W^Q * hₜ,
   kₜ = W^K * hₜ,
   vₜ = W^V * hₜ,

where W^Q, W^K, W^V ∈ ℝ^(D×D). Then attention for a single head is:

   Attn(qₜ, {kⱼ}, {vⱼ})
     =  ∑( softmax( (qₜ · kⱼ) / sqrt(dₖ ) ) * vⱼ ),

where dₖ = D / (number of heads).  
In multi-head attention, we do this for each head h, then concatenate:

   oₜ = [Attn¹; Attn²; …; Attn^H] * W^O

(H heads, plus a final projection W^O).

In **Multi-Head Latent Attention**, we may also incorporate:
- **RoPE** (Rotary Positional Embeddings): transforms kₜ and qₜ by rotation.  
- **Latent vectors** cₜ^Q, cₜ^K (extra “latent” queries/keys) that can add further context or compression.

The net result is an updated hidden state uₜ, which merges the standard attention with specialized latent transformations.

**Mini-Example**  
Let’s do a very small multi-head attention with:  
- D = 4,  
- 2 heads (H = 2), so dₖ = 2.  
- Suppose you have 2 tokens, so t ∈ {1,2}.

(1) Projecting the inputs  
Let W^Q, W^K, W^V each be 4×4. In a real system they are large, but we’ll choose a small sample:

   W^Q = [ [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1] ],
   W^K = …,
   W^V = …,

(Identity for demonstration.) For token 1: h₁ = [1, 2, 3, 4]. Then

   q₁ = [1, 2, 3, 4],
   k₁ = [1, 2, 3, 4],
   v₁ = [1, 2, 3, 4].

For token 2: h₂ = [2, 0, 1, 0]. Then

   q₂ = [2, 0, 1, 0],
   k₂ = [2, 0, 1, 0],
   v₂ = [2, 0, 1, 0].

(2) Compute attention scores (per head).  
- Head 1 uses first half of each vector: q₁^1 = [1, 2], k₂^1 = [2, 0], etc.  
- Head 2 uses second half, etc.

(3) Softmax over the token dimension to gather values from each token.  

By the end, we get uₜ for each token after merging heads and possibly applying the latent transformations (RoPE, etc.).

------------------------------------------------------------
1.3 Residual Connection
------------------------------------------------------------

After the attention block, we add the original hidden state back:

   hₜ^(after attn) = hₜ + uₜ.

This helps gradient flow and stabilizes training.

------------------------------------------------------------
1.4 Feed-Forward Network (FFN) or Mixture-of-Experts (MoE)
------------------------------------------------------------

**Purpose**  
- Applies a position-wise nonlinear transformation to each token’s embedding.  
- In a conventional Transformer, this is usually a 2-layer MLP.  
- In **DeepSeekMoE**, it becomes a mixture-of-experts block, where the router chooses which feed-forward “expert(s)” to apply to each token.

**Conventional FFN**  
A typical feed-forward might be:

   zₜ = σ(W₁ * hₜ + b₁),
   fₜ = W₂ * zₜ + b₂,

where σ is a nonlinear activation (e.g. GELU).

**MoE FFN**  
Instead of one FFN, we have multiple “experts” FFN₁, FFN₂, …, FFN₍Nr₎. A router produces gating weights gₜ to select top-k experts:

   fₜ = ∑( r ∈ Top-k ) [ g₍t,r₎ * FFNᵣ(hₜ ) ].

DeepSeek uses a large set of experts but activates only a small subset per token, thus saving compute.

**Mini-Example**  
- Suppose 3 experts E₁, E₂, E₃.  
- The router outputs a gating vector gₜ = [0.7, 0.3, 0.0].  
- Then the token’s feed-forward output is 0.7 * E₁(hₜ) + 0.3 * E₂(hₜ). Expert 3 is dormant for this token.

**Residual Connection**  
We again add the output of the FFN/MoE back to hₜ:

   hₜ' = hₜ^(after attn) + fₜ.

------------------------------------------------------------
2. Overall DeepSeek R1 Architecture
------------------------------------------------------------

The second image shows a broader schematic:

1. Input Data flows into the Router Network.  
2. The router decides which experts in the Mixture-of-Experts (MoE) block activate.  
3. Only the “active” experts handle the token’s forward pass.  
4. The outputs from experts are merged in the Expert Aggregator.  
5. Next is Multi-Head Latent Attention with a KV cache (for up to 128k tokens), plus latent vector compression.  
6. The resulting representations pass to the Attention Processing and final heads that produce the model’s output.

Below is a layer-by-layer breakdown.

------------------------------------------------------------
2.1 Router Network
------------------------------------------------------------

**Purpose**  
- Takes token representations as input.  
- Computes a routing distribution so that each token is sent to a small subset of experts.  
- Greatly reduces computational cost at inference because only some portion of the total parameters is “active” for each token.

**Routing Math**  
Let the hidden embedding for token t be hₜ ∈ ℝ^D. The router’s function can be:

   rₜ = W^(router) * hₜ + b^(router) ∈ ℝ^(Nr).

We then convert rₜ into a probability or gating vector gₜ ∈ ℝ^(Nr). One common approach is a softmax top-k gating:

1) pₜ = softmax(rₜ).  
2) Sort pₜ to identify top-k.  
3) Zero out all but top-k elements.

**Example**  
- Suppose Nᵣ = 3. Let rₜ = [3.4, 1.1, 2.2].  
- softmax(rₜ) ≈ [0.70, 0.06, 0.24].  
- If k = 2, then gₜ = [0.70, 0, 0.24].  
- Experts 1 and 3 are active; expert 2 is dormant.

------------------------------------------------------------
2.2 Mixture of Experts (MoE)
------------------------------------------------------------

**Purpose**  
- Houses a large number of feed-forward sub-networks (experts).  
- Each expert can specialize in a portion of the representation space.  
- Only top-k experts run for each token, but all experts are trained (via backprop) over the entire dataset, each time for whichever tokens are routed to them.

**Sub-Layers**  
Inside each “MoE block,” we effectively have the same Transformer sublayers described above (Attention + FFN), except the FFN is replaced by the MoE feed-forward. The key difference is that after the router picks experts for each token, the token is processed through:

   fₜ = ∑( r ∈ Top-k ) [ g₍t,r₎ * FFNᵣ(hₜ) ].

Then aggregated to form the final hidden representation.

------------------------------------------------------------
2.3 Expert Aggregator
------------------------------------------------------------

**Purpose**  
- Collects the outputs of whichever experts were active and merges them back into a single hidden vector per token.  
- May include load-balancing or other constraints to keep the distribution of tokens per expert well-spread.

After the aggregator, you have the next hidden layer representation hₜ'.

------------------------------------------------------------
2.4 KV Cache
------------------------------------------------------------

**Purpose**  
- Stores key–value projections for previously processed tokens so you don’t need to recompute them repeatedly (especially useful at inference time or for very long contexts).  
- Enables context lengths up to 128k tokens, as the second diagram notes.

**Example**  
When processing a new token at position t:

   kₜ = W^K * hₜ,
   vₜ = W^V * hₜ.

We append kₜ, vₜ to the cache. For subsequent attention calculations, we can do:

   Attn(qₜ, [k₁,…,kₜ], [v₁,…,vₜ]).

------------------------------------------------------------
2.5 Latent Vector Compression
------------------------------------------------------------

**Purpose**  
- Compresses or projects the keys/values into a lower-dimensional latent space if necessary.  
- Maintains representational capacity without ballooning memory usage for extremely long contexts.

**Possible Math**  

   k̃ₜ = W^(compress)ₖ * kₜ,
   ṽₜ = W^(compress)ᵥ * vₜ,

reducing dimension from D to D'. The aggregator can be similarly adapted for the compressed dimension.

------------------------------------------------------------
2.6 Attention Processing → Model Output
------------------------------------------------------------

After going through the MoE + KV caching + latent compression, you do final multi-head attention steps, possibly additional Transformer blocks, and eventually a projection into the output dimension (e.g. for language modeling logits).

In the second figure, once the tokens have been processed through the MoE, the aggregator, and the “Multi-Head Latent Attention” block, the model continues with standard Transformer decoders or final layers, culminating in the **model’s output** (e.g. the next-token distribution or some classification head, depending on the application).

At each Transformer block, we do:

   hₜ  --(RMSNorm)--> (Multi-Head Latent Attention) --> +hₜ
        --(RMSNorm)--> (MoE FFN) --> +previous --> hₜ'

…where **MoE** is orchestrated by the router. That block is repeated L times, producing increasingly deep representations, ultimately outputting the final layer’s hidden states.

------------------------------------------------------------
Putting It All Together
------------------------------------------------------------

**DeepSeek R1** is a large-scale Transformer-type model that combines:

1) **Mixture of Experts** for the feed-forward portion of each Transformer block, which drastically expands parameter count while limiting active parameters per token.  
2) **Multi-Head Latent Attention** plus **KV caching** (up to 128k tokens) to handle extremely long sequences.  
3) **Router Network** to dynamically select experts.  
4) **RMSNorm** layers for stable training.  
5) **Residual connections** around both Attention and Feed-Forward sublayers.

The big picture flow (matching the second diagram) is:

1) Input Data → (Token Embeddings + Positional Embeddings)  
2) Router Network determines top-k experts for each token.  
3) Experts (each is a feed-forward sub-network) are run only for those tokens.  
4) Expert Aggregator merges the outputs.  
5) KV Cache is used/updated to store k,v states for attention over long contexts.  
6) Latent Vector Compression used to keep large context memory feasible.  
7) Attention is computed using the updated queries qₜ, compressed keys/values, etc.  
8) The final hidden states pass to Attention Processing or subsequent Transformer blocks if deeper layers exist.  
9) Model Output (e.g. next-token probability distribution or final representation).

At each Transformer block, we do:

   hₜ → RMSNorm → Multi-Head Latent Attention → (add hₜ)
       → RMSNorm → MoE FFN → (add previous) → hₜ'

…with residuals around attention and MoE feed-forward. That block repeats L times, culminating in the final output.

------------------------------------------------------------
Summary of Key Ideas
------------------------------------------------------------

1) Layer-by-Layer  
   - RMSNorm normalizes the hidden vectors.  
   - Multi-Head Latent Attention transforms them via queries/keys/values (potentially also specialized latent vectors or RoPE).  
   - Feed-Forward (MoE) uses a router to pick top-k experts out of many.  
   - Residual connections keep the original signal and help training.

2) Router + MoE  
   - The router is crucial to selecting which feed-forward experts to run.  
   - This allows the overall parameter count to be huge, but only a fraction is active per token.

3) KV Cache + Latent Compression  
   - Speeds up inference by reusing previously computed k, v.  
   - Allows extremely long contexts (up to 128k tokens) without blowing up memory usage.

4) Final Output  
   - After going through multiple stacked Transformer + MoE blocks, the model produces either next-token logits (language modeling) or a final hidden representation for downstream tasks.

This combination of dynamic routing, expert specialization, and long-context attention is what defines **DeepSeek R1** and yields a large parameter model (236 B total) with only ~21 B active per token, as indicated in the second image.

================== END OF TEXT FILE ==================
