# Speaker notes and timing

Target: **30 minutes**, including **2:30 of discussion**. Main slides take 27:30. The backup slides do not belong in the timed route.

Assumed audience: basic Python, tensor shapes, and gradient descent. Introduce all tokenizer and Transformer-specific notation. Here, “encoding” includes both discrete text tokenization and positional representation inside attention. The model is a modern causal Transformer, rather than a new architecture proposed by this project.

| Slide | Topic | Time | Cumulative |
|---|---|---:|---:|
| 1 | Title and scope | 0:30 | 0:30 |
| 2 | Language modeling objective | 1:00 | 1:30 |
| 3 | Bytes, IDs, and embeddings | 1:30 | 3:00 |
| 4 | Tokenization variants | 1:00 | 4:00 |
| 5 | BPE worked example | 2:00 | 6:00 |
| 6 | BPE training implementation | 1:30 | 7:30 |
| 7 | Encoding implementation | 1:30 | 9:00 |
| 8 | Embedding and output matrices | 1:00 | 10:00 |
| 9 | Architecture context | 1:00 | 11:00 |
| 10 | Decoder stack | 1:00 | 12:00 |
| 11 | RMSNorm and residuals | 1:30 | 13:30 |
| 12 | Causal attention | 2:00 | 15:30 |
| 13 | Multi-head tensor shapes | 1:30 | 17:00 |
| 14 | RoPE | 2:00 | 19:00 |
| 15 | SwiGLU | 1:30 | 20:30 |
| 16 | Architecture comparison | 1:00 | 21:30 |
| 17 | Targets and loss | 1:00 | 22:30 |
| 18 | Training loop | 1:30 | 24:00 |
| 19 | Generation | 1:00 | 25:00 |
| 20 | Scaling variants | 1:00 | 26:00 |
| 21 | Implementation checks | 1:00 | 27:00 |
| 22 | Recap and discussion prompt | 0:30 | 27:30 |
| — | Discussion | 2:30 | 30:00 |

## 1. Title and scope

“We will follow text all the way through the implementation: how it becomes token IDs, how a modern Transformer predicts the next ID, and how training and sampling use those predictions.” Explain that the same architectural ingredients appear in larger language models, while this repository provides a small educational implementation. Scale, data, and training quality determine capabilities beyond the architecture itself.

## 2. The language modeling problem

Read the conditional probability as “the probability of the next token given everything before it.” The product follows the probability chain rule. Maximum likelihood becomes a sum of negative log probabilities, which gives the cross-entropy loss. Model parameters are theta here. Later, the code also calls the RoPE frequency base `theta`, a separate hyperparameter.

Training provides the true prefix and target at every position. During generation the model must continue with its own sampled predictions. A pretrained next-token predictor alone does not imply instruction following. Instruction tuning and preference-based post-training are outside this repository and this talk.

Code: `transformer.py::Transformer`, `transformer.py::cross_entropy`, `train.py::train`.

## 3. Three representations

Use “café” to separate characters from bytes. Four characters occupy five UTF-8 bytes. The last character is the two-byte sequence C3 A9. Base byte tokens can split that character, while a merged token may hold the whole character or a larger string.

Token IDs are dictionary keys. ID 200 being close to ID 201 says nothing about their meanings. Embedding vectors are learned parameters. After the decoder layers, a vector represents a position in its prefix context, so the same token ID can have different hidden representations in different sentences.

The tokenizer is trained separately by pair-frequency statistics. The model does not backpropagate through the BPE merge decisions.

## 4. Tokenization variants

Explain the vocabulary/sequence tradeoff. Word vocabularies can become large and need an unseen-word policy. Raw bytes cover any valid UTF-8 string using just 256 base symbols, but create longer sequences. Byte BPE keeps coverage while allocating vocabulary entries to frequent strings.

Subword unigram models optimize a probabilistic segmentation model and can choose segmentations using their scores. BPE instead produces an ordered series of pair merges. SentencePiece is a toolkit, with its own normalization and whitespace conventions, that can train BPE or unigram models directly on raw sentences. Do not identify SentencePiece with unigram alone.

Tokenizer efficiency differs across languages. A fixed number of tokens does not mean a fixed number of words, characters, or bytes. Avoid comparing raw token perplexity across different tokenizers.

## 5. BPE training example

The example starts after pre-tokenization, so no pair crosses a pre-token boundary. Count each adjacent pair weighted by the pre-token's frequency. Both `(l,o)` and `(o,w)` initially have count five. The inspected code uses Python's tuple ordering in a min-heap of `(-count, pair)`, so it chooses `(l,o)` on this tie. After the first merge, `(lo,w)` has count five, making `low` the next merged symbol.

Each selected merge creates a fresh ID whose byte value is the concatenation of the pair's byte strings. Training stores both the vocabulary mapping and the ordered merge list. A target vocabulary size is a stopping budget, and training can stop earlier if no pairs remain.

Distinguish this code's tie rule from the assignment handout's lexicographically greatest-pair rule. The difference can change the learned vocabulary. Slide 21 returns to this as an implementation/specification issue.

## 6. BPE training implementation

Walk through `BPE_tokenizer_training` in stages. Multiprocessing counts regex pre-tokens in chunks. Special-token boundaries help choose independent chunks. Ordinary text does not merge through special tokens, and the reserved special strings get their own IDs.

The regex handles contractions, Unicode letters and numbers, punctuation, and whitespace. It can include one leading ASCII space with a word. For example, `Hello, world!` pre-tokenizes as `Hello`, `,`, ` world`, `!`. This makes leading spaces part of learned byte strings without allowing unrestricted merging across the regex boundaries.

`counts_words` stores frequencies. `words_tokens` stores the current symbol list for each distinct pre-token. `counts_pairs` stores weighted adjacency counts. `pairs_to_words` avoids scanning all pre-tokens after every merge. `counts_pairs_update` modifies local neighbors around a merged occurrence.

The heap uses lazy deletion: an entry can remain after its count changes, so the pop loop checks it against the current dictionary count. A heap operation is logarithmic in heap size, but total merge cost also includes affected occurrences, pushes, and discarded stale entries. Do not claim the entire training loop is logarithmic or memory is proportional only to final BPE vocabulary size. Each worker reads its current chunk, and distinct pre-token statistics and stale heap entries also consume memory. Code comments asserting large speedups are not measurements established in this presentation.

## 7. Encoding implementation

The slide shows pseudocode rather than a literal runnable replacement. Real methods are `_encode_word`, `encode`, and `decode`. The algorithm starts from byte IDs, scans adjacent pairs, selects the lowest learned merge rank, applies that merge, and repeats. It does not recompute frequencies from the input sentence. It also does not simply choose the longest dictionary string at every step.

The byte-pair-to-rank dictionary makes rank lookup constant time. Repeated scans and list deletion can still produce quadratic work in a pre-token's byte length. `_word_cache` reuses results for repeated pre-token strings, trading memory for less repeated work. The cache is unbounded in this implementation.

Special tokens are split before the ordinary regex path. The implementation sorts special strings by length so a longer overlapping special can match first. `decode` concatenates byte strings before UTF-8 decoding because individual tokens can hold incomplete multibyte characters. Its `errors="replace"` policy handles arbitrary invalid byte sequences. `encode_file` separately reads malformed input with `errors="ignore"`, so do not promise lossless preservation of arbitrary malformed files. `encode_iterable` encodes each supplied chunk independently, so arbitrary chunk boundaries can change segmentation relative to one whole-string call.

## 8. Embedding and output matrices

Introduce B for batch size, T for token count, d for hidden width, and V for vocabulary size. Indexing `E` produces one vector per token, resulting in `(B,T,d)`. LinearTransform stores matrices in `(out_features,in_features)` order and uses `x @ weight.T`. There are no bias vectors.

After the decoder, a separate `(V,d)` matrix creates one logit for every candidate next token. The implementation leaves embeddings and the output matrix untied. Weight tying would reuse the same parameter matrix and reduce their combined parameter count from 2Vd to Vd.

## 9. Architecture context

RNNs propagate a recurrent state through the sequence. Self-attention gives each position direct access to permitted earlier positions and supports training across positions in parallel. Dense attention also creates quadratic sequence-length costs, which we revisit later.

The original Transformer solved conditional sequence generation with an encoder, a decoder, and cross-attention. Encoder models commonly learn bidirectional representations with masked-token objectives. The repository's decoder sees only its current and previous positions and predicts the following token. The name “decoder” here does not imply that an encoder exists elsewhere in the code.

## 10. Decoder stack

Read the pseudocode once from top to bottom. The real class names are `Embedding`, `TransformerBlock`, `RMSNorm`, `CausalMultiHeadSelfAttention`, and `SwiGLU`. Attention mixes information between positions. The FFN mixes features at each individual position using shared weights across positions. Both return the residual width d.

The block builds positions from 0 through T−1 when none are supplied. The model applies a final RMSNorm and outputs logits, not probabilities. There is no cross-attention, learned input-position embedding table, or dropout in this implementation.

## 11. RMSNorm and residuals

The mean square runs across the feature dimension independently at every batch/sequence location. Epsilon avoids division by zero. The learned vector g provides a scale per feature. LayerNorm also subtracts the feature mean and usually learns a shift. RMSNorm omits that recentering step.

In pre-norm, the residual stream bypasses normalization and the sublayer through the explicit addition. In post-norm, the normalization wraps that addition. This changes gradient paths and training behavior, although it does not guarantee successful optimization on every setup.

`RMSNorm.forward` promotes activations to float32 for its calculations and casts back to the original dtype. If all parameters are converted to a lower precision, g itself has that stored precision even though the activation arithmetic promotes to float32.

## 12. Causal attention

Use one head to explain the equation before returning to multiple heads. A query from one position compares against keys at all allowed positions. Softmax along the key axis produces weights summing to one. The weighted sum retrieves values. These are learned projections of the same input in self-attention, rather than separate external databases.

For components with roughly unit variance, an unscaled dot product's variance grows with head dimension. Dividing by its square root helps keep score scale under control. This is a motivation, not a universal assumption about trained activations.

The triangular mask includes the diagonal. A position can see its own input token because its target is the following token. Future positions receive negative infinity before softmax and therefore zero weight. The implementation subtracts the row maximum before exponentiating for numerical stability. All-masked rows would need extra handling in more general mask settings.

## 13. Multi-head shapes

Keep the numerical example visible and narrate each axis. Projecting Q produces `(2,4,512)`. Splitting 512 into eight heads of width 64 and swapping sequence/head axes gives `(2,8,4,64)`. Dotting queries with transposed keys creates `(2,8,4,4)`. The final two axes mean query position and key position.

RoPE, omitted from this abbreviated shape snippet, applies to Q and K before their dot product. Masking is also omitted here and was shown on the previous slide. After weighted value retrieval, swap axes back, make memory contiguous, concatenate heads, and apply the separate output projection.

Each head can learn different projections, but heads need not develop easily interpretable roles. Confirm d divides H exactly. For the adjacent even/odd rotation used here, the per-head width must also be even. The code does not explicitly check both conditions.

## 14. RoPE

Introduce a two-dimensional rotation as a way to attach position to a query/key pair of coordinates. Each coordinate pair has a different angular frequency. Position m supplies the rotation angle m times that frequency. Orthogonality gives the relative-position identity on the slide.

The identity says the rotation contribution depends on the displacement n−m. It does not mean attention depends only on position: the content vectors q and k remain essential. Unlike adding an absolute embedding at the input, this operation acts inside every attention layer on queries and keys.

In `ROPE`, sine and cosine tables have shape `(max_seq_len,d_h/2)`. `register_buffer(..., persistent=False)` moves them with the module while excluding them from saved state dictionaries. The code broadcasts the selected tables across heads, splits even and odd coordinates, then uses `stack` and `flatten` to restore the interleaved feature order. The default base is 10,000. Tables cap supported indices in this implementation, and changing context length or frequency scaling is not a guarantee of reliable long-context behavior.

## 15. SwiGLU

SiLU is z times its sigmoid. One branch passes through SiLU and multiplies the other branch elementwise. It can amplify, attenuate, or change the sign of features. Avoid describing it as a gate strictly constrained to [0,1], because SiLU itself is not.

For column-vector notation the matrices are W1/W2 of shape `(f,d)` and W3 of shape `(d,f)`. The code applies the equivalent row-vector convention. An ordinary two-projection FFN uses approximately 2df parameters. SwiGLU uses 3df, which motivates reducing hidden width to approximately 8d/3 when matching an ordinary width of 4d.

The `SwiGLU` class has an automatic width path when d_ff is −1, including custom rounding to multiples of 64. The training CLI passes 2048 by default with d=512, which means that automatic path does not run. Do not accidentally describe the CLI as using parameter-matched 8d/3 width.

## 16. Architecture comparison

Use the table to organize ideas already explained. The repository's choices resemble the architecture described in the original LLaMA paper: pre-normalization with RMSNorm, SwiGLU, and rotary position representation. This does not imply identical tokenization, initialization, data mixture, training recipe, or model behavior.

The comparison concerns the original 2017 Transformer, not every later Transformer variant. The original also used weight sharing in its embedding/output setup. This implementation has an independent output matrix. Improvements claimed in cited papers are evidence from those papers' experimental settings, not results measured for this repository.

## 17. Training examples and loss

The input `[A,B,C,D]` pairs with target `[B,C,D,E]`. Say “targets start one token later in the corpus” to avoid confusion about left/right shifting terminology. `get_batch` samples random contiguous windows. The model returns a vocabulary distribution for each input position, and the loss averages all B×T targets after flattening.

The stable log-sum-exp form subtracts the maximum and adds it back algebraically. The custom implementation accepts flattened `(B*T,V)` logits and one-dimensional target IDs. It is not a general arbitrary-rank cross-entropy API.

Perplexity exponentiates average loss in nats. Comparable perplexity requires the same tokenizer, data split, context treatment, and evaluation protocol. No training curves or perplexity measurements are supplied in this deck.

## 18. Training loop

Explain zeroing gradients before backward, because PyTorch accumulates gradients. AdamW's first moment smooths gradients and its second moment tracks squared gradients. Bias correction accounts for their zero initialization. Global gradient clipping rescales the gradient vector when its norm exceeds the threshold. Linear warmup followed by cosine decay controls learning rate.

The custom optimizer applies an adaptive update followed by multiplicative decay of the resulting parameter. This detail follows the current code. Avoid presenting its operation order as identical to every AdamW implementation. It applies decay to every supplied parameter, including normalization scales, because the training script does not create separate parameter groups.

`load_dataset` memory-maps `.npy` or raw integer arrays. `encode_file` writes raw uint16/uint32 based on maximum token ID, so raw file loading needs the matching dtype. The train script tracks validation loss and saves a best checkpoint in addition to periodic/final checkpoints. Its resume saves model, optimizer, and iteration, but not random number generator state, so it is not an exact replay guarantee.

Fit BPE on training text only. The tokenizer training module's current example constants name `owt_valid.txt`, so they should be changed before treating that script as a clean training/validation experiment. This deck does not run a full training job.

## 19. Generation

At each step, the helper keeps the final context window, runs a full model forward pass, reads the final position's logits, and samples an ID. Temperature zero takes an explicit argmax branch rather than dividing by zero. Positive temperatures change distribution sharpness. Top-k masks logits below the kth threshold, with ties potentially retaining more than k entries.

The helper uses no gradients and sets evaluation mode. It runs for the requested number of new tokens, without stopping on an end-of-text ID. It also recomputes all prefix positions instead of storing K and V. Cropping resets supplied positions to zero for the new window through the block's default position construction.

## 20. Scaling variants

Distinguish changing the model from changing how an equivalent computation executes. KV caching reuses prior per-layer keys/values during inference. GQA shares KV projections among groups of query heads, while MQA uses one KV head. These change KV head structure and cache size. FlashAttention tiles the exact attention operation to avoid full score-matrix materialization in high-bandwidth device memory. Floating-point results can differ slightly due to operation ordering.

Sparse attention changes which positions can interact and can reduce pairwise work depending on the sparsity pattern. FlashAttention alone does not remove dense attention's quadratic arithmetic complexity. The source here explicitly computes full scores and allocates a new causal mask on every attention call.

## 21. Implementation checks

These points distinguish the actual source from a conceptual architecture. Focus on the BPE tie rule and shape requirements if time is tight. Python's tuple comparison gives ascending pair order for equal negative counts, while the assignment handout explicitly requests the lexicographically greatest pair. This is a specification mismatch, not a reason to silently rewrite the presentation example.

Calling a full block supplies positions and activates RoPE. Calling attention directly with `token_positions=None` skips it. The misleading comment after `output_proj` says it shares V weights, but a separate `LinearTransform` owns those parameters. For checkpoints, `train.py` stores `iteration` and `val_loss`, while `training_utils.py` stores `epoch` and uses different signatures. Use the matching save/load pair.

The main task was to explain the repository, so the presentation documents these details without modifying the implementation.

## 22. Recap and discussion

End with the relationship among segmentation, representation, and optimization. Ask which arrays grow when context doubles and which parameter matrices grow when vocabulary doubles. Expected answer: fixed-width model parameters do not grow with context length, but dense attention scores grow quadratically. Embedding and output weights grow linearly with vocabulary size. Use backup slide 25 if discussion needs a concrete comparison.

## Backup 23. Parameter count

The formula exactly describes the current bias-free architecture with untied embeddings. With V=10,000 and CLI default widths/depth, the total is 35,412,480 parameters. This is an illustrative instantiated configuration. The CLI requires an explicit vocabulary size, so 10,000 is not a vocabulary default. RoPE tables are buffers rather than trainable parameters. Parameter count excludes optimizer state, gradients, activations, and data storage.

## Backup 24. Masked attention

The excerpt combines the source's attention function with its custom softmax for readability. In this API True means allowed. A causal row always includes itself. If extending to padding or arbitrary masks, handle fully masked rows rather than assuming the same invariant holds.

## Backup 25. Vocabulary and context costs

The table holds B, d, H, f, L, and the other axis fixed. Doubling vocabulary doubles embedding/head parameters and logits, but leaves attention score dimensions unchanged. Doubling T doubles most positionwise work and logits, while quadrupling dense attention score count. Tokenization can change the token count for a fixed raw document, so real end-to-end comparisons couple V and T.

## Backup 26. Walkthrough checks

The listed checks are useful for a future live demonstration. They are not a claim that the entire repository test suite passes. For a causality check, change tokens after position t and compare outputs up to t. A tiny-batch overfit checks local learnability, while held-out evaluation tests a separate property. For Unicode round trips, use valid input strings and join all token bytes before decoding.

## Backup 27–28. Sources

The slide hyperlinks point to primary papers. Implementation facts come from this checkout. BPE's deterministic tie specification comes from Section 2.3 of `cs336_assignment1_basics.pdf`. Bibliographic dates distinguish preprint dates where appropriate. If the code changes later, recheck the implementation-specific comparisons, defaults, and caveats before reusing the talk.
