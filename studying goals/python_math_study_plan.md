# Practical study plan for the Python and mathematics partner

## Objective

Your role is to build the executable mathematical specification of the project. You will help turn every formula into a small numerical example and a NumPy reference that the C implementation can test against.

By the end, you should also understand enough C to trace tensor storage, checkpoint offsets, ownership and the quantized matrix multiplication. The project is complete when both implementations agree, not when the Python implementation works alone.

For every new operation, use this sequence:

```text
calculate by hand
-> implement with explicit Python loops
-> implement with NumPy
-> inspect the corresponding repository code
-> compare with C
-> integrate into the model
```

Because you already know Python, mathematics and NumPy, avoid spending time on introductory syntax. Concentrate on numerical precision, explicit tensor shapes and translating high-level array expressions into loops and memory layouts.

## Working rules

- Annotate every tensor with its shape and dtype.
- Keep the NumPy reference simple; clarity matters more than speed.
- Do not use `torch.nn.MultiheadAttention` or similar high-level layers in the reference.
- Use `float32` deliberately rather than NumPy's implicit `float64` where C comparison matters.
- Compare arrays using absolute and relative tolerances.
- Use greedy decoding before randomized sampling.
- Explain equations with a concrete numeric example before introducing notation.

## Phase 1 — Numerical foundations and the C bridge

Review and demonstrate:

- vectors, matrices and tensor shapes;
- dot products and matrix-vector multiplication;
- element-wise versus matrix operations;
- broadcasting;
- `float32`, `int8` and `int32`;
- rounding, clipping, overflow and accumulated error;
- binary serialization and endianness.

Build a small Python test-data generator that writes:

```text
input vectors
weights
expected output
shape and dtype metadata
```

The C partner should be able to load this data without guessing its representation.

For each test fixture, record:

```text
operation
input shapes
output shape
dtype
absolute tolerance
relative tolerance
reason for the tolerance
```

## Phase 2 — Required inference mathematics

### Linear algebra

Teach and implement:

1. dot product;
2. matrix-vector multiplication;
3. transpose as a change of indexing;
4. element-wise multiplication;
5. vector norms and root mean square.

Always connect notation to storage. For example:

```text
W has shape [n, d]
x has shape [d]
y = W @ x has shape [n]

y[i] = sum(W[i, j] * x[j] for j in range(d))
```

### Normalization and probabilities

Teach and implement:

- RMSNorm;
- exponentials;
- stable softmax using maximum subtraction;
- logits and probabilities;
- argmax;
- categorical sampling;
- temperature and top-p sampling.

Include pathological inputs:

- all zeros;
- very large positive logits;
- very large negative logits;
- equal logits;
- a vector with a single dominant value.

### Activations and residual paths

Implement and plot:

- sigmoid;
- SiLU;
- the gated element-wise product used by SwiGLU;
- residual addition.

Use the repository's `RMSNorm`, `Attention`, `FeedForward` and `TransformerBlock` in `model.py` only after your minimal versions work.

## Phase 3 — Attention from first principles

Start with one head and two tokens. Choose small integer inputs and weights so every intermediate result can be inspected.

Implement:

```text
q = Wq @ x
k = Wk @ x
v = Wv @ x
scores = q @ previous_keys.T / sqrt(head_size)
probabilities = softmax(scores)
output = probabilities @ previous_values
```

Produce a trace containing:

```text
x, q, k, v, raw scores, scaled scores,
softmax probabilities and attention output
```

Then extend in this order:

1. more previous tokens;
2. multiple heads;
3. grouped-query attention (`n_kv_heads < n_heads`);
4. causal access to previous positions only;
5. a KV cache.

Your explanation of the KV cache should distinguish learned model weights from runtime state.

## Phase 4 — RoPE and a complete Llama block

Explain RoPE first as an ordinary 2D rotation:

```text
x0' = x0*cos(theta) - x1*sin(theta)
x1' = x0*sin(theta) + x1*cos(theta)
```

Write tests for:

- position zero;
- a known angle;
- preservation of the pair's magnitude;
- different frequency pairs;
- the difference between query heads and KV heads.

Build one complete block:

```text
x
-> RMSNorm
-> Q/K/V projections
-> RoPE
-> cached attention
-> output projection
-> residual addition
-> RMSNorm
-> SwiGLU feed-forward
-> residual addition
```

Create a shape table for every intermediate tensor. Your C partner should review how each tensor maps into a flat buffer.

## Phase 5 — NumPy reference inference engine

Suggested layout:

```text
python_ref/
  checkpoint.py
  ops.py
  transformer.py
  tokenizer.py
  sampling.py
  generate.py
  tests/
```

Implementation order:

1. numerical primitives;
2. checkpoint reader;
3. embedding lookup;
4. RoPE;
5. attention and KV cache;
6. SwiGLU;
7. one Transformer block;
8. all blocks and final logits;
9. greedy generation;
10. tokenizer;
11. temperature and top-p.

Use explicit functions rather than framework layers. A useful interface is:

```python
logits, trace = model.forward(token_id, position, debug=True)
```

The trace should optionally expose selected arrays from each layer. Establish a stable naming scheme that the C engine can reproduce.

### Required validation

- Compare your primitives with hand-calculated examples.
- Compare your model with `model.py`.
- Compare selected arrays with `run.c`.
- Compare greedy token IDs, not only decoded text.
- Document tolerances and investigate the first divergent layer.

Completion means at least 200 matching greedy tokens on the 260K model and close agreement of selected intermediate tensors.

## Phase 6 — Quantization mathematics

Start with symmetric signed 8-bit quantization. For a group of floating-point values:

```text
scale = max(abs(x)) / 127
q = clip(round(x / scale), -127, 127)
x_hat = q * scale
```

Define the zero-vector behavior explicitly.

Build an experiment notebook or script that compares:

- per-tensor quantization;
- per-row quantization;
- group sizes 64, 32, 16 and 8;
- symmetric and asymmetric quantization;
- distributions with and without outliers.

Report:

- maximum absolute error;
- mean absolute error;
- mean squared error;
- relative error where meaningful;
- storage in bytes;
- compression ratio.

Explain the quantized group dot product:

```text
x ~= sx * qx
w ~= sw * qw
dot(x, w) ~= sx * sw * sum(qx[i] * qw[i])
```

Derive an upper bound on the integer accumulator and check that the C type is safe.

## Phase 7 — Python Q8_0 reference

Read and reproduce `quantize_q80` from `export.py`, but keep your implementation independent enough to catch errors.

Implement:

1. offline weight quantization;
2. serialized integer values and group scales;
3. checkpoint metadata and version validation;
4. dynamic activation quantization;
5. quantized dot product;
6. quantized matrix-vector multiplication;
7. a quantized forward pass.

Maintain three comparisons:

```text
NumPy float32
NumPy Q8_0
C Q8_0
```

The NumPy and C Q8_0 implementations should closely agree. Their difference from float32 is expected and must be measured.

Do not use token equality as the only quality measure. A small logit perturbation can change a near-tie. Also record logit error and, later, perplexity on a fixed corpus.

## Phase 8 — Evaluation and later quantization study

Create a reproducible evaluation script with:

```text
model identifier
checkpoint format
quantization type and group size
fixed input tokens
float and quantized logits
token agreement rate
error statistics
model size
runtime supplied by the C benchmark
```

After Q8_0 works, study these topics in order:

1. weight-only versus weight-and-activation quantization;
2. per-channel and per-group scaling;
3. sensitivity of different tensors;
4. calibration datasets;
5. perplexity as a model-quality measure;
6. simulated 4-bit packing;
7. modern methods such as SmoothQuant;
8. `llama.cpp` block formats and mixed quantization.

Do not begin with GPTQ, AWQ or complex 4-bit formats. Q8_0 gives you the concepts and a complete implementation with much less machinery.

## What to learn from the C partner

For every NumPy expression, ask them to show:

- the equivalent nested loops;
- the flat-memory offset calculation;
- the number of bytes read and written;
- which buffers can be reused;
- whether access is contiguous;
- whether rows can be computed independently;
- which integer or float type holds each intermediate value.

You should eventually be able to trace this yourself:

```c
y[i] += w[i * d + j] * x[j];
```

and explain why it implements a row of matrix-vector multiplication.

## Recommended primary resources

Read only the section needed for the current implementation:

- NumPy quickstart: <https://numpy.org/doc/stable/user/quickstart.html>
- PyTorch tensor tutorial: <https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>
- Attention Is All You Need, especially sections 3.1–3.2: <https://arxiv.org/abs/1706.03762>
- RMSNorm: <https://arxiv.org/abs/1910.07467>
- GLU Variants Improve Transformer: <https://arxiv.org/abs/2002.05202>
- RoFormer/RoPE: <https://arxiv.org/abs/2104.09864>
- Llama 2 architecture: <https://arxiv.org/abs/2307.09288>
- SmoothQuant, after Q8_0: <https://arxiv.org/abs/2211.10438>
- `llama.cpp` quantization documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md>

Your main deliverable is not a notebook. It is a trustworthy executable specification that makes every disagreement with C local, observable and testable.

