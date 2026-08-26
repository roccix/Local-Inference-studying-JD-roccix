# Shared project and study plan

## Mission

Build two compatible Llama 2 inference engines:

1. a clear NumPy reference;
2. an independent C implementation;
3. a Q8_0 version of both;
4. a measured and optimized C engine.

The project is practical: every study topic must produce code, tests, a trace or a benchmark. The Python/math partner owns the first mathematical specification; the C partner owns the first systems design. Ownership rotates so both people learn the other discipline.

## Definition of success

The float32 milestone is complete when:

- both engines load the same 260K checkpoint;
- selected intermediate tensors agree within documented tolerances;
- at least 200 greedy token IDs match;
- decoded output matches;
- the C program passes sanitizer checks.

The quantized milestone is complete when:

- Python and C implement the same Q8_0 format;
- the quantized checkpoint contains explicit version and group metadata;
- Python Q8_0 and C Q8_0 closely agree;
- error against float32 is measured;
- file size, memory and tokens per second are reported.

## Study rhythm

Use three sessions per week. A session can be approximately two hours.

### Session A — Concept and experiment

```text
15 min  choose one question and one deliverable
25 min  explain with a hand-calculated example
50 min  implement the plain Python/NumPy reference
20 min  write expected values and tests
10 min  recap in your own words
```

### Session B — C implementation

```text
15 min  review shapes, dtype and tolerance
60 min  pair-program the C implementation
25 min  compare Python and C
10 min  inspect memory and lifetime
10 min  record the result
```

### Session C — Integration

```text
20 min  run all tests
50 min  integrate one component
25 min  diagnose the first mismatch
15 min  benchmark or improve diagnostics
10 min  choose the next milestone
```

Switch driver and navigator every 30–45 minutes. In odd weeks the C partner drives Python/math work; in even weeks the Python/math partner drives C work. The specialist navigates and teaches without taking over the keyboard.

## Communication agreement

- Code, identifiers, commits and permanent documentation are in English.
- Conversation may switch language whenever that improves understanding.
- Maintain a shared English–Italian glossary in the repository.
- Avoid saying only “it is a tensor operation.” State the exact shapes and loops.
- The person receiving an explanation must restate it and implement a tiny example.
- Disagreements are settled with a hand calculation, trace or test.

Use this vocabulary consistently:

| English | Italian | Meaning |
|---|---|---|
| weight | peso | learned model value |
| activation | attivazione | runtime value produced by the model |
| embedding | embedding/rappresentazione | vector associated with a token |
| logit | logit/punteggio | score before normalization |
| head | testa | one attention subspace |
| shape | forma/dimensioni | number of elements along each axis |
| stride | passo | memory distance between indexed elements |
| scale | scala/fattore di scala | multiplier used by quantization |
| residual | connessione residuale | addition of an earlier value |
| cache | cache | stored K/V values from earlier positions |

## Roadmap

The week numbers are guidance, not deadlines. Move forward only when the exit test passes.

### Weeks 1–2: repository and numerical bridge

Deliverables:

- build `run` and run local tests;
- trace `main -> generate -> forward -> sample`;
- write/read a small `float32` binary fixture in Python and C;
- create shared conventions for shape, dtype and tolerance;
- implement dot product and matrix-vector multiplication in both languages.

Exit test: both partners can explain how `W[n,d] @ x[d]` becomes C indexing and can predict the output of a tiny case.

### Weeks 3–4: numerical operations

Deliverables:

- RMSNorm;
- stable softmax;
- SiLU and gated multiplication;
- argmax and categorical sampling;
- cross-language unit tests including edge cases.

Exit test: all primitive C results match the NumPy reference within documented tolerances.

### Weeks 5–7: attention

Deliverables:

- one-head, two-token attention trace;
- multi-head attention;
- RoPE for one pair and complete vectors;
- KV cache for three positions;
- grouped-query attention shape explanation.

Exit test: both partners can draw attention, state all shapes and explain every value in a tiny trace.

### Weeks 8–9: complete Transformer block

Deliverables:

- attention sublayer and residual;
- SwiGLU sublayer and residual;
- complete single-block implementation in NumPy;
- corresponding C implementation;
- layer-by-layer comparison tool.

Exit test: one block agrees between Python and C.

### Weeks 10–12: complete float32 engine

Deliverables:

- checkpoint loading;
- all model layers and final logits;
- greedy generation;
- tokenizer and decoding;
- temperature and top-p only after greedy correctness;
- C sanitizer build and baseline benchmark.

Exit test: at least 200 identical greedy tokens on the 260K model.

### Weeks 13–14: quantization laboratory

Deliverables:

- hand-calculated symmetric `int8` example;
- Python and C quantize/dequantize functions;
- per-tensor, per-row and grouped experiments;
- outlier experiment;
- error and compression report;
- safe accumulator analysis.

Exit test: both partners can explain scale, clipping, rounding, group size, reconstruction error and accumulator width.

### Weeks 15–17: Q8_0 engine

Deliverables:

- Python checkpoint quantizer;
- versioned quantized format;
- C quantized checkpoint loader;
- dynamic activation quantization;
- quantized dot product and matvec;
- quantized forward pass;
- float32 versus Q8_0 comparison.

Exit test: Python Q8_0 and C Q8_0 agree, and quality/performance differences from float32 are recorded.

### Weeks 18–20: measurement and optimization

Deliverables:

- reproducible benchmark command;
- operation-level timings;
- removal of unnecessary token-loop allocations;
- OpenMP experiment;
- compiler vectorization report;
- final comparison of reference, float32 C and Q8_0 C.

Exit test: every claimed speed improvement is supported by the same repeatable benchmark and retains correctness.

### Optional weeks 21–24: training pipeline

Deliverables:

- understand cross-entropy and backpropagation conceptually;
- train a very small TinyStories model with PyTorch;
- export it to your format;
- run it in both float32 and Q8_0 engines;
- evaluate generation and perplexity on fixed inputs.

## Source-reading order

### First pass: inference

```text
README overview
-> run.c main and structures
-> checkpoint and runtime memory
-> rmsnorm, softmax, matmul
-> forward in conceptual blocks
-> greedy sampling
-> tokenizer
-> generate
-> corresponding model.py components
-> tests
```

### Second pass: quantization

```text
export.py quantize_q80
-> export version 2
-> runq.c QuantizedTensor
-> quantize/dequantize
-> quantized matmul
-> quantized forward
```

### Third pass: training

```text
tokenizer.py
-> tinystories.py
-> model.py
-> train.py
-> export.py
```

## Cross-language testing protocol

Every numerical component should have:

1. a hand-calculated case;
2. a NumPy expected result stored as `float32`;
3. a C unit test;
4. absolute and relative tolerances;
5. a failure message containing index, expected value and actual value.

For integration failures, find the first divergence in this order:

```text
input token and position
-> embedding
-> normalized input
-> q, k, v
-> RoPE output
-> attention scores
-> attention probabilities
-> attention result
-> residual
-> feed-forward result
-> final logits
-> selected token
```

Never debug only the final generated sentence.

## Branch and review practice

Keep tasks small enough to review in one sitting. A feature is done only when it contains:

- implementation;
- tests;
- a short shape/dtype explanation;
- no unexplained warnings;
- comparison with the reference when numerical.

Suggested commit scope:

```text
Add stable softmax reference and edge-case tests
Implement C RMSNorm and compare with NumPy
Add debug trace for layer-zero attention
Add grouped Q8_0 quantization fixtures
```

Do not mix refactoring, optimization and a new numerical operation in one change.

## Progress board

Track each component using:

| Component | Hand example | Python | C | Cross-test | Integrated | Explained by both |
|---|---:|---:|---:|---:|---:|---:|
| Dot product |  |  |  |  |  |  |
| Matvec |  |  |  |  |  |  |
| RMSNorm |  |  |  |  |  |  |
| Softmax |  |  |  |  |  |  |
| RoPE |  |  |  |  |  |  |
| Attention |  |  |  |  |  |  |
| KV cache |  |  |  |  |  |  |
| SwiGLU |  |  |  |  |  |  |
| Q8_0 quantization |  |  |  |  |  |  |
| Quantized matvec |  |  |  |  |  |  |

## Final demonstration

Finish the project with a short demonstration that both partners can present:

1. generate text with the repository reference;
2. generate the same greedy tokens with your NumPy engine;
3. generate them with your float32 C engine;
4. run your Q8_0 engine;
5. show one tensor trace across Python and C;
6. show model size, memory, accuracy/error and tokens per second;
7. explain one optimization and its measured effect;
8. swap speakers so each person presents the area they originally knew least.

The real deliverable is shared understanding. If only one person can explain a component, that component is not finished.

