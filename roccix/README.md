# Roccix study notebook

This directory records what Roccix has understood, not merely what has been
covered. Each lesson contains definitions, a concrete example, misconceptions
that were corrected, verification questions and the next recommended topic.

## Learning method

1. Introduce one concept at a time in concise Italian.
2. Put the English technical term next to the Italian term.
3. Explain every symbol before using a formula.
4. Use formulas that remain readable in a terminal.
5. Ask short verification questions.
6. Continue only after the answers show understanding.
7. Offer optional approfondimenti without introducing them prematurely.

## Lessons

| Status | Lesson | Verified understanding |
|---|---|---|
| Complete | [01 - Inference and token generation](01_inference_and_generation.md) | Can distinguish input, parameters, logits, vocabulary and selected token |
| Next | 02 - Logits and greedy decoding | Not started |

## Current next step

Understand what a logit is and implement the simplest selection rule:

```text
next_token_id = argmax(logits)
```

Do not introduce probabilities, softmax, temperature or random sampling until
the distinction between logits and the selected token is secure.
