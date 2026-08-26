# Lesson 01: inference and token generation

## Objective

Understand formally, but without unnecessary detail, what inference is and how
an autoregressive language model generates one token at a time.

## Basic mathematical objects

```text
scalar: one number
vector: an ordered list of numbers
matrix: a table of numbers arranged in rows and columns
```

Examples:

```text
scalar: 5
vector: [2, 4, 7]
matrix: [[1, 2, 3], [4, 5, 6]]
```

## Inference

**Inference** (*inference*) is the process in which an already trained model
uses fixed learned parameters to transform an input into an output.

Terminal-readable notation:

```text
y = f(x; theta)
```

Meaning:

- `x`: input;
- `f`: model, meaning the sequence of mathematical operations;
- `theta`: all learned parameters used by the model;
- `y`: model output.

For now, this is a sufficient abstraction:

```text
theta = container of all numbers learned by the model
```

These numbers are called **parameters** (*parameters*) or, less precisely,
**weights** (*weights*). They are numbers used by the model's operations; not
every parameter is necessarily multiplied directly by the original input.

During inference:

```text
theta_before = theta_after
```

The semicolon in `f(x; theta)` does not enforce this mathematically. It is a
notation convention that distinguishes the input `x` from the parameters that
define the function. The inference procedure is what keeps `theta` unchanged.

## Vocabulary and tokens

A **token** is a unit of text recognized by the tokenizer. It may be a word, a
part of a word, punctuation, a space-related unit or a byte sequence.

The **vocabulary** (*vocabulary*) maps every supported token to an integer ID:

```text
ID  token
0   "Il"
1   "cane"
2   "corre"
3   "dorme"
4   "."
```

The vocabulary is not `x`, `y` or `theta`. It is a separate mapping used by the
tokenizer and decoder.

```text
vocabulary = mapping between tokens and IDs
x          = sequence of token IDs
theta      = learned model parameters
```

## The model output

For next-token prediction, the model does not normally return the selected
token directly. It returns one score for every token in the vocabulary. These
scores are called **logits**.

```text
logits = f(x; theta)
```

If the vocabulary contains 10,000 tokens, the final logits vector contains
10,000 elements.

## Generation

**Generation** (*generation*) repeatedly predicts and selects a next token.

```text
logits = f(x; theta)
selected_token_id = selection(logits)
new_x = x with selected_token_id appended
```

Then the same model and parameters are used again:

```text
new_logits = f(new_x; theta)
```

Example:

```text
x = [0, 1]                 # "Il", "cane"
selected_token_id = 2      # "corre"
new_x = [0, 1, 2]
```

The logits are therefore not used directly as the new input. A token ID is
selected from them and appended to the existing sequence.

## Misconceptions corrected

- A vector is a list of numbers, not a single number.
- `theta` is not necessarily one matrix: it is a collection of learned values.
- The semicolon does not mathematically prevent `theta` from changing.
- `y` is normally a vector of logits, not the next token itself.
- Generation reuses `f`; it does not require a new function `f'` at every step.
- The next input is the previous token sequence with the selected ID appended.

## Verified understanding

Given this vocabulary:

```text
0 = "Il", 1 = "cane", 2 = "corre"
```

Roccix correctly determined that appending `"corre"` to `[0, 1]` produces:

```text
[0, 1, 2]
```

Roccix also correctly determined that a vocabulary of 10,000 tokens produces
10,000 final logits for one next-token prediction.

## Next lesson

Study **logits** and **greedy decoding**, including why:

```text
next_token_id = argmax(logits)
```

selects one token without yet introducing probability or randomness.
