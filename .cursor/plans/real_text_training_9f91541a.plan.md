---
name: Real Text Training
overview: Replace synthetic data with Tiny Shakespeare corpus and a custom BPE tokenizer, keeping the model architecture intact while adapting hyperparameters for real language modeling.
todos:
  - id: install-deps
    content: Add cell to install tokenizers library and download Tiny Shakespeare
    status: pending
  - id: train-bpe
    content: Add cell to train BPE tokenizer (vocab=512) on Shakespeare text
    status: pending
  - id: tokenize-data
    content: Replace synthetic data cell with real tokenization + train/val split + chunking
    status: pending
  - id: update-config
    content: "Update ArchitectureConfig: vocab_size=512, batch_size=16, seq_len=32"
    status: pending
  - id: update-training
    content: Adjust training loop for larger dataset (step-based or more epochs)
    status: pending
isProject: false
---

# Real Text Training with Tiny Shakespeare + BPE

## Key Constraint

The model processes each sequence position **sequentially** (not in parallel like a Transformer), with 3 transport blocks x 5 Langevin steps = 15 iterative operations per position. Sequence length directly multiplies wall-clock time, so we must keep it short enough to train on CPU.

## Changes

### 1. Install dependencies and download data (new cell after imports)

- `pip install tokenizers` (HuggingFace Tokenizers — fast BPE training)
- Download Tiny Shakespeare from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Save to workspace as `data/input.txt`

### 2. Train a BPE tokenizer (new cell)

- Use `tokenizers.Tokenizer` with `tokenizers.models.BPE` to train a custom BPE on the Shakespeare text
- **vocab_size = 512** — matches current `n_dictionary_atoms` and keeps the model small (~3M params). Going larger (e.g., 50k with tiktoken) would make the embedding/decoder dominate the parameter budget.
- Save the tokenizer for reuse

### 3. Tokenize and create train/val splits (replaces cell 15)

- Tokenize full corpus into a single long token ID sequence
- 90/10 train/val split
- Chunk into fixed-length sequences of `seq_len` tokens with a sliding window
- This replaces `generate_synthetic_data()` entirely

### 4. Update `ArchitectureConfig` (cell 3)

- `vocab_size: 256 -> 512` (to match BPE vocab)
- `max_seq_len: 128 -> 64` (keep unchanged, just use 64 for actual training)
- `batch_size: 32 -> 16` (longer sequences use more memory per batch)
- `n_epochs` will likely need to increase or switch to step-based training since we'll have much more data

### 5. Sequence length consideration

- **seq_len = 32** is the recommended starting point. At 32 positions with 15 ops each, that's 480 iterative steps per sequence — already ~2x more work than the current 16-token sequences. Going to 64 or 128 would be 4-8x slower.
- We can increase later once we confirm the model learns on real text.

### 6. Training loop adjustments (cell 16)

- The training loop structure is fine — it already takes token_ids and does next-token prediction
- May want to switch from epoch-based to step-based training with a max_steps limit, since the dataset will be much larger
- Reduce logging frequency (every N steps instead of every epoch)

## What stays the same

- The entire model architecture (SpectralGaugeCLM, transport blocks, Langevin settling, ContextGate)
- The loss function (CE + dict coherence + sparsity)
- The optimizer and scheduler

## Estimated parameter impact

With vocab_size=512: embedding grows from 256x128=32K to 512x128=65K params, decoder output grows similarly. Total model goes from ~3M to ~3.1M — negligible increase.