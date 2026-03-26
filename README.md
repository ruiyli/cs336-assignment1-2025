# CS336 Assignment 1: Building a Transformer LM from Scratch

Stanford CS336 (Language Models from Scratch) Spring 2025 - Assignment 1

## What's implemented

- **Byte-Pair Encoding (BPE) tokenizer** — training and encoding/decoding
- **Transformer LM** — RMSNorm, SwiGLU, RoPE, Causal Multi-Head Attention
- **Training components** — AdamW optimizer, cosine LR schedule, gradient clipping
- **Training infrastructure** — data loading, checkpointing, text generation

## Results

| Model | Dataset | Val Loss |
|-------|---------|----------|
| 22.7M params | TinyStories | 1.69 |
| 45.2M params | OpenWebText | 4.22 |

## Structure
```
cs336_basics/
├── bpe.py              # BPE tokenizer
├── transformer.py      # Transformer LM and all components
├── optimizer.py        # AdamW, LR schedule, gradient clipping
├── training.py         # Data loading, checkpointing
├── train.py            # Training script
├── train_bpe_script.py # BPE training script
├── encode_dataset.py   # Dataset encoding script
└── generate.py         # Text generation script
```

## Usage
```bash
# Install dependencies
uv sync

# Train BPE tokenizer
python cs336_basics/train_bpe_script.py \
    --input_path data/TinyStoriesV2-GPT4-train.txt \
    --vocab_size 10000 \
    --vocab_out vocab.json \
    --merges_out merges.txt

# Train language model
python cs336_basics/train.py \
    --train_path tinystories_train.npy \
    --val_path tinystories_val.npy \
    --vocab_size 10000 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --device cuda:0

# Generate text
python cs336_basics/generate.py \
    --checkpoint checkpoints/ckpt_final.pt \
    --vocab_path vocab.json \
    --merges_path merges.txt \
    --prompt "Once upon a time"
```

## Course

[CS336: Language Models from Scratch](https://stanford-cs336.github.io/spring2025/)
— Percy Liang, Tatsunori Hashimoto, Stanford University, Spring 2025
