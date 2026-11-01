# Import required libraries:
import os
import random
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Normalization Rules:
## Text normalization: strip whitespace and convert to lowercase
def normalize_text(text):
    return text.strip().lower()

## Deduplication: remove exact duplicates based on normalized text
def deduplicate(stories, seen_hashes):
    unique_stories = []
    for story in stories:
        norm_story = normalize_text(story)
        if norm_story not in seen_hashes:
            seen_hashes.add(norm_story)
            unique_stories.append(story)
    return unique_stories

#Data Preprocessing:
## Download TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories")

## Prepare training, validation and test splits
train_stories = dataset['train']['text']
official_val_stories = dataset['validation']['text']

##  Divide the validation splits by whole stories into our own validation and final test sets
midpoint = len(official_val_stories) // 2
val_stories = official_val_stories[:midpoint]
test_stories = official_val_stories[midpoint:]

## Remove duplicates within each split
seen_hashes = set() # Track globally to prevent train/test overlap
train_clean = deduplicate(train_stories, seen_hashes)
val_clean = deduplicate(val_stories, seen_hashes)
test_clean = deduplicate(test_stories, seen_hashes)

# Record how many were removed
removed_train = len(train_stories) - len(train_clean)
print(f"Removed {removed_train} duplicate stories from the training set.")

## Construct nested training pools
## Shuffle the training stories once using a fixed data-selection seed
random.seed(42)
random.shuffle(train_clean)

# The small pool contains an initial subset and the large pool contains the same subset plus additional stories, tinystories has about 2.1M stories
small_pool_size = 10000    # approx 2.5 MB of text
large_pool_size = 1000000  # approx 250 MB of text

small_pool = train_clean[:small_pool_size]
large_pool = train_clean[:large_pool_size]

# Save large pool temporarily to train tokenizer
with open("temp_train_large.txt", "w", encoding="utf-8") as f:
    f.write("<|endoftext|>".join(large_pool))

## Train Byte-level BPE Tokenizer (about 8,000 vocab) on large pool
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["<|endoftext|>", "<|unk|>"])

## Train it only on training text and freeze it[cite: 5]
tokenizer.train(files=["temp_train_large.txt"], trainer=trainer)
tokenizer.save("shared_bpe_tokenizer.json")
os.remove("temp_train_large.txt") # Clean up

## Encode datasets to binary for nanoGPT
eos_id = tokenizer.token_to_id("<|endoftext|>")

def encode_and_save(stories, filename):
    all_ids = []
    for s in stories:
        all_ids.extend(tokenizer.encode(s).ids)
        all_ids.append(eos_id) # End-of-story token
    
    # vocab is 800 so uint16 (max 65535)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(filename)
    return len(stories), len(arr)

s_count, s_tokens = encode_and_save(small_pool, "train_small.bin")
l_count, l_tokens = encode_and_save(large_pool, "train_large.bin")
v_count, v_tokens = encode_and_save(val_clean, "val.bin")
t_count, t_tokens = encode_and_save(test_clean, "test.bin")

## Generate split/pool manifest
## Output the table of stories and tokens in each set
manifest = f"""Dataset Manifest: TinyStories (roneneldan/TinyStories)
Normalization Rule: text.strip().lower()
Duplicates Removed (Train): {removed_train}

| Split           | Story Count | Token Count | File              |
|-----------------|-------------|-------------|-------------------|
| Train (Small)   | {s_count:<11} | {s_tokens:<11} | train_small.bin   |
| Train (Large)   | {l_count:<11} | {l_tokens:<11} | train_large.bin   |
| Validation      | {v_count:<11} | {v_tokens:<11} | val.bin           |
| Test (Held-out) | {t_count:<11} | {t_tokens:<11} | test.bin          |

Tokenizer: Byte-level BPE, 8000 Vocab (shared_bpe_tokenizer.json)
"""

with open("manifest.txt", "w") as f:
    f.write(manifest)