
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## DSA4212 TinyStories evaluation

Run `eval.py` separately after training. It reads the explicit `final.pt` (end
of the training allowance) or `best.pt` (selected by training validation loss),
using its saved `model_args` and weights. It does not train or select checkpoints.
Dependencies are PyTorch and NumPy. From the notebook's `/content/repo` directory,
with Drive mounted and the prepared data already available, run these Colab cells:

```python
%cd /content/repo
!cp /content/drive/MyDrive/dsa4212/data/val.bin data/tinystories/val.bin
run = '/content/drive/MyDrive/dsa4212/runs/small_small_seed1337'

# Offline synthetic checks (no dataset download or training).
!python -m unittest -v test_eval.py

# Validation smoke test: first 4 target blocks, explicitly marked in the JSON.
!python eval.py --checkpoint {run}/final.pt --split val --device cuda --precision float32 --batch-size 8 --context-length 256 --max-blocks 4 --output {run}/eval_final_val_smoke.json

# Full validation: compare final and best using exactly the same protocol.
!python eval.py --checkpoint {run}/final.pt --split val --device cuda --precision float32 --batch-size 8 --context-length 256 --output {run}/eval_final_val.json
!python eval.py --checkpoint {run}/best.pt --split val --device cuda --precision float32 --batch-size 8 --context-length 256 --output {run}/eval_best_val.json
```

For final evaluation, after development choices are fixed, run:

```python
!cp /content/drive/MyDrive/dsa4212/data/test.bin data/tinystories/test.bin
!python eval.py --checkpoint {run}/final.pt --split test --device cuda --precision float32 --batch-size 8 --context-length 256 --output {run}/eval_final_test.json
!python eval.py --checkpoint {run}/best.pt --split test --device cuda --precision float32 --batch-size 8 --context-length 256 --output {run}/eval_best_test.json
```

Set `run` to each notebook run name (`{size}_{pool}_seed{seed}`, for small,
medium, large model sizes, small/large pools and seeds 1337/2024). Use the same
context length and precision for every comparison. CPU uses `--device cpu
--precision float32`; CUDA also supports explicit `float16`/`bfloat16` autocast
(T4 does not support bfloat16). Lower `--batch-size` if GPU memory is insufficient.
The script never falls back to another split if the selected file is absent.
Existing output files require explicit `--overwrite`; training artifacts are
protected even with that flag. Evaluate only checkpoints you trust.

The protocol reads the prepared raw native `uint16` stream, including EOS tokens.
For context length L (default 256, shared by all three configurations), block k
uses inputs `tokens[k*L:k*L+L]` and shifted targets
`tokens[k*L+1:k*L+L+1]`. Context and position indices reset at each block;
story boundaries are treated as ordinary EOS-separated stream boundaries, as in
training. Targets never overlap: every token except stream token 0 is scored
once. The final short block is evaluated at its actual length, without padding;
no tail tokens are dropped, and a final incomplete batch is retained. Consequently
the available context for each prediction ranges from 1 to L tokens. Contexts
larger than the checkpoint's saved block size are rejected rather than silently
reduced. `--max-blocks N` evaluates only the first N blocks for smoke testing and
records excluded tokens; omit it for complete split evaluation.

Evaluation uses `model.eval()` and inference mode. It sums float32 per-token
next-token cross-entropies with float64 accumulation and divides by the valid
target count (ignoring target -1 if present; prepared uint16 files contain no
padding/ignored targets). Perplexity is `exp(mean_cross_entropy_nats)`; overflow
is represented by JSON `null` with `perplexity_overflow: true`. JSON also records
input SHA-256 hashes, saved model/run metadata, token counts, window settings,
device, precision and elapsed time including loading/hashing. These complete
split metrics differ from `train.py`'s bounded validation estimate. Deterministic
ordering/kernels are enabled; floating-point results can still differ across
hardware, precision, PyTorch versions and batch sizes.

Synthetic checks cover shifting, boundary coverage, uneven batches, weighted
loss with ignored targets, repeatability, compiled checkpoint prefixes, output
protection, missing splits and perplexity overflow. Recorded full-test CUDA
results are included below; reproducing them requires the original local inputs.

## DSA4212 report reproducibility

The 24 recorded held-out evaluations are in `results/test/`, named
`{size}_{pool}_seed{seed}_{final|best}_test.json`: three model sizes, two data
pools, two seeds (1337 and 2024), and two checkpoint types. The JSONs are exact
copies of the files in the local source archive `test_results.zip`, including
the original Kaggle paths and metadata. These historical paths identify the
original environment; they do not need to exist on a new machine.

Every result records the test split, context length 256, float32 evaluation,
batch size 8, device `cuda:0`, and the complete test stream with no block limit.
The stream has 1,551,129 raw uint16 tokens; 1,551,128 targets are scored. All
results share this test-data SHA-256 (also verified against the local binary):

```text
fd3da30feb765b7611e31178d4544421c0ef793a0ae98a4f4abf4c7563f00ccd
```

### Regenerate the validation figure

From the repository root, using a Python environment with Matplotlib:

```sh
python -m pip install matplotlib
python plot_validation_loss.py
```

This saves `figures/validation_loss_all_runs.png` at 300 dpi. All 12 input logs
are included under `runs/`; their downloaded filenames are retained:

| Run | CSV path |
| --- | --- |
| small_small_seed1337 | `runs/small_small_seed1337/log (7).csv` |
| small_small_seed2024 | `runs/small_small_seed2024/log (8).csv` |
| small_large_seed1337 | `runs/small_large_seed1337/log (5).csv` |
| small_large_seed2024 | `runs/small_large_seed2024/log (6).csv` |
| medium_small_seed1337 | `runs/medium_small_seed1337/log (3).csv` |
| medium_small_seed2024 | `runs/medium_small_seed2024/log (4).csv` |
| medium_large_seed1337 | `runs/medium_large_seed1337/log (1).csv` |
| medium_large_seed2024 | `runs/medium_large_seed2024/log (2).csv` |
| large_small_seed1337 | `runs/large_small_seed1337/log (3).csv` |
| large_small_seed2024 | `runs/large_small_seed2024/log (4).csv` |
| large_large_seed1337 | `runs/large_large_seed1337/log (1).csv` |
| large_large_seed2024 | `runs/large_large_seed2024/log (2).csv` |

The script reports absent canonical `log.csv` names and explicitly selects
the unique numbered download in each folder. It uses `train_time` in seconds
(training steps only, excluding validation), converts to minutes, and plots
recorded validation losses without smoothing or seed averaging. The four panels
share scales, model colors and one legend. Original CSVs are never edited.
In memory only, it excludes one identical duplicate step-0 row from
`medium_small_seed2024` and original-segment steps 135000 and 135440 from
`small_large_seed1337`, keeping the resumed segment from step 135000 onward.
Unexpected non-increasing steps or backwards time cause an error.

### Reproduce the held-out evaluations

Checkpoints and binary datasets are local inputs, excluded from Git. Preserve
the original files and place them in the existing layout below (a fresh clone
requires obtaining these original inputs separately):

```text
final_folder/test.bin
final_folder/{size}_{pool}_seed{seed}.pt
best_folder/{size}_{pool}_seed{seed}_best.pt
```

Use the original prepared `test.bin`; do not regenerate the tokenizer or splits
to reproduce these results. Alternatively, put that same binary in
`data/tinystories/test.bin` and change `--data-dir` to `data/tinystories`.
From the repo root on a CUDA machine with PyTorch and NumPy installed:

```sh
python -m pip install torch numpy
python -m unittest -v test_eval.py
python -c "import hashlib; from pathlib import Path; p=Path('final_folder/test.bin'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'fd3da30feb765b7611e31178d4544421c0ef793a0ae98a4f4abf4c7563f00ccd'"

# One run, evaluating final and best separately with the same full-test protocol.
python eval.py --checkpoint final_folder/small_small_seed1337.pt --data-dir final_folder --split test --context-length 256 --precision float32 --batch-size 8 --device cuda:0 --output results/reproduced/test/small_small_seed1337_final_test.json
python eval.py --checkpoint best_folder/small_small_seed1337_best.pt --data-dir final_folder --split test --context-length 256 --precision float32 --batch-size 8 --device cuda:0 --output results/reproduced/test/small_small_seed1337_best_test.json
```

To reproduce all 24 results, run this Bash loop instead of the two example calls:

```bash
for size in small medium large; do
  for pool in small large; do
    for seed in 1337 2024; do
      run="${size}_${pool}_seed${seed}"
      python eval.py --checkpoint "final_folder/${run}.pt" --data-dir final_folder --split test --context-length 256 --precision float32 --batch-size 8 --device cuda:0 --output "results/reproduced/test/${run}_final_test.json" || exit 1
      python eval.py --checkpoint "best_folder/${run}_best.pt" --data-dir final_folder --split test --context-length 256 --precision float32 --batch-size 8 --device cuda:0 --output "results/reproduced/test/${run}_best_test.json" || exit 1
    done
  done
done
```

There is no `--max-blocks` limit in these commands. New outputs go to a separate
directory so the recorded JSONs remain untouched. Existing output paths are
rejected unless `--overwrite` is explicitly supplied. The recorded environment
was a Tesla T4 with PyTorch `2.11.0+cu130`; numerical results may vary with
hardware and software versions. Each JSON records checkpoint and dataset hashes
for checking exact input identity.

### Known checkpoint/summary discrepancy

For `small_large_seed1337`, the saved **best checkpoint evaluated on test is
from step 135440**, as recorded by `run_metadata.iter_num` in
`results/test/small_large_seed1337_best_test.json`. Its training summary lists
step **134500** as best (reported by the experiment owner; the summary file is
not present in this repository). Therefore the best test metric describes the
saved step-135440 checkpoint, not a step-134500 model. The resumed final
checkpoint's recorded step is 135001. This provenance discrepancy is retained;
neither evaluation JSONs nor original logs have been rewritten, and the plot's
in-memory resume repair does not change which checkpoint was evaluated.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepe