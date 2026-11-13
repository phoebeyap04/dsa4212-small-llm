# Model Size and Data Repetition under a Fixed Compute Budget

DSA4212 group project (Group 38). We train small GPT-style language models from
scratch on TinyStories and ask: **within one hour of training on a single NVIDIA
T4 GPU, does a smaller model that processes more text outperform a larger one,
and does the answer change when unique training data is limited?**

The complete Overleaf source is in [`report/`](report/), with `main.tex`,
`references.bib`, `validation_loss_all_runs.png`, and `nus_logo.jpg`. A compiled
final PDF has not yet been included or checked for the 10-page limit. Authors are listed in
[`authors.md`](authors.md). The code builds on Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) (MIT licence); see *Credit*
at the end of this file.

## Main findings

Held-out test cross-entropy (nats per token, lower is better), averaged over
seeds 1337 and 2024. *Final* = end of the one-hour budget; *best* = checkpoint
with the lowest validation loss.

| Model (non-embedding params) | 10k stories, final | 10k stories, best | 1M stories, final | 1M stories, best |
|---|---|---|---|---|
| Small (3.1M)  | 6.093 | 2.483 | 1.640 | 1.639 |
| Medium (10.6M)| 5.220 | 2.451 | 1.518 | 1.517 |
| Large (25.2M) | 4.816 | 2.463 | 1.507 | 1.507 |

- With 1,000,000 stories, the largest model achieved the lowest test loss, with
  only a small gain over the medium model.
- With 10,000 stories, all sizes peaked after about 7-9 passes over the data and
  then overfitted; keeping the best checkpoint reduced test loss by 2.35-3.61
  nats per token, far more than the choice of model size.

## Experimental setup

| | |
|---|---|
| Data | TinyStories (`roneneldan/TinyStories`), deduplicated; nested pools of 10,000 and 1,000,000 stories; separate validation (8,039) and test (7,350) stories |
| Tokeniser | Byte-level BPE, 8,000 tokens, trained on the large pool only |
| Models | nanoGPT, trained from random initialisation: 4/6/8 layers, width 256/384/512 |
| Runs | 3 sizes x 2 pools x 2 seeds = 12 runs |
| Budget | 3,600 s of training time per run on one T4 (Google Colab or Kaggle); evaluation and checkpointing excluded |
| Evaluation | Token-weighted cross-entropy on the full test set, for both the final and best checkpoint of each run (24 evaluations) |

## Repository structure

| Path | Contents |
|---|---|
| `data/tinystories/prepare.py` | Downloads TinyStories, splits, deduplicates, builds the pools, trains the tokeniser, writes token files |
| `data/tinystories/manifest.txt`, `shared_bpe_tokenizer.json` | Dataset statistics and the shared tokeniser |
| `config/ts_small.py`, `ts_medium.py`, `ts_large.py` | The three model configurations (all other settings identical) |
| `train.py` | nanoGPT training script, modified for the time budget, time-based learning-rate schedule, logging, best checkpoint and budget-preserving resume |
| `model.py`, `configurator.py`, `sample.py` | nanoGPT model, config loader and text sampling (sampling uses our tokeniser) |
| `run_experiment.ipynb`, `run_experiment_kaggle.ipynb` | Notebooks used to run the experiments on Colab and on Kaggle |
| `eval.py`, `test_eval.py` | Deterministic held-out evaluation and its tests |
| `collect_results.py` | Copies each run's `summary.json` and `log.csv` into `results/runs/`, builds `results/runs_summary.csv`, and builds `results/test_results.csv` from `results/test/` (leaves `results/test/` untouched) |
| `plot_validation_loss.py`, `figures/` | Validation-loss figure used in the report |
| `results/runs/<run>/` | `summary.json` and `log.csv` for each of the 12 runs |
| `results/runs_summary.csv` | One row per run (speed, tokens seen, epochs, losses) |
| `results/test/` | The 24 test-evaluation JSONs |
| `results/test_results.csv` | All 24 test scores in one table |
| `runs/` | Training logs read by `plot_validation_loss.py`; see the duplicate-row exception below |

`LICENSE` is nanoGPT's MIT licence. Example scripts and data folders from the
original nanoGPT that this project does not use have been removed.

## Reproducing the experiments

**Requirements.** Python 3 with PyTorch (pre-installed on Colab and Kaggle) and
`pip install numpy datasets tokenizers tiktoken`; `matplotlib` for the figure.
Training was run on an NVIDIA T4.

**1. Prepare the data** (about 20-40 minutes on CPU):

```bash
cd data/tinystories
python prepare.py --reuse-tokenizer
cd ../..
```

The `--reuse-tokenizer` option loads the committed
`data/tinystories/shared_bpe_tokenizer.json` and leaves its bytes unchanged. It
does not fit a new tokenizer. Omit this flag only when deliberately training a
new tokenizer; that default overwrites the saved tokenizer and may produce
token IDs incompatible with the original checkpoints.

This writes `train_small.bin`, `train_large.bin`, `val.bin` and `test.bin`
(not stored in the repository because of their size) and `manifest.txt`. Check
that the manifest reports 10,000 / 1,000,000 / 8,039 / 7,350 stories.

**2. Train.** Open `run_experiment.ipynb` in Colab (or
`run_experiment_kaggle.ipynb` on Kaggle) with a T4 GPU, set `size`, `pool` and
`seed` in the Settings cell, keep `budget = 3600.0`, and change the data
preparation call to `python prepare.py --reuse-tokenizer` before running the
cells in order. Set a new output directory to preserve recorded runs. Equivalently, from the repository root:

```bash
python train.py config/ts_small.py --train_file=train_small.bin \
    --seed=1337 --time_budget=3600.0 --out_dir=reruns/small_small_seed1337
```

This concrete example trains one run. To rerun all 12 main configurations,
use the following Bash loop from the repository root (requires the prepared
training and validation binaries and a CUDA GPU):

```bash
for size in small medium large; do
  for pool in small large; do
    for seed in 1337 2024; do
      python train.py "config/ts_${size}.py" --train_file="train_${pool}.bin" \
        --seed="$seed" --time_budget=3600.0 \
        --out_dir="reruns/${size}_${pool}_seed${seed}" || exit 1
    done
  done
done
```

Use new output directories for reruns to preserve the recorded logs. Each run writes `log.csv`, `summary.json`, `best.pt`,
`final.pt` and `ckpt.pt` (for resuming) to its output folder.

**3. Collect results:**

```bash
python collect_results.py --runs reruns --out results/reproduced
```

**4. Evaluate and plot:** see the evaluation and reproducibility sections below.

## Reproducibility notes

- Model checkpoints (`*.pt`, up to about 350 MB each) are not in the
  repository because of GitHub's file-size limit; the saved logs, summaries and
  test JSONs are.
- The data files can be regenerated with the saved tokenizer as above;
  verify the test-data hash below before evaluating the original checkpoints.
  The dataset revision and NumPy/datasets/tokenizers training versions were
  not recorded, so regenerated data are not guaranteed to match byte-for-byte.
- Throughput on the same GPU type varied between sessions (large model:
  58,000-77,000 tokens/s), so runs of one configuration processed different
  numbers of tokens within the same budget; `tokens_seen` is recorded per run.
- One run was accidentally resumed after finishing; see *Known
  checkpoint/summary discrepancy* below.

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

The log collections differ in one recorded detail: `runs/medium_small_seed2024/log (4).csv`
contains two identical step-0 rows; `results/runs/medium_small_seed2024/log.csv`
contains one. The plot drops the duplicate only in memory. Other log pairs are
identical; keep the original files as recorded.

### Reproduce the held-out evaluations

Download the original inputs to reproduce the recorded held-out test scores:

- [Original run files](https://drive.google.com/drive/folders/1pa1h6WPk8r-OcfmWkuoDUm7Afk4ATYvA):
  download `best.pt` and `final.pt` from each of the 12 run folders, preserving
  the run names (`{size}_{pool}_seed{seed}`). Keep these original folders intact.
  For the evaluation commands below, copy each run's `final.pt` to
  `final_folder/{run}.pt` and its `best.pt` to `best_folder/{run}_best.pt`.
  For example, `small_small_seed1337/final.pt` becomes
  `final_folder/small_small_seed1337.pt`.
- [Evaluation data](https://drive.google.com/drive/folders/1VSbVlqNnSK7sQzVfFKL3t8bXUCxLXZab?usp=drive_link):
  download `test.bin` and place it at `final_folder/test.bin`, the path used by
  the hash check and `--data-dir final_folder` evaluation commands below.

Create `final_folder/` and `best_folder/` in the repository root as needed.
Verify the downloaded inputs using the SHA-256 instructions below.

Rerunning the training protocol is different from evaluating the original saved
checkpoints: a new one-hour run can stop at a different step and produce different
weights and scores, even with the same seed. For protocol reruns, checkpoints
are `reruns/<run>/final.pt` and `reruns/<run>/best.pt`; use those explicit paths with
`eval.py --data-dir data/tinystories`. For exact recorded-score reproduction,
use the original checkpoint files below and compare their SHA-256 values with
`checkpoint.sha256` in the matching `results/test/<run>_<kind>_test.json`.

Checkpoints and binary datasets are local inputs, excluded from Git. Preserve
the original files and place them in the existing layout below (a fresh clone
requires obtaining these original inputs separately):

```text
final_folder/test.bin
final_folder/{size}_{pool}_seed{seed}.pt
best_folder/{size}_{pool}_seed{seed}_best.pt
```

Prefer the original prepared `test.bin`. Alternatively regenerate it using
`prepare.py --reuse-tokenizer` as above, and accept it as the same evaluation
input only if its SHA-256 matches the recorded value. Do not refit the tokenizer
or alter the split policy to evaluate the original checkpoints. Alternatively, put that same binary in
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
hardware and software versions. `2.11.0+cu130` is the recorded evaluation
version, not a verified training-environment version. No exact Python, NumPy,
datasets or tokenizers version or Hugging Face dataset revision is evidenced
in the saved metadata, so none is asserted here. Each JSON records checkpoint and dataset hashes
for checking exact input identity.

### Known checkpoint/summary discrepancy

`small_large_seed1337` was accidentally resumed after it had finished. The
original run trained to step 135440 (3600.0 s of training time) and saved that
end-of-budget model as `best.pt`. A resumed session then reloaded the step-135000
checkpoint, trained one further step and overwrote `final.pt` and
`summary.json` (step 135001, 3603.8 s, best step restored as 134500); the
recompilation after resuming fell inside that timed step. This is visible in
`results/runs/small_large_seed1337/log.csv`, whose last three rows are steps
135440, 135000 and 135001. The test JSONs record the evaluated checkpoints
(`run_metadata.iter_num` 135440 for best, 135001 for final); their test losses
differ by 0.0006 nats per token. No evaluation JSON or log has been rewritten.


## Build and check the final report

The source was copied unchanged from the final Overleaf export
`Assignment_Group38.zip`. With a LaTeX distribution that includes `natbib`,
`libertine`, `fullpage`, `hyperref`, `booktabs` and the standard graphics/math
packages, compile in a temporary directory:

```bash
report_build_dir=$(mktemp -d)
cp report/main.tex report/references.bib report/validation_loss_all_runs.png report/nus_logo.jpg "$report_build_dir/"
(cd "$report_build_dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
pdfinfo "$report_build_dir/main.pdf" | grep '^Pages:'
```

`pdfinfo` is supplied by Poppler. The PDF must be at most 10 pages **including
references and appendices**. Check the build log for undefined references or
citations and inspect the resulting PDF before adding the final PDF to
`report/`. The source inputs and citation/reference keys were checked, but no
LaTeX installation was available for compilation or page-count verification.

---

## Credit

This repository is a fork of Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), released under the MIT licence
(see `LICENSE`). We use its model definition (`model.py`), configuration loader
(`configurator.py`) and training loop (`train.py`, modified as described above).
No pretrained GPT-2 weights are used. The full original README is available in
the upstream repository.
