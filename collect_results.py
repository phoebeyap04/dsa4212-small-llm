"""Collect the small result files of every training run into results/.

For each run folder (e.g. large_small_seed1337) it copies summary.json and
log.csv into results/runs/<run>/, skipping model checkpoints (*.pt) and
preliminary test runs (folders ending in _test), and writes
results/runs_summary.csv with one row per run. If results/test/ holds the
evaluation JSONs written by eval.py, it also writes results/test_results.csv
with one row per evaluated checkpoint.

Only results/runs/ and the two CSV files are rewritten; results/test/ is left
untouched.

Usage (in Colab, after mounting Google Drive):
    python collect_results.py --runs /content/drive/MyDrive/dsa4212/runs --out results
"""
import argparse
import csv
import glob
import json
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='folder that contains one subfolder per run')
parser.add_argument('--out', default='results', help='results folder')
args = parser.parse_args()

# 1. per-run summaries and logs
runs_out = os.path.join(args.out, 'runs')
shutil.rmtree(runs_out, ignore_errors=True)  # only the copied run files are rebuilt
rows = []
for name in sorted(os.listdir(args.runs)):
    run_dir = os.path.join(args.runs, name)
    if name.endswith('_test') or not os.path.isdir(run_dir):
        continue
    os.makedirs(os.path.join(runs_out, name), exist_ok=True)
    for f in ('summary.json', 'log.csv'):
        if os.path.exists(os.path.join(run_dir, f)):
            shutil.copy(os.path.join(run_dir, f), os.path.join(runs_out, name, f))
    summary_path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            rows.append({'run': name, **json.load(fh)})

if rows:
    with open(os.path.join(args.out, 'runs_summary.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(len(rows), 'runs collected:', [r['run'] for r in rows])

# 2. one table of all test evaluations (from eval.py's JSON outputs)
tests = []
for path in sorted(glob.glob(os.path.join(args.out, 'test', '*_test.json'))):
    with open(path) as fh:
        d = json.load(fh)
    run, checkpoint = os.path.basename(path)[:-len('_test.json')].rsplit('_', 1)
    size, pool, seed = run.split('_')
    m = d['metrics']
    tests.append({'run': run, 'size': size, 'pool': pool, 'seed': seed.replace('seed', ''),
                  'checkpoint': checkpoint,
                  'test_cross_entropy_nats': round(m['mean_cross_entropy_nats'], 4),
                  'test_perplexity': round(m['perplexity'], 3),
                  'scored_tokens': m['scored_tokens']})
if tests:
    with open(os.path.join(args.out, 'test_results.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tests[0].keys()))
        writer.writeheader()
        writer.writerows(tests)
print(len(tests), 'test evaluations tabulated')
