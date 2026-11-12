"""Collect the small result files of every training run into results/.

For each run folder (e.g. large_small_seed1337) it copies summary.json,
log.csv and any eval_*.json files, skipping model checkpoints (*.pt) and
preliminary test runs (folders ending in _test). It also writes
results/runs_summary.csv, with one row per run built from its summary.json.

Usage (in Colab, after mounting Google Drive):
    python collect_results.py --runs /content/drive/MyDrive/dsa4212/runs --out results
"""
import argparse
import csv
import json
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--runs', required=True, help='folder that contains one subfolder per run')
parser.add_argument('--out', default='results', help='where to write the collected results')
args = parser.parse_args()

shutil.rmtree(args.out, ignore_errors=True)  # only clears the output folder
rows = []
for name in sorted(os.listdir(args.runs)):
    run_dir = os.path.join(args.runs, name)
    if name.endswith('_test') or not os.path.isdir(run_dir):
        continue
    os.makedirs(os.path.join(args.out, 'runs', name), exist_ok=True)
    for f in os.listdir(run_dir):
        if f.endswith('.json') or f == 'log.csv':
            shutil.copy(os.path.join(run_dir, f), os.path.join(args.out, 'runs', name, f))
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
