"""Plot the 12 recorded training validation curves without editing source logs.

Run from the repository root: python plot_validation_loss.py
Numbered download filenames are accepted only when log.csv is absent and
exactly one log (*).csv is present; every substitution is printed explicitly.
"""

import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIZES = ('small', 'medium', 'large')
POOLS = ('small', 'large')
SEEDS = (1337, 2024)
COLORS = dict(small='#0072B2', medium='#D55E00', large='#009E73')


def read_log(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {'iter', 'train_time', 'val_loss'} <= set(fields):
            raise ValueError(f'{path}: unexpected header {fields}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty log')
    print(f'{path.relative_to(ROOT)}: columns={fields}, rows={len(rows)}')
    for line, row in enumerate(rows, 2):
        try:
            step = int(row['iter'])
            elapsed, loss = float(row['train_time']), float(row['val_loss'])
            if step < 0 or elapsed < 0 or not all(map(math.isfinite, (elapsed, loss))):
                raise ValueError('negative step/time or nonfinite value')
        except (ValueError, TypeError) as error:
            raise ValueError(f'{path}: invalid CSV line {line}: {row}') from error
    return rows


def clean_rows(rows, name):
    rows = list(rows)
    if name == 'medium_small_seed2024':
        if (len(rows) >= 2 and int(rows[0]['iter']) == 0
                and rows[0] == rows[1]):
            print(f'{name}: excluded identical duplicate step 0 (CSV line 3)')
            rows.pop(1)

    reversals = [i for i in range(1, len(rows))
                 if int(rows[i]['iter']) < int(rows[i - 1]['iter'])]
    if name == 'small_large_seed1337' and reversals:
        if (len(reversals) != 1
                or int(rows[reversals[0] - 1]['iter']) != 135440
                or int(rows[reversals[0]]['iter']) != 135000):
            raise ValueError(f'{name}: unexpected backwards steps: ' + str([
                (int(rows[i - 1]['iter']), int(rows[i]['iter'])) for i in reversals]))
        boundary = reversals[0]
        excluded = [row for row in rows[:boundary] if int(row['iter']) >= 135000]
        print(f'{name}: confirmed 135440 -> 135000 resume; excluded original '
              f'segment steps {[int(row["iter"]) for row in excluded]}')
        rows = [row for row in rows[:boundary] if int(row['iter']) < 135000] + rows[boundary:]

    for i in range(1, len(rows)):
        previous, current = rows[i - 1], rows[i]
        if (int(current['iter']) <= int(previous['iter'])
                or float(current['train_time']) < float(previous['train_time'])):
            raise ValueError(f'{name}: unexpected non-increasing steps or backwards time: '
                             f'{previous} -> {current}; refusing to guess a repair')
    return rows


def main():
    expected = {(size, pool, seed): ROOT / 'runs' / f'{size}_{pool}_seed{seed}' / 'log.csv'
                for size in SIZES for pool in POOLS for seed in SEEDS}
    missing = [path for path in expected.values() if not path.is_file()]
    if missing:
        print('Missing expected log.csv paths:')
        for path in missing:
            print(f'  {path.relative_to(ROOT)}')
    paths = {}
    problems = []
    for key, path in expected.items():
        if path.is_file():
            paths[key] = path
        else:
            candidates = sorted(path.parent.glob('log (*).csv'))
            if len(candidates) == 1:
                paths[key] = candidates[0]
                print(f'Using numbered download: {path.relative_to(ROOT)} -> '
                      f'{candidates[0].relative_to(ROOT)}')
            else:
                problems.append(f'{path}: found {len(candidates)} numbered alternatives')
    if problems:
        raise SystemExit('\n'.join(problems))
    # Validate all 12 inputs before creating any figure. Hash source bytes so an
    # accidental source modification would be detected before saving the plot.
    original_hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in paths.values()}
    logs = {key: clean_rows(read_log(path), path.parent.name) for key, path in paths.items()}
    print('Confirmed all 12 headers. train.py records train_time in seconds '
          '(accumulated training-step time, excluding validation); plotting minutes.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for row, seed in enumerate(SEEDS):
        for column, pool in enumerate(POOLS):
            ax = axes[row, column]
            for size in SIZES:
                records = logs[size, pool, seed]
                ax.plot([float(record['train_time']) / 60 for record in records],
                        [float(record['val_loss']) for record in records],
                        color=COLORS[size], linestyle='-', linewidth=1.6,
                        label=size.capitalize())
            ax.set_title(f'Seed {seed} · {pool.capitalize()} data pool')
            ax.grid(alpha=0.25)
            ax.tick_params(labelbottom=True, labelleft=True)
    fig.supxlabel('Elapsed training time (minutes)', y=0.025)
    fig.supylabel('Validation loss', x=0.025)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, title='Model size')
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.87, hspace=0.3, wspace=0.17)
    assert len(fig.axes) == 4 and all(len(ax.lines) == 3 for ax in fig.axes)
    for path, digest in original_hashes.items():
        if hashlib.sha256(path.read_bytes()).digest() != digest:
            raise RuntimeError(f'Source changed during plotting: {path}')
    output = ROOT / 'figures' / 'validation_loss_all_runs.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f'Saved {output} at 300 dpi (4 panels, 3 curves per panel). Sources unchanged.')


if __name__ == '__main__':
    main()
