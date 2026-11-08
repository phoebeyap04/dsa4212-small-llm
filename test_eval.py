"""Offline checks: python -m unittest -v test_eval.py"""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import eval as evaluator
from model import GPT, GPTConfig


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_shift_boundaries_and_partial_batches(self):
        data = np.arange(24, dtype=np.uint16)
        batches = list(evaluator.batches(data, 4, 3, 23))
        self.assertEqual([tuple(x.shape) for x, _ in batches], [(3, 4), (2, 4), (1, 3)])
        targets = torch.cat([y.flatten() for _, y in batches])
        self.assertEqual(targets.tolist(), list(range(1, 24)))
        for x, y in batches:
            torch.testing.assert_close(y, x + 1)

    def test_weighted_loss_ignores_and_determinism(self):
        class Toy(torch.nn.Module):
            def forward(self, x, y):
                # Different target losses in the tail expose batch-mean averaging.
                logits = torch.stack((x.float(), -x.float()), dim=-1)
                return logits, None

        data = np.array([0, 1, 1, 0, 1, 0, 1, -1], dtype=np.int64)
        expected_nll = 0.0
        expected_count = 0
        for x, y in evaluator.batches(data, 3, 1, 7):
            logits, _ = Toy()(x, y)
            expected_nll += F.cross_entropy(logits.reshape(-1, 2), y.flatten(),
                                           ignore_index=-1, reduction='sum').item()
            expected_count += (y != -1).sum().item()
        results = [evaluator.evaluate(Toy(), data, 3, bs, 7, torch.device('cpu'), 'float32')
                   for bs in (1, 2, 3, 2)]
        for result in results:
            self.assertEqual(result['scored_tokens'], expected_count)
            self.assertAlmostEqual(result['mean_cross_entropy_nats'], expected_nll / expected_count, places=6)
        self.assertEqual(results[1], results[3])

    def test_checkpoint_cli_and_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = GPTConfig(n_layer=1, n_head=1, n_embd=8, vocab_size=11,
                            block_size=4, dropout=0.5)
            model = GPT(cfg)
            state = {'_orig_mod.' + k: v for k, v in model.state_dict().items()}
            checkpoint = root / 'final.pt'
            torch.save(dict(model=state, model_args=asdict(cfg), config=dict(seed=1337,
                       train_file='train_small.bin'), iter_num=7), checkpoint)
            np.array([i % 11 for i in range(24)], dtype=np.uint16).tofile(root / 'val.bin')
            before = evaluator.sha256(checkpoint)
            loaded, _ = evaluator.load_model(checkpoint, torch.device('cpu'))
            self.assertFalse(loaded.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            best = root / 'best.pt'
            torch.save(dict(model=model.state_dict(), model_args=asdict(cfg)), best)
            uncompiled, _ = evaluator.load_model(best, torch.device('cpu'))
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, uncompiled.state_dict()[key])
            args = ['--checkpoint', str(checkpoint), '--split', 'val', '--data-dir', str(root),
                    '--output', str(root / 'result.json'), '--context-length', '4', '--batch-size', '3']
            with redirect_stdout(io.StringIO()):
                evaluator.main(args)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['metrics']['scored_tokens'], 23)
            self.assertEqual(result['windowing']['blocks'], 6)
            self.assertFalse(result['smoke_test'])
            self.assertEqual(result['run_metadata']['config']['seed'], 1337)
            self.assertEqual(evaluator.sha256(checkpoint), before)
            with self.assertRaises(FileExistsError):
                evaluator.main(args)
            with redirect_stdout(io.StringIO()):
                evaluator.main(args + ['--overwrite'])
            repeated = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['metrics'], repeated['metrics'])
            with redirect_stdout(io.StringIO()):
                evaluator.main(args + ['--overwrite', '--max-blocks', '2'])
            smoke = json.loads((root / 'result.json').read_text())
            self.assertTrue(smoke['smoke_test'])
            self.assertEqual(smoke['metrics']['scored_tokens'], 8)
            with self.assertRaises(FileNotFoundError):
                evaluator.main(args + ['--split', 'test', '--overwrite'])
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                evaluator.main(args + ['--output', str(root / 'summary.json'), '--overwrite'])

    def test_perplexity_overflow_json(self):
        ppl, overflow = evaluator.perplexity(1000)
        self.assertIsNone(ppl)
        self.assertTrue(overflow)
        self.assertEqual(json.dumps({'perplexity': ppl}, allow_nan=False), '{"perplexity": null}')


if __name__ == '__main__':
    unittest.main()
