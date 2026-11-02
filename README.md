# Quantum Hashing by Amplitude Amplification

Code and archived data accompanying *Quantum Hashing by Amplitude Amplification*.
The repository evaluates the linear amplitude-amplification hash (AAH), Feistel
AAH, an XOR parity baseline, and width-matched truncated SHA-256/MD5 baselines.

The structural result is exact: for distinct targets, the linear AAH's modal
output is the first message block's target. The Feistel construction applies a
fixed public permutation to that output. This code is research software, not a
cryptographic hash implementation suitable for security applications.

## Reproduce offline

Requires Python 3.12. The pinned dependency versions are those used for the
paper. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/make_report.py
python scripts/make_assets.py
pytest -q
```

`make_report.py` reads the archived counts and writes:

- `data/hardware_report.md`: hardware cost/agreement, avalanche, and chi-square tables.

`make_assets.py` performs the exact state-vector analysis and writes:

- `data/structural.json`: model validation, exhaustive target-pair statistics,
  Feistel equivalence, chi-square floors, surviving-signal floors, gate-error
  budgets, and the 8-bit iteration sweep.
- `figures/aah_schematic.svg`
- `figures/feistel_round.svg`
- `figures/iteration_sweep.svg`
- `figures/hardware_determinism.svg`

No IBM credentials or network access are needed for these commands. The exact
analysis is deterministic; sampled noiseless results use seed 42 and the
multinomial noise floor uses seed 20260908.

## Repository layout

| Path | Contents |
|---|---|
| `quantum_circuits/` | Linear AAH, reversible Feistel oracle, and XOR parity circuits |
| `experiments/ibm/` | Noiseless/hardware runner, metrics, backend selection, and classical baselines |
| `scripts/` | Offline paper table, analysis, and figure generators |
| `results/classical_baselines.json` | Seeded SHA-256/MD5 reference results |
| `results/ibmq_20260908T220749Z/` | Hashing counts, per-shot bitstrings, circuit metrics, session job IDs, manifest, calibration, and the cited 16-probe relaxation artifact |
| `tests/` | Feistel circuit contracts and paper-result regression tests |

Each quantum configuration contains 40 circuits: 12 seeded messages, all 16
single-bit flips of message 0, and four samples each of 2-, 3-, and 4-bit flips.
Every circuit used 1,024 shots. Hardware execution used `ibm_fez` on 8 September
2026; transpilation used optimization level 3 and seed 1337.

## Run a new experiment

Set `IBM_CLOUD_API_KEY` in the environment or in an ignored `.env` file. The
runner uses the IBM Quantum Platform `qpus` instance in `us-east`.

```bash
python experiments/ibm/select_backend.py
python experiments/ibm/classical_baselines.py
python experiments/ibm/exp_hashing.py --run-id NEW_RUN_ID
python experiments/ibm/exp_hashing.py --run-id NEW_RUN_ID --submit
```

Without `--submit`, the runner transpiles for the selected QPU, runs and saves
noiseless references, and prints the QPU-time estimate. It still contacts IBM
to obtain the backend and calibration. **`--submit` spends IBM QPU time.**

Use `--algorithms grover,feistel,simple_xor`, `--widths 3,4,8`, and `--shots` to
select a subset. The report and paper-asset commands intentionally target the
archived campaign; new campaign files are inputs for further analysis rather
than an automatic replacement for the published results.

## Provenance

Extracted from the MSc thesis prototype at
`/home/isati/dev/uni/projects/protocol`, Git revision
`2d55cf68e343194f1f057467eadf825c77b2d037`, using the source working tree as
found on 2 November 2026. The paper's original asset generator came from
`/home/isati/university/quantumhashingpaper/scripts/make_assets.py`.

Algorithm implementations and archived results are preserved. Changes for this
repository are limited to standalone paths, focused packaging/documentation,
the offline paper-table report, and reproduction tests. Unrelated token,
BB84, superdense-coding, clock-synchronization, and web-demo code is excluded.
No source license was specified; no new license is asserted here.
