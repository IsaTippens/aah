"""Generate the paper's numerical hardware and baseline tables offline.

    python scripts/make_report.py

Reads the archived JSON campaign and writes ``data/hardware_report.md``. No IBM
credentials or network access are required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_assets import ALGORITHMS, LABEL, WIDTHS, signal_vs_uniform, uniform_tvd  # noqa: E402

RUN_ID = "ibmq_20260908T220749Z"
RUN = ROOT / "results" / RUN_ID
OUT = ROOT / "data" / "hardware_report.md"


def load(algorithm: str, width: int, ideal: bool = False) -> dict:
    suffix = "_ideal" if ideal else ""
    path = RUN / "hashing" / f"{algorithm}_{width}bit{suffix}.json"
    return json.loads(path.read_text())


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def surviving_signal(hardware: dict, ideal: dict, width: int) -> tuple[float, float]:
    ideal_tvd = signal_vs_uniform(ideal, width)
    sigma = signal_vs_uniform(hardware, width) / ideal_tvd
    sigma_0 = uniform_tvd(width) / ideal_tvd
    return sigma, sigma_0


def hardware_table() -> list[str]:
    lines = [
        "## Hardware execution",
        "",
        "| Construction | n | K | Qubits | CZ | Depth | Duration (us) | Hellinger fidelity | Top-1 | sigma | sigma_0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        for width in WIDTHS:
            hardware = load(algorithm, width)
            ideal = load(algorithm, width, ideal=True)
            metrics = hardware["isa_metrics"]
            analysis = hardware["analysis"]["vs_ideal"]
            sigma, sigma_0 = surviving_signal(hardware, ideal, width)
            iterations = hardware["grover_iterations"] or "-"
            lines.append(
                f"| {LABEL[algorithm]} | {width} | {iterations} | "
                f"{hardware['logical_metrics'][0]['num_qubits']} | "
                f"{max(m['two_qubit_gates'] for m in metrics)} | "
                f"{max(m['depth'] for m in metrics)} | "
                f"{hardware['estimate']['max_circuit_duration_s'] * 1e6:.1f} | "
                f"{f(analysis['mean_hellinger_fidelity'])} | "
                f"{f(analysis['top1_agreement_rate'], 2)} | {f(sigma)} | {f(sigma_0)} |"
            )
    return lines


def avalanche_table() -> list[str]:
    lines = [
        "## Modal-digest avalanche",
        "",
        "Normalised Hamming distance; the final column is the hardware single-flip result.",
        "",
        "| Construction | n | 1 flip, noiseless | 2 flips | 3 flips | 4 flips | 1 flip, ibm_fez |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        for width in WIDTHS:
            ideal_curve = {
                row["flips"]: row["top1_normalised"]
                for row in load(algorithm, width, ideal=True)["analysis"]["avalanche"]["per_flip_count"]
            }
            hardware_curve = {
                row["flips"]: row["top1_normalised"]
                for row in load(algorithm, width)["analysis"]["avalanche"]["per_flip_count"]
            }
            lines.append(
                f"| {LABEL[algorithm]} | {width} | "
                + " | ".join(f(ideal_curve[count]) for count in range(1, 5))
                + f" | {f(hardware_curve[1])} |"
            )

    classical = json.loads((ROOT / "results" / "classical_baselines.json").read_text())
    for algorithm, label in (("sha256", "SHA-256"), ("md5", "MD5")):
        for width in WIDTHS:
            curve = {
                row["flips"]: row["normalised"]
                for row in classical["table"][algorithm][str(width)]["avalanche"]["per_flip_count"]
            }
            lines.append(
                f"| {label} | {width} | "
                + " | ".join(f(curve[count]) for count in range(1, 5))
                + " | -- |"
            )
    return lines


def uniformity_table() -> list[str]:
    lines = [
        "## Pooled-shot uniformity",
        "",
        "Chi-square statistics over 12 messages x 1,024 shots.",
        "",
        "| Construction | Source | n = 3 | n = 4 | n = 8 |",
        "|---|---|---:|---:|---:|",
    ]
    critical = [load("grover", width)["analysis"]["uniformity"]["critical_value_95"] for width in WIDTHS]
    lines.append("| Critical chi-square (95%) | -- | " + " | ".join(f(v, 2) for v in critical) + " |")
    for algorithm in ALGORITHMS:
        for ideal in (True, False):
            values = [
                load(algorithm, width, ideal)["analysis"]["uniformity"]["chi_square"]
                for width in WIDTHS
            ]
            source = "noiseless" if ideal else "ibm_fez"
            lines.append(
                f"| {LABEL[algorithm]} | {source} | "
                + " | ".join(f(v, 1) for v in values) + " |"
            )
    return lines


def build_report() -> str:
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    relaxation = json.loads((RUN / "qcs" / "t1_relaxation.json").read_text())
    backend = manifest["backend"]
    lines = [
        f"# Archived AAH hardware results: `{RUN_ID}`",
        "",
        "Generated by `python scripts/make_report.py` from the archived raw counts.",
        "",
        "## Device",
        "",
        f"- Backend: `{backend['name']}` ({backend['num_qubits']} qubits, "
        f"{backend['processor_type']['family']} r{backend['processor_type']['revision']})",
        f"- Calibration: {backend['last_calibration_utc']}",
        f"- Median T1 / T2: {backend['median_t1_us']} / {backend['median_t2_us']} us",
        f"- Mean fitted T1 over 16 probes: {relaxation['analysis']['summary']['mean_t1_us']:.1f} us",
        f"- Median CZ error: {backend['median_2q_error']}",
        f"- Median readout error: {backend['median_readout_error']}",
        "",
    ]
    for section in (hardware_table, avalanche_table, uniformity_table):
        lines.extend(section())
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    report = build_report()
    OUT.write_text(report)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
