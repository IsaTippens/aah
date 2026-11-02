"""Reproduce the exact analysis and figures for the quantum hashing paper.

Run from any directory:

    python scripts/make_assets.py

Outputs are written to ``figures/*.svg`` and ``data/structural.json`` in this
repository. The archived hardware campaign is read from ``results/``.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
RUN = ROOT / "results" / "ibmq_20260908T220749Z"
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Gate  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from experiments.ibm.exp_hashing import ITERATIONS, messages  # noqa: E402
from quantum_circuits.grover_hash import GroverHash, GroverNonLinearHash  # noqa: E402
from quantum_circuits.non_linear_oracle import (  # noqa: E402
    feistel_permutation,
    feistel_permute,
    feistel_round_key,
    recommended_rounds,
)

WIDTHS = (3, 4, 8)
SHOTS = 1024
CZ_ERROR = 0.00285  # median CZ error of ibm_fez at the campaign calibration
GATE_COUNTS = (16, 32, 40, 79, 227, 630, 2625, 3339, 9023)
ALGORITHMS = ("grover", "feistel", "simple_xor")
LABEL = {"grover": "Linear AAH", "feistel": "Feistel AAH", "simple_xor": "XOR parity"}
COLOUR = {"grover": "#4C72B0", "feistel": "#C44E52", "simple_xor": "#55A868"}
# Surviving signal reported in Table IV; recomputed below as a consistency check.
TABLE_SIGMA = {
    ("grover", 3): 0.703, ("grover", 4): 0.227, ("grover", 8): 0.723,
    ("feistel", 3): 0.059, ("feistel", 4): 0.061, ("feistel", 8): 0.725,
    ("simple_xor", 3): 0.742, ("simple_xor", 4): 0.782, ("simple_xor", 8): 0.937,
}

MESSAGES = messages()

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "svg.fonttype": "path",
    "svg.hashsalt": "aah-paper",
})

SVG_METADATA = {"Date": None}


# ------------------------------------------------------------------ exact model
def aah_probabilities(n: int, t1: int, t2: int, K: int) -> np.ndarray:
    """Outcome distribution of (D O_t2)^K (D O_t1)^K |s> on n qubits."""
    size = 1 << n
    psi = np.full(size, 1.0 / np.sqrt(size))
    for t in (t1, t2):
        for _ in range(K):
            psi[t] *= -1.0
            psi = 2.0 * psi.mean() - psi
    return psi ** 2


def block_targets(msg: str, n: int) -> tuple[int, int]:
    return int(msg[:8], 2) % (1 << n), int(msg[8:], 2) % (1 << n)


def register_marginal(qc: QuantumCircuit, n: int) -> np.ndarray:
    """Noiseless distribution of the first n qubits (the last n key characters)."""
    unitary_part = qc.remove_final_measurements(inplace=False)
    probs = np.zeros(1 << n)
    for key, p in Statevector(unitary_part).probabilities_dict().items():
        probs[int(key[-n:], 2)] += p
    return probs


# ------------------------------------------------------------------- analyses
def validate_model() -> dict:
    """The 2^n-dimensional model against Qiskit statevectors of the campaign circuits."""
    out = {}
    for n in WIDTHS:
        K = ITERATIONS[n]
        worst = 0.0
        for msg in MESSAGES:
            circuit = register_marginal(GroverHash(n, K).build_circuit(msg), n)
            model = aah_probabilities(n, *block_targets(msg, n), K)
            assert np.allclose(circuit, model, rtol=0.0, atol=1e-9), (n, msg)
            worst = max(worst, float(np.abs(circuit - model).max()))
        out[str(n)] = {"K": K, "messages": len(MESSAGES), "max_abs_difference": worst}
    return out


def target_statistics(n: int, K: int) -> dict:
    """Modal outcome over every ordered target pair, and the t1 = t2 case."""
    size = 1 << n
    modal = {"t1": 0, "t2": 0, "other": 0, "tie": 0}
    reference = None
    for t1 in range(size):
        for t2 in range(size):
            if t1 == t2:
                continue
            p = aah_probabilities(n, t1, t2, K)
            rest = np.delete(p, [t1, t2])
            assert rest.max() - rest.min() <= 1e-12, (n, K, t1, t2)
            triple = np.array([p[t1], p[t2], rest[0]])
            if reference is None:
                reference = triple
            assert np.allclose(triple, reference, rtol=0.0, atol=1e-12), (n, K, t1, t2)
            winners = np.flatnonzero(np.isclose(p, p.max(), rtol=0.0, atol=1e-12))
            if len(winners) > 1:
                modal["tie"] += 1
            elif winners[0] == t1:
                modal["t1"] += 1
            elif winners[0] == t2:
                modal["t2"] += 1
            else:
                modal["other"] += 1

    same = None
    for t in range(size):
        p = aah_probabilities(n, t, t, K)
        p_max = p.max()
        record = (p[t], p_max, int(np.isclose(p, p_max, rtol=0.0, atol=1e-12).sum()))
        if same is None:
            same = record
        assert abs(record[0] - same[0]) <= 1e-12 and abs(record[1] - same[1]) <= 1e-12
        assert record[2] == same[2], (n, K, t)
    return {
        "n": n,
        "K": K,
        "pairs": size * (size - 1),
        "P_t1": float(reference[0]),
        "P_t2": float(reference[1]),
        "P_other": float(reference[2]),
        "modal_counts": modal,
        "same_target": {
            "P_target": float(same[0]),
            "P_max": float(same[1]),
            "states_at_P_max": same[2],
        },
    }


def feistel_equivalence(n: int) -> dict:
    """P_F(y) = P_L(f(y)) and f(h_F) = h_L on the campaign messages."""
    rounds = recommended_rounds(n)
    K = ITERATIONS[n]
    f = np.array([feistel_permute(y, n, rounds) for y in range(1 << n)])
    worst, agree = 0.0, 0
    for msg in MESSAGES:
        p_feistel = register_marginal(GroverNonLinearHash(n, K).build_circuit(msg), n)
        p_linear = register_marginal(GroverHash(n, K).build_circuit(msg), n)
        worst = max(worst, float(np.abs(p_feistel - p_linear[f]).max()))
        agree += int(f[p_feistel.argmax()] == p_linear.argmax())
    return {"rounds": rounds, "K": K, "max_abs_deviation": worst,
            "modal_agreement": agree, "messages": len(MESSAGES)}


def chi2_floor(n: int, n_messages: int = 12, shots: int = SHOTS) -> float:
    """Smallest pooled-shot chi-square a deterministic function can reach."""
    buckets = 1 << n
    q, r = divmod(n_messages, buckets)
    observed = np.array([q + 1] * r + [q] * (buckets - r), dtype=float) * shots
    expected = n_messages * shots / buckets
    return float(((observed - expected) ** 2 / expected).sum())


def signal_vs_uniform(data: dict, n: int) -> float:
    """Mean TVD between each message circuit's output and the flat distribution.

    Same computation as make_report.signal_vs_uniform in the protocol repository.
    """
    buckets = 1 << n
    uniform = {format(v, f"0{n}b"): 1.0 / buckets for v in range(buckets)}
    scores = []
    for row in data["analysis"]["per_circuit"]:
        if row["kind"] != "message":
            continue
        total = sum(row["counts"].values())
        dist = {k: v / total for k, v in row["counts"].items()}
        keys = set(dist) | set(uniform)
        scores.append(0.5 * sum(abs(dist.get(k, 0.0) - uniform.get(k, 0.0)) for k in keys))
    return sum(scores) / len(scores)


def uniform_tvd(n: int, draws: int = 2000, shots: int = SHOTS) -> float:
    """Mean TVD from uniform of a structureless output sampled with `shots` shots."""
    buckets = 1 << n
    rng = np.random.default_rng(20260908)
    counts = rng.multinomial(shots, np.full(buckets, 1.0 / buckets), size=draws)
    return float((0.5 * np.abs(counts / shots - 1.0 / buckets).sum(axis=1)).mean())


def load_run(algorithm: str, n: int) -> tuple[dict, dict]:
    hashing = RUN / "hashing"
    hardware = json.loads((hashing / f"{algorithm}_{n}bit.json").read_text())
    noiseless = json.loads((hashing / f"{algorithm}_{n}bit_ideal.json").read_text())
    assert hardware["messages"] == MESSAGES, (algorithm, n)
    return hardware, noiseless


def surviving_floor() -> dict:
    out = {}
    for algorithm in ALGORITHMS:
        for n in WIDTHS:
            hardware, noiseless = load_run(algorithm, n)
            noiseless_tvd = signal_vs_uniform(noiseless, n)
            sigma = signal_vs_uniform(hardware, n) / noiseless_tvd
            assert abs(sigma - TABLE_SIGMA[(algorithm, n)]) <= 0.0005, (algorithm, n, sigma)
            floor_tvd = uniform_tvd(n)
            out[f"{algorithm}_{n}"] = {
                "sigma": sigma,
                "noiseless_tvd": noiseless_tvd,
                "uniform_tvd": floor_tvd,
                "sigma_0": floor_tvd / noiseless_tvd,
            }
    return out


def gate_budget() -> dict:
    return {str(g): float(np.exp(-CZ_ERROR * g)) for g in GATE_COUNTS}


def sweep() -> dict:
    rows = []
    for K in range(1, 21):
        p = aah_probabilities(8, 0, 1, K)
        rows.append({"K": K, "P_t1": float(p[0]), "P_t2": float(p[1])})
    return {"n": 8, "t1": 0, "t2": 1, "rows": rows}


# -------------------------------------------------------------------- figures
def fig_aah_schematic() -> None:
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])
    qc.append(Gate("G1", 3, [], label=r"$(D\,O_{t_1})^K$"), [0, 1, 2])
    qc.append(Gate("G2", 3, [], label=r"$(D\,O_{t_2})^K$"), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    fig = qc.draw(output="mpl", fold=-1, style={"fontsize": 18, "subfontsize": 14})
    # The drawer numbers the wires inside multi-qubit boxes and shifts the label
    # right to make room, but the mathtext label is wider than the drawer's width
    # estimate and runs into the numbers. Both blocks act on the whole register,
    # so the numbers carry no information: drop them and centre the labels.
    ax = fig.axes[0]
    blocks = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 1.0]
    assert len(blocks) == 2, len(blocks)
    for text in list(ax.texts):
        x, y = text.get_position()
        for box in blocks:
            if (box.get_x() <= x <= box.get_x() + box.get_width()
                    and box.get_y() <= y <= box.get_y() + box.get_height()):
                if text.get_text().isdigit():
                    text.remove()
                else:
                    text.set_x(box.get_x() + 0.5 * box.get_width())
                break
    fig.savefig(FIGURES / "aah_schematic.svg", bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)


def fig_feistel_round() -> None:
    # The caption states that round 0 has no round-key X gates at n = 4 and
    # counts 6 CNOT and 4 Toffoli gates.
    assert feistel_round_key(0, 2) == 0
    qc = feistel_permutation(4, rounds=1)
    assert dict(qc.count_ops()) == {"cx": 6, "ccx": 4}, qc.count_ops()
    fig = qc.draw(output="mpl", fold=-1, style={"fontsize": 22, "subfontsize": 16})
    fig.savefig(FIGURES / "feistel_round.svg", bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)


def fig_iteration_sweep(data: dict) -> None:
    K = [row["K"] for row in data["rows"]]
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.plot(K, [row["P_t1"] for row in data["rows"]], "-o", color=COLOUR["grover"],
            markersize=3, linewidth=1.2, label="$P(t_1)$")
    ax.plot(K, [row["P_t2"] for row in data["rows"]], "--", color=COLOUR["grover"],
            linewidth=1.2, label="$P(t_2)$")
    ax.axvline(2, color="grey", linestyle=":", linewidth=1.0, label="campaign, $K=2$")
    ax.axvline(12, color="black", linestyle=":", linewidth=1.0, label=r"$K_{\mathrm{opt}}=12$")
    ax.set_xlabel("Iterations per block, $K$")
    ax.set_ylabel("Probability")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(2, 21, 2))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    # Above the axes: inside, every placement crosses one of the curves.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.savefig(FIGURES / "iteration_sweep.svg", bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)


def fig_hardware_determinism() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for algorithm in ALGORITHMS:
        for n in WIDTHS:
            hardware, _ = load_run(algorithm, n)
            cz = max(m["two_qubit_gates"] for m in hardware["isa_metrics"])
            vs = hardware["analysis"]["vs_ideal"]
            top1, fidelity = vs["top1_agreement_rate"], vs["mean_hellinger_fidelity"]
            colour = COLOUR[algorithm]
            ax.plot(cz, top1, "o", color=colour, markersize=4.5)
            ax.plot(cz, fidelity, "s", markerfacecolor="none", markeredgecolor=colour,
                    markersize=4.5, markeredgewidth=1.0)
            # The XOR points at 16, 32 and 40 gates sit too close for labels on the
            # right, so their labels go above the points, the 4- and 3-bit ones
            # pushed apart so that they do not read as one number.
            offset = {("simple_xor", 3): (1, 4), ("simple_xor", 4): (-6, 4),
                      ("simple_xor", 8): (-2, 4)}.get((algorithm, n), (4, 3))
            ax.annotate(str(n), (cz, top1), xytext=offset, textcoords="offset points",
                        fontsize=7)
    gates = np.logspace(1, np.log10(2e4), 300)
    ax.plot(gates, np.exp(-CZ_ERROR * gates), "--", color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(10, 2e4)
    ax.set_ylim(-0.03, 1.1)  # headroom for the width labels above the points at 1.00
    ax.set_xlabel("Two-qubit (CZ) gates after transpilation")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)
    handles = [Patch(facecolor=COLOUR[a], label=LABEL[a]) for a in ALGORITHMS]
    handles += [
        Line2D([], [], marker="o", linestyle="none", color="dimgrey", markersize=4.5,
               label="top-1 agreement (filled)"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor="none",
               markeredgecolor="dimgrey", markersize=4.5, label="Hellinger fidelity (hollow)"),
        Line2D([], [], linestyle="--", color="black", linewidth=1.0,
               label=r"$e^{-\varepsilon G}$"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False,
              handlelength=1.5, labelspacing=0.3, borderaxespad=0.3)
    fig.savefig(FIGURES / "hardware_determinism.svg", bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    results = {
        "model_validation": validate_model(),
        "target_statistics": {f"{n},{K}": target_statistics(n, K)
                              for n, K in ((3, 2), (4, 3), (8, 2), (8, 12))},
        "feistel_equivalence": {str(n): feistel_equivalence(n) for n in WIDTHS},
        "chi2_floor": {str(n): chi2_floor(n) for n in WIDTHS},
        "surviving_floor": surviving_floor(),
        "gate_budget": gate_budget(),
        "sweep_n8": sweep(),
    }
    text = json.dumps(results, indent=2)
    (DATA / "structural.json").write_text(text + "\n")
    print(text)

    fig_aah_schematic()
    fig_feistel_round()
    fig_iteration_sweep(results["sweep_n8"])
    fig_hardware_determinism()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
