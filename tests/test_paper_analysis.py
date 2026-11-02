import json
from pathlib import Path

import numpy as np
import pytest

from experiments.ibm.classical_baselines import baseline_table
from experiments.ibm.common import run_ideal
from experiments.ibm.exp_hashing import ITERATIONS, analyse, messages
from quantum_circuits.grover_hash import GroverHash
from quantum_circuits.simple_xor_hash import SimpleXORHash
from scripts.make_assets import aah_probabilities, block_targets, register_marginal

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "ibmq_20260908T220749Z" / "hashing"


@pytest.mark.parametrize("width", (3, 4, 8))
def test_exact_aah_model_matches_qiskit(width):
    message = messages()[0]
    circuit = register_marginal(
        GroverHash(width, ITERATIONS[width]).build_circuit(message), width
    )
    model = aah_probabilities(width, *block_targets(message, width), ITERATIONS[width])
    np.testing.assert_allclose(circuit, model, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("width,iterations", ((3, 2), (4, 3), (8, 2), (8, 12)))
def test_distinct_targets_leave_first_block_target_modal(width, iterations):
    probabilities = aah_probabilities(width, 0, 1, iterations)
    assert int(probabilities.argmax()) == 0


@pytest.mark.parametrize("algorithm,width", (("grover", 3), ("feistel", 4), ("simple_xor", 8)))
def test_archived_analysis_recomputes_from_raw_counts(algorithm, width):
    hardware = json.loads((RUN / f"{algorithm}_{width}bit.json").read_text())
    ideal = json.loads((RUN / f"{algorithm}_{width}bit_ideal.json").read_text())
    recomputed = analyse(hardware["records"], hardware["circuit_index"], width, ideal["records"])
    assert recomputed == hardware["analysis"]


def test_classical_baselines_reproduce_archived_table():
    archived = json.loads((ROOT / "results" / "classical_baselines.json").read_text())
    assert baseline_table() == archived["table"]


@pytest.mark.parametrize("width", (3, 4, 8))
def test_xor_parity_is_exact_for_every_contiguous_group(width):
    message = "0110100110010110"
    hasher = SimpleXORHash(width)
    parity = [message[start:stop].count("1") % 2 for start, stop in (
        (group[0], group[-1] + 1) for group in hasher._groups()
    )]
    expected = "".join(str(bit) for bit in reversed(parity))
    records = run_ideal([hasher.build_circuit(message)], shots=64)
    assert records[0]["counts"] == {expected: 64}



def test_relation_between_fitted_probe_t1_and_reported_mean():
    path = ROOT / "results" / "ibmq_20260908T220749Z" / "qcs" / "t1_relaxation.json"
    relaxation = json.loads(path.read_text())
    analysis = relaxation["analysis"]
    fitted = [row["fit"]["t1_us"] for row in analysis["per_qubit"]]
    assert len(fitted) == analysis["summary"]["n_fitted"] == 16
    assert sum(fitted) / len(fitted) == pytest.approx(analysis["summary"]["mean_t1_us"])