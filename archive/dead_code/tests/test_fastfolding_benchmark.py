from protein_jepa_frame.fastfolding.benchmark import (
    assign_group_splits,
    build_event_samples,
    build_hitting_time_samples,
    build_transition_paths,
    evaluate_predictions,
    persistence_predictions,
    scaffold_id,
)


def _rows():
    rows = []
    sequences = {
        "A3D-0-protein": ["unfolded", "transition", "folded", "folded", "folded"],
        "A3D-1-protein": ["unfolded", "transition", "folded", "folded", "folded"],
        "NTL9-0-protein": ["folded", "transition", "unfolded", "unfolded", "unfolded"],
        "CLN025-0-protein": ["unfolded", "transition", "folded", "folded", "folded"],
    }
    for protein, states in sequences.items():
        for frame, state in enumerate(states):
            rows.append(
                {
                    "protein_id": protein,
                    "trajectory_id": protein,
                    "frame": frame,
                    "state": state,
                    "q_native": {"unfolded": 0.1, "transition": 0.5, "folded": 0.9}[state],
                    "direction": "folding" if frame == 1 else "stationary",
                    "committor_proxy": 1 if state != "folded" and protein != "NTL9-0-protein" else 0,
                }
            )
    return rows


def test_scaffold_split_keeps_replicas_together():
    assert scaffold_id("A3D-1-protein") == "A3D"
    assignments = assign_group_splits(_rows(), seed=4)
    assert assignments["A3D"] == assignments["A3D"]
    assert len(set(assignments.values())) == 3


def test_fastfolding_event_samples_and_persistence_baseline_are_scored():
    rows = _rows()
    samples = build_event_samples(rows, context_frames=2, horizons=[1])
    predictions = persistence_predictions(samples, ["unfolded", "transition", "folded"])
    result = evaluate_predictions(samples, predictions, bootstrap_samples=20)

    assert result["n_predictions"] == len(predictions)
    assert "1" in result["by_horizon"]
    assert "transition_operator" in result


def test_fastfolding_paths_are_extracted_from_observed_sequences():
    paths = build_transition_paths(_rows())
    assert paths
    assert paths[0][0] == "unfolded"
    assert paths[0][-1] == "folded"


def test_hitting_time_samples_support_survival_scoring():
    from protein_jepa_frame.fastfolding.benchmark import evaluate_survival_predictions

    samples = build_hitting_time_samples(_rows(), context_frames=2)
    predictions = [
        {"sample_id": sample["sample_id"], "event_probabilities": [0.2, 0.8]}
        for sample in samples
    ]
    result = evaluate_survival_predictions(samples, predictions, horizons=[1, 2])
    assert result["n_predictions"] == len(samples)
    assert "integrated_brier" in result
