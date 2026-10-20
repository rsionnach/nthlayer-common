from typing import get_args

import nthlayer_common.manifest as manifest
from nthlayer_common.cloudevents import ASSESSMENT_KINDS, _ce_type_for_assessment

_PUBLIC = ["evaluate_threshold", "ThresholdOutcome", "BoundResult", "SegmentResult",
           "Samples", "Missing", "AtLeast", "THRESHOLD_SERIES", "ThresholdSeries",
           "FEEDBACK_LATENCY_WINDOW", "combine",
           "Status", "MissingReason", "ResultReason"]


def test_evaluation_api_is_public():
    for name in _PUBLIC:
        assert name in manifest.__all__, name
        assert hasattr(manifest, name), name


def test_assessment_kind_is_canonical():
    assert "judgment_threshold_evaluation" in ASSESSMENT_KINDS
    assert _ce_type_for_assessment("judgment_threshold_evaluation") == (
        "io.nthlayer.assessment.judgment_threshold_evaluation.v1"
    )


def _literal_values(tp) -> set[str]:
    args = get_args(tp)
    if all(isinstance(a, str) for a in args):
        return set(args)
    return set().union(*(_literal_values(a) for a in args))


def test_reason_types_name_the_documented_reasons():
    # The five result reasons documented in docs/upgrading-5.0.md ("New in
    # 5.2.0") and nthlayer/docs/metrics-contract.md; MissingReason is the
    # subset a sampler may supply.
    assert _literal_values(manifest.MissingReason) == {
        "no_data_source", "no_data", "insufficient_window"}
    assert _literal_values(manifest.ResultReason) == {
        "no_data_source", "no_data", "insufficient_window",
        "lower_bound", "beyond_top_bucket"}
    assert _literal_values(manifest.Status) == {"within", "breach", "not_evaluable"}
