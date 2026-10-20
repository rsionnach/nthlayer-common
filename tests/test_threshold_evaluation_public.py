import nthlayer_common.manifest as manifest
from nthlayer_common.cloudevents import ASSESSMENT_KINDS, _ce_type_for_assessment

_PUBLIC = ["evaluate_threshold", "ThresholdOutcome", "BoundResult", "SegmentResult",
           "Samples", "Missing", "AtLeast", "THRESHOLD_SERIES", "ThresholdSeries",
           "FEEDBACK_LATENCY_WINDOW"]


def test_evaluation_api_is_public():
    for name in _PUBLIC:
        assert name in manifest.__all__, name
        assert hasattr(manifest, name), name


def test_assessment_kind_is_canonical():
    assert "judgment_threshold_evaluation" in ASSESSMENT_KINDS
    assert _ce_type_for_assessment("judgment_threshold_evaluation") == (
        "io.nthlayer.assessment.judgment_threshold_evaluation.v1"
    )
