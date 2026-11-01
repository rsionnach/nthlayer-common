"""SloSeries: the classical-SLO recording-rule contract (opensrm-h0at).

The pinned name and labels come from what nthlayer-generate writes
(recording_rules/manifest_builder.py, `_build_availability_rules`, and its
recording_rules goldens), not from any reader: measure's reader was the
side that had drifted. The escaping cases come from PromQL's string-literal
rules (Go escapes inside double quotes), not from the implementation.
"""

import dataclasses

import pytest

from nthlayer_common.manifest import ERROR_BUDGET_REMAINING, SloSeries


def test_error_budget_name_is_what_generate_writes():
    assert ERROR_BUDGET_REMAINING.metric == "slo:error_budget_remaining:ratio"


def test_error_budget_labels_are_what_generate_attaches():
    assert ERROR_BUDGET_REMAINING.labels == ("service", "slo", "objective")


def test_selector_with_service_and_slo():
    assert (
        ERROR_BUDGET_REMAINING.selector(service="checkout", slo="checkout-availability")
        == 'slo:error_budget_remaining:ratio{service="checkout",slo="checkout-availability"}'
    )


def test_selector_keeps_caller_order():
    assert ERROR_BUDGET_REMAINING.selector(slo="a", service="b") == (
        'slo:error_budget_remaining:ratio{slo="a",service="b"}'
    )


def test_selector_with_no_matchers_is_the_bare_metric():
    assert ERROR_BUDGET_REMAINING.selector() == "slo:error_budget_remaining:ratio"


def test_selector_refuses_a_label_the_series_does_not_carry():
    # `name` is the SLO's field name, not the label generate writes (`slo`):
    # filtering on it would match nothing, which reads as no data.
    with pytest.raises(ValueError, match="name"):
        ERROR_BUDGET_REMAINING.selector(service="checkout", name="availability")


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        ('a"b', r'"a\"b"'),
        ("a\\b", r'"a\\b"'),
        ("a\nb", r'"a\nb"'),
        ('\\"', r'"\\\""'),
    ],
)
def test_selector_escapes_label_values(value, literal):
    assert ERROR_BUDGET_REMAINING.selector(service=value) == (
        f"slo:error_budget_remaining:ratio{{service={literal}}}"
    )


def test_selector_renders_a_non_str_value_as_generate_writes_it():
    # srm/v1's parser only checks metadata.name is truthy, so `name: 404`
    # loads as an int; generate's rule file dumps the label as YAML `404`.
    assert ERROR_BUDGET_REMAINING.selector(service=404) == (
        'slo:error_budget_remaining:ratio{service="404"}'
    )


def test_selector_renders_objective_as_generate_writes_it():
    # generate: labels={"objective": str(objective)}, objective = slo.target,
    # a float on the parsed model, so 99 is written "99.0".
    target = 99.0
    assert ERROR_BUDGET_REMAINING.selector(objective=target) == (
        f'slo:error_budget_remaining:ratio{{objective="{str(target)}"}}'
    )
    assert ERROR_BUDGET_REMAINING.selector(objective=target).endswith('"99.0"}')


def test_series_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ERROR_BUDGET_REMAINING.metric = "slo:error_budget:ratio"  # type: ignore[misc]


def test_error_budget_is_an_slo_series():
    assert isinstance(ERROR_BUDGET_REMAINING, SloSeries)
