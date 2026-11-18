# Legacy manifests with plural `indicators:` lists (snapshot)

Four legacy-format service files copied unchanged from
`rsionnach/nthlayer-generate`, `examples/services/`, at commit `8230fb6`.

They are the corpus files whose latency SLOs declare their bound only as
`indicators: [{type: latency, threshold_ms: N, ...}]`, which the legacy
loader did not read (opensrm-wo62.4.24). `tests/test_legacy_plural_indicators.py`
loads them. They were written for generate's examples, not for this
loader's tests: a fixture written beside the loader agrees with the loader,
bugs included.

This is a snapshot, so a change to generate's examples does not reach it.
To refresh, copy the files again and update the commit above.
