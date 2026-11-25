# load_manifest shape matrix: srm/v1 and v2 (opensrm-wo62.4.28)

What `tests/test_load_manifest_shape_contract.py` mutates, and what loaded
before `load_manifest` stopped leaking `AttributeError`, `TypeError`,
`KeyError` and `OverflowError` on the srm/v1 and v2 paths.

## Documents (snapshots, copied unchanged)

| Here | From | At |
|---|---|---|
| `v2-spec-examples/` | `rsionnach/opensrm`, `spec/v2/examples/`: every `kind: ServiceManifest` file, and `judgment-slos/` | `d713553` |
| `v2-spec-examples/templates/payment-services-base.yaml` | the same repo's `spec/v2/examples/service-manifest-template.yaml`, under the name `template-extension.yaml` extends | `d713553` |
| `v2-schema.json` | `rsionnach/opensrm`, `spec/v2/schema.json` | `d713553` |
| `srm-v1-examples/` | `rsionnach/nthlayer-generate`, `examples/opensrm/` | `2010f6d` |

The one change is the template's location and file name: the spec keeps it
beside the examples, and the parser looks for `templates/<name>.yaml`.

They are documents written for the spec and for generate's users, not for
this parser's tests: a fixture written beside the parser agrees with the
parser, bugs included. The test adds documents of its own only where no
snapshot reaches (it says which, and why, beside each), and for v2 holds the
whole set to the schema: every property `v2-schema.json` declares for a
ServiceManifest is in at least one document.

These are snapshots, so a change to the spec does not reach them. To
refresh, copy the files again, update the commits above, and give any new
cell its answer as described below. The live check that the spec's examples
and schema agree with this parser is `tests/test_manifest_v2_archetypes.py`.

## `loaded_before.json`

Position -> the wrong values (space-separated names from `WRONG_VALUES`)
that loaded at that position on commit `adaa35b`, the last one before the
fix. On that commit, of the 48,737 cells, 15,948 loaded, 26,375 raised
`ManifestLoadError` or `ValueError`, and 6,414 crashed: 4,671 with
`AttributeError`, 1,608 with `TypeError`, 78 with `OverflowError` and 57
with `KeyError`.

It was produced by running the matrix against the parsers as they were on
`adaa35b`, not written from the fix: `git archive adaa35b src` unpacked into
a scratch directory and put first on `PYTHONPATH`, then each cell loaded
(`build_work`, `load_cell` and `classify` from the test module) and the ones
that returned a manifest recorded. So it says what the old parsers accepted,
and the test can hold the new ones to "nothing that loaded stops loading,
and nothing that failed starts to load".

A new document, position or wrong value adds cells. Give the new cells
their answer the same way, from `adaa35b`. Do not regenerate the file from
the current parsers: it would then agree with whatever they do.
