# Demo specs (snapshot)

The four `srm/v1` manifests the NthLayer demo runs on, copied unchanged from
the front door: `rsionnach/nthlayer`, `demo/specs/`, at commit `23c7ea3`
(the directory last changed in `a50ee7a`, 2026-11-16).

`tests/test_manifest_real_specs.py` parses them. They are manifests written
for the demo, not for this parser's tests, which is the point: a fixture
written beside the parser agrees with the parser, bugs included.

This is a snapshot, so a change to the demo does not reach it. To refresh,
copy the files again and update the commit above. The live check, that the
demo's current specs load through this package, belongs to the front door
(opensrm-t93o).
