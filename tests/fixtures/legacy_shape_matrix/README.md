# Legacy loader shape matrix: what loaded before

`loaded_before.json` lists the cells of the matrix in
`tests/test_legacy_loader_shape_contract.py` that loaded on commit `ec44215`,
the last one before `load_manifest` stopped leaking `AttributeError`,
`TypeError`, `OverflowError` and `RecursionError` for a malformed legacy
manifest (opensrm-wo62.4.10). On that
commit, of the 9,426 cells, 4,794 loaded, 2,564 raised `ManifestLoadError` or
`ValueError`, and 2,068 crashed: 1,583 with `AttributeError` or `TypeError`,
465 with `RecursionError` and 20 with `OverflowError`.

It was produced by running the matrix against the loader as it was on
`ec44215`, not written from the fix: `git archive ec44215 src` unpacked
into a scratch directory and put first on `PYTHONPATH`, then each cell
loaded and the ones that returned a manifest recorded. So it says what the
old loader accepted, and the test can hold the new one to "nothing that
loaded stops loading, and nothing that failed starts to load unnoticed".

A new position or a new wrong value in the matrix adds cells. Give the new
cells their answer the same way, from `ec44215`. Do not regenerate the file
from the current loader: it would then agree with whatever the loader does.

One departure is deliberate and lives in the test, not in this file: since
opensrm-wo62.4.24 the loader reads `indicators[].threshold_ms`, so 36 cells
that put a value that is no threshold there loaded on `ec44215` (the key was
not read) and are now refused (`REFUSED_SINCE_WO62_4_24`). The baseline
still records what `ec44215` did.
