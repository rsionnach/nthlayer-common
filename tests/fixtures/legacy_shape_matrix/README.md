# Legacy loader shape matrix: what loaded before

`loaded_before.json` lists the cells of the matrix in
`tests/test_legacy_loader_shape_contract.py` that loaded on commit `ec44215`,
the last one before `load_manifest` stopped leaking `AttributeError`,
`TypeError`, `OverflowError` and `RecursionError` for a malformed legacy
manifest (opensrm-wo62.4.10). On that
commit, of the 8,280 cells, 4,561 loaded, 1,778 raised `ManifestLoadError` or
`ValueError`, and 1,941 crashed: 1,460 with `AttributeError` or `TypeError`,
461 with `RecursionError` and 20 with `OverflowError`.

It was produced by running the matrix against the loader as it was on
`ec44215`, not written from the fix: `git archive ec44215 src` unpacked
into a scratch directory and put first on `PYTHONPATH`, then each cell
loaded and the ones that returned a manifest recorded. So it says what the
old loader accepted, and the test can hold the new one to "nothing that
loaded stops loading, and nothing that failed starts to load unnoticed".

A new position or a new wrong value in the matrix adds cells. Give the new
cells their answer the same way, from `ec44215`. Do not regenerate the file
from the current loader: it would then agree with whatever the loader does.
