# Legacy loader shape matrix: what loaded before

`loaded_before.json` lists the cells of the matrix in
`tests/test_legacy_loader_shape_contract.py` that loaded on commit `ec44215`,
the last one before `load_manifest` stopped leaking `AttributeError` and
`TypeError` for a malformed legacy manifest (opensrm-wo62.4.10). On that
commit 819 of the 1,233 cells loaded, 139 raised `ManifestLoadError` or
`ValueError`, and 275 crashed with `AttributeError` or `TypeError`.

It was produced by running that matrix against the loader as it then was,
not written from the fix: it records what the old loader accepted, so the
test can hold the new one to "nothing that loaded stops loading, and
nothing that failed starts to load unnoticed".

A new position in the matrix's documents adds cells. Decide for each new
cell whether it should load and add the ones that should; do not regenerate
the whole file from the current loader, which would make it agree with
whatever the loader does.
