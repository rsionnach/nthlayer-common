# Upgrading to nthlayer-common 3.0.0

Two beads land in this release and both change behaviour that callers depend on.
This file exists because the generated `CHANGELOG.md` cannot hold it: release-please
captures only the FIRST PARAGRAPH of a `BREAKING CHANGE:` footer, so a multi-item
list is truncated on every regeneration. The footer links here instead.

## opensrm-ocvu — SLO targets convert at the inbound boundary

`v1` and `v2` parsed the same SLO to values 100x apart: `availability` came out as
`99.9` from a v1 manifest and `0.999` from a v2 one. Hard rule 1 makes
`SLODefinition.target` a 0-100 percentage for classical *and* judgment SLOs, with the
OpenSLO surface at 0.0-1.0 and conversion **at the boundary**. The outbound boundary
always converted; the inbound one did not.

1. **`SLODefinition.target` VALUES CHANGE on the v2 path.** A v2 OpenSLO objective
   target of `0.999` previously parsed to `0.999` and now parses to `99.9`. Judgment
   targets likewise: `maximum_reversal_rate: 0.05` was `0.05` and is now the `95.0`
   SLI floor. **Any consumer comparing `manifest.slos[].target` against a measured
   value gets a 100x different answer than before.** `nthlayer-workers`' measure
   adapter compares targets directly.
2. **A judgment target outside the Ratio domain `[0, 1]` now raises.** It was
   silently complemented: `maximum_reversal_rate: 5` became `-400.0`, and the
   load-time validator skips targets `<= 0`, so nothing flagged it.
3. **A non-finite target now raises.** `NaN` previously reached
   `SLODefinition.target`, and because every `NaN` comparison is `False`,
   `validate_contracts()` reported a contract breach as **clean**.
4. **A non-numeric target raises `ValueError` instead of `TypeError`.** `TypeError`
   was never a declared type for either parser and escaped callers' `except` clauses
   as a stack trace. Strings are still accepted: `target: "0.999"` has always parsed.
5. **`convert_v1_to_v2` emits judgment targets as RATIOS**, matching the classical
   path and opensrm v2's `Ratio` type. It previously copied the v1 percentage
   verbatim, which was schema-illegal by ~100x and made the documented round trip
   produce `-9750.0` from `98.5`. `nthlayer-generate`'s `migrate_manifest_command`
   writes this document to disk, so **migrated output changes shape.**

Only the three judgment MAXIMA complement (`reversal_rate`,
`high_confidence_failure`, `escalation`). `outcomes` and `audit_sampling` are already
floors and are scaled only. The error magnitudes (`segments`, `stability`,
`calibration`) are left unconverted pending decision 3c (opensrm-l33e).

## opensrm-xvwt — one manifest per stem

6. **`iter_manifest_files()` returns FEWER files for a colliding directory, and
   warns.** A directory holding both `svc.yaml` and `svc.yml` previously yielded
   both. Both are valid manifests, so nothing errored: measure's `load_specs`
   appended a service's SLOs twice and `count_consecutive_breaches` counts verdicts
   rather than windows, so a 3-window hysteresis threshold was reached in 2 real
   windows; observe inflated per-service SLO counts.

   `.yaml` now wins, then the exactly-lowercase spelling over a case variant, then
   the name. Stems are NFC-normalised, since APFS preserves rather than enforces
   normalisation. Each dropped file raises a **`ManifestCollisionWarning`**, which a
   host running `-W error` will turn into an exception.

   Grouping is by STEM, so a collision may be two DIFFERENT services rather than a
   duplicate — `payments.yaml` declaring `payments` beside `payments.yml` declaring
   `payments-api` loads one and drops the other. New `scan_manifest_files()` returns
   `ManifestScan(files, suffix_collisions)` for callers that should surface that to
   an operator; migrating the three `nthlayer-workers` call sites is opensrm-j9wq.

## Upgrade order

Release `nthlayer-common` **before** any consumer regenerates artefacts, per the
ecosystem's shared-repo-first rule. Migrated v1→v2 output changes shape, so anything
that regenerates against 2.x and then loads under 3.x will disagree with itself.
