# Changelog — nthlayer-common

`nthlayer-common` is the shared utility library for the NthLayer ecosystem.
Imported by `nthlayer-core`, `nthlayer-workers`, and `nthlayer-bench`. License:
Apache-2.0.

## [5.6.0](https://github.com/rsionnach/nthlayer-common/compare/v5.5.0...v5.6.0) (2026-10-27)


### Features

* legacy loader applies env overlays, expands templates, substitutes ${var} (opensrm-wo62.4.1) ([b50bad8](https://github.com/rsionnach/nthlayer-common/commit/b50bad8da4c2372257470d9a076d4ae0c15fd8f4))
* legacy loader applies env overlays, expands templates, substitutes ${var} (opensrm-wo62.4.1) ([a2c8b22](https://github.com/rsionnach/nthlayer-common/commit/a2c8b226feebae2497da469cc4f38daf02fc581d))

## [5.5.0](https://github.com/rsionnach/nthlayer-common/compare/v5.4.0...v5.5.0) (2026-10-27)


### Features

* **manifest:** legacy manifests parse judgment SLOs and thresholds like srm/v1 (opensrm-wo62.1) ([3a8b960](https://github.com/rsionnach/nthlayer-common/commit/3a8b960207572376350e646a1977db512605edde))


### Bug Fixes

* **manifest:** legacy judgment thresholds ignore indicator with a warning (opensrm-wo62.1) ([9813b34](https://github.com/rsionnach/nthlayer-common/commit/9813b3477c922c23ead76b8dfc5d8d379b3eb2d6))
* **manifest:** legacy judgment thresholds require exactly one objective (opensrm-wo62.1) ([89dfba5](https://github.com/rsionnach/nthlayer-common/commit/89dfba5f39b380e61c45b98969112181d9984f39))
* **manifest:** legacy parser lifts calibration/feedback_latency into thresholds (opensrm-wo62.1) ([a8d1b17](https://github.com/rsionnach/nthlayer-common/commit/a8d1b170447dcc40b88fbbeffa1fc1cbb132b5aa))
* **manifest:** legacy parser sets judgment_type like srm/v1 (opensrm-wo62.1) ([95db69c](https://github.com/rsionnach/nthlayer-common/commit/95db69c00e2cf6b091853efb6b86655124e1adc9))

## [5.4.0](https://github.com/rsionnach/nthlayer-common/compare/v5.3.1...v5.4.0) (2026-10-24)


### Features

* **manifest:** ReliabilityManifest.authored_type keeps the authored service type (opensrm-wo62.1) ([9830cac](https://github.com/rsionnach/nthlayer-common/commit/9830cac6e24148e718e88eff2330c8d3f1d79e91))


### Documentation

* bare nthlayer belongs to nthlayer-generate from core 4.0.0 (opensrm-dww5) ([ee69714](https://github.com/rsionnach/nthlayer-common/commit/ee69714fc57f9319405c35b630adc126eb74543a))
* **manifest:** document ReliabilityManifest.authored_type (opensrm-wo62.1) ([2b24161](https://github.com/rsionnach/nthlayer-common/commit/2b2416186e26e9d26acb87af8866556296663e97))

## [5.3.1](https://github.com/rsionnach/nthlayer-common/compare/v5.3.0...v5.3.1) (2026-10-23)


### Bug Fixes

* **manifest:** opensrm/v1 rejection names nthlayer-generate migrate-manifest (opensrm-dww5) ([aeebabc](https://github.com/rsionnach/nthlayer-common/commit/aeebabc9b9817d7f0e6974c2a0a887f223cddc7c))

## [5.3.0](https://github.com/rsionnach/nthlayer-common/compare/v5.2.0...v5.3.0) (2026-10-22)


### Features

* **manifest:** converted v1 dependencies keep their type (opensrm-pnmv) ([54aeaef](https://github.com/rsionnach/nthlayer-common/commit/54aeaef67f48f8bd78fee3a2329b2bfacb59868d))
* **manifest:** directory scans surface opensrm/v1 files instead of skipping them (opensrm-pnmv) ([f7ab272](https://github.com/rsionnach/nthlayer-common/commit/f7ab2720575a476e007d69985afa5ece28b9acb9))
* **manifest:** distinguish dropped from assumed migration fields (opensrm-pnmv) ([69b91cd](https://github.com/rsionnach/nthlayer-common/commit/69b91cdc84baa589f4685d17e3eae3398b5f47ba))
* **manifest:** key inline v1 sections by schema path; max_rps follows the v1 minimum (opensrm-pnmv) ([9617215](https://github.com/rsionnach/nthlayer-common/commit/9617215e9e52de861e2e57f63a04c36704e9b21e))
* **manifest:** load_manifest rejects opensrm/v1 and names the migrate command (opensrm-pnmv) ([243a87f](https://github.com/rsionnach/nthlayer-common/commit/243a87f27dc49ad3ad6362406a7938758c093332))
* **manifest:** opensrm/v1 classical SLOs become OpenSLO documents (opensrm-pnmv) ([f9f434a](https://github.com/rsionnach/nthlayer-common/commit/f9f434a27c459d779bdd01829deb76c1b595fd4a))
* **manifest:** opensrm/v1 contract, dependencies, ownership and instrumentation (opensrm-pnmv) ([d02ac89](https://github.com/rsionnach/nthlayer-common/commit/d02ac89d4ce571e5ad1b3454633a937fb0108b7d))
* **manifest:** opensrm/v1 converter accepts every v1 tier and type-checks metadata (opensrm-pnmv) ([98121d4](https://github.com/rsionnach/nthlayer-common/commit/98121d443da212bca8f2b3aa2bfae6df62d8cd31))
* **manifest:** opensrm/v1 converter envelope, metadata and owner (opensrm-pnmv) ([d63ad99](https://github.com/rsionnach/nthlayer-common/commit/d63ad99d8d28ed9b0bdb915aa93b45e5c1b5d83e))
* **manifest:** opensrm/v1 judgment block becomes judgment thresholds (opensrm-pnmv) ([02a8dd9](https://github.com/rsionnach/nthlayer-common/commit/02a8dd9a4fcc1560da5d7b28cb69ca2d41413302))
* **manifest:** opensrm/v1 openslo references never drop input silently (opensrm-pnmv) ([8d6ce61](https://github.com/rsionnach/nthlayer-common/commit/8d6ce61e27b59bc823e67fff4285aa28e573ac35))
* **manifest:** report unknown keys in v1 pagerduty and expects by schema section (opensrm-pnmv) ([00d0251](https://github.com/rsionnach/nthlayer-common/commit/00d0251f4d9cf8299af8072918cb0509122cc5d3))
* **manifest:** v2 dependencies accept resource: entity refs (opensrm-pnmv) ([e9df822](https://github.com/rsionnach/nthlayer-common/commit/e9df822f07ceca9ac26dafe32ab4e2a7858d5035))


### Code Refactoring

* **manifest:** one key-classification path and one section table in opensrm_v1 (opensrm-pnmv) ([3fac559](https://github.com/rsionnach/nthlayer-common/commit/3fac5593fadae3d316e3e4638d10c39970c150e3))
* **manifest:** opensrm/v1 conversion fails loudly if a section walk is abandoned (opensrm-pnmv) ([b695338](https://github.com/rsionnach/nthlayer-common/commit/b695338d036f8861393694bc82716ee8acbe8600))
* **manifest:** opensrm/v1 reports unmapped fields in document order (opensrm-pnmv) ([871c92b](https://github.com/rsionnach/nthlayer-common/commit/871c92bc8388e047489d5edb2480cf4fbe00e89f))


### Documentation

* **manifest:** opensrm/v1 notes say what the code does (opensrm-pnmv) ([c3e9b53](https://github.com/rsionnach/nthlayer-common/commit/c3e9b5384b09923267026e8c4d1f5b5190753f3e))

## [5.2.0](https://github.com/rsionnach/nthlayer-common/compare/v5.1.0...v5.2.0) (2026-10-20)


### Features

* **manifest:** accept Decimal samples and reject malformed outcome payloads (opensrm-g32d) ([232e5cb](https://github.com/rsionnach/nthlayer-common/commit/232e5cbc931f3475e2acb0d72281c9e31c0f7731))
* **manifest:** evaluate segments thresholds per segment (opensrm-g32d) ([a998cba](https://github.com/rsionnach/nthlayer-common/commit/a998cba8ec09fe8c6f79cffaeebc16bbaf687ea8))
* **manifest:** evaluate_threshold E1-E4 for bound-valued thresholds (opensrm-g32d) ([9475ab6](https://github.com/rsionnach/nthlayer-common/commit/9475ab6835612dfec1c8c028b0a23f842fdbd7bb))
* **manifest:** harden threshold evaluation inputs and payload reading (opensrm-g32d) ([e2df72e](https://github.com/rsionnach/nthlayer-common/commit/e2df72e2d0e01428a74d952b67db3ee469c94cbe))
* **manifest:** make threshold outcome reasons forward-compatible and name the capped-histogram case (opensrm-g32d) ([dc8c0fe](https://github.com/rsionnach/nthlayer-common/commit/dc8c0fee76c911e10354128fe09ea58911f55c4c))
* **manifest:** public threshold evaluation API and judgment_threshold_evaluation assessment kind (opensrm-g32d) ([a0590c2](https://github.com/rsionnach/nthlayer-common/commit/a0590c230e477f797644196040aaac797521ff21))
* **manifest:** segment variance equal to its bound is within; bool samples are not data (opensrm-g32d) ([4e0cc10](https://github.com/rsionnach/nthlayer-common/commit/4e0cc10351634a4fe3073a87fed93eb0868db9b2))
* **manifest:** threshold outcome model and assessment payload (opensrm-g32d) ([0ebc8e5](https://github.com/rsionnach/nthlayer-common/commit/0ebc8e5e705bf237852752396ecb93b6675240d0))
* **manifest:** THRESHOLD_SERIES, the judgment-threshold metric contract (opensrm-g32d) ([48d1c77](https://github.com/rsionnach/nthlayer-common/commit/48d1c77f19148b7e4d8463c5e69c9803e7f83eb4))
* **manifest:** type outcome reasons as open strings for readers (opensrm-g32d) ([0189d25](https://github.com/rsionnach/nthlayer-common/commit/0189d2529298a9a8aa11f1b9ebd7f88eacfb0c3a))


### Documentation

* **manifest:** document threshold evaluation reasons and lower bounds (opensrm-g32d) ([8193ed5](https://github.com/rsionnach/nthlayer-common/commit/8193ed5264d6ac540e85892ec032d7258b4ac0d2))
* threshold evaluation is public API (opensrm-g32d) ([470f54d](https://github.com/rsionnach/nthlayer-common/commit/470f54d1d41f8b293b5b12f54b78f4adc0a958d1))

## [5.1.0](https://github.com/rsionnach/nthlayer-common/compare/v5.0.0...v5.1.0) (2026-10-12)


### Features

* **manifest:** threshold_from_v2_entry, the public inverse of threshold_to_v2_entry (opensrm-908x) ([02faf08](https://github.com/rsionnach/nthlayer-common/commit/02faf087077e2478c567f61dc71b1d2abb3e953a))


### Documentation

* **manifest:** say what to pass as where; file threshold_from_v2_entry as new in 5.1 (opensrm-908x) ([ca97b5f](https://github.com/rsionnach/nthlayer-common/commit/ca97b5f0349ab0c50f62650a1b8011532d5f8fcd))
* **manifest:** threshold_from_v2_entry checks per-entry rules only; catch both error types ([9b2af1b](https://github.com/rsionnach/nthlayer-common/commit/9b2af1b296afed75a6463b2a455fe51d23eb234a))

## [5.0.0](https://github.com/rsionnach/nthlayer-common/compare/v4.0.0...v5.0.0) (2026-10-12)


### ⚠ BREAKING CHANGES

* **manifest:** srm/v1 manifests whose calibration or feedback_latency entry carries a key outside target/window/unit/description/labels, a non-30d feedback_latency window, a non-Duration unit, or a unit on calibration are now rejected; a feedback_latency `unit` other than s now changes the lifted bound instead of being ignored. See docs/upgrading-5.0.md.
* **manifest:** manifests that 4.x accepted are now rejected: a segments threshold without `metric`, and any threshold entry carrying a key its type does not define (previously ignored). See docs/upgrading-5.0.md.

### Features

* **manifest:** close judgment_threshold entries; segments name a metric (opensrm-7zpz) ([a503f40](https://github.com/rsionnach/nthlayer-common/commit/a503f4086fbe8047339f3c4381db233a04932bd6))
* **manifest:** export threshold_to_v2_entry for consumers (opensrm-7zpz) ([8656776](https://github.com/rsionnach/nthlayer-common/commit/865677643b775aa6793738c1298740f8844cc84b))
* **manifest:** OutcomeSignal and segments metric fields on ThresholdMeasurement (opensrm-7zpz) ([2759377](https://github.com/rsionnach/nthlayer-common/commit/2759377cee8cb7845713bc823d15bc36b924ff2b))
* **manifest:** srm/v1 feedback_latency targets are capped at 2**53 (opensrm-1rnr) ([fd85791](https://github.com/rsionnach/nthlayer-common/commit/fd857915ca2622b2c4c97716421debbaec598d97))
* **manifest:** a threshold error names the document that declares it (template, manifest or overrides) (opensrm-7zpz) ([c3fe0b5](https://github.com/rsionnach/nthlayer-common/commit/c3fe0b5b4e648da29659d1ac29277fc7cf2a1c87))
* **manifest:** threshold_to_v2_entry validates what it emits (opensrm-7zpz) ([1fbd09d](https://github.com/rsionnach/nthlayer-common/commit/1fbd09dc9f52ebbb1909012dcb176369b85a70b8))


### Bug Fixes

* **manifest:** srm/v1 threshold lift rejects keys it would drop; honours unit (opensrm-1rnr) ([394a7c3](https://github.com/rsionnach/nthlayer-common/commit/394a7c3018cb8146b95cf0320db68f9878697a01))


### Documentation

* threshold_to_v2_entry raises ValueError on an unparseable threshold ([2197104](https://github.com/rsionnach/nthlayer-common/commit/2197104aa1aa272f0031519d5eb9f94e4d6e518d))
* unit is accepted on srm/v1 feedback_latency only ([52e4819](https://github.com/rsionnach/nthlayer-common/commit/52e4819ce6f86757bc0dda1b7900290e5a4138b7))
* upgrading to 5.0 — closed threshold entries, segments metric, srm/v1 lift (opensrm-7zpz, opensrm-1rnr) ([5fb9482](https://github.com/rsionnach/nthlayer-common/commit/5fb94826a631db104e8aacf437e374c5689dc14b))

## [4.0.0](https://github.com/rsionnach/nthlayer-common/compare/v3.0.0...v4.0.0) (2026-10-11)


### ⚠ BREAKING CHANGES

* **manifest:** JUDGMENT_SLO_TYPES is removed. Use V2_JUDGMENT_SLO_TYPES for the five judgment SLO types and V2_JUDGMENT_THRESHOLD_TYPES for the five threshold types. segments, stability and calibration are no longer SLODefinitions; read ReliabilityManifest.judgment_thresholds. Contract promises on threshold or unknown judgment types are parse errors. See docs/upgrading-4.0.md.

### Features

* **manifest:** add judgment threshold bound validation ([45b31e9](https://github.com/rsionnach/nthlayer-common/commit/45b31e919666d4cb6c71da9b658dfcf3a9ee15d5))
* **manifest:** give judgment thresholds their own measurement type ([d58a462](https://github.com/rsionnach/nthlayer-common/commit/d58a462c27dcb58619dcdc24f2b1c6957fc51863))
* **manifest:** migrate srm/v1 thresholds to v2 judgment_threshold ([d1ac419](https://github.com/rsionnach/nthlayer-common/commit/d1ac419d3794d39ea7eeb6c7c56b9fdd4de93463))
* **manifest:** model judgment thresholds apart from SLOs ([4e1dc3b](https://github.com/rsionnach/nthlayer-common/commit/4e1dc3b96a8c84fa40a8f2f7a913d17d6e23032b))
* **manifest:** parse v2 judgment_threshold blocks ([19d1c5a](https://github.com/rsionnach/nthlayer-common/commit/19d1c5a03823d26d267e71db38dbfbace2a3e46f))
* **manifest:** read srm/v1 calibration and feedback_latency as thresholds ([66761ed](https://github.com/rsionnach/nthlayer-common/commit/66761edafe464d5c8885ce34590dea99e8558418))
* **manifest:** remove JUDGMENT_SLO_TYPES for the 3c vocabularies ([678ce6d](https://github.com/rsionnach/nthlayer-common/commit/678ce6df3d14dc4ce4f4380ec0a64bb6d44e5241))


### Notes

* **manifest:** the judgment-threshold code above was hardened during review before
  release: schema/parser agreement on malformed input, parse errors in place of raw
  Python exceptions, a loadable fix-it entry for pre-3c manifests, exact millisecond
  durations, and srm/v1 keys the lift cannot carry rejected rather than dropped. These
  fix code that is new in 4.0.0; **no 3.x behaviour is affected.** See
  [docs/upgrading-4.0.md](docs/upgrading-4.0.md).


### Documentation

* **manifest:** say unset threshold measurement fields keep their defaults ([172ac23](https://github.com/rsionnach/nthlayer-common/commit/172ac232a63411d352a696b0719ae388f6bb6ce4))
* **manifest:** separate evaluation guidance and flag the measurement type change ([f0cc2a4](https://github.com/rsionnach/nthlayer-common/commit/f0cc2a4dc61176b3a42923e07c29c5f1fd9149df))
* **manifest:** show what JudgmentThreshold holds and what consumers change ([1fcdc11](https://github.com/rsionnach/nthlayer-common/commit/1fcdc11f6b4ec3263d59e5197887bacbf5348b67))
* **manifest:** state the JudgmentThreshold type contract and invariant owner ([e8b3940](https://github.com/rsionnach/nthlayer-common/commit/e8b39409d6a74503b69f635a002397e5acdec365))

## [3.0.0](https://github.com/rsionnach/nthlayer-common/compare/v2.1.2...v3.0.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* iter_manifest_files() returns FEWER paths than before for a directory holding two manifests with the same stem, and raises a new ManifestCollisionWarning for each file it sets aside — which a host running `-W error` turns into an exception. `.yaml` wins, then the exactly-lowercase spelling over a case variant, then the name; stems are NFC-normalised. Grouping is by STEM, so a collision may be two DIFFERENT services rather than a duplicate. Callers that should surface a set-aside file to an operator want the new scan_manifest_files(), which returns ManifestScan(files, suffix_collisions). THE FULL LIST OF BREAKING CHANGES IN 3.0.0, for this bead and for opensrm-ocvu, is in docs/upgrading-3.0.md.
* nthlayer-common now converts SLO targets at the INBOUND manifest boundary, so parsed values and accepted inputs both change — five behaviour changes, the first of which is that **`SLODefinition.target` values change on the v2 path** (a v2 objective target of `0.999` previously parsed to `0.999` and now parses to `99.9`), so any consumer comparing a target against a measured value gets a 100x different answer than before. All five are listed in [docs/upgrading-3.0.md](docs/upgrading-3.0.md) (opensrm-ocvu).

### Features

* iter_manifest_files returns one manifest per stem (opensrm-xvwt) ([99e45a9](https://github.com/rsionnach/nthlayer-common/commit/99e45a99d5271d9e5e7d90c3e6afa2a1afa6653e))
* SLO targets are 0-100 at the inbound boundary too (opensrm-ocvu) ([45ad7bc](https://github.com/rsionnach/nthlayer-common/commit/45ad7bc255ef3a426f6dd10574d7f39f95bd8c23))


### Bug Fixes

* a same-stem .yaml/.yml pair loads a service once (opensrm-xvwt) ([33e6490](https://github.com/rsionnach/nthlayer-common/commit/33e6490e2956509851530c3da1576faaa532ebe9))
* convert judgment targets outbound too, and un-skip the demo-spec test (opensrm-ocvu) ([2adbeaf](https://github.com/rsionnach/nthlayer-common/commit/2adbeaf0ea1528fb26e18a26a7db5fda13e486c7))
* convert SLO targets at the inbound boundary (opensrm-ocvu sections 1-2) ([4d6c6e6](https://github.com/rsionnach/nthlayer-common/commit/4d6c6e6ec116d46c8ac00b6b8dc6b227f1ae495a))
* derive judgment promise direction from the field, in one place (opensrm-ocvu) ([a1fca7b](https://github.com/rsionnach/nthlayer-common/commit/a1fca7b6fe6aed5170bb13990a596aab94452336))
* guard the classical target boundary the same way (opensrm-ocvu) ([69a1a75](https://github.com/rsionnach/nthlayer-common/commit/69a1a75fb0f7cdb0ee18384e6b172339d52eea68))
* guard the last unguarded writer of SLODefinition.target (opensrm-ocvu) ([28e0fc0](https://github.com/rsionnach/nthlayer-common/commit/28e0fc041ce142125abe2a17b21447d2785dc6ca))
* log a dropped manifest instead of discarding it silently (opensrm-xvwt) ([aae27d2](https://github.com/rsionnach/nthlayer-common/commit/aae27d2a6a029e9b746205a745f1120d97f1b38b))
* one stem per service, and stop a library writing to stdout (opensrm-xvwt) ([11b3791](https://github.com/rsionnach/nthlayer-common/commit/11b37914d6fc2a8b0680a163fc7d5f76202d9930))
* validate the judgment target domain at the conversion boundary (opensrm-ocvu) ([2050ec1](https://github.com/rsionnach/nthlayer-common/commit/2050ec110d2a7c3fd264d0d56ed5787842c34666))


### Code Refactoring

* drop a redundant float() cast (opensrm-ocvu) ([30a5340](https://github.com/rsionnach/nthlayer-common/commit/30a5340a131bf1a34afbf584ae1f2454b54156d7))
* one factory for judgment promises, one place for the rationale (opensrm-ocvu) ([5f9f2c1](https://github.com/rsionnach/nthlayer-common/commit/5f9f2c1be13d8b1b362afda86ef61333525a588a))
* one home for each argument, and correct the scan catalogue (opensrm-xvwt) ([781cf11](https://github.com/rsionnach/nthlayer-common/commit/781cf11f17e265b633c930d98bdf440ed05bce91))


### Documentation

* correct an overclaim — test_load_demo_specs still skips in CI (opensrm-ocvu) ([0c15a25](https://github.com/rsionnach/nthlayer-common/commit/0c15a257bb603f00a83b9e72f643daf3ce7be612))
* correct what the demo-spec test actually binds (opensrm-ocvu) ([7a87892](https://github.com/rsionnach/nthlayer-common/commit/7a87892441dd65185b6372eae9a0a124caf154f3))
* note the dead inner sort and the class-patch residual (opensrm-xvwt) ([1a9c8f5](https://github.com/rsionnach/nthlayer-common/commit/1a9c8f566e68362ae7a8d388e92278ba95240fa9))

## [2.1.2](https://github.com/rsionnach/nthlayer-common/compare/v2.1.1...v2.1.2) (2026-08-28)


### Bug Fixes

* **manifest:** spec.type is only a manifest marker if it is a valid type ([0d4ee5a](https://github.com/rsionnach/nthlayer-common/commit/0d4ee5a96e313ac5d9585c3f76864cb7d5287ce8))
* **manifest:** spec.type is only a manifest marker if it is a valid type ([303a2f1](https://github.com/rsionnach/nthlayer-common/commit/303a2f1d8ac71d712be0b629eb435ef42a89d3f1))

## [2.1.1](https://github.com/rsionnach/nthlayer-common/compare/v2.1.0...v2.1.1) (2026-08-28)


### Bug Fixes

* **manifest:** body recovery must check the keys the parsers require ([142b583](https://github.com/rsionnach/nthlayer-common/commit/142b58384d80463053a3327380a16cbf01e2c877))
* **manifest:** body recovery must check the keys the parsers require ([2dcbce7](https://github.com/rsionnach/nthlayer-common/commit/2dcbce706a59ff592f33bdd0d52e6a1e04ea25ed))
* **manifest:** skip directories in iter_manifest_files (opensrm-3470) ([9119393](https://github.com/rsionnach/nthlayer-common/commit/9119393ef2bdcf8bd5acd7bf3dd7d46c8cc8d108))
* **manifest:** skip directories in iter_manifest_files (opensrm-3470) ([008db84](https://github.com/rsionnach/nthlayer-common/commit/008db84065ca48afe2be5060d7be132653bc36b2))
* **manifest:** the body-recovery step was inert (opensrm-3470) ([907da3a](https://github.com/rsionnach/nthlayer-common/commit/907da3abef2d8581e200474e899791e13948a686))
* **manifest:** the body-recovery step was inert (opensrm-3470) ([c8f2762](https://github.com/rsionnach/nthlayer-common/commit/c8f27628458ec152c7e2a3545c68193337ddfb38))


### Documentation

* **manifest:** document scan.py and fix the __all__ grouping (opensrm-3470) ([8608205](https://github.com/rsionnach/nthlayer-common/commit/860820572484b41e58da5af11d0a3c8b16ee10d9))
* **manifest:** document scan.py and fix the __all__ grouping (opensrm-3470) ([f97ff02](https://github.com/rsionnach/nthlayer-common/commit/f97ff0243bd9fe0a96e4a86708253048c9f9bc23))

## [2.1.0](https://github.com/rsionnach/nthlayer-common/compare/v2.0.0...v2.1.0) (2026-08-28)


### Features

* **manifest:** promote the foreign-YAML scan gate into nthlayer-common ([674321e](https://github.com/rsionnach/nthlayer-common/commit/674321e4ac38585d5c9bfbcff7bd8a5ac063f817))
* **manifest:** promote the foreign-YAML scan gate into nthlayer-common ([49594de](https://github.com/rsionnach/nthlayer-common/commit/49594dea247b9b1ea4e929a1831226677d63acc6))

## [2.0.0](https://github.com/rsionnach/nthlayer-common/compare/v1.7.0...v2.0.0) (2026-08-25)


### ⚠ BREAKING CHANGES

* **manifest:** v2 manifests must declare spec.service.type; metadata.labels.type is no longer consulted. convert_v1_to_v2 raises on a v1 manifest without spec.type. VALID_SERVICE_TYPES no longer contains 'web' — it is now the alias web -> x-web.

### Features

* **manifest:** read spec.service.type instead of inferring it ([28d7add](https://github.com/rsionnach/nthlayer-common/commit/28d7adda0e3d9c46e6b92f68fe12da5e97bdd8b5))


### Documentation

* add contributing guide (opensrm-tu04.4) ([f42ce8d](https://github.com/rsionnach/nthlayer-common/commit/f42ce8dabf8775f3f4160eeb3966dd06450fa955))

## [1.7.0](https://github.com/rsionnach/nthlayer-common/compare/v1.6.0...v1.7.0) (2026-06-17)


### Features

* **api-client:** add CoreAPIClient.apply_override · opensrm-jmy.18 ([c5ee550](https://github.com/rsionnach/nthlayer-common/commit/c5ee5502c09f9739645c76acda29321a981e34b2))
* **governance-bridge:** foundation — signal schemas + emitter (opensrm-jmy.5) ([75b7b71](https://github.com/rsionnach/nthlayer-common/commit/75b7b71bba662c86787127474be17217f2d91873))
* **manifest:** extract_declared_dependencies polymorphic helper ([fbc50bf](https://github.com/rsionnach/nthlayer-common/commit/fbc50bf01565ea7bd2a3881c32023fc5ae0025c9))
* **overrides:** add OverrideEvent.to_dict for HTTP wire · opensrm-jmy.18 ([2a43432](https://github.com/rsionnach/nthlayer-common/commit/2a434328a04f353c649cf5a589bc3e67ee04dfe3))
* **overrides:** add pre_redacted flag with plaintext_reviewer alias · opensrm-jmy.18 ([71e477f](https://github.com/rsionnach/nthlayer-common/commit/71e477f0cc63b742114542e0cd9dc14d3d5d525e))


### Bug Fixes

* **ci:** add tag-push + workflow_dispatch triggers to release.yml ([c8a0213](https://github.com/rsionnach/nthlayer-common/commit/c8a0213ea85fc5bf6257a3cba1eae8d2f727f4c0))
* **overrides:** correctness fixes from jmy.4 dry-run (opensrm-jmy.11) ([6610fc0](https://github.com/rsionnach/nthlayer-common/commit/6610fc0c8b593adbdde6e43d805bc366d335678c))
* remove orphan timezone/Enum imports left by 33ff UP017 sweep ([fc1d9e3](https://github.com/rsionnach/nthlayer-common/commit/fc1d9e3dffebea40baeae0c56efcfa500d1f3716))
* resolve 9 ruff findings causing red CI on main · opensrm-5qn3 ([09d0833](https://github.com/rsionnach/nthlayer-common/commit/09d08330b5ced0048b93625bf8ddf5abd610d3ac))


### Code Refactoring

* **manifest:** tighten extract_declared_dependencies types + test imports ([085106e](https://github.com/rsionnach/nthlayer-common/commit/085106e240458260d90ff570a5ef8a7360f6a32e))


### Documentation

* **CLAUDE.md:** record jmy.18 verdict-binding additions · opensrm-jmy.18 ([c34435d](https://github.com/rsionnach/nthlayer-common/commit/c34435d0c4b4d11e9d573fadbdbd720175936c3b))
* link to ecosystem testing conventions (opensrm-2wkc) ([919f4de](https://github.com/rsionnach/nthlayer-common/commit/919f4de993d9f6ce4e655c985368b5cfaad9c762))
* thin CLAUDE.md; move detail to AGENTS.md + docs/ ([125911e](https://github.com/rsionnach/nthlayer-common/commit/125911e9b2ea1e3219bfa853250d2d9df098162f))

## [1.6.0](https://github.com/rsionnach/nthlayer-common/compare/v1.5.0...v1.6.0) (2026-05-10)


### Features

* add BudgetExplanation data model and formatter (nthlayer-hmj) ([e0a8266](https://github.com/rsionnach/nthlayer-common/commit/e0a82666461b1a950ac1293fda9c0081604bdc3e))
* add retry with backoff for transient LLM failures ([adefc7c](https://github.com/rsionnach/nthlayer-common/commit/adefc7cd13121fa5f7e7a7700b57c2cafc1c8f1c))
* add SlackWebClient for interactive messages (buttons, updates) ([a4a037b](https://github.com/rsionnach/nthlayer-common/commit/a4a037bc1e910237f25b92f6faf77f7183d644c0))
* add status code classification and Retry-After parsing for llm_call ([e50c2ef](https://github.com/rsionnach/nthlayer-common/commit/e50c2ef3fc681f50ca25c30cc4c82b092ae28f62))
* add token counts to LLMResponse, add CLAUDE.md ([3b66abf](https://github.com/rsionnach/nthlayer-common/commit/3b66abfaddd21594b3a888c4af802e294ef8407c))
* **cloudevents:** CloudEvents v1.0 envelope helpers ([0086d07](https://github.com/rsionnach/nthlayer-common/commit/0086d0721c2b9775a0c7ab34dae9d8ab6b59c1bb))
* **config:** unified Config dataclass with env override resolution ([9b16795](https://github.com/rsionnach/nthlayer-common/commit/9b16795c122419ff858593c8573dc31ba0c26bf1))
* content-addressed decision records package ([5d2b605](https://github.com/rsionnach/nthlayer-common/commit/5d2b605f451bec69d0a96b60ed5786f1798d201e))
* content-addressed decision records package (records/) ([e98cb9b](https://github.com/rsionnach/nthlayer-common/commit/e98cb9b4ac465c4c4d34067819e629cd1ff10b2f))
* **errors:** add tier-boundary error hierarchy ([ab6f2d1](https://github.com/rsionnach/nthlayer-common/commit/ab6f2d18fcdac01b1ab5f2e6a38df18f07c358a2))
* expose Phase 1 subsystem APIs from package root ([c274cf1](https://github.com/rsionnach/nthlayer-common/commit/c274cf1eea3f84cae181873c45febe69e454e732))
* **llm_stub:** NTHLAYER_LLM_STUB=canned mode for CI integration tests ([9018f6f](https://github.com/rsionnach/nthlayer-common/commit/9018f6fcd463c33a8ea06a7831b3e4e8f8bc1090))
* **llm-stub:** canned factories for respond agent response models ([ee91822](https://github.com/rsionnach/nthlayer-common/commit/ee91822827db6990ac633b0be93a5ae382b57b99))
* **manifest:** convert_v1_to_v2 library function (P2-A.2) ([a65b519](https://github.com/rsionnach/nthlayer-common/commit/a65b519bae66def9b116bf6a5e87d901c10d24b5))
* **manifest:** SLO target convention validator + honest documentation (opensrm-pa2w) ([57b9de1](https://github.com/rsionnach/nthlayer-common/commit/57b9de123cb1b84dc3bd0e9c655895a30cb5efb8))
* **manifest:** unified OpenSRM manifest parsing — v1, v2, legacy ([d340f28](https://github.com/rsionnach/nthlayer-common/commit/d340f28c504aa42193c3a967e2da2b7ebf2a775b))
* **metrics:** self-observability Prometheus metrics ([3b9e666](https://github.com/rsionnach/nthlayer-common/commit/3b9e666875bc428c51e36254646230d5deb002af))
* model-agnostic LLM wrapper for NthLayer ecosystem ([a8ad6b0](https://github.com/rsionnach/nthlayer-common/commit/a8ad6b07eb59bf767fdd7d026e2aa73c0577a30f))
* **outcomes:** business outcome binding foundation (opensrm-jmy.1) ([437eb33](https://github.com/rsionnach/nthlayer-common/commit/437eb337a018d8953b3e707d4ff1f59fd1c6fcc4))
* **overrides:** human override ingestion foundation (opensrm-jmy.4) ([f03c3cf](https://github.com/rsionnach/nthlayer-common/commit/f03c3cf22fce52d185c19de25507dedb8106601d))
* shared infrastructure migration + py.typed + README ([f36b49b](https://github.com/rsionnach/nthlayer-common/commit/f36b49bd9a202f67508161edbb0c380900c3382b))
* shared prompt loader with YAML schemas and response validation ([8bc5555](https://github.com/rsionnach/nthlayer-common/commit/8bc555529e35f9848d352926a4e315f7d9bea262))
* SlackNotifier transport — Block Kit messages via webhook, fail-open ([5b1f328](https://github.com/rsionnach/nthlayer-common/commit/5b1f328b0c67bb504ec773ca00e685bd7397a89a))
* SlackWebClient + LLM retry with backoff (v0.1.4) ([3f0801e](https://github.com/rsionnach/nthlayer-common/commit/3f0801e988a6c628af1e82920189e633af48d270))
* **verdicts:** atomic AI judgment model migrated from nthlayer-learn ([21d5f79](https://github.com/rsionnach/nthlayer-common/commit/21d5f79d06c802765af1d0b4ca96daed328edbbe))
* **verdicts:** wire-canonical names + 4 respond-agent verdict types ([dccb520](https://github.com/rsionnach/nthlayer-common/commit/dccb520878e737f5bd4f5d61a567963f2fe6c0e7))


### Bug Fixes

* add missing runtime dependencies to pyproject.toml ([11c36c2](https://github.com/rsionnach/nthlayer-common/commit/11c36c2d0d234022a0f2d282d73da49e1e48aa82))
* add shared parsing utilities, fix Azure URL, add timeout docs ([15fcded](https://github.com/rsionnach/nthlayer-common/commit/15fcdedf05f5a8007d75d1325602ec70e39bee2f))
* catch BaseHTTPClient error types in MimirRulerProvider (R5 finding) ([b7977ea](https://github.com/rsionnach/nthlayer-common/commit/b7977ea28af4d0dfe888d33880ac86a9c88679cb))
* **deps:** pin pygments&gt;=2.20.0 to clear CVE-2026-4539 ([65880e5](https://github.com/rsionnach/nthlayer-common/commit/65880e5cf3ae3dd683a966b47cd0cf0595b2c1ab))
* detect API keys accidentally used as model names ([3a44f5b](https://github.com/rsionnach/nthlayer-common/commit/3a44f5b09b62a392beeb60650f1cf5e738c41f95))
* export error hierarchy from __init__.py + bump to 0.1.7 ([b77cc51](https://github.com/rsionnach/nthlayer-common/commit/b77cc5122ab269f1507ba983e7aeeb412ce714c2))
* handle non-UTF-8 body in verify_signature ([d73e924](https://github.com/rsionnach/nthlayer-common/commit/d73e9246af3221da984ab334db03d927b16b9dac))
* harden llm.py from R5 review findings ([b1a103a](https://github.com/rsionnach/nthlayer-common/commit/b1a103a7115a75a7a6afb0a338e9b594c1d6a77e))
* **llm-stub:** include confidence in SnapshotSummary canned response ([992c0c3](https://github.com/rsionnach/nthlayer-common/commit/992c0c378f03cac02289fe8c67f5e9992b7bec75))
* **manifest/v1:** read indicator.query from nested form (opensrm-0i5h) ([0fb5096](https://github.com/rsionnach/nthlayer-common/commit/0fb5096d10a1d477e67f72763f54477ad965477b))
* **prometheus:** distinguish empty-result from 0.0 in get_sli_value ([1874e2b](https://github.com/rsionnach/nthlayer-common/commit/1874e2b8d9092da4c1d65a382557e70362908c9d))
* R5 edge case findings in BaseHTTPClient + MimirRulerProvider ([e702447](https://github.com/rsionnach/nthlayer-common/commit/e702447b694f1a4543b521c30cbbebcea35e9e37))


### Code Refactoring

* move MimirRulerProvider to clients/, extend BaseHTTPClient (nthlayer-2xe) ([2403945](https://github.com/rsionnach/nthlayer-common/commit/2403945ce70092951601c1fd737f16c1d200e9f7))
* **overrides:** clarity nits — comment + docstring trim (opensrm-jmy.12) ([11de001](https://github.com/rsionnach/nthlayer-common/commit/11de001a847abce77284b7d8a0b682635b9600a5))
* **overrides:** R5 review followups (opensrm-jmy.4) ([8139492](https://github.com/rsionnach/nthlayer-common/commit/8139492897c8d5e4b24ca0637ae99aa944a86308))
* share httpx client in SlackWebClient for connection pooling ([0a9ecf3](https://github.com/rsionnach/nthlayer-common/commit/0a9ecf35eb9e0720748ca4699dbe104571cc9eb1))
* **slo:** adopt 0-100 percentage canonical for SLO target convention ([0ca1296](https://github.com/rsionnach/nthlayer-common/commit/0ca129688c72b8e8c6745db85c2345fdf74c3b9f))


### Documentation

* add README.md for PyPI description, bump to 0.1.1 ([f642ff2](https://github.com/rsionnach/nthlayer-common/commit/f642ff2efda3c14a545a5c4a0c05b4209b803f2a))
* **CLAUDE.md:** catalogue Outcomes module + financial impact tests (opensrm-jmy.1) ([4bbca0d](https://github.com/rsionnach/nthlayer-common/commit/4bbca0d371bbd881a4955ea180a609341c7bf9ad))
* **CLAUDE.md:** document 8-judgment-type parametrised parser test ([5ad876a](https://github.com/rsionnach/nthlayer-common/commit/5ad876a8d80c8ea09960f380acd058a0a44a29f0))
* **CLAUDE.md:** document convert_v1_to_v2 and migration tests ([0da4c54](https://github.com/rsionnach/nthlayer-common/commit/0da4c5480f451cc2dc0326c2dc89bfb1e383f210))
* **CLAUDE.md:** document get_sli_value None-vs-zero semantics + tests ([674e1fe](https://github.com/rsionnach/nthlayer-common/commit/674e1fe1a01d8a490e605d2fb0b61f0fb7de3086))
* **CLAUDE.md:** document pygments transitive pin for CVE-2026-4539 ([6f672fe](https://github.com/rsionnach/nthlayer-common/commit/6f672fe6aeac9d0e39912a9c9764f5d775fe189f))
* **CLAUDE.md:** document release-please + smoke gate + Dependabot ([392ba4d](https://github.com/rsionnach/nthlayer-common/commit/392ba4df6bcc467aeaddcdfb680ef8fb1c0c0d35))
* **CLAUDE.md:** document respond-agent stub factories ([6ff38f5](https://github.com/rsionnach/nthlayer-common/commit/6ff38f5308bb7cff8e096ade3176a8994af4fdf2))
* **CLAUDE.md:** document target_validation module + SLO convention divergence ([60bf7da](https://github.com/rsionnach/nthlayer-common/commit/60bf7da7546b6684a261e81a47b3e540627dae1f))
* **CLAUDE.md:** expand terminal-status enumeration (opensrm-jmy.12) ([588f786](https://github.com/rsionnach/nthlayer-common/commit/588f786f2a3caaca86f176a8cc00d15b0a23a486))
* **CLAUDE.md:** note confidence=0.5 in SnapshotSummary stub factory ([0df2256](https://github.com/rsionnach/nthlayer-common/commit/0df2256b37e61c47d4e297fd976c0b66faf09f11))
* **CLAUDE.md:** tighten override test catalogue post-R5 ([726384e](https://github.com/rsionnach/nthlayer-common/commit/726384ea36eae8698e3097138ed3186233b48d50))
* **CLAUDE.md:** update SLO target convention + target_validation entries ([0a5a227](https://github.com/rsionnach/nthlayer-common/commit/0a5a2272592b6234dfe2e7c295874769611e9734))
* **comments:** inline pointers to verdict-vs-assessment decision ([008941e](https://github.com/rsionnach/nthlayer-common/commit/008941eabcfae5cd30bd3989d1cfa57703e8fb27))
* fix self-referential _headers() docstring (R5 clarity) ([384eab8](https://github.com/rsionnach/nthlayer-common/commit/384eab8ccc539062b058fbd09b98c4bc73d24d91))
* update README to reflect full shared infrastructure scope ([7e74923](https://github.com/rsionnach/nthlayer-common/commit/7e74923e300f04074658938034ee6aa4bfcd8922))

## v1.5.0 — 2026-05-03

First lockstep release with the rest of the v1.5 ecosystem. Major changes:

**LLM stub for CI integration tests** (opensrm-saun.1.1).
`nthlayer_common.llm_stub` short-circuits `llm_call()`, `structured_call()`,
and `structured_call_with_usage()` when `NTHLAYER_LLM_STUB=canned` is set.
Returns deterministic canned responses indexed by detected agent role
(triage / investigation / communication / remediation) for raw text and by
`response_model` class name (`EvaluationResult`, `SnapshotSummary`) for
structured calls. Activation predicate `is_stub_enabled()` is
case-insensitive and trim-tolerant. Three call sites in `llm.py` and
`llm_structured.py` import the stub lazily for cycle safety. Documented in
the package's CLAUDE.md; not advertised on PyPI as it is intended for CI
integration tests only.

**Wire-canonical field names in `to_dict(Verdict)`** (opensrm-saun.1.2).
`to_dict` now emits HTTP-canonical `type` and `created_at` (renamed from
internal `verdict_type` and `timestamp`) at the serialisation boundary.
The `Verdict` dataclass keeps its internal field names. `from_dict` accepts
both name pairs for round-trip compatibility with stored data; precedence
rule is "explicit None on canonical name falls back to legacy". This
aligns the wire format with what `nthlayer-core`'s `POST /verdicts` API
contract expects.

**CloudEvents type taxonomy consolidated** (opensrm-saun.1.2).
`nthlayer_common.cloudevents._VERDICT_TYPES` now imports
`VALID_VERDICT_TYPES` from `nthlayer_common.verdicts.models` rather than
maintaining a parallel set — one canonical source. The verdict-type set
itself was reshaped: dropped `"assessment"` (a category error — verdicts
are decisions, assessments are continuous observations) and added the four
respond agent role names (`triage`, `investigation`, `communication`,
`remediation`).

**v1 SRM manifest parser reads canonical `indicator.query`** (opensrm-saun.1).
`parser.v1.py` previously read top-level `config["query"]` (a non-canonical
shape that no published spec or example produces). The fix routes through
a new `_extract_indicator_query()` helper that reads the canonical
`indicator.query` nested form per `nthlayer-generate/documentation/SCHEMA.md`.
Discovered when the saun.1 three-tier integration test exercised the
worker → core `/manifests` path for the first time and found every SLO
was being collected as `NO_DATA` because the indicator PromQL was being
silently dropped at parse time. Regression test
(`tests/test_manifest_real_specs.py`) loads each `demo/specs/*.yaml`
from disk and asserts `indicator_query` populates.

**Provenance.** This package was created during the six-repo
consolidation decided 2026-04-21 (see
`docs/superpowers/specs/2026-04-21-repo-consolidation-recommendation.md`
in the `opensrm` repo). It hosts the unified LLM wrapper, provider
infrastructure (Prometheus, Grafana, PagerDuty, Mimir), identity
resolution, HTTP clients, Slack notifiers, error hierarchy, tier
definitions, shared data models, prompt loader, content-addressed
decision records, verdict model, manifest parsing, CloudEvents envelope
helpers, and self-observability metrics. Detailed surface documented in
this package's `CLAUDE.md`.
