# CS7367 Machine Vision Research

Research and dataset qualification for **Image Enhancement as Preprocessing for Industrial Anomaly Detection Under Challenging Illumination**. The validation notebook performs sample-based qualification, not exhaustive integrity certification. The separate research notebook implements the approved MVTec AD 2 PaDiM comparison.

Open `notebooks/CS7367_Dataset_Validation.ipynb` in Google Colab, select a CPU runtime, and choose **Runtime → Run all**. Authorize the Drive mount. Pillow is included in Colab. The configuration cell expects:

```text
My Drive/ColabData/CS7367_Vision/
├── mvtec_ad_2/
├── VisA_20220922/
└── output/validation/
```

Category roots are discovered beneath the configured dataset directories, allowing an extraction wrapper. Ambiguous copies fail validation. Datasets are read only; the notebook never downloads, extracts, reorganizes, or edits them. Use Hardware accelerator **None**; no GPU is needed.

All required category/split image directories and all filenames/counts are inventoried. All annotation records are checked against the filename inventory without decoding their referenced contents. A fixed seed (`7367`) selects up to **15 images per category across all available split/class groups**, plus up to **15 available masks per category**, without replacement. Image groups and filenames are shuffled, then groups are visited in rounds so each nonempty split/class is represented where feasible. Selection is reproducible for the same relative inventory and seed. Categories with fewer files select all available files and record a limitation.

Only selected images and masks are fully decoded. For each sampled mask, its paired image is checked for correspondence and matching dimensions; an unsampled paired image has only its header read, preserving the 15-image decode cap. Header reads are counted separately. Sample processing prints flushed progress every five files with dataset/category, processed/total, and elapsed time, plus stage start/completion messages.

The output directory receives `dataset_inventory.csv`, `validation_exceptions.csv`, and `validation_summary.json`. A new run first invalidates prior evidence with an incomplete FAIL checkpoint. Only a completed run can report **PASS — sample-based qualification**. Do not treat artifacts as current unless the summary is complete and its timestamps match the run. Inventory rows use dataset-specific groups (for example `test_public/bad` and `Anomaly`), with full and sampled image/mask counts and JSON-encoded relative filename lists. The JSON summary records full totals, sample sizes by category, seed, runtime versions, errors, and limitations. Counts describe this installation, not proof of archive checksum identity.

MVTec requires all eight categories, normal train/validation data, and public normal/anomalous test data. Absent private image partitions are explicit scope limitations; present private partitions must be nonempty and their sampled images readable. Private labels are intentionally withheld. Missing inferred MVTec mask paths and blank VisA mask annotations are limitations that exclude those images from pixel-level qualification. Broken explicit CSV references, missing required image annotation coverage, duplicates, unmatched available masks, structural errors, and sampled unreadable/inconsistent files are critical failures. VisA requires all 12 original categories and valid per-category CSV image coverage; normal masks are intentionally absent. Original VisA does not require train/test folders or downloaded split definitions. Unsupported annotation schemas fail with an explicit exception rather than guessing. Installed release identifiers remain unverified unless independently established.

**Residual risk:** unsampled corrupt images, unreadable masks, and dimension inconsistencies can remain undetected even after PASS. Missing/withheld masks and unavailable private images are outside the qualified scope. The notebook makes no exhaustive integrity or statistical confidence claim.

Official sources consulted for layout and annotation rules:

- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), including the linked official code utilities (`mvtec_ad_2_public_offline.py`).
- [VisA AWS registry](https://registry.opendata.aws/visa/), which identifies the `VisA_20220922.tar` archive.
- [VisA reference repository](https://github.com/amazon-science/spot-diff), including its original data tree and `utils/prepare_data.py` category definitions.

Implementation and synthetic tests do not establish dataset acceptance. Carlos executes in Colab; Sam Chatley reviews the three artifacts and recommends PASS/FAIL; Carlos retains final acceptance authority. Keep datasets, generated evidence, and executed notebook outputs out of Git. Do not merge to main or push acceptance-related changes without Carlos's explicit Git authorization.

## Research experiments

Open `notebooks/CS7367_Research_Experiments.ipynb` in a fresh Colab Python 3 runtime, choose CPU or an optional GPU, keep `workloadSize = 'small'` for the approved new pilot, and select **Runtime → Run all**. Production now uses Team-approved protocol **`512-coverage-layer2-v1`**: 512×512 RGB, exact foreground-coverage masks, and a layer2-derived 64×64 feature grid. The runtime needs PyTorch/torchvision, Pillow, OpenCV, NumPy, SciPy, and scikit-learn, normally included in Colab. First use downloads torchvision's ImageNet ResNet18 weights; loading failure stops the run rather than substituting random weights.

MVTec AD 2 only. The default small workload includes all eight categories: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`. Edit the obvious `selectedCategories` list near the workload selector to choose a subset. Fit spatial Gaussian normal-reference distributions from `train/good`; evaluate only labeled `test_public/good` and `test_public/bad`. Validation and private tests are excluded. One fixed manifest is reused across baseline, gamma, global histogram equalization, and CLAHE. Enhancements apply equally to reference and evaluation images; masks are not enhanced. Small screening does not establish final category selection or a winning method; Team review determines follow-up scope.

| Workload | Reference cap per category | Evaluation cap per category |
| --- | ---: | ---: |
| `small` | 25 | 20 |
| `medium` | 250 | 100 |
| `large` | All eligible | All eligible |

Small loads `output/experiments/small/20261115T000643_647170Z/sample_manifest.csv`, checks pinned SHA-256 `0588df4ceceabe99b7d4fab5a6798d13df467b02863d9da42517a1edd907c7c0`, validates roles/splits/labels and existence, and preserves exact identities and order. It requires the archived 25 reference/10 good/10 bad counts per selected category. Missing/changed identities or archive cause a clear stop, never resampling or workload reduction. Subsets retain archived row order. Medium/large keep the original seed-stratified selector and caps, but are not authorized for automatic execution. Small/medium AUROC is a sample metric, not an estimate claimed equivalent to full-set AUROC.

All conditions use 512×512 RGB inputs without cropping, unchanged bilinear RGB resizing, ImageNet normalization, layers 1–3 of frozen ResNet18, a seed-fixed 32-channel subset, covariance regularization 0.01, Mahalanobis scores, and Gaussian smoothing σ=4. The 64×64 alignment grid is derived from the actual layer2 tensor; shapes and finite features/covariance/precision/scores are checked. Batch-Welford moments avoid storing reference features for the entire dataset. Gamma uses exponent 0.7 on normalized CIELAB L; histogram equalization and CLAHE (clip 2, 8×8 tiles) use the same channel after resizing. These fixed resource choices are documented implementations, not a reproduction claim for published PaDiM results.

Metrics are image-level AUROC and normalized **AU-PRO0.05**. The latter gives each 8-connected target-mask region equal weight, sweeps exact score thresholds with tied scores grouped, interpolates at FPR 0.05, and divides the partial area by 0.05. Background pixels from normal and anomalous images enter FPR. Both evaluation classes are required; AU-PRO additionally requires all selected anomaly masks to be valid. Every selected mask is audited before scoring: dimensions/binary values, native/target area, components, boxes, edge alignment and support for every native component. Missing/invalid masks or an unrepresented component produce blank category AU-PRO with reasons; evaluation identities and valid image AUROC remain intact. Exact source-footprint coverage projects binary occupancy at 512, permitting accepted region merging/area inflation. It is not native-resolution official benchmark certification. Private illumination-shift conclusions remain unsupported by public-only metrics.

Each run saves `metrics.csv`, `experiment_summary.json`, and `sample_manifest.csv` under `My Drive/ColabData/CS7367_Vision/output/experiments/<mode>/<UTC timestamp>/`. Results are saved after each condition with incomplete status until all configurations finish. Earlier runs are never overwritten. A restart creates a new folder; model checkpoint/resume is outside scope. Covariance memory is bounded by the fixed feature grid/dimension; exact metric sorting scales with the evaluation pixels of one category. Start with `small` before a larger run. No real dataset results are claimed by the implementation's synthetic tests.

Batch 3 confirmed that the known can/fabric tiny annotations survive 512 coverage. Across 80 archived anomaly masks, all 172 native components were represented, becoming 150 target components with merging in 12 masks. Coverage declares a target cell occupied only if its exact edge-aligned source footprint overlaps a positive native pixel. It does not reconstruct exact shape: native-equivalent area can increase and square resizing retains aspect distortion (including 4:1 sheet_metal). The Team explicitly accepted these semantics for Batch 4. `maskDiagnostics` and `maskAuditOutcomes` record detailed results for all selected anomalies, not just losses. Packed-bit audit caches avoid keeping every full mask array resident while scoring categories sequentially.

UTC ISO-8601 stage logs use flushed START/PROGRESS/COMPLETE output and `time.perf_counter()` durations, including five-image updates in substantive loops. `stageTimings` records actual intervals; incomplete stages keep null end/duration. The JSON also records protocol/version, exact file manifest/comparison, selected counts, mask audits, model grid/layer shapes, parameters, environment and optional CPU/GPU peak memory. CPU peak RSS is measured via `resource` where available (including Colab Linux); Windows records unavailable rather than inventing a value. GPU peaks use PyTorch when available. Resource failures stop with incomplete evidence; there is no automatic feature/sample reduction. Existing CSV schemas and output directory routing are unchanged.

**Keep the archived 256/NEAREST/grid32 series separate.** Reusing image identities does not make the new protocol's metric values combinable with old values. Batch 3 fabric AUROC regressed materially; no universal 512 accuracy gain is claimed. Carlos runs the fresh all-eight small Colab pilot and provides its new evidence folder for Sam's review before medium. The accepted validation notebook is unchanged.

## Separate 512 feasibility investigation

`diagnostics/resolution_512.py` and its offline tests preserve the bounded Batch 3 investigation, outside the production notebook. The archived investigation used can/fabric only. **Batch 4 subsequently promoted the explicitly approved uniform protocol; a new real all-eight production pilot still requires Carlos's execution/review.** Historical diagnostic evidence remains distinct and unchanged.

The audit compares NEAREST512 with exact foreground-occupancy coverage: a target cell is positive only when its edge-aligned source footprint overlaps positive annotation. Coverage prevents missing positives but can inflate small regions and merge components; it does not reconstruct precise shape. No arbitrary dilation or original-file edits occur. The feature grid is derived from actual layer-2 output (64×64 at 512); other scientific parameters stay fixed. Mask-policy sensitivity is assessed on the same fixed scores without another model run. Comparison with archived 256 metrics is descriptive and explicitly records protocol/library/environment differences.

Investigation evidence goes only to `output/investigations/resolution_512/<unique UTC run>/`. On Windows the script defaults to `G:/My Drive`; `CS7367_DRIVE_ROOT` can specify another mounted Drive root, and `CS7367_TORCH_CACHE` can specify a separate untracked weights cache. Required dependencies are the research notebook's existing dependencies. No generated evidence, images, or weights belong in Git. The mandatory Batch 3 report in the shared `_Reports` folder records the findings and decision gate.
