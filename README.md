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

Open `notebooks/CS7367_Research_Experiments.ipynb` in a fresh Colab Python 3 runtime, choose CPU or an optional GPU, set `workloadSize` near the beginning, and select **Runtime → Run all**. The runtime needs PyTorch/torchvision, Pillow, OpenCV, NumPy, SciPy, and scikit-learn, normally included in Colab. First use downloads torchvision's ImageNet ResNet18 weights; loading failure stops the run rather than substituting random weights.

MVTec AD 2 only. Batch 2 defaults to exploratory small-workload screening of all eight categories: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`. Edit the obvious `selectedCategories` list near the workload selector to choose a subset. Fit spatial Gaussian normal-reference distributions from `train/good`; evaluate only labeled `test_public/good` and `test_public/bad`. Validation and private tests are excluded. One fixed-seed manifest is reused across baseline, gamma, global histogram equalization, and CLAHE. Enhancements apply equally to reference and evaluation images; masks are not enhanced. Small screening does not establish final category selection or a winning method; Team review determines follow-up scope.

| Workload | Reference cap per category | Evaluation cap per category |
| --- | ---: | ---: |
| `small` | 25 | 20 |
| `medium` | 250 | 100 |
| `large` | All eligible | All eligible |

Evaluation sampling prioritizes both classes and reports actual balance. Reference/evaluation filenames are disjoint. Small/medium AUROC is a sample metric, not an estimate claimed equivalent to full-set AUROC.

All conditions use 256×256 RGB inputs without cropping, ImageNet normalization, layers 1–3 of frozen ResNet18, a seed-fixed 32-channel subset aligned to 32×32, covariance regularization 0.01, Mahalanobis scores, and Gaussian smoothing σ=4. Batch-Welford moments avoid storing reference features for the entire dataset. Gamma uses exponent 0.7 on normalized CIELAB L; histogram equalization and CLAHE (clip 2, 8×8 tiles) use the same channel after resizing. These fixed resource choices are documented implementations, not a reproduction claim for published PaDiM results.

Metrics are image-level AUROC and normalized **AU-PRO0.05**. The latter gives each 8-connected ground-truth region equal weight, sweeps exact score thresholds with tied scores grouped, interpolates at FPR 0.05, and divides the partial area by 0.05. Background pixels from normal and anomalous images enter FPR. Both evaluation classes are required; AU-PRO additionally requires all selected anomaly masks to be valid. Missing/degenerate masks produce a blank AU-PRO with an explicit reason while valid image labels can still support AUROC. Metrics use nearest-neighbor masks at **256×256**; small defects can disappear, and these values are not native-resolution official benchmark scores. Private illumination-shift conclusions remain unsupported by public-only metrics.

Each run saves `metrics.csv`, `experiment_summary.json`, and `sample_manifest.csv` under `My Drive/ColabData/CS7367_Vision/output/experiments/<mode>/<UTC timestamp>/`. Results are saved after each condition with incomplete status until all configurations finish. Earlier runs are never overwritten. A restart creates a new folder; Batch 1 does not checkpoint/resume fitted models. Covariance memory is bounded by the fixed feature grid/dimension; exact metric sorting scales with the evaluation pixels of one category. Start with `small` before a larger run. No real dataset results are claimed by the implementation's synthetic tests.

Read-only Batch 2 inspection confirmed that `can/test_public/bad/002_shift_1.png` has a matching 2232×1024 binary mask with 25 positive native pixels, all lost under the existing 256×256 nearest-neighbor resize. The native annotation is valid. Batch 2 preserves the resampling policy and leaves category AU-PRO blank when any selected anomaly disappears; it does not discard that image, dilate masks, or manufacture positives. Image AUROC remains supported by valid public labels. `maskDiagnostics` records native/resized areas, values, dimensions, and bounding boxes for selected masks that vanish. A higher/native-resolution protocol needs separate approval and fresh comparisons; increasing workload alone does not repair lost mask detail.

UTC ISO-8601 stage logs use flushed START/PROGRESS/COMPLETE output and `time.perf_counter()` durations, including five-image updates in substantive loops. `experiment_summary.json` adds `stageTimings` entries with `stage`, `category`, `condition`, `startUtc`, `endUtc`, `elapsedSeconds`, and `status`. Started but interrupted stages retain null end/duration and incomplete status. Enhancement/features are timed together with reference fitting or inference to avoid nested per-batch log spam. `implementationVersion`, selected categories, mask policy, and comparison policy make scope/settings explicit; existing CSV schemas and compact UTC run-directory identifiers remain unchanged. Earlier-category metrics retain Batch 1 methodology and are directly comparable when manifests, parameters, weights, and environment match. No model, parameter, mask-resampling, scoring, or AU-PRO-mathematics change is introduced by this screening update.

## Separate 512 feasibility investigation

`diagnostics/resolution_512.py` and its offline tests investigate 512 only, outside the production notebook. **The production experiment remains 256; Team acceptance is required before any promotion.** Run the offline test/preflight first, then the read-only `audit` phase. Only after reviewing a passing audit, `model <audit_run_directory>` uses exactly the archived small pilot's can/fabric identities (25 normal references and 20 public evaluations each) across four conditions. It never launches medium, large, or an eight-category model run.

The audit compares NEAREST512 with exact foreground-occupancy coverage: a target cell is positive only when its edge-aligned source footprint overlaps positive annotation. Coverage prevents missing positives but can inflate small regions and merge components; it does not reconstruct precise shape. No arbitrary dilation or original-file edits occur. The feature grid is derived from actual layer-2 output (64×64 at 512); other scientific parameters stay fixed. Mask-policy sensitivity is assessed on the same fixed scores without another model run. Comparison with archived 256 metrics is descriptive and explicitly records protocol/library/environment differences.

Investigation evidence goes only to `output/investigations/resolution_512/<unique UTC run>/`. On Windows the script defaults to `G:/My Drive`; `CS7367_DRIVE_ROOT` can specify another mounted Drive root, and `CS7367_TORCH_CACHE` can specify a separate untracked weights cache. Required dependencies are the research notebook's existing dependencies. No generated evidence, images, or weights belong in Git. The mandatory Batch 3 report in the shared `_Reports` folder records the findings and decision gate.
