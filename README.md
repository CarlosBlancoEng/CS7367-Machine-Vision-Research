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

MVTec AD 2 only: `can`, `wallplugs`, `sheet_metal`. Fit spatial Gaussian normal-reference distributions from `train/good`; evaluate only labeled `test_public/good` and `test_public/bad`. Validation and private tests are excluded. One fixed-seed manifest is reused across baseline, gamma, global histogram equalization, and CLAHE. Enhancements apply equally to reference and evaluation images; masks are not enhanced.

| Workload | Reference cap per category | Evaluation cap per category |
| --- | ---: | ---: |
| `small` | 25 | 20 |
| `medium` | 250 | 100 |
| `large` | All eligible | All eligible |

Evaluation sampling prioritizes both classes and reports actual balance. Reference/evaluation filenames are disjoint. Small/medium AUROC is a sample metric, not an estimate claimed equivalent to full-set AUROC.

All conditions use 256×256 RGB inputs without cropping, ImageNet normalization, layers 1–3 of frozen ResNet18, a seed-fixed 32-channel subset aligned to 32×32, covariance regularization 0.01, Mahalanobis scores, and Gaussian smoothing σ=4. Batch-Welford moments avoid storing reference features for the entire dataset. Gamma uses exponent 0.7 on normalized CIELAB L; histogram equalization and CLAHE (clip 2, 8×8 tiles) use the same channel after resizing. These fixed resource choices are documented implementations, not a reproduction claim for published PaDiM results.

Metrics are image-level AUROC and normalized **AU-PRO0.05**. The latter gives each 8-connected ground-truth region equal weight, sweeps exact score thresholds with tied scores grouped, interpolates at FPR 0.05, and divides the partial area by 0.05. Background pixels from normal and anomalous images enter FPR. Both evaluation classes are required; AU-PRO additionally requires all selected anomaly masks to be valid. Missing/degenerate masks produce a blank AU-PRO with an explicit reason while valid image labels can still support AUROC. Metrics use nearest-neighbor masks at **256×256**; small defects can disappear, and these values are not native-resolution official benchmark scores. Private illumination-shift conclusions remain unsupported by public-only metrics.

Each run saves `metrics.csv`, `experiment_summary.json`, and `sample_manifest.csv` under `My Drive/ColabData/CS7367_Vision/output/experiments/<mode>/<UTC timestamp>/`. Results are saved after each condition with incomplete status until all configurations finish. Earlier runs are never overwritten. A restart creates a new folder; Batch 1 does not checkpoint/resume fitted models. Covariance memory is bounded by the fixed feature grid/dimension; exact metric sorting scales with the evaluation pixels of one category. Start with `small` before a larger run. No real dataset results are claimed by the implementation's synthetic tests.
