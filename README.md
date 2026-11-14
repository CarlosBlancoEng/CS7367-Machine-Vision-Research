# CS7367 Machine Vision Research

One-time **sample-based dataset qualification** for **Image Enhancement as Preprocessing for Industrial Anomaly Detection Under Challenging Illumination**. This is not exhaustive image integrity certification. No experiments, enhancement, or anomaly detection are implemented.

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
