# CS7367 Machine Vision Research

One-time dataset qualification for **Image Enhancement as Preprocessing for Industrial Anomaly Detection Under Challenging Illumination**. No experiments, enhancement, or anomaly detection are implemented.

Open `notebooks/CS7367_Dataset_Validation.ipynb` in Google Colab, select a CPU runtime, and choose **Runtime → Run all**. Authorize the Drive mount. Pillow is included in Colab. The configuration cell expects:

```text
My Drive/ColabData/CS7367_Vision/
├── mvtec_ad_2/
├── VisA_20220922/
└── output/validation/
```

Category roots are discovered beneath the configured dataset directories, allowing an extraction wrapper. Ambiguous copies fail validation. Datasets are read only; the notebook never downloads, extracts, reorganizes, or edits them. All source image files and available masks are decoded one at a time.

The output directory receives `dataset_inventory.csv`, `validation_exceptions.csv`, and `validation_summary.json`. A new run first invalidates prior evidence with an incomplete FAIL checkpoint. Only a completed run can report PASS. Do not treat artifacts as current unless the summary is complete and its timestamps match the run. Inventory rows use dataset-specific groups (for example `test_public/bad` and `Anomaly`); counts describe this installation, not proof of archive checksum identity.

MVTec requires all eight categories, normal train/validation data, and public normal/anomalous test data with anomaly masks. Absent private image partitions are explicit scope limitations, while present private partitions must contain readable images. Private labels are intentionally withheld. A PASS excludes unavailable private images. VisA requires all 12 original categories and valid per-category CSV coverage; normal masks are intentionally absent. Original VisA does not require train/test folders or downloaded split definitions. Unsupported annotation schemas fail with an explicit exception rather than guessing. Installed release identifiers remain unverified unless independently established.

Official sources consulted for layout and annotation rules:

- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), including the linked official code utilities (`mvtec_ad_2_public_offline.py`).
- [VisA AWS registry](https://registry.opendata.aws/visa/), which identifies the `VisA_20220922.tar` archive.
- [VisA reference repository](https://github.com/amazon-science/spot-diff), including its original data tree and `utils/prepare_data.py` category definitions.

Implementation and synthetic tests do not establish dataset acceptance. Carlos executes in Colab; Sam Chatley reviews the three artifacts and recommends PASS/FAIL; Carlos retains final acceptance authority. Keep datasets, generated evidence, and executed notebook outputs out of Git. Do not merge to main or push acceptance-related changes without Carlos's explicit Git authorization.
