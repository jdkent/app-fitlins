# brainlife/app-fitlins-firstlevel

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)

Brainlife app wrapper for FitLins with a user-centered, human-readable first-level GLM interface.

This app is designed for the common workflow:
1. preprocessed BOLD from fMRIPrep derivatives,
2. confounds from fMRIPrep (`desc-confounds_timeseries.tsv`),
3. task events from raw BIDS (`events.tsv`),
4. simple first-level model setup without manually editing full BIDS Stats Models JSON.

The app generates a BIDS Stats Model JSON from the `design` + `confounds` blocks in `config.json`, stages a minimal BIDS + derivatives layout, and runs FitLins.

## Interface Highlights
- `runs[]`: one object per run with `bold`, `events_tsv`, and `confounds_tsv`.
- `design`: readable model controls (`condition_column`, `hrf_model`, and contrast weights by condition name).
- `confounds`: strategy-based nuisance selection (`acompcor6`, `motion_24`, `basic`, `custom`, `none`).
- `advanced.model_json_path`: escape hatch for full manual BIDS Stats Models JSON.

See `config.json.example` for a complete example.

## Local Test Run
```bash
cp config.json.example config.json
python3 main.py
```

To validate config/model generation only:
```json
{
  "advanced": { "dry_run": true }
}
```

## Outputs
- `out_dir/model/model-generated_smdl.json`: generated (or copied) BIDS Stats Model.
- `out_dir/model/model-summary.md`: readable summary of conditions/confounds/analysis level.
- `out_dir/fitlins/`: FitLins derivative outputs and reports.

## Scientific Defaults
- First-level run model with `Factor` + `Convolve` transforms on `trial_type` by default.
- Confounds default to `acompcor6` plus motion, FD, and DVARS (dropping missing columns unless `strict: true`).
- App is intentionally first-level only for now (`analysis_level=run`).

## Authors
- FitLins and Brainlife contributors

## Citations
Please cite:

1. Hayashi, S., Caron, B. A., et al. (2023). *brainlife.io: A decentralized and open source cloud platform to support neuroscience research*. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10274934/
2. Markiewicz, C. J., et al. (2021). *The OpenNeuro resource for sharing of neuroscience data*. https://doi.org/10.7554/eLife.71774
3. BIDS Stats Models specification. https://bids-standard.github.io/stats-models/

#### MIT Copyright (c) 2021 brainlife.io The University of Texas at Austin and Indiana University
