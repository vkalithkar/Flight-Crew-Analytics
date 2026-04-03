ml_classifier.py

Purpose
- Train a classifier to predict pilot mental state stages (Channelized Attention, Diverted Attention, Startle/Surprise) from physiological signals (EEG, ECG, Respiration, GSR).

Goal:

Quick start
1. Prepare a cleaned CSV with a target column (default name: `stage`) and numeric physiological features. Place it under `data/clean/` or provide a path.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python ./scripts/ml_classifier.py --data data/clean/processed.csv --target stage --out-dir ./output
```

Outputs
- `./output/metrics.json`: Evaluation metrics and CV scores.

Notes
- A separate data cleaning script should be run before this.
