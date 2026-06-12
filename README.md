# RealEssate Forecast

MLOps pipeline for **China Real Estate Demand Prediction** — forecast monthly new-house transaction volumes across 96 sectors using CatBoost, served via FastAPI (ONNX) and a Streamlit dashboard, with MLflow tracking and drift-based retraining.

**Data:** [Kaggle — China Real Estate Demand Prediction](https://www.kaggle.com/competitions/china-real-estate-demand-prediction/data)

---

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — Python package & environment manager (same tool used in CI)
- Git
- (Optional) Docker & Docker Compose
- Kaggle account to download competition CSVs

---

## Project structure

```
RealEssate_forecast/
├── README.md
├── pyproject.toml                 # Dependencies & pytest config
├── configs/
│   └── config.yaml                # Data paths, CV, Optuna, orchestration gates
├── data/
│   ├── data_source.md             # Dataset notes
│   └── train/                     # Competition CSVs (gitignored — you add these)
│       ├── new_house_transactions.csv
│       ├── new_house_transactions_nearby_sectors.csv
│       └── pre_owned_house_transactions.csv
├── docker/
│   ├── api.Dockerfile             # FastAPI image
│   ├── app.Dockerfile             # Streamlit image
│   └── docker-compose.yml         # API + dashboard services
├── notebooks/
│   ├── eda.ipynb
│   ├── main_nb.ipynb
│   └── variable_dictionary.md
├── src/
│   ├── api/                       # FastAPI REST service
│   │   ├── main.py
│   │   ├── routes.py              # /health, /forecast, /predict, /drift, …
│   │   └── schemas.py
│   ├── app/
│   │   ├── streamlit_app.py       # Home page
│   │   ├── utils.py               # Upload/merge CSV, local predict helpers
│   │   └── pages/
│   │       ├── 1_📤_predict.py    # Upload 3 raw CSVs → merge → predict
│   │       ├── 2_📈_forecast.py   # Multi-month sector forecast
│   │       └── 3_📊_monitoring.py # Drift & metrics dashboard
│   ├── models/
│   │   ├── model_config.py        # Artifact paths (artifacts/)
│   │   ├── model_registry.py      # ONNX Runtime inference
│   │   └── retrain.py             # Final fit + save ONNX/pickles
│   ├── monitoring/
│   │   ├── detect_drift.py        # PSI, KS, concept drift
│   │   ├── reference.py           # Reference parquet for drift baseline
│   │   └── log_report.py          # Drift report JSON export
│   ├── pipeline/
│   │   ├── ingest_preprocess.py   # Load/merge CSVs, impute, train/test split
│   │   ├── features.py            # Lag, rolling, regime, sector features
│   │   ├── training.py            # Optuna + TimeSeriesSplit CV + MLflow
│   │   ├── evaluation.py          # Competition score & holdout metrics
│   │   ├── predict.py             # Recursive multi-month forecast
│   │   └── orchestrator.py        # Drift → retrain → registry workflow
│   └── utils/
│       └── config.py
├── tests/                         # pytest suite (unit + integration)
├── artifacts/                     # Model outputs (gitignored — created by training)
│   ├── model.onnx
│   ├── feature_list.pkl
│   ├── sector_stats.pkl
│   ├── sector_profile.pkl
│   ├── zero_sectors.pkl
│   └── reference.parquet
├── reports/                       # Drift monitoring JSON reports
├── mlruns/                        # MLflow experiment tracking
└── .github/workflows/
    ├── pr-checks.yml              # Tests, lint, Docker build on PR
    ├── build-and-push.yml         # Build & push images to GHCR on main
    └── orchestration.yml          # Scheduled drift check + optional retrain
```

---

## End-to-end workflow

### 1. Clone and set up environment (uv)

Install **uv** if you do not have it yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repo and create the project environment:

```bash
git clone https://github.com/lfcbsk/RealEssate_forecast.git
cd RealEssate_forecast

# Install Python 3.10 and create a local .venv (matches CI)
uv python install 3.10
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Activate the virtual environment (optional — you can use `uv run` instead):

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Run any command through uv without activating:

```bash
uv run python -m src.pipeline.training
uv run pytest tests/ -v
```

> All commands below assume you are in the project root with the uv environment installed. Prefix with `uv run` if the venv is not activated.

### 2. Download data

Place the three Kaggle CSVs under `data/train/`:

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle competitions download -c china-real-estate-demand-prediction -p data/train/
cd data/train && unzip china-real-estate-demand-prediction.zip
```

Expected files (see `configs/config.yaml` → `data.train_dir`):

| File | Description |
|------|-------------|
| `new_house_transactions.csv` | Main target sector transactions |
| `new_house_transactions_nearby_sectors.csv` | Nearby sector features |
| `pre_owned_house_transactions.csv` | Pre-owned market features |

### 3. Train the model

Full pipeline: ingest → feature engineering → Optuna tuning → 5-fold time-series CV → holdout eval → production model → save artifacts.

```bash
uv run python -m src.pipeline.training
```

This will:

1. Load and preprocess data from `data/train/`
2. Tune CatBoost hyperparameters (50 Optuna trials by default)
3. Run leakage-safe TimeSeriesSplit cross-validation
4. Evaluate on temporal holdout
5. Retrain on full data and write artifacts to `artifacts/`:
   - `model.onnx` — production ONNX model
   - `feature_list.pkl`, `sector_stats.pkl`, `sector_profile.pkl`, `zero_sectors.pkl`
6. Log experiments to MLflow (`mlruns/`, experiment: `catboost_timeseries`)

**Tune settings** in `configs/config.yaml`:

```yaml
optimization:
  n_trials: 50      # reduce for faster runs, e.g. 5
cv:
  n_splits: 5
```

**Programmatic training** (from Python):

```python
from src.pipeline.training import run_pipeline

results = run_pipeline(tune=True, n_trials=10)
print(results["test_results"])
```

### 4. Drift reference baseline (automatic)

Training **automatically** saves the drift reference after Step 6:

- `artifacts/reference.parquet` — baseline dataset for drift detection
- `artifacts/reference_stats.json` — column statistics

No manual step needed if you run:

```bash
uv run python -m src.pipeline.training
```

To refresh the baseline manually (e.g. after uploading new data without retraining):

```python
from src.pipeline.ingest_preprocess import run as ingest_run
from src.monitoring.reference import save_reference_dataset, save_reference_statistics

df_train, _ = ingest_run(test_ratio=0.2, save_outputs=False)
save_reference_dataset(df_train)
save_reference_statistics(df_train)
```

### 5. Run drift → retrain → registry orchestration

Triggered by GitHub Action (`orchestration.yml`) or manually:

```
GitHub Action
      ↓
orchestrator.py
      ↓
load data (CSVs)
      ↓
load prod model (artifacts/model.onnx)
      ↓
detect_data_drift()          ← src/monitoring/detect_drift.py
      ↓
severity == low?
      ├── YES → stop
      └── NO → retrain → evaluate → registry gate → promote
                                    ↓
                         artifacts/reference.parquet updated
```

```bash
# Fast retrain (no Optuna) — recommended for routine checks
uv run python -m src.pipeline.orchestrator --tune false --promote true

# Full retrain with Optuna tuning
uv run python -m src.pipeline.orchestrator --tune true --promote true --n-trials 10
```

- **Reference baseline:** `artifacts/reference.parquet` (auto-created on first run if missing; refreshed on promote)
- **Drift reports:** saved under `reports/`

**Registry gates** (`configs/config.yaml` → `orchestration.registry`):

| Gate | Default | Meaning |
|------|---------|---------|
| `min_competition_score` | 0.55 | Minimum holdout competition score |
| `min_r2` | 0.0 | Minimum R² |
| `max_mape` | 100.0 | Maximum MAPE (%) |
| `require_improvement_over_current` | false | New model must beat current score |

### 6. Start the API

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/forecast` | POST | Multi-month forecast (`{"n_months": 12}`) |
| `/api/v1/predict` | POST | Single-row prediction from features |
| `/api/v1/sectors` | GET | Sector list and zero-sector info |
| `/api/v1/metrics` | GET | MLflow run metrics |
| `/api/v1/drift` | GET | Drift report |
| `/api/v1/upload/raw` | POST | Upload 3 raw CSVs → merge into `data/train/` → predict |
| `/api/v1/upload` | POST | Batch predict from pre-engineered features (single file) |
| `/docs` | GET | Swagger UI |

**Example forecast:**

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"n_months": 12}'
```

### 7. Start the Streamlit dashboard

```bash
uv run streamlit run src/app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

| Page | Purpose |
|------|---------|
| **Home** | Overview & quick start |
| **📤 Upload & Predict** | Upload 3 raw CSVs → append/overwrite `data/train/` → merge → feature engineer → ONNX predict |
| **📈 Sector Forecast** | Recursive multi-month forecast from merged CSV data |
| **📊 Monitoring** | Drift, MLflow metrics, data file status |

**Upload flow (no separate database):**

```
User uploads 3 raw CSV files
        ↓
Append + overwrite duplicates (month, sector) → data/train/*.csv
        ↓
load_and_merge → create_training_features → ModelRegistry.predict
        ↓
Download predictions CSV
```

### 8. Run with Docker (optional)

```bash
cd docker
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Dashboard | http://localhost:8501 |

> **Note:** Training writes to `artifacts/`. Mount or copy `artifacts/` into containers before serving. Docker Compose currently mounts `models/` — symlink or copy `artifacts/` → `models/` if needed.

---

## Tests and CI

```bash
# Run all tests (same as CI)
uv run pytest tests/ -v -m "not e2e"

# With coverage (CI gate: 50%)
uv run pytest tests/ --cov=src --cov-fail-under=50
```

**GitHub Actions workflows:**

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `pr-checks.yml` | PR / push to `main`, `develop` | pytest, lint, Docker build |
| `build-and-push.yml` | push to `main`, tags `v*` | Build & push API/app images to GHCR |
| `orchestration.yml` | Weekly + manual dispatch | Drift check and optional retrain |

---

## Configuration reference

`configs/config.yaml`:

```yaml
data:
  train_dir: "../data/train/"

target:
  column: amount_new_house_transactions
  transform: log1p

cv:
  n_splits: 5

optimization:
  n_trials: 50

orchestration:
  drift:
    feature_drift_ratio_threshold: 0.2
    severity_for_retrain: ["medium", "high"]
  registry:
    min_competition_score: 0.55
    min_r2: 0.0
    max_mape: 100.0
```

---

## Typical first-time checklist

1. Install uv → `uv python install 3.10` → `uv venv --python 3.10` → `uv pip install -e ".[dev]"`
2. Download CSVs → `data/train/`
3. `uv run python -m src.pipeline.training` (also saves drift reference)
4. `uv run uvicorn src.api.main:app --reload`
5. `uv run streamlit run src/app/streamlit_app.py`
6. `uv run pytest tests/ -v` to verify everything works

---

## License

See repository for license details.
