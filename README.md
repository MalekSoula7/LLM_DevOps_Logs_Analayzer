# DevOps Log Analyzer

A production-grade MLOps system that ingests CI/CD pipeline logs, classifies
their failure root cause using sentence-transformer embeddings and a scikit-learn
classifier, monitors for data drift using Evidently, auto-triggers retraining via
Prefect when drift is detected, and explains each prediction via the HuggingFace
Inference API — all exposed through a FastAPI backend and a Streamlit dashboard.

---

## Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │              Data Layer                       │
                      │  data/raw/logs.csv  ←  generate_data.py      │
                      │  data/raw/embeddings.npy  ←  embed.py        │
                      │  data/drift/  ←  drift simulation            │
                      └───────────────┬──────────────────────────────┘
                                      │
                      ┌───────────────▼──────────────────────────────┐
                      │           Training Pipeline (Prefect)         │
                      │  generate → embed → train → evaluate → reg.  │
                      │  flows/training_flow.py                       │
                      │  ↓ logs to MLflow  ↓ registers model         │
                      └───────────────┬──────────────────────────────┘
                                      │
               ┌──────────────────────┼────────────────────────────┐
               │                      │                            │
   ┌───────────▼────────┐  ┌──────────▼───────────┐  ┌────────────▼──────────┐
   │  MLflow Registry   │  │   Evidently Monitor   │  │  HuggingFace Explain  │
   │  localhost:5000    │  │   src/monitor.py       │  │  src/explain.py       │
   └───────────┬────────┘  └──────────┬────────────┘  └────────────┬──────────┘
               │                      │                             │
               └──────────────────────▼─────────────────────────────┘
                                      │
                      ┌───────────────▼──────────────────────────────┐
                      │           FastAPI Backend                     │
                      │  api/main.py  ←→  localhost:8000             │
                      │  POST /classify  GET /drift-report           │
                      │  GET /metrics   GET /pipeline-status         │
                      └───────────────┬──────────────────────────────┘
                                      │
                      ┌───────────────▼──────────────────────────────┐
                      │        Streamlit Dashboard                    │
                      │  dashboard/app.py  ←→  localhost:8501        │
                      │  Classify Log | Drift | Metrics | History    │
                      └──────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer          | Technology                                 | Version   |
|----------------|--------------------------------------------|-----------|
| Embeddings     | sentence-transformers (all-MiniLM-L6-v2)   | 3.0.1     |
| Classifier     | scikit-learn (LogisticRegression + RF)     | 1.5.1     |
| Experiment Tracking | MLflow                                | 2.14.1    |
| Orchestration  | Prefect                                    | 3.0.0     |
| Drift Detection | Evidently                                 | 0.4.30    |
| API            | FastAPI + Uvicorn                          | 0.112.0   |
| Dashboard      | Streamlit                                  | 1.37.0    |
| Explainability | HuggingFace Inference API (Mistral-7B)     | REST API  |
| Runtime        | Python                                     | 3.11+     |

---

## Setup (Windows)

### 1. Prerequisites

- Python 3.11+ (https://python.org)
- Git

### 2. Clone and install

```bat
git clone <your-repo-url>
cd devops-log-analyzer

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bat
copy .env.example .env
```

Edit `.env` and set your HuggingFace token:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free token at https://huggingface.co/settings/tokens

### 4. Run everything at once

```bat
run_all.bat
```

Open http://localhost:8501 for the dashboard.

---

## Running Components Individually

### Generate synthetic training data
```bat
python src/generate_data.py
```

### Compute embeddings
```bat
python src/embed.py
```

### Train the classifier
```bat
python src/train.py
```

### Run quality gate
```bat
python src/evaluate.py
```

### Run full training pipeline (Prefect)
```bat
python flows/training_flow.py
```

### Run inference pipeline
```bat
python flows/inference_flow.py data/raw
```

### Detect drift
```bat
python src/monitor.py
```

### Start the API server
```bat
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the dashboard
```bat
streamlit run dashboard/app.py
```

### Run tests
```bat
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint            | Description                                          |
|--------|---------------------|------------------------------------------------------|
| GET    | /health             | API health, model loaded, MLflow connected           |
| POST   | /classify           | Classify a log → label, confidence, explanation      |
| GET    | /drift-report       | Latest Evidently drift result (JSON)                 |
| GET    | /drift-report/html  | Latest Evidently HTML report (served as HTML page)   |
| GET    | /pipeline-status    | Last 10 inference runs with metadata                 |
| GET    | /metrics            | Current model performance (F1, per-class breakdown)  |
| POST   | /simulate-drift     | Run inference on drift dataset and detect drift      |

Interactive docs: http://localhost:8000/docs

---

## Project Structure

```
devops-log-analyzer/
├── data/
│   ├── raw/               ← Training logs + embeddings + model artifacts
│   └── drift/             ← Drift simulation logs + embeddings
├── src/
│   ├── generate_data.py   ← Synthetic log generator (600 training + 150 drift)
│   ├── embed.py           ← Sentence-transformer embedding with caching
│   ├── train.py           ← MLflow-instrumented classifier training
│   ├── evaluate.py        ← Quality gate: fails if F1 < 0.85
│   ├── monitor.py         ← Evidently drift detection
│   ├── explain.py         ← HuggingFace Inference API explanations
│   └── retrain_trigger.py ← Reads drift result, triggers Prefect flow
├── flows/
│   ├── training_flow.py   ← Prefect: generate → embed → train → evaluate → register
│   └── inference_flow.py  ← Prefect: load → embed → classify → explain → save → monitor
├── api/
│   ├── main.py            ← FastAPI application
│   └── schemas.py         ← Pydantic v2 request/response models
├── dashboard/
│   └── app.py             ← Streamlit 4-page dashboard
├── reports/               ← Evidently HTML reports + drift_result.json
├── results/               ← Inference output JSON files
├── tests/
│   ├── test_data.py       ← Data generation tests
│   ├── test_model.py      ← Embedding + classifier tests
│   └── test_api.py        ← FastAPI endpoint tests
├── .env.example           ← Environment variable template
├── requirements.txt       ← Pinned dependencies
└── run_all.bat            ← Windows one-shot launcher
```

---

## Screenshots

<!-- screenshot: Streamlit dashboard — Classify Log page showing build_failure prediction with confidence bar -->

<!-- screenshot: Streamlit dashboard — Drift Monitor page showing red DRIFT DETECTED banner -->

<!-- screenshot: MLflow UI showing experiment runs with F1 metrics and confusion matrix artifact -->

<!-- screenshot: Prefect UI showing completed training-pipeline flow run -->

<!-- screenshot: FastAPI /docs page with all endpoints expanded -->

---

## Measured Outcomes

*Fill in after first successful run:*

| Metric                         | Value     |
|--------------------------------|-----------|
| Best model algorithm           | —         |
| Weighted F1 (test set)         | —         |
| Lowest per-class F1            | —         |
| Training time (600 samples)    | —         |
| Embedding time (600 samples)   | —         |
| API classify latency (p50)     | —         |
| Drift detected on drift set    | —         |

---

## Author

**Malek Soula** — DevOps / MLOps Engineer  
soulamalek00@gmail.com
