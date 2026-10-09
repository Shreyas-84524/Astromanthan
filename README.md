# Astromanthan — Deep Sky Divide 🌌

**Machine Learning for Astronomical Star–Galaxy Classification**

Astromanthan is our project workspace for the **Deep Sky Divide: Star–Galaxy Classification Challenge**. Our goal is to build a reliable and interpretable machine-learning pipeline that distinguishes stars from galaxies using astronomical survey measurements.

The project combines data cleaning, feature analysis, model training, evaluation, and visualization to explore how machine learning can help classify large astronomical catalogues.

## 🎯 The Problem

Astronomical surveys contain enormous numbers of observed objects. Stars and galaxies may look similar in raw measurements, but their brightness and structural properties can contain useful patterns.

Real survey data can also contain missing values, invalid measurements, repeated sources, irrelevant attributes, and possible data leakage.

Our objective is to build a pipeline that can:

- Explore and preprocess astronomical survey data.
- Identify informative photometric and structural features.
- Classify stars and galaxies.
- Detect and handle problematic observations.
- Evaluate performance on unseen data.
- Explain model predictions and limitations.

## 🚀 Our Approach

```text
Astronomical Dataset
        ↓
Data Inspection and Cleaning
        ↓
Feature and Leakage Analysis
        ↓
Train and Compare ML Models
        ↓
Select the Best Model
        ↓
Evaluate on Unseen Test Data
        ↓
Feature Importance and Visualizations
        ↓
Predictions and Final Report
```

We use the existing Universal ML Engine as the reusable ML foundation and build the astronomy-specific workflow around the supplied datasets.

## 🧠 Technology Stack

- **Python** — core programming language
- **pandas and NumPy** — data handling and numerical analysis
- **scikit-learn** — preprocessing, baseline models, evaluation, and metrics
- **HistGradientBoosting, Random Forest, and Logistic Regression** — candidate classification approaches used in the current astronomy experiment
- **AutoGluon Tabular** — available in the general-purpose ML Engine when the required environment is installed
- **FastAPI** — local web interface and API
- **HTML, CSS, and JavaScript** — dashboard
- **matplotlib and seaborn** — plots and visual analysis

The exact models and dependencies used in a run may differ depending on the available environment and experiment configuration.

## 📂 Repository Structure

| Path | Purpose |
|---|---|
| `selected_datasets/` | Selected organizer-provided datasets |
| `outputs/` | Astronomy workflow, experiment reports, metrics, predictions, and plots |
| `ML Model/` | Reusable Universal ML Engine |
| `inspect_datasets.py` | Dataset inspection script |

## 🔬 Current Astronomy Experiment

The current workflow uses Dataset 29 as the primary structural classification dataset and Dataset 24 for a separate five-band transfer evaluation.

For the binary Star/Galaxy task, QSO objects are excluded rather than silently relabelled as stars.

The latest checked-in experiment report records the following results for its specific Dataset 29 split:

| Metric | Reported Result |
|---|---:|
| Final held-out macro-F1 | 0.9946 |
| Final held-out accuracy | 0.9948 |
| Final held-out ROC-AUC | 0.9978 |
| Selected model | HistGB_deeper |

**Important limitation:** The separate five-band model achieved a macro-F1 of 0.4780 when evaluated on Dataset 24. A strong result on Dataset 29 therefore does not establish reliable generalization across catalogues.

These are experiment-specific results, not guarantees for future runs. Refer to `outputs/REPORT.md` for the methodology, evaluation details, and limitations.

## ⚙️ Reproducing the Astronomy Workflow

### Requirements

- Python 3.12
- A compatible Python environment
- The required files from this repository

### 1. Create the environment

Run these commands from the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venvs\astronomy
.\.venvs\astronomy\Scripts\python.exe -m pip install --upgrade pip
.\.venvs\astronomy\Scripts\python.exe -m pip install -r .\outputs\requirements-lock.txt
```

### 2. Run the pipeline

Use a separate output directory so the checked-in experiment results are not overwritten:

```powershell
.\.venvs\astronomy\Scripts\python.exe -u .\outputs\pipeline.py --output outputs/reproduction
```

Generated reports, metrics, predictions, and plots will be written to the specified output directory.

### 3. Predict using the saved astronomy model

```powershell
.\.venvs\astronomy\Scripts\python.exe .\outputs\predict.py --csv ".\selected_datasets\Dataset 29.csv"
```

This uses the saved astronomy model rather than retraining it. Verify that the input dataset has the expected feature schema before using it.

## 🖥️ Universal ML Engine Dashboard

The reusable ML Engine has a local web dashboard for uploading datasets, inspecting columns, selecting a target, training and evaluating models, viewing visualizations, downloading reports, and making predictions.

Its documentation is available in [`ML Model/README.md`](ML%20Model/README.md).

Start with the [astronomy workflow instructions](outputs/README.md) for experiment-specific commands.

## 🛡️ Evaluation Principles

- Fit data-dependent preprocessing using training data only.
- Use development/validation data for model selection.
- Keep final-test data out of model selection and tuning.
- Treat feature importance as predictive evidence, not proof of causation.
- Report actual measured results and document limitations.
- Do not claim discovery of rare astronomical objects unless the task and evidence support that claim.

## 🔭 Future Improvements

- Validate catalogue labels and measurement units using authoritative data documentation.
- Improve cross-catalogue identity matching.
- Investigate domain shift and robustness.
- Evaluate class-wise errors and probability calibration.
- Improve the presentation of scientific findings and limitations.

## 🔗 Related Project

**Universal ML Engine:**  
https://github.com/SamSurve/universal-ml-engine

This repository contains the reusable ML platform used by the astronomy-specific workflow.

---

*Built for the Astromanthan Deep Sky Divide challenge.*
