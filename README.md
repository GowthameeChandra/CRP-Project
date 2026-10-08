# Model training: CatBoost Classification with Snowflake Integration

## Overview

Optune Model is a comprehensive machine learning pipeline for building and monitoring a **CatBoost classification model** trained on data from Snowflake. The project integrates:
- **Snowflake data ingestion** with secure credential management
- **CatBoost classifier** with hyperparameter tuning via Optuna
- **Threshold optimization** for balanced classification performance
- **Model monitoring** using Evidently for production drift detection
- **Cross-validation** and detailed evaluation metrics

This is a production-ready pipeline designed for change failure prediction on ServiceNow change management data, extensible for other classification use cases.

## Folder Structure

```
Optune_model/
├── app/                          # Main application modules
│   ├── __init__.py
│   ├── database/                 # Snowflake connectivity
│   │   ├── __init__.py
│   │   ├── snowflake_connection.py   # Snowflake session management
│   │   ├── snowflake_creds.py        # Credentials loading
│   │   └── snowflake_decrypt.py      # Private key decryption
│   └── training/                 # Model training
│       ├── __init__.py
│       ├── model_algorithm.py    # CatBoost wrapper with Optuna/GridSearch support
│       ├── preprocessed_data.py  # Data preprocessing & feature handling
│       └── trainer.py            # Main training orchestration
│
├── monitoring/                   # Production monitoring & drift detection
│   ├── evidently_monitor.py      # Evidently report generation
│   ├── monitoring_config.json    # Monitoring configuration (dates, splits)
│   ├── input/                    # Input predictions for monitoring
│   │   ├── development_prediction.csv
│   │   └── production_current_prediction.csv
│   └── output/                   # Generated monitoring reports
│       ├── evidently_report.html
│       ├── evidently_report.json
│       ├── evidently_monitoring_report.csv
│       └── evidently_monitoring_report.xlsx
│
├── artifacts/                    # Trained models and results
│   ├── model.cbm                 # Serialized CatBoost model
│   └── threshold_tuning.csv      # Threshold tuning results
│
├── catboost_info/                # CatBoost training logs & metrics
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── test_error.tsv
│   ├── time_left.tsv
│   ├── fold-0/, fold-1/, fold-2/ # Cross-validation fold logs
│   └── learn/, test/             # TensorBoard event files
│
├── train.py                      # Main training entry point
├── train_config.json             # Dataset & feature configuration
├── threshold_tuning.py           # Threshold optimization utility
├── dep_check.json                # Dependency verification config
├── pyproject.toml                # Project metadata & dependencies
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## What the Pipeline Does

### Training Pipeline (`train.py`)
1. **Load Configuration**: Reads `train_config.json` for dataset, features, and model settings
2. **Connect to Snowflake**: Authenticates using private key from `app/database/snowflake_creds.py`
3. **Fetch Training Data**: Executes configured SQL queries from Snowflake
4. **Preprocess Data**:
   - Removes duplicates and handles missing values
   - Auto-detects categorical features (or uses configured list)
   - Encodes categorical variables for CatBoost
5. **Data Splitting**:
   - Production split: Last 3 months → production data
   - Remaining data: Stratified split into train/validation/test sets
6. **Train CatBoost Model**:
   - Uses hyperparameter configuration from `train_config.json`
   - Supports Optuna hyperparameter optimization
   - Performs cross-validation (default: 3-fold)
7. **Evaluate Model**:
   - Computes metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision
   - Generates evaluation reports
8. **Save Artifacts**: Serializes model to `artifacts/model.cbm`

### Threshold Tuning (`threshold_tuning.py`)
- Evaluates model performance across probability thresholds (0.05 to 0.55)
- Computes weighted scores combining Precision, Recall, and F1
- Identifies optimal threshold for balanced classification
- Generates interactive Plotly visualization (`artifacts/threshold_tuning.html`)

### Monitoring Pipeline (`monitoring/evidently_monitor.py`)
- Compares baseline (development) vs. production predictions
- Detects data drift, prediction drift, and performance degradation
- Generates reports in HTML, JSON, and CSV formats
- Supports rolling 6-week batch monitoring

## Prerequisites

- **Python**: 3.10 or newer
- **Windows PowerShell** or compatible terminal
- **Snowflake Account**: Access to target database and tables
- **Private Key**: For Snowflake authentication (if using key-pair)

## Setup in Windows PowerShell

### 1. Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure Snowflake Credentials

Edit `app/database/snowflake_creds.py`:

```python
SF_ACCOUNT = "your_account"
SF_USER = "your_user"
SF_WAREHOUSE = "your_warehouse"
SF_DATABASE = "XDS"
SF_SCHEMA = "public"
SF_ROLE = "your_role"
SF_PRIVATE_KEY_PATH = "path/to/private_key"
SF_PRIVATE_KEY_PASSPHRASE_PATH = "path/to/passphrase.txt"
SF_TEST_TABLE = "snow_change_failure_x"
```

### 4. Configure Training Dataset

Edit `train_config.json`:
- Database name, table, and joins
- Date column for time-based splits
- Production split window (months)
- Feature definitions (numeric, categorical, text)
- Business ID and key columns

Example:
```json
{
  "dataset": {
    "dbconfig": {
      "db_name": "XDS",
      "table_name": "snow_change_failure_x",
      "table_alias": "cf",
      "filters": "cf.change_failure_flag in ('Y','N') and cf.__sys_soft_deleted_flag='N'"
    },
    "date_column": "OPENED_DATETIME",
    "production_split": { "months": 3 },
    "features": [
      { "name": "change_lead_duration", "type": "numeric" },
      { "name": "change_type", "type": "categorical" }
    ]
  }
}
```
## Running the Pipeline

### Train the Model

```powershell
python train.py
```

This will:
1. Load and validate configuration from `train_config.json`
2. Connect to Snowflake and fetch training data
3. Preprocess features and handle missing values
4. Split data (production / train / validation / test)
5. Train CatBoost with cross-validation
6. Evaluate on test set and save model to `artifacts/model.cbm`
7. Generate training logs in `catboost_info/`

### Run Threshold Tuning

```powershell
python threshold_tuning.py
```

Outputs:
- `artifacts/threshold_tuning.csv`: Threshold evaluation results
- `artifacts/threshold_tuning.html`: Interactive visualization

### Run Monitoring

```powershell
python monitoring/evidently_monitor.py
```

Generates comprehensive drift reports comparing development vs. production predictions.

## Configuration Files

### `train_config.json` - Main Training Configuration

**Key Sections:**

1. **`dataset.dbconfig`** - Snowflake query definition:
   - `db_name`, `table_name`, `table_alias`
   - `joins`: Optional table joins with conditions
   - `filters`: WHERE clause for data filtering

2. **`dataset.features`** - Feature definitions:
   ```json
   {
     "name": "feature_name",
     "type": "numeric|categorical|text"
   }
   ```

3. **`dataset.date_column`** - Column for time-based train/prod split

4. **`dataset.production_split.months`** - Lookback window for production data (default: 3)

5. **`catboost`** - CatBoost hyperparameters:
   - `iterations`: Number of boosting rounds
   - `learning_rate`: Gradient boosting learning rate
   - `depth`: Tree depth
   - `loss_function`: "Logloss" for binary classification
   - See [CatBoost docs](https://catboost.ai/docs/concepts/python-reference_parameters_training.html)

### `monitoring/monitoring_config.json` - Monitoring Setup

- `split_option`, `split_type`: Partitioning strategy
- `monitoring_mode`: "fortnight" or other frequency
- `batch_mode`: "rolling_6_week" for batch-based monitoring
- `baseline_prediction_path`, `current_prediction_path`: CSV file paths
- Date ranges for baseline and current monitoring windows

## Dependencies

All dependencies are listed in `requirements.txt`:

- **catboost**: Gradient boosting classifier
- **snowflake-snowpark-python**: Snowflake Snowpark API
- **snowflake-connector-python**: Snowflake connection
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML utilities (train/test split, metrics)
- **optuna**: Hyperparameter optimization
- **plotly**: Interactive visualizations
- **evidently**: ML monitoring & drift detection
- **cryptography**: Encrypt/decrypt Snowflake credentials

Install all at once:
```powershell
pip install -r requirements.txt
```

## Output Artifacts

| File | Purpose |
|------|---------|
| `artifacts/model.cbm` | Serialized CatBoost model (binary format) |
| `artifacts/threshold_tuning.csv` | Threshold metrics across probability ranges |
| `artifacts/threshold_tuning.html` | Interactive visualization of threshold performance |
| `catboost_info/catboost_training.json` | Training metadata and parameters |
| `catboost_info/learn_error.tsv`, `test_error.tsv` | Error metrics per iteration |
| `catboost_info/fold-*/` | Per-fold training logs (cross-validation) |
| `monitoring/output/evidently_report.html` | Drift detection report (interactive) |
| `monitoring/output/evidently_report.json` | Machine-readable drift metrics |

## Key Classes & Modules

### `app/training/trainer.py` - CatBoostTrainer
Orchestrates the entire training workflow:
- Data loading and validation
- Preprocessing and feature engineering
- Model training with configurable CV strategy
- Evaluation and metric computation

### `app/training/model_algorithm.py` - SupervisedModel
Wraps CatBoost training logic:
- Supports Optuna hyperparameter tuning
- GridSearchCV for discrete parameter grids
- Cross-validation with StratifiedKFold
- Metric computation (Precision, Recall, F1, ROC-AUC)

### `app/training/preprocessed_data.py` - PreprocessedData
Handles data preparation:
- Duplicate removal
- Missing value handling
- Categorical feature detection & encoding
- Train/validation/test splitting

### `app/database/snowflake_connection.py` - SnowflakeConnection
Manages Snowflake connectivity:
- Session initialization with private key auth
- Query execution and result caching
- Connection pooling and cleanup

### `threshold_tuning.py` - Threshold Optimization
Utility for finding optimal decision threshold:
- Evaluates performance across threshold grid
- Computes weighted metrics (Precision + Recall + F1)
- Detects inflection points and curvature
- Generates Plotly-based visualizations

### `monitoring/evidently_monitor.py` - Evidently Monitoring
Drift detection and model performance monitoring:
- Classification preset metrics
- Data drift detection
- Prediction drift monitoring
- Comparison reports (baseline vs. current)

## Troubleshooting

### Snowflake Connection Issues
- Verify credentials in `app/database/snowflake_creds.py`
- Check private key file path and passphrase
- Ensure Snowflake account, warehouse, and database are accessible

### CatBoost Training Failures
- Check for missing values in features
- Verify categorical feature names match data
- Review CatBoost logs in `catboost_info/`
- Ensure sufficient memory for dataset size

### Model Not Saving
- Verify `artifacts/` directory exists
- Check disk space availability
- Ensure write permissions in artifacts folder

### Monitoring Errors
- Verify prediction CSV files exist at paths in `monitoring_config.json`
- Check date formats in prediction files
- Ensure baseline and current date ranges don't overlap

## Contributing

To extend the pipeline:
1. Add new preprocessing steps in `app/training/preprocessed_data.py`
2. Implement custom metrics in training modules
3. Extend Evidently monitoring with custom checks
4. Add new hyperparameter configurations to `train_config.json`

## License

This project is proprietary and intended for internal use only.

## Support

For issues or questions, refer to project documentation or contact the ML team.

## Commands

Ingest data only:

```powershell
python main.py ingest
```

Train using the latest raw snapshot. If no local snapshot exists, the pipeline ingests from Snowflake first:

```powershell
python main.py train
```

Evaluate the saved model against the latest raw snapshot:

```powershell
python main.py evaluate
```

Run full ingestion plus training end to end:

```powershell
python main.py run
```

Run tests:

```powershell
pytest
```

## Generated Artifacts

After training or evaluation, the pipeline writes these files to `artifacts/`:

- `model.cbm`
- `metrics.csv`
- `feature_importance.csv`
- `prediction_probabilities.csv`
- `classification_report.txt`
- `confusion_matrix.png`

## Logging

The pipeline writes logs to `logs/training.log` and records these events:

- Database Connection
- SQL Execution
- Data Loading
- Preprocessing
- Training Started
- Training Completed
- Evaluation
- Artifact Saving
- Errors raised during execution

## Expected Workflow

1. Create and activate the virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and fill in Snowflake credentials.
4. Update `sql/training_query.sql` with your production query.
5. Update `app/config/config.json` with the correct target column and CatBoost parameters.
6. Run `python main.py run`.
7. Review outputs in `artifacts/` and logs in `logs/training.log`.

## Notes for Future Enhancements

The current layout isolates ingestion, preprocessing, training, evaluation, and artifact persistence so you can add Optuna tuning, SHAP analysis, threshold optimization, or model versioning later without changing the package structure.
