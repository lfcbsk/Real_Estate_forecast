# RealEssate_forecast

## Project Structure

```
RealEssate_forecast/
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore file
├── Makefile                           # Make commands for development
├── pyproject.toml                     # Python project configuration
├── requirement.txt                    # Python dependencies
├── .github/                           # GitHub configuration
│   └── workflows/                     # CI/CD workflows
│       ├── pr-checks.yml              # Pull request checks
│       ├── ci.yml                     # Continuous Integration
│       ├── cd.yml                     # Continuous Deployment
│       └── build-and-push.yml         # Docker build and push
├── configs/
│   └── config.yaml                    # Configuration settings
├── data/
│   ├── data_source.md                 # Data source documentation
│   
├── docker/                            # Docker configuration
│   ├── api.Dockerfile
│   ├── app.Dockerfile
│   └── docker-compose.yml
├── k8s/                               # Kubernetes configuration
│   ├── configmap.yaml
│   ├── cronjob.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── monitor/                       # Monitoring configuration
│       ├── grafana-dashboard.json
│       ├── grafana.yaml
│       └── prometheus.yaml
├── notebooks/                         # Jupyter notebooks
│   ├── eda.ipynb                      # Exploratory Data Analysis
│   ├── main_nb.ipynb                  # Main analysis notebook
│   ├── submission.csv
│   ├── variable_dictionary.md
├── src/                               # Source code
│   ├── api/                           # API service
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── app/                           # Streamlit application
│   │   └── streamlit_app.py
│   ├── models/                        # Model management
│   │   ├── model_config.py
│   │   ├── model_registry.py
│   │   └── retrain.py
│   ├── monitoring/                    # Monitoring utilities
│   │   ├── dectect_drift.py
│   │   ├── log_report.py
│   │   └── reference.py
│   └── pipeline/                      # Data pipeline
│       ├── evaluation.py
│       ├── features.py
│       ├── ingest.py
│       ├── predict.py
│       ├── preprocess.py
│       └── training.py
└── tests/                             # Test suite
    ├── conftest.py
    ├── test_evaluate.py
    └── test_features.py
```

## Directory Descriptions

- **.gitignore**: Git configuration to exclude files from version control
- **Makefile**: Development commands and task automation
- **pyproject.toml**: Python project metadata and build configuration
- **requirement.txt**: Python package dependencies
- **.github/**: GitHub configuration and CI/CD workflows
  - **workflows/**: Automated workflow files for PR checks, CI, CD, and Docker builds
- **configs/**: Project configuration files
- **data/**: Training and test datasets
- **docker/**: Docker and Docker Compose configuration for containerization
- **k8s/**: Kubernetes manifests for deployment and monitoring
- **notebooks/**: Jupyter notebooks for analysis and experimentation
- **src/**: Main source code organized by module
  - **api/**: FastAPI application endpoints and schemas
  - **app/**: Streamlit web application
  - **models/**: Machine learning model management and retraining
  - **monitoring/**: Data drift detection and monitoring
  - **pipeline/**: Data processing and model training pipeline
- **tests/**: Unit tests for the project
