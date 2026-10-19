# StockPilot ML — Retail Demand Forecasting & Inventory Intelligence

StockPilot ML is an end-to-end machine learning and inventory decision-support application built around the UCI Online Retail II dataset. It combines historical retail transaction processing, demand forecasting experiments, inventory policy simulation, a FastAPI backend, and a React dashboard.

The goal is to investigate how historical retail sales can support inventory planning while evaluating forecasting models against simple baselines.

> **Research status:** This is a portfolio prototype, not a live warehouse management system. Inventory quantities and stock inputs are simulated. Historical sales are an imperfect proxy for true customer demand.

## Features

- **Retail data processing:** Clean transaction records, aggregate daily product demand, and construct product-level daily calendars.
- **Forecasting experiments:** Compare a zero-demand baseline, seasonal-naive forecasting, Random Forest, XGBoost, and Croston SBA.
- **Inventory policy evaluation:** Compare baseline, safety-stock, and activity-aware safety-stock policies using validation and held-out test periods.
- **Inventory recommendations:** Browse product-level recommendations, search and filter records, inspect product details, and export inventory records.
- **Evaluation API:** Serve saved model metrics, inventory simulation results, cost comparisons, and robustness reports through FastAPI.
- **Interactive dashboard:** Explore inventory recommendations and model evaluation using a React/Vite frontend.
- **Automated tests:** Validate API behavior and important edge cases.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas and NumPy
- **Machine learning:** scikit-learn, XGBoost, and Croston SBA baseline implementation
- **Backend:** FastAPI and Uvicorn
- **Frontend:** React, Vite, and Tailwind CSS
- **Testing:** pytest
- **Version control:** Git and GitHub

## Architecture

```text
UCI Online Retail II
        |
        v
Transaction cleaning and preprocessing
        |
        v
Daily product-demand calendar
        |
        v
Chronological train / validation / test split
        |
        +-------------------------------+
        |                               |
        v                               v
Forecasting model evaluation     Inventory policy simulation
        |                               |
        +---------------+---------------+
                        |
                        v
               Saved evaluation reports
                        |
                        v
                 FastAPI services
                        |
                        v
                 React dashboard
```

## Dataset

StockPilot ML uses the UCI Online Retail II dataset, containing retail transactions from two periods between 2009 and 2011.

The processed daily-demand calendar contains approximately 2.3 million product-day records across 4,984 products. The calendar includes dates with no recorded positive sales.

The chronological evaluation periods are:

| Split | Date range |
|---|---|
| Training | 2009-12-01 to 2011-05-01 |
| Validation | 2011-05-02 to 2011-08-20 |
| Test | 2011-08-21 to 2011-12-09 |

The original workbook is not committed to Git. Download the dataset from the UCI Machine Learning Repository and place it at the path expected by the preprocessing scripts.

**Data caveat:** A day with no recorded positive sale is not necessarily a day with zero underlying demand. Stockouts, product availability, cancellations, and returns can affect the observed sales data.

## Forecasting Evaluation

The following are the recorded validation results for a controlled comparison involving 500 products and 55,500 evaluation rows.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Zero baseline | 2.7435 | 28.8512 | 100.00% |
| XGBoost | 3.6873 | 28.0306 | 134.40% |
| Random Forest V2 | 3.7040 | 27.7569 | 135.01% |
| Seasonal Naive | 3.7840 | 37.6540 | 137.92% |
| Croston SBA | 4.1148 | 27.8478 | 149.98% |

Lower values are better for these metrics.

The zero baseline performs best on overall MAE and WAPE in this comparison. Other models show different trade-offs in RMSE and positive-day MAE. These results do not establish that machine learning improves overall demand forecasting; they identify where further experiments and error analysis are needed.

## Inventory Policy Simulation

Recorded fill-rate results:

| Policy | Validation fill rate | Test fill rate |
|---|---:|---:|
| Baseline average demand | 71.30% | 58.69% |
| Safety stock | 86.58% | 78.41% |
| Activity-aware safety stock | 84.96% | 73.53% |

The safety-stock policy reaches the 85% target on validation but not on the held-out test period.

The simulation uses assumed inventory inputs, lead times, review periods, service levels, and hypothetical cost parameters. These results are simulated comparisons, not measurements from a real warehouse.

## Project Structure

```text
stockpilot-ml/
├── app/
│   ├── main.py
│   └── services/
│       ├── evaluation_service.py
│       └── inventory_service.py
├── data/
│   └── raw/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── reports/
│   └── figures/
├── scripts/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Local Setup

### Prerequisites

- Python compatible with the project's dependencies
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Charan-dev247/stockpilot-ml.git
cd stockpilot-ml
```

### 2. Create and activate a Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download Online Retail II from the UCI Machine Learning Repository. Place the workbook at the location expected by the preprocessing scripts. Do not commit the raw dataset to Git.

Run preprocessing scripts in the documented pipeline order. Check each script's arguments and expected inputs before executing it.

### 4. Start the backend

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

The local API is available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, typically `http://localhost:5173`.

### 6. Run tests and build the frontend

From the project root:

```powershell
python -m pytest tests -v
```

For the frontend production build:

```powershell
cd frontend
npm run build
```

The build output is generated in `frontend/dist/`.

## API Evaluation Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/evaluation/models` | Retrieve forecasting model comparisons |
| `GET /api/evaluation/inventory?period=validation` | Retrieve validation inventory simulation results |
| `GET /api/evaluation/inventory?period=test` | Retrieve test inventory simulation results |
| `GET /api/evaluation/costs?period=validation` | Retrieve validation cost comparisons |
| `GET /api/evaluation/costs?period=test` | Retrieve test cost comparisons |
| `GET /api/evaluation/robustness` | Retrieve available robustness comparisons |

Consult `/docs` for the complete endpoint schemas and response formats.

## Limitations and Future Work

- Improve demand-target construction and distinguish observed sales from underlying demand.
- Investigate intermittent-demand forecasting and stronger baselines.
- Perform segment-level error analysis and investigate the effect of product inactivity.
- Tune inventory policies using validation data and evaluate them on held-out test data.
- Replace assumed inventory levels and costs with realistic operational inputs where available.
- Improve API observability, configuration management, deployment automation, and production error handling.

## Disclaimer

StockPilot ML is an educational and portfolio research prototype. It does not provide verified purchasing advice or guarantee inventory availability, service levels, or cost savings.

## Author

**Charan Sai**

GitHub: [Charan-dev247](https://github.com/Charan-dev247)
