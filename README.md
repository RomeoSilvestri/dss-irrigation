# ML - DSS: Soil Moisture Prediction & Decision Support System for Irrigation Scheduling

**Repository Name:** dss-irrigation
**Author:** Romeo Silvestri

## Overview

This repository contains machine learning models developed to predict soil moisture levels using tensiometer data and to create a Decision Support System (DSS) for irrigation management. The project is part of the IRRITRE initiative by FBK (Fondazione Bruno Kessler), aimed at optimizing irrigation practices for various agricultural consortia in the Trentino-Alto Adige region.

## Project Structure

The project follows a standard Python project structure:

- `data/`: Contains all datasets (CSV, XLSX) used for training and validation.
- `notebooks/`: Jupyter notebooks for research, demonstration, and evaluation.
- `scripts/`: Python scripts for training models and running the main application.
- `src/dss_irrigation/`: Core package containing the logic for different DSS approaches:
    - `anfis/`: Adaptive Neuro-Fuzzy Inference System implementation.
    - `mamdani/`: Mamdani Fuzzy Logic implementation and supporting packages.
    - `utils/`: Utility functions.
- `pyproject.toml`: Project configuration and dependency management using `uv`.

```
dss-irrigation
├── data/
├── notebooks/
├── scripts/
├── src/
│   └── dss_irrigation/
│       ├── anfis/
│       ├── mamdani/
│       └── utils/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Requires Python 3.11 or higher.

1. Clone the repository:
   ```bash
   git clone https://github.com/romeo_silvestri/dss-irrigation.git
   cd dss-irrigation
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### Running Scripts
You can run the main scripts from the root directory:
```bash
uv run scripts/main.py
```

### Jupyter Notebooks
To explore the notebooks, start a Jupyter server:
```bash
uv run jupyter lab
```
Navigate to the `notebooks/` directory.

## Content

This project employs various machine learning models to predict soil moisture levels, including:
- Linear Regression
- XGBoost
- ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Mamdani Fuzzy Logic DSS

The DSS provides actionable insights to farmers by defining irrigation rules based on predicted soil moisture and weather data.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

For more information or any queries, please contact Romeo Silvestri at rsilvestri@fbk.eu.