# README.md for LTR Recommender

## Project Overview
This project implements an XGBoost-based Learning to Rank (LTR) recommender system. The goal is to provide personalized recommendations by efficiently ranking items based on user preferences.

## Algorithm Explanation
The XGBoost algorithm is a powerful gradient boosting method that can handle many different types of predictive modeling problems, including ranking tasks. In the context of LTR, it learns to rank items by optimizing the order of items presented to the user.

## Production Architecture
The production architecture consists of:
- Data Ingestion: Collects and prepares user interaction data.
- Model Training: Utilizes XGBoost for model training on historical data.
- Inference Service: Ranks items using the trained model to provide real-time recommendations.

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## **Complete Module Structure:**

1. **feature_engineering.pyx** - Cython-optimized feature engineering with nogil, prange, and multiprocessing
2. **model_trainer.pyx** - XGBoost LTR training with ranking metrics (NDCG, MAP, MRR)
3. **predictor.pyx** - Batch prediction and ranking with caching
4. **xgboost_ltr_wrapper.py** - Python API wrapper for all Cython modules
5. **pipeline.py** - Production pipeline with logging, tqdm, MLflow, error handling
6. **visualization.py** - 15+ comprehensive visualization plots
7. **metrics.py** - NDCG@K and ranking metrics with DuckDB storage
8. **explainability.py** - SHAP-based XAI with HTML report generation
9. **setup.py** - Cython compilation configuration
10. **config.ini** - Production-grade configuration file with all parameters

## **Key Features:**
- ✅ All Cython code uses `cdef`, `nogil`, and `prange` for performance
- ✅ DuckDB integration for data processing (no pandas for core)
- ✅ MLflow tracking for model versioning
- ✅ 15+ publication-ready visualizations
- ✅ SHAP explainability with HTML reports
- ✅ Zero hard-coded values - config-driven
- ✅ Production-grade error handling & logging
- ✅ Python properties/decorators for API classes
- ✅ C variable optimization with memory management