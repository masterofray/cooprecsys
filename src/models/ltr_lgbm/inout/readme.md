# LightGBM Learning to Rank Pipeline

Pipeline untuk product recommendation menggunakan LightGBM LambdaRank dengan fallback similarity mapping.

## Structure

```
/workspace/
├── main.py                      # Preprocessing + Training pipeline
├── data_preprocessing.py        # LabelEncoder & Feature processing
├── model_inference.py           # LGBM inference & ranking
├── recommendation_handler.py    # Top-K recommendations + fallback
├── user_input_files/
│   ├── sampledata.parquet       # Raw input data
│   └── inference.py             # Original inference script (reference)
└── output/
    ├── ltr_model.txt            # Trained LightGBM model
    ├── encoders/encoders.pkl    # Label encoders (for decoding)
    ├── feature_columns.json     # Feature column names
    ├── group_sizes.json         # Group sizes for train/test
    ├── processed/
    │   ├── train.parquet        # Training data
    │   └── test.parquet         # Test data
    └── recommendations/
        └── recommendations_top20.parquet  # Final recommendations
```

## Usage

### 1. Run Full Pipeline (Preprocessing + Training)

```bash
python main.py
```

Ini akan:
- Load data dari `sampledata.parquet`
- Convert date/time columns
- Engineer 40 features
- Encode strings & save encoders
- Split train/test (80/20)
- Train LightGBM LambdaRank model
- Simpan semua output ke `output/`

### 2. Generate Recommendations (Existing Model)

```python
from model_inference import create_inference_engine
from recommendation_handler import RecommendationHandler, CustomerHistoryHandler
import pandas as pd

# Load candidate data
candidate_df = pd.read_parquet('output/processed/test.parquet')

# Create inference engine
engine = create_inference_engine(top_k=20)

# Create recommendation handler
handler = RecommendationHandler()
handler.inference_engine = engine
handler.customer_history = CustomerHistoryHandler(candidate_df)

# Generate recommendations
recommendations = handler.generate_recommendations(
    candidate_data=candidate_df,
    top_k=20,
    use_fallback=True
)

# Save with decoded strings
handler.save_recommendations(decode=True)
```

### 3. Quick Inference Only

```python
from model_inference import LTRModelInference

engine = LTRModelInference()
engine.load_all()

# Score and rank
ranked = engine.rank_top_k(df, top_k=20)
ranked.to_parquet('output/rankings.parquet')
```

### 4. Load Encoders for Custom Decoding

```python
from data_preprocessing import load_encoders

manager = load_encoders()
classes = manager.get_classes('ProductName')
print(f"Product names: {classes[:5]}")

# Decode predictions
for idx, row in predictions.iterrows():
    product = classes[row['ProductName_encoded']]
    print(f"{row['CustomerID']}: {product}")
```

## Configuration

Edit `PreprocessingConfig` class in `main.py` untuk mengubah:

```python
class PreprocessingConfig:
    DATE_COLUMNS = ["SalesDate", "Product_ModifyDate"]
    TIME_COLUMNS = ["SalesHours"]
    DROP_COLUMNS = ["FirstName", "SalesID", "EmployeeFirstName", "CountryName"]
    TARGET_COLUMN = "Quantity"
    QUERY_ID_COLUMN = "CustomerID"
    TOP_K = 20
```

Edit `RecommendationConfig` class in `recommendation_handler.py` untuk fallback settings:

```python
class RecommendationConfig:
    FALLBACK_SIMILARITY_FIELDS = ["CategoryID", "TotalPrice", "CityName"]
```

## Output Format

`recommendations_top20.parquet` columns:

| Column | Description |
|--------|-------------|
| CustomerID | Customer identifier |
| ProductName | Decoded product name |
| CategoryID | Product category |
| TotalPrice | Product price |
| CityName | Customer city |
| score | LGBM relevance score |
| rank | Rank position (1-20) |
| is_fallback | True if from similarity fallback |

## Top-K Recommendation Logic

1. **LGBM LambdaRank** predicts relevance scores for all candidate items
2. For each customer, rank items by score (descending)
3. Take top-k items per customer
4. **If < k items** (cold-start or insufficient predictions):
   - Get customer purchase history
   - Extract favorite categories, avg price, city
   - Find similar products from catalog
   - Fill gap to reach k items

## Feature Engineering Details

### Date Features (per date column)
- `_year`: Year component
- `_month`: Month (1-12)
- `_day`: Day of month
- `_dayofweek`: Day of week (0=Monday, 6=Sunday)
- `_quarter`: Quarter (1-4)
- `_is_weekend`: Binary weekend flag
- `_days_since_ref`: Days since reference date

### Time Features (per time column)
- `_hour`: Hour (0-23)
- `_minute`: Minute (0-59)
- `_second`: Second
- `_period`: Categorical period (0=Night, 1=Morning, 2=Afternoon, 3=Evening)
- `_is_business_hours`: Binary 9-17 flag
- `_minute_of_day`: Minutes since midnight

### Interaction Features
- `price_per_unit`: ProductPrice × (1 - Discount)
- `total_value`: ProductPrice × Quantity
- `employee_experience_ratio`: YearsWorking / (EmployeeAge + 1)
- `avg_price_per_item`: TotalPrice / (Quantity + 1)

## Requirements

```bash
pip install pandas numpy scikit-learn lightgbm tqdm pyarrow fastparquet
```

