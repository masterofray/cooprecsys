'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

import pandas as pd
import duckdb
import os
from pathlib import Path

def load_data(data_path):
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Get file extension
    file_extension = Path(data_path).suffix.lower()
    
    try:
        # Load based on file extension
        if file_extension == '.csv':
            print(f"Loading CSV file: {data_path}")
            df = pd.read_csv(data_path)
            
        elif file_extension == '.parquet':
            print(f"Loading Parquet file: {data_path}")
            df = pd.read_parquet(data_path)
            
        elif file_extension == '.db':
            print(f"Loading DuckDB file: {data_path}")
            conn = duckdb.connect(data_path)
            tables = conn.execute("SHOW TABLES").fetchall()
            if not tables:
                raise ValueError(f"No tables found in database: {data_path}")
            
            # If multiple tables, use the first one or you can modify logic
            # to handle specific table names
            table_name = tables[0][0]
            print(f"Loading table '{table_name}' from database")
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            conn.close()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .parquet, .db")
        
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {str(e)}")

if __name__ == "__main__":
    pass
