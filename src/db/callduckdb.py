#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"

import gc
import os
import sys
import duckdb
import configparser
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Union

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger

class DuckDBManager:
    """
    Manager class for DuckDB connections and queries.

    Example:
        db = DuckDBManager('mydb.duckdb')
        df = db.query("SELECT * FROM users")
        db.close()

        # Or use context manager:
        with DuckDBManager('mydb.duckdb') as db:
            df = db.query("SELECT * FROM users")
    """
    def __init__(self,
                 db_path      : Union[str, Path] = None,
                 read_only    : bool             = True,
                 threads      : Optional[int]    = None,
                 memory_limit : Optional[str]    = '4GB'):
        """
        Initialize DuckDB manager.

        Args:
            db_path      : Path to database file (use ':memory:' for in-memory DB)
            read_only    : Open in read-only mode
            threads      : Number of CPU threads to use
            memory_limit : Memory limit (e.g., '4GB', '1TB')
        """
        if db_path is None:
            logger.warning('Your dpath is Empty, send data into memory!')
            db_path    = ':memory:'
        self.db_path   = str(db_path)
        self.read_only = read_only
        self.conn      = None
        self._create_connection(threads, memory_limit)


    def _create_connection(self, 
        threads      : Optional[int] = None,
        memory_limit : Optional[str] = None):
        """Create DuckDB connection with optional settings."""
        is_read_only = False if self.db_path == ':memory:' else self.read_only
        self.conn    = duckdb.connect(database  = self.db_path, 
                                      read_only = is_read_only)
        if threads:
            self.conn.execute(f"SET threads = {threads}")
        if memory_limit:
            self.conn.execute(f"SET memory_limit = '{memory_limit}'")

    def query(self,
              query      : str,
              params     : Optional[Union[List, Dict, tuple]] = None,
              fetch_size : Optional[int] = None,
        ) -> pd.DataFrame:
        """
        Execute query and return DataFrame.
        Args:
            query      : SQL query string
            params     : Query parameters
            fetch_size : Number of rows to fetch (None for all)
        """
        if self.conn is None:
            raise ValueError("Connection is closed. Please reconnect.")
        try:
            if params is not None:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
            if fetch_size:
                return result.fetch_df_chunk(fetch_size)
            return result.fetchdf()
        except Exception as e:
            raise Exception(f"Query failed: {e}\nQuery: {query}")

    def query_arrow(self, 
                    query : str, 
                    params: Optional[Union[List, Dict]] = None,
                   ):
        """Execute query and return Arrow table (faster for large datasets)."""
        if params is not None:
            return self.conn.execute(query, params).fetch_arrow_table()
        return self.conn.execute(query).fetch_arrow_table()

    def register_dataframe(self, 
                           name: str, 
                           df  : pd.DataFrame,
                          ):
        """Register pandas DataFrame as temporary table (table_view)."""
        try:
            self.conn.register(name, df)
        except Exception as arch:
            logger.info('Register Dataframe already error, lets try other method.')
            changetype = {col: str for col in df.select_dtypes(include=['object', 'string']).columns}
            df = df.astype(changetype)
            df = df.convert_dtypes()
            self.conn.register(name, df)
            logger.debug('Register Dataframe is success.')

    def ListedTable(self) -> pd.DataFrame:
        data      = list()
        tables    = self.conn.execute("SELECT table_name FROM duckdb_tables()").df()
        for table in tables['table_name']:
            count = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            data.append({'table_name': table, 'row_count': count})
        info_df   = pd.DataFrame(data, columns=['table_name', 'row_count'])
        if not info_df.empty:
            info_df              = info_df.sort_values(by = 'row_count', ascending = False)
            info_df['row_count'] = info_df['row_count'].apply(lambda x: f"{x:,}")
        logger.info('Here is the DataFrame of table listed.')
        logger.info(info_df)
        return info_df

    def execute(self, 
                query: str, 
                params: Optional[Union[List, Dict]] = None):
        """Execute query without returning results."""
        if params is not None:
            self.conn.execute(query, params)
        else:
            self.conn.execute(query)

    def table_exists(self, 
                     table_name: str) -> bool:
        """Check if table exists in database."""
        result = self.query(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name])
        return result.iloc[0, 0] > 0

    def get_tables(self) -> List[str]:
        """Get list of all tables in database."""
        df = self.query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
        return df['table_name'].tolist()

    def get_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table."""
        return self.query(f"DESCRIBE {table_name}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@contextmanager
def duckdb_connection(db_path   : Union[str, Path] = ':memory:',
                      read_only : bool = False,
                      **kwargs) -> DuckDBManager:
    """
    Context manager for DuckDB connection.
    use ':memory:' for in-memory DB
    Example:
        with duckdb_connection('./dbprocess.duckdb') as db:
            df = db.query("SELECT * FROM users")
    """
    db = DuckDBManager(db_path, read_only, **kwargs)
    try:
        yield db
    finally:
        db.close()

if __name__ == '__main__':
    data_produk = {
        'product_id'  : [101, 102, 103, 104],
        'product_name': ['Beras 5kg', 'Minyak Goreng 2L', 'Gula Pasir 1kg', 'Susu Kental Manis'],
        'category'    : ['Sembako', 'Sembako', 'Sembako', 'Minuman'],
        'price'       : [65000, 34000, 15000, 12000]}
    df_products = pd.DataFrame(data_produk)
    logger.info("Initializing DuckDB Manager...")

    with duckdb_connection(':memory:') as db:
        logger.info("Registering local DataFrame as 'koperasi_products'...")
        db.register_dataframe('koperasi_products', df_products)
        db.execute("CREATE TABLE inventory AS SELECT * FROM koperasi_products")
        if db.table_exists('inventory'):
            logger.info("Success: Table 'inventory' created.")

        # Get Schema
        logger.info("\nTable Schema for 'inventory':")
        logger.info(db.get_schema('inventory')[['column_name', 'column_type']])

        # Query back data with a filter
        expensive_items = db.query(
            "SELECT product_name, price FROM inventory WHERE price > ?", 
            params=[20000])
        logger.info("\nProducts priced over 20,000:")
        logger.info(expensive_items)

        # Show table stats
        db.ListedTable()
    logger.warning("\nDatabase connection closed.")
