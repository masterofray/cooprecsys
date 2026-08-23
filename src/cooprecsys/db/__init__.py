#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"
__modified__   = "2026-08-22"


try:
    from .callduckdb import DuckDBManager, duckdb_connection
except ImportError as exc:
    DuckDBManager     = None
    duckdb_connection = None
    _IMPORT_ERROR     = exc
else:
    _IMPORT_ERROR = None

__all__ = ["__version__", "DuckDBManager", "duckdb_connection"]
