#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-11"


from pathlib import Path
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from typing import get_type_hints, get_origin, get_args, Union


@dataclass
class FallbackConfig:
    strategy                : str = "content"
    fallback_score_mode     : str = "min"
    fallback_score_value    : float = 0.0
    fallback_score_quantile : float = 0.05
    max_candidates_scan     : Optional[int] = 5000
    diversity_weight        : float = 0.0
    random_state            : int = 42
    top_k                   : Optional[int] = None
    item_id_col             : Optional[str] = "ProductID"
    use_ann                 : bool = True
    ann_threshold           : int = 50_000
    ann_library             : str = "faiss"
    use_gpu                 : bool = False
    cache_dir               : Optional[str] = "./vector_cache"
    cache_key               : Optional[str] = None
    cold_start_threshold    : int = 5
    cold_start_strategy     : str = "popularity"
    mark_fallback           : bool = True
    use_duckdb              : bool = True
    n_jobs                  : int = -1
    batch_size              : str = "auto"

    @classmethod
    def from_configparser(cls, 
            cfg     : ConfigParser, 
            section : str = "FALLBACK",
        ) -> "FallbackConfig":
        """Automatically populate fields from INI 
        section using dataclass defaults as fallback."""
        field_values = dict()
        hints        = get_type_hints(cls)
        for f in fields(cls):
            key = f.name
            expected_type = hints[key]
            default = f.default if f.default is not f.default_factory else None
            
            # Determine reader method based on type
            raw = cfg.get(section, key, fallback=None)
            if raw is None:
                value = default
            else:
                origin = get_origin(expected_type)
                inner_type = expected_type
                if origin is Union:
                    args = [a for a in get_args(expected_type) if a is not type(None)]
                    inner_type = args[0] if args else str
                # Convert
                if inner_type == bool:
                    value = cfg.getboolean(section, key)
                elif inner_type == int:
                    value = cfg.getint(section, key)
                elif inner_type == float:
                    value = cfg.getfloat(section, key)
                else:
                    value = raw
            field_values[key] = value
        return cls(**field_values)


    def validate(self) -> None:
        if self.fallback_score_mode not in {'min', 'quantile', 'fixed'}:
            raise ValueError(f"fallback_score_mode must be min/quantile/fixed, got '{self.fallback_score_mode}'")
        valid_strategies = {'content', 'popularity', 'collaborative', 'hybrid'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got '{self.strategy}'")
        if self.cold_start_strategy not in valid_strategies:
            raise ValueError(f"cold_start_strategy must be one of {valid_strategies}, got '{self.cold_start_strategy}'")
        if self.ann_library not in {'faiss', 'annoy'}:
            raise ValueError(f"ann_library must be 'faiss' or 'annoy', got '{self.ann_library}'")
        if self.cold_start_threshold < 0:
            raise ValueError(f"cold_start_threshold must be >= 0, got {self.cold_start_threshold}")
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {self.n_jobs}")

if __name__ == '__main__':
    pass