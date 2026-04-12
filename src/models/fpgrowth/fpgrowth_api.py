'''
Create by Aryanto
at 20260412
email me : aryanto.dandan@gmail.com
'''

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, FrozenSet
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import cloudpickle as pickle
import os
from collections import Counter
import warnings
import gc

try:
    from fpgrowth_core import CFPTree, c_predict_one
except ImportError:
    raise ImportError("Run 'python setup.py build_ext --inplace' first!")

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Python helpers (data prep + frequent 1-items) - kept in Python because
# they use Pandas + joblib. The heavy tree work is in Cython.
# --------------------------------------------------------------------
def prepare_transactions(
    df: pd.DataFrame,
    **kwargs
) -> List[List[str]]:
    logger.info("Preparing transactions (Python wrapper)")
    group_by = kwargs.get('group_by', 'user_id')
    item_col = kwargs.get('item_col', 'product_id')
    filter_col = kwargs.get('filter_col', 'purchase_count')
    min_val = kwargs.get('min_val', 1)
    unique_items = kwargs.get('unique', True)
    chunk_size = kwargs.get('chunk_size', None)

    if chunk_size and len(df) > chunk_size:
        transactions = list()
        for start in tqdm(range(0, len(df), chunk_size), desc="Low-RAM chunking", colour='green'):
            chunk = df.iloc[start:start+chunk_size].copy()
            if filter_col in chunk.columns:
                chunk = chunk[chunk[filter_col] >= min_val]
            chunk_trans = chunk.groupby(group_by)[item_col].apply(list).tolist()
            transactions.extend(chunk_trans)
            del chunk
            gc.collect()
    else:
        if filter_col in df.columns:
            df = df[df[filter_col] >= min_val]
        transactions = df.groupby(group_by)[item_col].apply(list).tolist()

    if unique_items:
        transactions = [list(dict.fromkeys(t)) for t in transactions]
    transactions = [[str(item) for item in t] for t in transactions]

    logger.info(f"Prepared {len(transactions)} transactions")
    gc.collect()
    return transactions


def _count_in_chunk(chunk: List[List[str]]) -> Dict[str, int]:
    return dict(Counter(item for transaction in chunk for item in transaction))
    
def find_frequent_items(
    transactions: List[List[str]],
    min_support: float,
    **kwargs
) -> Dict[str, int]:
    logger.info("Finding frequent 1-itemsets (parallel joblib)")
    num_transactions = len(transactions)
    min_count = int(min_support * num_transactions)
    n_jobs = kwargs.get('n_jobs', -1)

    n_chunks = min(os.cpu_count() or 8, len(transactions))
    chunks = [transactions[i::n_chunks] for i in range(n_chunks)]

    chunk_counts_list: List[Dict[str, int]] = Parallel(n_jobs=n_jobs)(
        delayed(_count_in_chunk)(chunk)
        for chunk in tqdm(chunks, desc="Parallel counting", colour='green')
    )

    item_count: Dict[str, int] = dict()
    for chunk_count in tqdm(chunk_counts_list, desc="Merging counts", colour='green'):
        for item, cnt in chunk_count.items():
            item_count[item] = item_count.get(item, 0) + cnt

    frequent_items = {item: count for item, count in item_count.items() if count >= min_count}
    logger.info(f"Found {len(frequent_items)} frequent 1-itemsets")
    gc.collect()
    return frequent_items


def visualize_frequent_itemsets(
    frequent_itemsets: Dict[FrozenSet[str], int],
    output_dir: str = "."
) -> None:
    logger.info("Generating seaborn visualizations (saved, never shown)")
    os.makedirs(output_dir, exist_ok=True)
    data = [{'itemset': ' --> '.join(sorted(itemset)), 'support': supp, 'length': len(itemset)}
            for itemset, supp in frequent_itemsets.items()]
    vis_df = pd.DataFrame(data).sort_values('support', ascending=False)

    # Top 20
    plt.figure(figsize=(14, 8))
    sns.barplot(data=vis_df.head(20), x='support', y='itemset', palette='viridis')
    plt.title('Top 20 Frequent Itemsets')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_20_frequent_itemsets.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Length distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=vis_df, x='length', palette='viridis')
    plt.title('Itemset Length Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'itemset_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    gc.collect()
