'''
Create by Aryanto
at 20260412
email me : aryanto.dandan@gmail.com
'''

class FPGrowthAPI:
    def __init__(self, 
                 min_support: float = 0.01, 
                 **kwargs,
                ) -> None:
        self.min_support = min_support
        self.frequent_itemsets: Optional[Dict[FrozenSet[str], int]] = None
        self.tree: Optional[CFPTree] = None
        self.total_transactions: int = 0
        self.variant = kwargs.get('variant', 'full')  # full / incremental / streaming
        self.batch_size = kwargs.get('batch_size', 50000)

    # ----------------------------------------------------------------
    # Unified fit (routes to variant)
    # ----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        if self.variant == 'full':
            self._fit_full(df, **kwargs)
        elif self.variant == 'incremental':
            self._fit_incremental(df, **kwargs)
        elif self.variant == 'streaming':
            self._fit_streaming(df, **kwargs)
        else:
            raise ValueError("variant must be 'full', 'incremental' or 'streaming'")

    def _fit_full(self, df: pd.DataFrame, **kwargs):
        logger.info("=== FULL FP-Growth (Cython core) ===")
        transactions = prepare_transactions(df, **kwargs)
        self.total_transactions = len(transactions)

        frequent_1 = find_frequent_items(transactions, self.min_support, **kwargs)
        if not frequent_1:
            logger.warning("No frequent items found")
            return

        # Cython Step 3 + Step 4
        self.tree = CFPTree(frequent_1, self.min_support)
        self.tree.build_from_transactions(transactions)          # Step 3
        min_count = int(self.min_support * self.total_transactions)
        self.frequent_itemsets = dict()
        self.tree.mine_frequent_itemsets(list(), min_count, self.frequent_itemsets)  # Step 4

        visualize_frequent_itemsets(self.frequent_itemsets, kwargs.get('output_dir', '.'))
        gc.collect()

    def _fit_incremental(self, df: pd.DataFrame, **kwargs):
        logger.info("=== INCREMENTAL FP-Growth (Cython update) ===")
        transactions = prepare_transactions(df, **kwargs)
        self.total_transactions += len(transactions)

        if self.tree is None:
            frequent_1 = find_frequent_items(transactions, self.min_support, **kwargs)
            self.tree = CFPTree(frequent_1, self.min_support)
            self.tree.build_from_transactions(transactions)
        else:
            self.tree.update(transactions)   # Cython incremental insert

        min_count = int(self.min_support * self.total_transactions)
        self.frequent_itemsets = dict()
        self.tree.mine_frequent_itemsets(list(), min_count, self.frequent_itemsets)

        visualize_frequent_itemsets(self.frequent_itemsets, kwargs.get('output_dir', '.'))
        gc.collect()

    def _fit_streaming(self, df: pd.DataFrame, **kwargs):
        logger.info("=== STREAMING FP-Growth (Cython + low-RAM batches) ===")
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="Streaming batches", colour='green'):
            start = batch_idx * self.batch_size
            batch_df = df.iloc[start:start + self.batch_size].copy()
            transactions = prepare_transactions(batch_df, **kwargs)
            self.total_transactions += len(transactions)

            if self.tree is None:
                frequent_1 = find_frequent_items(transactions, self.min_support, **kwargs)
                self.tree = CFPTree(frequent_1, self.min_support)
                self.tree.build_from_transactions(transactions)
            else:
                self.tree.update(transactions)

            gc.collect()

        min_count = int(self.min_support * self.total_transactions)
        self.frequent_itemsets = dict()
        self.tree.mine_frequent_itemsets(list(), min_count, self.frequent_itemsets)

        visualize_frequent_itemsets(self.frequent_itemsets, kwargs.get('output_dir', '.'))
        gc.collect()

    def save_model(self, path: str = "fpgrowth_model.pkl") -> None:
        logger.info(f"Saving model --> {path}")
        model_data = {
            'frequent_itemsets': self.frequent_itemsets,
            'min_support': self.min_support,
            'total_transactions': self.total_transactions,
            'variant': self.variant
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info("Model saved (Cython tree not pickled - re-fit for continued incremental)")

    def load_model(self, path: str) -> None:
        logger.info(f"Loading model <-- {path}")
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.frequent_itemsets = model_data['frequent_itemsets']
        self.min_support = model_data['min_support']
        self.total_transactions = model_data.get('total_transactions', 0)
        self.variant = model_data.get('variant', 'full')
        self.tree = None  # tree must be rebuilt if you want to continue incremental
        logger.info("Model loaded")

    # ----------------------------------------------------------------
    # Cython-accelerated prediction on new dataset
    # ----------------------------------------------------------------
    def predict(self, new_df: pd.DataFrame, **kwargs) -> Dict[int, List[Tuple[FrozenSet[str], int]]]:
        logger.info("Cython prediction on new dataset")
        if self.frequent_itemsets is None:
            raise ValueError("Model not fitted")

        transactions = prepare_transactions(new_df, **kwargs)
        recommendations: Dict[int, List[Tuple[FrozenSet[str], int]]] = dict()

        for idx, trans in tqdm(enumerate(transactions), desc="Cython predict", colour='green', total=len(transactions)):
            recommendations[idx] = c_predict_one(trans, self.frequent_itemsets)

        logger.info(f"Prediction done for {len(transactions)} transactions")
        gc.collect()
        return recommendations
        
if __name__ == '__main__':
    
    api = FPGrowthAPI(min_support=0.005, variant='full')
    api.fit(mas_train, group_by='user_id', filter_col='purchase_count', output_dir='results')
    api.save_model("fpgrowth_full.pkl")

    api_inc = FPGrowthAPI(min_support=0.005, variant='incremental')
    api_inc.fit(mas_train, group_by='user_id', filter_col='purchase_count', output_dir='results')