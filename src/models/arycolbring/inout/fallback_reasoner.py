#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-18"


"""
fallback_reasoner.py
__________________________________________________
Purchase-aware recommendation cleanup and Item-to-Item fallback
companion for the AryColBring collaborative filtering model.

Ditempatkan satu direktori dengan inference.py agar bisa dipakai
langsung oleh AryColBringInference.clean_recommend().
"""

import numpy  as np
import pandas as pd
from   pathlib   import Path
from   datetime  import datetime
from   typing    import Optional, List, Tuple, Dict, Union

#LocDir = Path(__file__).resolve()
#sys.path.append(str(LocDir.parents[2]))
from ....configs import _cfg, logger
from ....db      import duckdb_connection
from ....prepare import DetectReco_Identifier


class AryInfFallBack:
    """Post-processing companion for AryColBringInference.recommend().

    Given a user's purchase history (passed as a dataframe, regardless of column names), 
    this class executes the following pipeline:
      1. Auto-detects 'user_id' & 'item_id' columns via DetectReco_Identifier, 
         then compiles a list of items already purchased by the user_id.
      2. Audits the model's Top-N recommendations to identify which items have 
         actually been purchased by that specific user.
      3. Filters out (removes) already-purchased items from the recommendation list.
      4. Backfills (patches) the vacant slots using fallback candidates scored by 
         the model (ranked below the original Top-N threshold / "overscan pool").
      5. If model candidates are insufficient, falls back to an Item-to-Item algorithm 
         (cosine similarity over model's item embeddings) seeded by user purchase history.
      6. Formats the final output as a pandas.DataFrame, or exports it into a 
         DuckDB flat-file (.db) via src/db/callduckdb.py.

    Important Note:
      The user_id/item_id used here MUST reside in the exact same ID space as the model 
      (i.e., label-encoded IDs whose rows correspond directly to the embedding matrix), 
      matching the contract enforced by AryColBringInference.recommend(). If your 
      purchase_data still uses raw IDs, you must encode them using the exact same 
      encoder from training (see src/features/encdec.py) before feeding them here.
    """
    SOURCE_MODEL     = "model_topn"
    SOURCE_PATCH     = "model_patch"
    SOURCE_ITEM2ITEM = "item2item_fallback"

    def __init__(self,
                 purchase_data   : pd.DataFrame,
                 item_embeddings : np.ndarray,
                 user_col        : Optional[str] = None,
                 item_col        : Optional[str] = None,
                ) -> None:
        if not isinstance(purchase_data, pd.DataFrame) or purchase_data.empty:
            msg = "purchase_data must be a non-empty pandas DataFrame."
            logger.error(msg)
            raise ValueError(msg)

        self.purchase_data   = purchase_data
        self.item_embeddings = np.asarray(item_embeddings, dtype = np.float32)

        # Point 1 - deteksi kolom user/item apapun nama aslinya di data.
        self.Collect  = DetectReco_Identifier(Dataprocess = purchase_data)
        self.user_col = user_col or self.Collect.get('user_col')
        self.item_col = item_col or self.Collect.get('item_col')
        if self.user_col is None or self.item_col is None:
            msg = ("Could not auto-detect user/item identifier columns from "
                   f"purchase_data. Detected mapping: {self.Collect}. "
                   "Please pass user_col/item_col explicitly.")
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("AryInfFallBack resolved columns: user_col = '%s', item_col = '%s'",
                      self.user_col, self.item_col)

        self._purchase_map  = self._build_purchase_map()
        self._item_unitnorm = self._normalize_embeddings()

    def _build_purchase_map(self) -> Dict:
        """Bangun peta user_id -> set(item_id yang pernah dibeli)."""
        grouped = self.purchase_data.groupby(self.user_col)[self.item_col].apply(
                  lambda s: set(s.tolist()))
        logger.debug("Purchase map built for %d unique users.", len(grouped))
        return grouped.to_dict()

    def _normalize_embeddings(self) -> np.ndarray:
        """Unit-normalize embedding item, untuk cosine similarity yang murah (dot product)."""
        norms          = np.linalg.norm(self.item_embeddings, axis = 1, keepdims = True)
        norms[norms == 0] = 1e-9
        return self.item_embeddings / norms

    def purchased_items(self, user_id: int) -> set:
        """(Point 1) Item apa saja yang pernah dibeli user_id."""
        return self._purchase_map.get(user_id, set())

    def item_to_item_candidates(self,
                                seed_items : List[int],
                                exclude    : set,
                                n          : int,
                               ) -> List[Tuple[int, float]]:
        """
        (Point 5) Fallback Item-to-Item Based: cari item paling mirip
        (cosine similarity) terhadap rata-rata embedding item seed
        (biasanya riwayat pembelian user), lalu buang item yang perlu
        dikecualikan (sudah dibeli / sudah masuk rekomendasi).
        """
        if not seed_items or n <= 0:
            return list()
        n_catalog   = self._item_unitnorm.shape[0]
        valid_seeds = [int(i) for i in seed_items if 0 <= int(i) < n_catalog]
        if not valid_seeds:
            return list()

        seed_vec  = self._item_unitnorm[valid_seeds].mean(axis = 0, keepdims = True)
        seed_norm = np.linalg.norm(seed_vec)
        if seed_norm == 0:
            return list()
        seed_vec  = seed_vec / seed_norm

        similarities = (self._item_unitnorm @ seed_vec.T).ravel()
        ranked_idx   = np.argsort(similarities)[::-1]

        exclude_set = set(int(x) for x in exclude) | set(valid_seeds)
        results     = list()
        for idx in ranked_idx:
            if len(results) >= n:
                break
            idx = int(idx)
            if idx in exclude_set:
                continue
            results.append((idx, float(similarities[idx])))
        logger.debug("Item-to-Item fallback produced %d candidate(s) from %d seed item(s).",
                      len(results), len(valid_seeds))
        return results

    def clean_recommendations(self,
                              user_id        : int,
                              candidate_pool : List[Tuple[int, float]],
                              n_items        : int = 10,
                             ) -> pd.DataFrame:
        """
        Orkestrasi Point 2-5.
        candidate_pool : hasil recommend() model, idealnya di-overscan
                          (lebih besar dari n_items) supaya ada cadangan
                          untuk menambal item yang dibuang.
        Returns tidy DataFrame: user_id, rank, item_id, score, source, is_fallback.
        """
        bought = self.purchased_items(user_id)
        logger.debug("User %s has %d previously purchased item(s).", user_id, len(bought))

        # Point 2 & 3 - saring kandidat yang belum pernah dibeli, tetap
        # mempertahankan urutan skor asli dari model.
        seen      = set()
        clean     = list()
        removed_n = 0
        for position, (item_id, score) in enumerate(candidate_pool):
            item_id = int(item_id)
            if item_id in bought:
                removed_n += 1
                continue
            if item_id in seen:
                continue
            seen.add(item_id)
            source = self.SOURCE_MODEL if position < n_items else self.SOURCE_PATCH
            clean.append((item_id, float(score), source))
            if len(clean) >= n_items:
                break
        logger.debug("Removed %d already-purchased item(s) for user %s; "
                      "%d/%d slot(s) filled from model pool.",
                       removed_n, user_id, len(clean), n_items)

        # Point 5 - Item-to-Item fallback kalau kandidat model masih kurang.
        if len(clean) < n_items:
            gap = n_items - len(clean)
            logger.info("Model candidate pool exhausted for user %s; "
                        "falling back to Item-to-Item for %d slot(s).", user_id, gap)
            exclude    = bought | seen
            seed_items = list(bought) if bought else [item_id for item_id, _, _ in clean]
            fallback_candidates = self.item_to_item_candidates(
                                  seed_items = seed_items,
                                  exclude    = exclude,
                                  n          = gap)
            for item_id, score in fallback_candidates:
                clean.append((item_id, score, self.SOURCE_ITEM2ITEM))

        result_df = pd.DataFrame(clean, columns = ["item_id", "score", "source"])
        result_df.insert(0, "user_id", user_id)
        result_df.insert(1, "rank", range(1, len(result_df) + 1))
        result_df["is_fallback"] = result_df["source"] != self.SOURCE_MODEL
        return result_df

    def export_duckdb(self,
                      result_df  : pd.DataFrame,
                      db_path    : Optional[Union[str, Path]] = None,
                      table_name : str  = "clean_recommendations",
                      mode       : str  = "append",
                     ) -> str:
        """(Point 6, opsi B) Simpan hasil rekomendasi bersih ke DuckDB flat-file (.db)."""
        if mode not in ("append", "replace"):
            msg = f"mode must be 'append' or 'replace', got '{mode}'."
            logger.error(msg)
            raise ValueError(msg)

        if db_path is None:
            out_dir = Path(_cfg.get('PATHS', 'output_dir', fallback = 'artifacts'))
            out_dir.mkdir(parents = True, exist_ok = True)
            db_path = out_dir / "clean_recommendations.db"
        db_path = Path(db_path)
        db_path.parent.mkdir(parents = True, exist_ok = True)

        export_df = result_df.copy()
        export_df["generated_at"] = datetime.now().isoformat()

        with duckdb_connection(str(db_path), read_only = False) as db:
            db.register_dataframe("incoming_recs", export_df)
            if mode == "append" and db.table_exists(table_name):
                db.execute(f"INSERT INTO {table_name} SELECT * FROM incoming_recs")
            else:
                db.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM incoming_recs")
        logger.info("Clean recommendations exported to DuckDB: %s (table = %s)",
                     db_path, table_name)
        return str(db_path)


if __name__ == '__main__':
    pass
