# ary2tower residual fallback policy

The serving path follows this order:

1. Score the full eligible catalogue with the learned two-tower model.
2. Remove explicitly excluded and already-purchased items.
3. Take the first `n_items` remaining model-ranked items.
4. If the result is still short, fill only the missing slots with a Bayesian-smoothed global popularity prior. When an event-time column is available (`timestamp`, `event_time`, `created_at`, `datetime`, or `date`), the prior is time-decayed so recent demand receives more weight.

This fallback intentionally does **not** use item-to-item cosine similarity, user-user similarity, or a truncate-then-filter heuristic. It is a residual candidate generator: the personalized two-tower model remains the primary ranker, while the prior supplies robust candidates for the missing tail.

The contract is exact-N whenever at least `n_items` unique catalogue items remain eligible after exclusions. When fewer unique eligible items physically exist, returning fewer is unavoidable and is logged explicitly.
