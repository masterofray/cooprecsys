# FP‑Growth: motivation and problem

The goal of **frequent itemset mining** is, given a collection of transactions $T = \{t_1, t_2, \dots, t_n\}$ over a set of items $I$, to find all itemsets $X \subseteq I$ whose **support** $\text{supp}(X)$ is at least a user‑defined minimum support threshold $\sigma$. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417410011917)

Let $D$ be the dataset of $n$ transactions. The absolute support of an itemset $X$ is:

$$
\text{supp}(X) = \left| \{ t \in D \mid X \subseteq t \} \right|
$$

and its **relative support** (fraction of transactions) is:

$$
S(X) = \frac{\text{supp}(X)}{n}
$$

We say $X$ is **frequent** if $S(X) \ge \sigma$. [emergentmind](https://www.emergentmind.com/topics/fp-growth-algorithm)

Classical algorithms such as **Apriori** generate candidate itemsets of increasing size and repeatedly scan the database, which becomes costly for large datasets. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/frequent-pattern-growth-algorithm/)

***

### FP‑Tree and FP‑Growth overview

FP‑Growth avoids candidate generation by building a **compact prefix tree** called an **FP‑Tree** and then mining frequent patterns directly from that tree. [mitu.co](https://mitu.co.in/wp-content/uploads/2024/07/46.-FP-Growth-Algorithm.pdf)

#### Step 1: Build the FP‑Tree

1. **Compute item frequencies**  
   For each item $i \in I$, compute its support count:

   $$
   \text{supp}(i) = \left| \{ t \in D \mid i \in t \} \right|
   $$

2. **Filter infrequent items**  
   Remove all items for which $\text{supp}(i) < \sigma n$. [mitu.co](https://mitu.co.in/wp-content/uploads/2024/07/46.-FP-Growth-Algorithm.pdf)

3. **Sort items by frequency**  
   Fix an ordering of items by decreasing frequency and apply this sort to each transaction. [mitu.co](https://mitu.co.in/wp-content/uploads/2024/07/46.-FP-Growth-Algorithm.pdf)

4. **Build the FP‑Tree**  
   The tree is initialized with a root node $\text{root}$. Each transaction is inserted as a path starting from the root; nodes are shared whenever possible. Each node stores:
   - an item label,
   - a count (how many times this prefix occurs),
   - links to the same item in other branches (via a **header table**). [en.wikibooks](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm)

   After building, the tree encodes the **conditional pattern base** of each frequent item in a compact way.

#### Step 2: Mine frequent patterns

For each frequent item $z$ (in frequency order), the algorithm:

1. Builds the **conditional pattern base** of $z$, i.e., all prefixes of paths ending in $z$ with their counts.  
2. From this base, builds a **conditional FP‑tree** $FP_{z}$. [en.wikibooks](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm)
3. Recursively mines frequent patterns of the form $X \cup \{z\}$ by applying the same procedure over $FP_{z}$.

By working bottom‑up and using the tree’s prefix‑sharing, FP‑Growth avoids generating explicit candidate itemsets and reduces the number of database scans to **two** (one for the first scan, one implicitly via the tree). [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/frequent-pattern-growth-algorithm/)

***

### Key variants of FP‑Growth

Here are some important variants and improvements:

#### 1. **IFP‑Growth (Improved FP‑Growth)**
Adds an **address‑table** and modified FP‑tree structure to reduce the need to rebuild full conditional trees and to lower memory usage. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417410011917)
- Uses **FP‑tree$+$** to store more structured node pointers.
- Reduces rebuilds of conditional trees and speeds up mining of large datasets. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417410011917)

#### 2. **Non‑ord FP‑Growth (Nonordfp)**
Discards the fixed frequency‑order projection and instead uses a more compact trie‑style FP‑tree; this can reduce both space and time, especially when the usual “most frequent first” ordering is expensive to maintain. [en.wikibooks](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm)

#### 3. **FP‑Bonsai and DynFP‑Growth**
- **FP‑Bonsai** compresses the tree structure further and is tuned for high‑dimensional, sparse datasets. [en.wikibooks](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm)
- **DynFP‑Growth** adapts the tree dynamically when new transactions arrive, making it suitable for incremental mining. [en.wikibooks](https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm)

#### 4. **GFP‑Growth (Generalized FP‑Growth)**
Extends FP‑Growth to handle **targeted mining** (e.g., rare‑class rules) by synchronously traversing a separate tree for the target and the FP‑tree, thus pruning irrelevant branches early. [emergentmind](https://www.emergentmind.com/topics/fp-growth-algorithm)
- Can be **up to $80\times$** faster than vanilla FP‑Growth for rare‑target workloads. [emergentmind](https://www.emergentmind.com/topics/fp-growth-algorithm)

***

### Best alternatives to FP‑Growth

Depending on your use‑case (size, memory, streaming, distributed), these are often considered strong alternatives:

#### 1. **Eclat / Depth‑first search (DFS) based methods**

Instead of prefix trees, **Eclat** uses a **vertical tidset representation**: each item is mapped to the set of transaction IDs (tids) where it appears. [easychair](https://easychair.org/publications/preprint/bS8f/download)

The support of an itemset $\{i_1,\dots,i_k\}$ is:

$$
\text{supp}(\{i_1,\dots,i_k\}) = \left| \bigcap_{j=1}^{k} \text{TID}(i_j) \right|
$$

where $\text{TID}(i_j)$ is the list of transactions containing item $i_j$. [easychair](https://easychair.org/publications/preprint/bS8f/download)
Eclat efficiently intersects tidsets and can be very fast on dense datasets with moderate numbers of items, often using **less memory** than FP‑Growth for some settings. [easychair](https://easychair.org/publications/preprint/bS8f/download)

#### 2. **COFI / other tidset‑based methods**

**COFI** is a tidset‑based frequent‑itemset miner that uses aggressive pruning strategies. It typically uses **less memory** than FP‑Growth while remaining competitive in runtime. [easychair](https://easychair.org/publications/preprint/bS8f/download)
It is a good choice if you want to avoid tree‑based overhead and can afford storing compact tidsets.

#### 3. **Parallel / distributed FP‑Growth (MR‑PFP, PFP, etc.)**

For large‑scale data, parallel FP‑Growth variants such as **PFP (Parallel FP‑Growth)** and **MR‑PFP (MapReduce‑based PFP)** run FP‑Growth on distributed clusters (e.g., Hadoop/Spark). [easychair](https://easychair.org/publications/preprint/bS8f/download)
These trade single‑node simplicity for **scalability** on very large datasets.

#### 4. **When to prefer FP‑Growth vs others**

- **FP‑Growth is excellent** when:
  - the dataset is large but not extremely wide,
  - you want **minimal scans** and **compact storage**,
  - you care about **response time** on already‑built structures. [jurnal.polibatam.ac](https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/11921/3383/38626)

- **Eclat / COFI / tidset methods** are better when:
  - the number of items is manageable,
  - tidset intersections remain efficient,
  - you want **lower memory** and **simpler data structures**. [easychair](https://easychair.org/publications/preprint/bS8f/download)

- **Parallel FP‑Growth / MR‑PFP** shine when:
  - data is **too large** for a single machine,
  - you already have a distributed infrastructure. [easychair](https://easychair.org/publications/preprint/bS8f/download)

***

### Suggested “TL;DR” section for your README

You can add a short comparison table like this:

```markdown
### Quick comparison

| Algorithm        | Main structure            | Pros                                      | When to prefer |
|------------------|---------------------------|-------------------------------------------|----------------|
| **Apriori**      | Candidate sets + scans    | Simple to implement                       | Very small datasets |
| **FP‑Growth**    | FP‑Tree (prefix tree)     | Compact, fast on large data, no candidates | Large transactional DBs |
| **IFP‑Growth**   | Enhanced FP‑tree + table  | Lower memory, faster rebuilds             | Memory‑constrained FP‑Growth usage |
| **Eclat / COFI** | Tidset lists              | Simple, often less memory                 | Moderate item counts |
| **MR‑PFP**       | Distributed FP‑trees      | Scales to huge data                       | Big‑data clusters |
```

***

### Most Practical & Reliable Modern Options

| Method / Library              | Type                          | Strengths (vs classic FP-Growth)                  | Best For                          | Reliability / Status |
|-------------------------------|-------------------------------|---------------------------------------------------|-----------------------------------|----------------------|
| **Spark MLlib FPGrowth (PFP)** | Parallel/distributed FP-Growth | True scalability on clusters, handles massive data | Big Data, production | Very high (mature, widely used) |
| **LCMFreq / LCM (Linear time Closed itemset Miner)** | Non-tree, array-based | Often faster on sparse data, lower memory for certain cases | Sparse datasets, closed/maximal itemsets | Excellent, strong benchmark performer |
| **CFP-Growth / CFP-Array**    | Memory-optimized FP variants | ~10x lower memory usage than standard FP-Tree | Large dense datasets | Good research track record |
| **H-Mine**                    | Hyper-structure (non-FP-tree) | Predictable memory, good on sparse + very large DBs | Mixed sparse/dense, out-of-core | Solid |
| **Improved variants (FP-TDA, GFP-Growth, etc.)** | Enhanced FP-Growth | Better tree compression, guided mining, less recursion | Specific big data or targeted mining | Emerging (2024–2025 papers) |

### Other Notable Modern Approaches
- **Guided FP-Growth (GFP-Growth)** — Better for targeted/minority-class mining (up to 80× faster in some cases).
- **Region-based / Projection-based** improvements — Reduce preprocessing and tree building cost.
- **Hybrid / Matrix-based (FP-TDA)** — Recent 2025 papers use Two-Dimensional Arrays for better compression on big data.

### Quick Advice for Your Code
Your current Cython FP-Tree is solid for medium datasets. For better reliability/scalability:
- Switch to **PySpark** if your data is growing or you need distributed computing.
- Try **LCMFreq** (available in some FIMI implementations or research code) if you stay single-machine and data is sparse.
- Add **conditional tree pruning** or switch to **iterative** (stack-based) mining instead of deep recursion to avoid stack overflow on very long patterns.


### Top Recommendations

| Repository | Language | Type | Highlights | Link |
|------------|----------|------|----------|------|
| **scikit-mine/scikit-mine** | Python | Pure Python (scikit-learn style) | Easiest to use for Python users, integrates well with pandas/numpy, supports closed itemsets | [https://github.com/scikit-mine/scikit-mine](https://github.com/scikit-mine/scikit-mine) |
| **slide-lig/jlcm** | Java | Multi-threaded (parallel) | Fast Java implementation of LCM (called PLCM internally), good performance | [https://github.com/slide-lig/jlcm](https://github.com/slide-lig/jlcm) |
| **bnegreve/plcm** | C/C++ | Parallel version | Parallel algorithm based on LCM for closed itemset mining | [https://github.com/bnegreve/plcm](https://github.com/bnegreve/plcm) |
| **david-duverle/rlcm** | C (R wrapper) | R integration | R package wrapper around original LCM by Takeaki Uno | [https://github.com/david-duverle/rlcm](https://github.com/david-duverle/rlcm) |

### Official / Original Source
The original LCM (versions up to 5.x) by **Takeaki Uno** is not hosted on GitHub but available directly from his page:  
--> http://research.nii.ac.jp/~uno/code/lcm.html (C implementation, very fast and widely considered the reference)
Many repositories (including the ones above) are based on or ports of this original code.

