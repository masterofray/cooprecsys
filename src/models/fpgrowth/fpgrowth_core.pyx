# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

from libc.stdlib cimport malloc, free
from libc.stdio cimport fprintf, stderr
from cython.parallel cimport prange, parallel, threadid
cimport numpy as np
import numpy as np
np.import_array()
import cython

cdef class CTreeNode:
    cdef public int item_id
    cdef public int count
    cdef public CTreeNode parent
    cdef public dict children
    cdef public CTreeNode next_same_item

    def __cinit__(self, int item_id=-1, int count=0, CTreeNode parent=None):
        self.item_id = item_id
        self.count = count
        self.parent = parent
        self.children = dict()
        self.next_same_item = None


cdef class CFPTree:
    cdef public dict header_table
    cdef public dict frequent_items
    cdef public dict item_to_id
    cdef public list id_to_item
    cdef public float min_support
    cdef CTreeNode root
    cdef int num_frequent_items

    def __cinit__(self, dict frequent_items, float min_support):
        self.root = CTreeNode()
        self.frequent_items = frequent_items
        self.min_support = min_support

        sorted_items = sorted(frequent_items.keys(), key=lambda x: frequent_items[x], reverse=True)
        self.item_to_id = {item: idx for idx, item in enumerate(sorted_items)}
        self.id_to_item = sorted_items
        self.num_frequent_items = len(sorted_items)

        self.header_table = {item: None for item in sorted_items}

        fprintf(stderr, "DEBUG [CFPTree init] Created %d item IDs, min_support=%.4f\n",
                self.num_frequent_items, min_support)

    # ----------------------------------------------------------------
    # Helper: list[str] --> temporary C intlist() array
    # ----------------------------------------------------------------
    cdef int* _convert_transaction_to_carray(self, list transaction, int* length):
        length[0] = 0
        if not transaction:
            return NULL

        cdef int n = len(transaction)
        cdef int* carray = <int*> malloc(n * sizeof(int))
        if carray == NULL:
            return NULL

        cdef int i, item_id
        cdef str item
        for i in range(n):
            item = <str>transaction[i]
            item_id = self.item_to_id.get(item, -1)
            if item_id != -1:
                carray[length[0]] = item_id
                length[0] += 1
        return carray

    # ----------------------------------------------------------------
    # Step 3: INSERT (parallel-called)
    # ----------------------------------------------------------------
    cpdef void insert_transaction(self, list transaction):
        fprintf(stderr, "DEBUG [insert_transaction] Thread %d | transaction len=%d\n",
                threadid(), len(transaction))

        cdef int length = 0
        cdef int* carray = self._convert_transaction_to_carray(transaction, &length)
        if carray == NULL or length == 0:
            if carray != NULL:
                free(carray)
            return

        cdef int i, j, tmp
        for i in range(length):
            for j in range(i + 1, length):
                if self.frequent_items[self.id_to_item[carray[i]]] < self.frequent_items[self.id_to_item[carray[j]]]:
                    tmp = carray[i]
                    carray[i] = carray[j]
                    carray[j] = tmp

        cdef CTreeNode current = self.root
        cdef CTreeNode new_node
        cdef CTreeNode last
        cdef int item_id
        cdef str item_str

        for i in range(length):
            item_id = carray[i]
            item_str = self.id_to_item[item_id]
            if item_str not in current.children:
                new_node = CTreeNode(item_id, 1, current)
                current.children[item_str] = new_node
                if self.header_table[item_str] is None:
                    self.header_table[item_str] = new_node
                else:
                    last = self.header_table[item_str]
                    while last.next_same_item is not None:
                        last = last.next_same_item
                    last.next_same_item = new_node
            else:
                (<CTreeNode>current.children[item_str]).count += 1
            current = <CTreeNode>current.children[item_str]

        free(carray)
        fprintf(stderr, "DEBUG [insert_transaction] Thread %d | insert finished\n", threadid())

    # PARALLEL build (your main heavy loop)
    cpdef void build_from_transactions(self, list transactions):
        fprintf(stderr, "DEBUG [build_from_transactions] Starting PARALLEL build | %d transactions\n",
                len(transactions))

        cdef int n = len(transactions)
        cdef int i

        with nogil, parallel():
            for i in prange(n, schedule='dynamic', chunksize=1000):
                with gil:
                    self.insert_transaction(transactions[i])

        fprintf(stderr, "DEBUG [build_from_transactions] PARALLEL build completed\n")

    cpdef void update(self, list new_transactions):
        fprintf(stderr, "DEBUG [update] Incremental/Streaming update | %d transactions\n",
                len(new_transactions))
        self.build_from_transactions(new_transactions)

    # ----------------------------------------------------------------
    # Step 4: MINING (sequential + full fprintf debugging)
    # ----------------------------------------------------------------
    cpdef void mine_frequent_itemsets(self, list prefix, int min_support_count, dict frequent_itemsets):
        fprintf(stderr, "DEBUG [mine_frequent_itemsets] Starting mining | prefix len=%d\n", len(prefix))

        cdef list items_to_process = list(self.header_table.keys())
        items_to_process.reverse()
        cdef int num_items = len(items_to_process)
        cdef int i
        cdef str item
        cdef int support_count
        cdef CTreeNode node
        cdef list cond_patterns
        cdef CFPTree cond_tree
        cdef bytes bitem   # safe bytes for fprintf

        for i in range(num_items):
            item = items_to_process[i]
            bitem = item.encode('utf-8')
            fprintf(stderr, "DEBUG [mine] Thread %d | Processing item %s (%d/%d)\n",
                    threadid(), <char*>bitem, i + 1, num_items)

            support_count = 0
            node = self.header_table.get(item)
            while node is not None:
                support_count += node.count
                node = node.next_same_item

            if support_count >= min_support_count:
                new_itemset = frozenset(prefix + [item])
                frequent_itemsets[new_itemset] = support_count

            cond_patterns = self.get_conditional_pattern_base(item)
            cond_tree = self._build_conditional_tree(cond_patterns, min_support_count)

            if cond_tree is not None and cond_tree.root.children:
                cond_tree.mine_frequent_itemsets(prefix + [item], min_support_count, frequent_itemsets)

        fprintf(stderr, "DEBUG [mine_frequent_itemsets] Mining finished\n")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    cpdef list get_conditional_pattern_base(self, str item):
        cdef bytes bitem = item.encode('utf-8')
        fprintf(stderr, "DEBUG [get_conditional_pattern_base] Item %s\n", <char*>bitem)

        cdef list patterns = list()
        cdef CTreeNode node = self.header_table.get(item)
        if node is None:
            return patterns
        cdef list path
        cdef CTreeNode parent
        while node is not None:
            path = list()
            parent = node.parent
            while parent is not None and parent.item_id != -1:
                path.append(self.id_to_item[parent.item_id])
                parent = parent.parent
            if path:
                patterns.append((path[::-1], node.count))
            node = node.next_same_item
        return patterns

    cpdef CFPTree _build_conditional_tree(self, list conditional_patterns, int min_support_count):
        if not conditional_patterns:
            return None
        fprintf(stderr, "DEBUG [_build_conditional_tree] Building for %d patterns\n",
                len(conditional_patterns))

        cdef dict cond_item_count = dict()
        cdef list path
        cdef int count
        cdef list sorted_path
        cdef CTreeNode current
        cdef CTreeNode new_node
        cdef CTreeNode last
        cdef str item
        cdef int i, j
        cdef str tmp_item

        for path, count in conditional_patterns:
            for item in path:
                if item in cond_item_count:
                    cond_item_count[item] += count
                else:
                    cond_item_count[item] = count

        cdef dict cond_frequent = {item: cnt for item, cnt in cond_item_count.items() if cnt >= min_support_count}
        if not cond_frequent:
            return None

        cdef CFPTree cond_tree = CFPTree(cond_frequent, 0.0)

        for path, count in conditional_patterns:
            sorted_path = [item for item in path if item in cond_frequent]
            if not sorted_path:
                continue

            # Manual bubble sort (no lambda)
            for i in range(len(sorted_path)):
                for j in range(i + 1, len(sorted_path)):
                    if cond_frequent[sorted_path[i]] < cond_frequent[sorted_path[j]]:
                        tmp_item = sorted_path[i]
                        sorted_path[i] = sorted_path[j]
                        sorted_path[j] = tmp_item

            current = cond_tree.root
            for item in sorted_path:
                if item not in current.children:
                    new_node = CTreeNode(cond_tree.item_to_id[item], count, current)
                    current.children[item] = new_node
                    if cond_tree.header_table[item] is None:
                        cond_tree.header_table[item] = new_node
                    else:
                        last = cond_tree.header_table[item]
                        while last.next_same_item is not None:
                            last = last.next_same_item
                        last.next_same_item = new_node
                else:
                    (<CTreeNode>current.children[item]).count += count
                current = <CTreeNode>current.children[item]
        return cond_tree


# --------------------------------------------------------------------
# Prediction (simple + fprintf)
# --------------------------------------------------------------------
cpdef list c_predict_one(list trans, dict frequent_itemsets):
    # Convert trans to a set for O(1) lookup speed in .issubset()
    cdef object trans_set = set(trans) 
    
    if not frequent_itemsets:
        return list()

    cdef np.ndarray keys = np.array(list(frequent_itemsets.keys()), dtype=object)
    cdef np.ndarray vals = np.array(list(frequent_itemsets.values()), dtype=np.int32)
    cdef int n = len(keys)
    cdef list matches = list()
    cdef int i
    cdef object current_key

    for i in prange(n, nogil=True, schedule='static'):
        with gil:
            current_key = keys[i]
            if current_key.issubset(trans_set):
                matches.append((current_key, vals[i]))

    return matches[:10]