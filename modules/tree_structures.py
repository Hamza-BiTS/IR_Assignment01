import bisect
import pandas as pd
import time

class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def insert(self, key):
        if key < self.key:
            if self.left is None:
                self.left = BSTNode(key)
            else:
                self.left.insert(key)
        elif key > self.key:
            if self.right is None:
                self.right = BSTNode(key)
            else:
                self.right.insert(key)

    def search(self, key):
        if self.key == key:
            return True
        if key < self.key:
            return self.left.search(key) if self.left else False
        return self.right.search(key) if self.right else False

class SimulatedBTree:
    def __init__(self, keys):
        self.keys = sorted(keys)

    def search(self, key):
        i = bisect.bisect_left(self.keys, key)
        return i < len(self.keys) and self.keys[i] == key

def run_tree_benchmark(dictionary_terms, query_terms):
    root = BSTNode(dictionary_terms[0])
    for term in dictionary_terms[1:]:
        root.insert(term)
    btree = SimulatedBTree(dictionary_terms)
    rows = []
    for q in query_terms:
        t1 = time.perf_counter()
        bst_found = root.search(q)
        bst_time = (time.perf_counter() - t1) * 1e6
        t2 = time.perf_counter()
        bt_found = btree.search(q)
        bt_time = (time.perf_counter() - t2) * 1e6
        rows.append({
            'query_term': q,
            'bst_found': bst_found,
            'bst_search_time_us': round(bst_time, 3),
            'btree_found': bt_found,
            'btree_search_time_us': round(bt_time, 3),
            'expected_bst_complexity': 'O(h)',
            'expected_btree_complexity': 'O(log n)',
        })
    df = pd.DataFrame(rows)
    note = 'The B-Tree style search is typically more stable for larger dictionaries because it keeps search balanced and scalable.'
    return df, note