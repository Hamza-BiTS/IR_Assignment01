import re
from collections import defaultdict

def build_kgram_index(terms, k=3):
    idx = defaultdict(set)
    for term in terms:
        padded = f"${term}$"
        for i in range(len(padded)-k+1):
            idx[padded[i:i+k]].add(term)
    return dict(idx)

def wildcard_search(pattern, terms):
    regex = '^' + pattern.replace('*', '.*') + '$'
    return sorted([t for t in terms if re.match(regex, t)])

def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1]

def suggest_corrections(term, vocabulary, kgram_index, k=3, top_n=5):
    padded = f"${term}$"
    candidates = set()
    for i in range(len(padded)-k+1):
        gram = padded[i:i+k]
        candidates |= kgram_index.get(gram, set())
    ranked = sorted([(cand, edit_distance(term, cand)) for cand in candidates], key=lambda x: (x[1], x[0]))
    return ranked[:top_n]

def tolerant_search(query, inverted_index, kgram_index, preprocessed_docs):
    q = query.lower().strip()
    if '*' in q:
        matches = wildcard_search(q, list(inverted_index.keys()))
        return {'mode': 'wildcard', 'matches': matches[:20]}
    if q in inverted_index:
        return {'mode': 'exact', 'matches': sorted(list(inverted_index[q]))}
    suggestions = suggest_corrections(q, list(inverted_index.keys()), kgram_index)
    if suggestions:
        best = suggestions[0][0]
        return {'mode': 'corrected', 'suggested_term': best, 'matches': sorted(list(inverted_index.get(best, []))), 'all_suggestions': suggestions}
    return {'mode': 'not_found', 'matches': []}