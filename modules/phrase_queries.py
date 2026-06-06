from collections import defaultdict
import re

def _norm_phrase(phrase):
    return [t.lower() for t in re.findall(r"\b[a-zA-Z]+\b", phrase.replace('-', ' '))]

def build_biword_index(preprocessed_docs):
    idx = defaultdict(set)
    for d in preprocessed_docs:
        toks = d['filtered_tokens']
        for i in range(len(toks)-1):
            idx[f"{toks[i]} {toks[i+1]}"].add(d['doc_id'])
    return dict(idx)

def build_positional_index(preprocessed_docs):
    idx = defaultdict(lambda: defaultdict(list))
    for d in preprocessed_docs:
        toks = d['filtered_tokens']
        for pos, tok in enumerate(toks):
            idx[tok][d['doc_id']].append(pos)
    return idx

def biword_phrase_search(phrase, biword_index, preprocessed_docs):
    terms = _norm_phrase(phrase)
    if len(terms) < 2:
        return []
    pairs = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms)-1)]
    posting_sets = [biword_index.get(p, set()) for p in pairs]
    if not posting_sets:
        return []
    docs = set.intersection(*map(set, posting_sets)) if posting_sets else set()
    return sorted(docs)

def positional_phrase_search(phrase, positional_index, preprocessed_docs):
    terms = _norm_phrase(phrase)
    if not terms:
        return []
    candidate_docs = None
    for t in terms:
        docs = set(positional_index.get(t, {}).keys())
        candidate_docs = docs if candidate_docs is None else candidate_docs & docs
    results = []
    for doc_id in candidate_docs or []:
        first_positions = positional_index[terms[0]][doc_id]
        for p in first_positions:
            ok = True
            for offset, term in enumerate(terms[1:], start=1):
                if (p + offset) not in positional_index[term][doc_id]:
                    ok = False
                    break
            if ok:
                results.append(doc_id)
                break
    return sorted(results)