import re
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def tokenize(text):
    text = text.replace('-', ' ')
    return re.findall(r"\b[a-zA-Z]+\b", text)

def preprocess_documents(docs):
    out = []
    for d in docs:
        original_tokens = tokenize(d['text'])
        lowered = [t.lower() for t in original_tokens]
        filtered = [t for t in lowered if t not in STOPWORDS]
        stemmed = [stemmer.stem(t) for t in filtered]
        lemmatized = [lemmatizer.lemmatize(t) for t in filtered]
        out.append({
            "doc_id": d['doc_id'],
            "title": d['title'],
            "text": d['text'],
            "original_tokens": original_tokens,
            "lower_tokens": lowered,
            "filtered_tokens": filtered,
            "processed_tokens": stemmed,
            "lemmatized_tokens": lemmatized,
        })
    return out

def build_inverted_index(preprocessed_docs):
    idx = {}
    for d in preprocessed_docs:
        for token in set(d['processed_tokens']):
            idx.setdefault(token, set()).add(d['doc_id'])
    return idx

def compare_stemming_vs_lemmatization(docs):
    rows = []
    query_terms = ['markets', 'running', 'policies', 'learning', 'vaccines']
    processed = preprocess_documents(docs)
    for q in query_terms:
        stem_q = stemmer.stem(q.lower())
        lemma_q = lemmatizer.lemmatize(q.lower())
        stem_hits = sum(stem_q in d['processed_tokens'] for d in processed)
        lemma_hits = sum(lemma_q in d['lemmatized_tokens'] for d in processed)
        rows.append({
            'query_term': q,
            'stem_form': stem_q,
            'lemma_form': lemma_q,
            'stem_hit_count': stem_hits,
            'lemma_hit_count': lemma_hits,
        })
    df = pd.DataFrame(rows)
    stem_total = df['stem_hit_count'].sum()
    lemma_total = df['lemma_hit_count'].sum()
    recommendation = 'Lemmatization preserved more meaningful root words for this dataset.' if lemma_total >= stem_total else 'Stemming produced broader matching and is more suitable for this dataset.'
    return df, recommendation