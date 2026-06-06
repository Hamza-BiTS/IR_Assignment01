import streamlit as st
import pandas as pd
from modules.data_loader import load_documents_from_uploads, load_sample_documents
from modules.preprocessing import preprocess_documents, build_inverted_index, compare_stemming_vs_lemmatization
from modules.phrase_queries import build_biword_index, build_positional_index, biword_phrase_search, positional_phrase_search
from modules.tree_structures import run_tree_benchmark
from modules.tolerant_retrieval import build_kgram_index, wildcard_search, suggest_corrections, tolerant_search

st.set_page_config(page_title="IR Assignment Dashboard", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px;}
[data-testid="stSidebar"] {background: #f7f8fa;}
.ir-card {background:#ffffff;border:1px solid #e6e8eb;padding:16px 18px;border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);margin-bottom:14px;}
.small-note {color:#5f6368;font-size:0.92rem;}
.title-wrap {padding: 0.2rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-wrap"><h1>Information Retrieval Dashboard</h1><p class="small-note">End-to-end Streamlit workflow for preprocessing, indexing, phrase querying, tree comparison, and tolerant retrieval.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    use_sample = st.toggle("Use built-in sample dataset", value=True)
    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    page = st.radio(
        "Navigate",
        [
            "Home",
            "Preprocessing",
            "Stemming vs Lemmatization",
            "Phrase Query",
            "BST vs B-Tree",
            "Tolerant Retrieval",
            "Inference",
        ],
    )

if use_sample:
    docs = load_sample_documents("data")
else:
    docs = load_documents_from_uploads(uploaded_files)

if not docs:
    st.warning("Please upload text documents or enable the sample dataset.")
    st.stop()

preprocessed = preprocess_documents(docs)
inverted_index = build_inverted_index(preprocessed)
biword_index = build_biword_index(preprocessed)
positional_index = build_positional_index(preprocessed)
kgram_index = build_kgram_index(list(inverted_index.keys()), k=3)
comparison_df, recommendation = compare_stemming_vs_lemmatization(docs)

if page == "Home":
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Vocabulary Size", len(inverted_index))
    c3.metric("Biword Terms", len(biword_index))

    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Uploaded Documents")
    doc_table = pd.DataFrame(
        [{"doc_id": d["doc_id"], "title": d["title"], "characters": len(d["text"])} for d in docs]
    )
    st.dataframe(doc_table, use_container_width=True)
    chosen_doc = st.selectbox("View document", options=[d["doc_id"] for d in docs])
    selected = next(d for d in docs if d["doc_id"] == chosen_doc)
    st.text_area("Document content", selected["text"], height=280)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Preprocessing":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Preprocessing Effects")
    doc_id = st.selectbox("Select document", options=[d["doc_id"] for d in preprocessed])
    row = next(d for d in preprocessed if d["doc_id"] == doc_id)
    st.write("Original tokens")
    st.code(", ".join(row["original_tokens"][:80]))
    st.write("After lowercasing + stopword removal + hyphen handling + stemming")
    st.code(", ".join(row["processed_tokens"][:80]))
    st.write("Inverted Index Sample")
    sample_terms = sorted(list(inverted_index.keys()))[:20]
    sample_df = pd.DataFrame({"term": sample_terms, "postings": [sorted(list(inverted_index[t])) for t in sample_terms]})
    st.dataframe(sample_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Stemming vs Lemmatization":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    st.info(recommendation)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Phrase Query":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Biword vs Positional Index")
    phrase = st.text_input("Enter phrase query", value="climate change")
    b_results = biword_phrase_search(phrase, biword_index, preprocessed)
    p_results = positional_phrase_search(phrase, positional_index, preprocessed)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Biword Results")
        st.write(b_results if b_results else "No matches")
    with col2:
        st.write("Positional Results")
        st.write(p_results if p_results else "No matches")
    st.caption("Biword index may produce false positives because adjacent biwords can appear without the full exact phrase at the same positions, while positional index checks token locations explicitly.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "BST vs B-Tree":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Performance Experiment")
    query_terms = st.text_input("Benchmark query terms (comma-separated)", value="climate, market, learning, vaccine, energy")
    terms = [t.strip().lower() for t in query_terms.split(",") if t.strip()]
    bench_df, bench_note = run_tree_benchmark(sorted(list(inverted_index.keys())), terms)
    st.dataframe(bench_df, use_container_width=True)
    st.info(bench_note)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Tolerant Retrieval":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Imperfect Query Handling")
    mode = st.selectbox("Method", ["Wildcard", "Spelling Correction", "Tolerant Search"])
    q = st.text_input("Enter imperfect query", value="comput*")
    if mode == "Wildcard":
        result = wildcard_search(q, list(inverted_index.keys()))
        st.write(result if result else "No wildcard matches")
    elif mode == "Spelling Correction":
        suggestions = suggest_corrections(q, list(inverted_index.keys()), kgram_index)
        st.write(suggestions if suggestions else "No suggestions")
    else:
        result = tolerant_search(q, inverted_index, kgram_index, preprocessed)
        st.write(result)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Inference":
    st.markdown('<div class="ir-card">', unsafe_allow_html=True)
    st.subheader("Inference and Discussion")
    st.markdown(
        f"""
1. Preprocessing improved retrieval quality by reducing noise through lowercasing, stopword removal, and normalization.
2. **Stemming vs Lemmatization:** {recommendation}
3. Positional index is more accurate for phrase queries because it validates exact term positions.
4. B-Tree style multi-way search is generally faster and more scalable than a plain BST for larger dictionaries.
5. Tolerant retrieval improves user experience by handling wildcards and misspellings.
6. Limitations: small dataset, simple scoring, and approximate B-Tree simulation.
7. Improvements: BM25 ranking, larger dataset, phonetic matching, and better relevance evaluation.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)