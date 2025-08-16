import json
import pandas as pd
import streamlit as st
from ir_models import CorpusIndex, VSM, BM25

def precision_at_k(ranked, rel_set, k):
    top = [d for d,_ in ranked[:k]]
    hits = sum(1 for d in top if d in rel_set)
    return hits / max(1, k)

def recall_at_k(ranked, rel_set, k):
    top = [d for d,_ in ranked[:k]]
    hits = sum(1 for d in top if d in rel_set)
    return hits / max(1, len(rel_set))

def average_precision(ranked, rel_set):
    hits = 0
    sum_prec = 0.0
    for i, (d, _) in enumerate(ranked, start=1):
        if d in rel_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / max(1, len(rel_set))

def mrr_score(ranked, rel_set):
    for i, (d, _) in enumerate(ranked, start=1):
        if d in rel_set:
            return 1.0 / i
    return 0.0

def run_eval(model_name, k=10):
    with open("corpus.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open("qrels.json", "r", encoding="utf-8") as f:
        qrels = json.load(f)

    index = CorpusIndex(docs)
    model = BM25(index) if model_name == "BM25" else VSM(index)

    results = []
    for q in queries:
        qid = q.get("qid") or q.get("query_id")
        qtext = q.get("query") or q.get("text")
        rel_set = set(qrels.get(str(qid), []))
        ranked = model.score(qtext, k=50)
        row = {
            "qid": qid,
            "query": qtext,
            "AP": average_precision(ranked, rel_set),
            "MRR": mrr_score(ranked, rel_set),
            f"P@{k}": precision_at_k(ranked, rel_set, k),
            f"R@{k}": recall_at_k(ranked, rel_set, k),
        }
        results.append(row)
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("IR Model: Search & Evaluation (BM25 vs VSM)")

# Search section
st.header("🔎 Search the Corpus")
with open("corpus.json", "r", encoding="utf-8") as f:
    docs = json.load(f)
index = CorpusIndex(docs)
query_text = st.text_input("Enter your search query", "bm25 vs tf idf for ad hoc retrieval")
top_k = st.slider("Top K Results", 1, 20, 10)
if st.button("Search"):
    bm25 = BM25(index)
    vsm = VSM(index)
    bm25_results = bm25.score(query_text, k=top_k)
    vsm_results = vsm.score(query_text, k=top_k)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BM25 Results")
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            doc = next(d for d in docs if d["doc_id"] == doc_id)
            st.markdown(f"**#{rank}. {doc['title']}**  \nScore: `{score:.4f}`")
            st.write(doc["text"])
            st.markdown("---")
    with col2:
        st.subheader("VSM Results")
        for rank, (doc_id, score) in enumerate(vsm_results, 1):
            doc = next(d for d in docs if d["doc_id"] == doc_id)
            st.markdown(f"**#{rank}. {doc['title']}**  \nScore: `{score:.4f}`")
            st.write(doc["text"])
            st.markdown("---")

# Evaluation section
st.header("📊 Model Evaluation (MAP, MRR, AP, Precision, Recall)")
model_name = st.selectbox("Select Model", ["BM25", "VSM"])
k = st.slider("k for P@k / R@k", 1, 20, 10, key="eval_k")
if st.button("Run Evaluation"):
    df = run_eval(model_name, k)
    st.dataframe(df.style.format({"AP": "{:.3f}", "MRR": "{:.3f}", f"P@{k}": "{:.3f}", f"R@{k}": "{:.3f}"}))
    st.markdown("### Mean Scores")
    st.write({
        "MAP": round(df["AP"].mean(), 3),
        "MRR": round(df["MRR"].mean(), 3),
        f"P@{k}": round(df[f"P@{k}"].mean(), 3),
        f"R@{k}": round(df[f"R@{k}"].mean(), 3),

    })

