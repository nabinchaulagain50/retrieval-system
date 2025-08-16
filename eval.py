import json
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

def run_eval(corpus_path="data/corpus.json",
             queries_path="data/queries.json",
             qrels_path="data/qrels.json",
             k_list=(5,10)):
    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    index = CorpusIndex(docs)
    vsm = VSM(index)
    bm25 = BM25(index)

    results = []
    for q in queries:
        qid = q["qid"]; qtext = q["query"]
        rel_set = set(qrels.get(qid, []))
        row = {"qid": qid, "query": qtext}

        ranked_vsm = vsm.score(qtext, k=50)
        ranked_bm25 = bm25.score(qtext, k=50)

        for k in k_list:
            row[f"VSM_P@{k}"]  = round(precision_at_k(ranked_vsm,  rel_set, k), 3)
            row[f"VSM_R@{k}"]  = round(recall_at_k(ranked_vsm,     rel_set, k), 3)
            row[f"BM25_P@{k}"] = round(precision_at_k(ranked_bm25, rel_set, k), 3)
            row[f"BM25_R@{k}"] = round(recall_at_k(ranked_bm25,    rel_set, k), 3)

        row["VSM_AP"]   = round(average_precision(ranked_vsm,  rel_set), 3)
        row["BM25_AP"]  = round(average_precision(ranked_bm25, rel_set), 3)
        row["VSM_MAP"]  = round(average_precision(ranked_vsm,  rel_set), 3)
        row["BM25_MAP"] = round(average_precision(ranked_bm25, rel_set), 3)
        row["VSM_MRR"]  = round(mrr_score(ranked_vsm,  rel_set), 3)
        row["BM25_MRR"] = round(mrr_score(ranked_bm25, rel_set), 3)

        results.append(row)
    return results

if __name__ == "__main__":
    rows = run_eval()
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Calculate and print mean metrics for both models
    print("\n=== Evaluation Summary ===")
    for model in ["VSM", "BM25"]:
        mean_map = df[f"{model}_MAP"].mean()
        mean_mrr = df[f"{model}_MRR"].mean()
        mean_ap = df[f"{model}_AP"].mean()
        mean_p5 = df[f"{model}_P@5"].mean()
        mean_r5 = df[f"{model}_R@5"].mean()
        mean_p10 = df[f"{model}_P@10"].mean()
        mean_r10 = df[f"{model}_R@10"].mean()
        print(f"{model} - MAP: {mean_map:.3f} | MRR: {mean_mrr:.3f} | AP: {mean_ap:.3f} | "
              f"P@5: {mean_p5:.3f} | R@5: {mean_r5:.3f} | P@10: {mean_p10:.3f} | R@10: {mean_r10:.3f}")