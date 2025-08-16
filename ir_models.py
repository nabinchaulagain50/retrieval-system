import json, math, re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]

class CorpusIndex:
    def __init__(self, docs):
        self.docs = docs
        self.N = len(docs)
        self.doclen = {}
        self.df = Counter()
        self.postings = {}  # term -> {doc_id: tf}
        for d in docs:
            terms = tokenize(d["title"] + " " + d["text"])
            self.doclen[d["doc_id"]] = len(terms)
            tf = Counter(terms)
            for term, c in tf.items():
                self.df[term] += 1
                self.postings.setdefault(term, {})[d["doc_id"]] = c
        self.avgdl = sum(self.doclen.values()) / self.N if self.N else 0.0

    def idf_smooth(self, term):
        # Natural log idf w/ smoothing (BM25-style)
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def idf_plain(self, term):
        # Plain idf for TF-IDF
        df = self.df.get(term, 0)
        return math.log((self.N + 1) / (df + 1))

class VSM:
    """TF-IDF + cosine similarity"""
    def __init__(self, index: CorpusIndex):
        self.idx = index

    def score(self, query, k=10):
        q_terms = tokenize(query)
        q_tf = Counter(q_terms)
        q_idf = {t: self.idx.idf_plain(t) for t in q_tf}

        numer = Counter()
        doc_norm = Counter()
        q_norm = 0.0
        for t, qf in q_tf.items():
            idf = q_idf[t]
            qw = (1 + math.log(qf)) * idf
            q_norm += qw * qw
            if t not in self.idx.postings:
                continue
            for doc_id, tf in self.idx.postings[t].items():
                dw = (1 + math.log(tf)) * idf
                numer[doc_id] += qw * dw
                doc_norm[doc_id] += dw * dw
        q_norm = math.sqrt(q_norm) if q_norm > 0 else 1.0

        scored = []
        for doc_id, num in numer.items():
            sim = num / (math.sqrt(doc_norm[doc_id]) * q_norm) if doc_norm[doc_id] > 0 else 0.0
            scored.append((doc_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

class BM25:
    def __init__(self, index: CorpusIndex, k1=1.5, b=0.75):
        self.idx = index
        self.k1 = k1
        self.b = b

    def score(self, query, k=10):
        q_terms = tokenize(query)
        scores = Counter()
        for t in q_terms:
            if t not in self.idx.postings:
                continue
            idf = self.idx.idf_smooth(t)
            for doc_id, tf in self.idx.postings[t].items():
                dl = self.idx.doclen[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.idx.avgdl)
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
