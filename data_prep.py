import json, random
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

root = Path("data")
root.mkdir(exist_ok=True)

random.seed(7)
topics = [
    ("information retrieval", [
        "tf idf cosine similarity vector space model document ranking tokens query relevance",
        "bm25 okapi retrieval function probabilistic model term frequency inverse document length",
        "indexing inverted index postings list skip pointers compression",
        "evaluation precision recall map mrr ndcg trec topics qrels",
        "query expansion relevance feedback pseudo relevance rm3",
        "stemming lemmatization tokenization stopwords normalization",
    ]),
    ("web search", [
        "search engine crawling indexing ranking pagerank link analysis",
        "clueweb gov2 wt10g trec web track ad hoc",
        "spam filtering web graph anchor text",
        "click models user behavior dwell time satisfaction",
        "query logs sessionization reformulation",
        "learning to rank pairwise listwise lambdamart xgboost",
    ]),
    ("nlp ml", [
        "word embeddings word2vec glove fasttext distributional semantics",
        "transformers bert roberta gpt attention mechanism fine tuning",
        "classification logistic regression svm naive bayes",
        "deep learning neural networks cnn rnn lstm gru",
        "token classification ner pos tagging chunking",
        "topic modeling lda nmf coherence perplexity",
    ]),
    ("programming", [
        "python javascript java csharp golang rust programming languages",
        "data structures algorithms complexity big o",
        "unit testing continuous integration deployment",
        "docker containers kubernetes orchestration",
        "rest api design oauth jwt",
        "software architecture microservices monolith hexagonal",
    ]),
    ("health", [
        "medical records privacy hipaa interoperability",
        "clinical trials pubmed mesh queries",
        "biomedical text mining entity linking",
        "genomics sequencing variant calling",
        "covid vaccination efficacy rna vaccines",
        "epidemiology incidence prevalence risk factors",
    ]),
]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def normalize(text):
    tokens = word_tokenize(text.lower())
    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return {
        "original": text,
        "stemmed": " ".join(stemmed),
        "lemmatized": " ".join(lemmatized)
    }

docs = []
doc_id = 1
for theme, sentences in topics:
    for s in sentences:
        title = f"{theme.title()} Doc {doc_id}"
        text = f"{s}. This document discusses {theme} and related ideas."
        norm = normalize(text)
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "text": norm["original"],
            "text_stemmed": norm["stemmed"],
            "text_lemmatized": norm["lemmatized"]
        })
        doc_id += 1

extra = [
    "football world cup fifa players goals match",
    "mountaineering everest nepal trekking hiking altitude",
    "ecommerce checkout payment gateway cart conversion",
    "climate change greenhouse gases co2 emissions mitigation",
    "education university coursework assignments grading",
    "economics inflation interest rates monetary policy",
]
for s in extra:
    norm = normalize(s + ". Short note.")
    docs.append({
        "doc_id": doc_id,
        "title": f"Misc Doc {doc_id}",
        "text": norm["original"],
        "text_stemmed": norm["stemmed"],
        "text_lemmatized": norm["lemmatized"]
    })
    doc_id += 1

with open(root/"corpus.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

queries = [
    {"qid": "Q1", "query": "bm25 vs tf idf for ad hoc retrieval"},
    {"qid": "Q2", "query": "trec web track clueweb gov2 wt10g"},
    {"qid": "Q3", "query": "evaluation metrics precision recall map mrr"},
    {"qid": "Q4", "query": "pagerank link analysis search engine ranking"},
    {"qid": "Q5", "query": "bert transformers fine tuning for text classification"},
]
for q in queries:
    norm = normalize(q["query"])
    q["query_stemmed"] = norm["stemmed"]
    q["query_lemmatized"] = norm["lemmatized"]

with open(root/"queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

qrels = {
    "Q1": [2, 1, 4],
    "Q2": [8, 9, 10],
    "Q3": [4, 1],
    "Q4": [7],
    "Q5": [14, 15],
}
with open(root/"qrels.json", "w", encoding="utf-8") as f:
    json.dump(qrels, f, ensure_ascii=False, indent=2)

print("Wrote data/corpus.json, data/queries.json, data/qrels.json")