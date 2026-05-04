# topic_model.py
"""
End-to-end topic modeling pipeline using Gensim LDA.
Inputs: data/feedback_sample.csv
Outputs:
 - outputs/topics_summary.csv
 - outputs/topic_freq.png
 - outputs/topic_wordcloud_<id>.png
 - outputs/lda_model.* (gensim save)
Runs fast with modest resources (N=1000).
"""
import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import spacy
from wordcloud import WordCloud

# Settings
DATA_PATH = "data/feedback_sample.csv"
OUTPUT_DIR = "outputs"
SPACY_MODEL = "en_core_web_sm"
NUM_TOPICS = 8
PASSES = 10
RANDOM_STATE = 42

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+|\S+@\S+"," ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_lemmatize(nlp, text):
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t)>2]
    return tokens

def build_corpus(texts, no_below=5, no_above=0.5, keep_n=5000):
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(t) for t in texts]
    return dictionary, corpus

def train_lda(corpus, dictionary, num_topics=8, passes=10, random_state=42):
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                   random_state=random_state, passes=passes, chunksize=200, minimum_probability=0.01)
    return lda

def get_coherence(lda, texts, dictionary):
    cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
    return cm.get_coherence()

def topic_top_words(lda, id2word, topn=10):
    topics = {}
    for tid in range(lda.num_topics):
        terms = lda.show_topic(tid, topn=topn)
        topics[tid] = [w for w, _ in terms]
    return topics

def doc_topic_df(lda, corpus, texts, original_df):
    rows = []
    for i, bow in enumerate(corpus):
        topics = lda.get_document_topics(bow, minimum_probability=0.0)
        topics = sorted(topics, key=lambda x:-x[1])
        top_id, top_prob = topics[0]
        rows.append({"id": original_df.iloc[i]["id"],
                     "text": original_df.iloc[i]["text"],
                     "rating": original_df.iloc[i].get("rating", np.nan),
                     "source": original_df.iloc[i].get("source", ""),
                     "dominant_topic": int(top_id),
                     "topic_prob": float(top_prob)})
    return pd.DataFrame(rows)

def plot_topic_freq(df, outpath):
    freq = df["dominant_topic"].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    sns.barplot(x=freq.index, y=freq.values, palette="muted")
    plt.xlabel("Topic ID"); plt.ylabel("Document Count"); plt.title("Topic Frequencies")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_wordclouds(topics, outdir):
    os.makedirs(outdir, exist_ok=True)
    for tid, words in topics.items():
        txt = " ".join(words*8)
        wc = WordCloud(width=600, height=400, background_color="white").generate(txt)
        wc.to_file(os.path.join(outdir, f"topic_wordcloud_{tid}.png"))

def save_topic_summary(topics, doc_df, outpath):
    rows = []
    freq = doc_df["dominant_topic"].value_counts().to_dict()
    for tid, words in topics.items():
        examples = doc_df[doc_df["dominant_topic"]==tid].sort_values("topic_prob", ascending=False).head(3)["text"].tolist()
        rows.append({
            "topic_id": tid,
            "top_words": " ".join(words),
            "frequency": int(freq.get(tid,0)),
            "examples": " || ".join(examples)
        })
    pd.DataFrame(rows).to_csv(outpath, index=False)

def main(args=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # NLP pipeline
    nlp = spacy.load(SPACY_MODEL, disable=["ner","parser"])
    texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
    tokenized = [tokenize_lemmatize(nlp, t) for t in texts]

    # build corpus
    dictionary, corpus = build_corpus(tokenized, no_below=5, no_above=0.5, keep_n=5000)
    print(f"Dictionary size: {len(dictionary)}. Sample tokens: {list(dictionary.token2id.keys())[:10]}")

    # train lda
    lda = train_lda(corpus, dictionary, num_topics=NUM_TOPICS, passes=PASSES, random_state=RANDOM_STATE)
    coherence = get_coherence(lda, tokenized, dictionary)
    print(f"Trained LDA with {NUM_TOPICS} topics. Coherence (c_v): {coherence:.4f}")

    # topics
    topics = topic_top_words(lda, dictionary, topn=10)
    docdf = doc_topic_df(lda, corpus, tokenized, df)

    # outputs
    plot_topic_freq(docdf, os.path.join(OUTPUT_DIR, "topic_freq.png"))
    save_wordclouds(topics, os.path.join(OUTPUT_DIR, "wordclouds"))
    save_topic_summary(topics, docdf, os.path.join(OUTPUT_DIR, "topics_summary.csv"))

    # save model and dictionary
    lda.save(os.path.join(OUTPUT_DIR, "lda.model"))
    dictionary.save(os.path.join(OUTPUT_DIR, "dictionary.dict"))
    with open(os.path.join(OUTPUT_DIR, "doc_topics.pkl"), "wb") as f:
        pickle.dump(docdf, f)

    print("Outputs saved in", OUTPUT_DIR)
    print("Top words per topic:")
    for tid, words in topics.items():
        print(f"{tid}: {', '.join(words)}")

if __name__ == "__main__":
    main()
