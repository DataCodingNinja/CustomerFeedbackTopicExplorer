# app.py
"""
Streamlit visualization for topic modeling outputs.
Run: streamlit run app.py
Requires outputs/ produced by topic_model.py
"""
import streamlit as st
import pandas as pd
import os
from PIL import Image

OUTPUT_DIR = "outputs"

st.set_page_config(page_title="Customer Feedback Topics", layout="wide")
st.title("Customer Feedback — Topic Explorer")

if not os.path.exists(OUTPUT_DIR):
    st.error("Run topic_model.py first to generate outputs in the outputs/ folder.")
    st.stop()

topics_csv = os.path.join(OUTPUT_DIR, "topics_summary.csv")
doc_topics_pkl = os.path.join(OUTPUT_DIR, "doc_topics.pkl")
wordcloud_dir = os.path.join(OUTPUT_DIR, "wordclouds")
freq_img = os.path.join(OUTPUT_DIR, "topic_freq.png")

topics_df = pd.read_csv(topics_csv)
doc_df = pd.read_pickle(doc_topics_pkl)

st.sidebar.header("Filters")
sel_topic = st.sidebar.selectbox("Select topic", options=["All"] + topics_df["topic_id"].astype(str).tolist())
min_rating = st.sidebar.slider("Minimum rating", 1, 5, 1)

st.header("Topic frequencies")
st.image(freq_img, use_column_width=True)

st.header("Topics")
for _, row in topics_df.iterrows():
    tid = str(row["topic_id"])
    if sel_topic != "All" and sel_topic != tid:
        continue
    st.subheader(f"Topic {tid} — top words: {row['top_words']}")
    wc_path = os.path.join(wordcloud_dir, f"topic_wordcloud_{tid}.png")
    if os.path.exists(wc_path):
        st.image(wc_path, width=400)
    st.markdown("**Examples:**")
    examples = row["examples"].split(" || ")
    for ex in examples:
        st.write(f"- {ex}")

st.sidebar.markdown("---")
st.sidebar.markdown("Quick actions")
if st.sidebar.button("Show sample docs for selected topic"):
    if sel_topic == "All":
        st.info("Select a specific topic in the sidebar.")
    else:
        t = int(sel_topic)
        subset = doc_df[(doc_df["dominant_topic"]==t) & (doc_df["rating"]>=min_rating)].head(10)
        st.write(subset[["id","rating","source","topic_prob","text"]])
