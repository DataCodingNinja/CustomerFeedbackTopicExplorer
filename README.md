Topic Modeling for Customer Feedback

One-line: End-to-end topic modeling pipeline (data generation → LDA modeling → outputs → Streamlit visualizer) that groups customer feedback into actionable topics.

Contents
- data_gen.py      — generate synthetic feedback CSV (data/feedback_sample.csv)
- topic_model.py   — preprocessing, gensim LDA training, outputs to outputs/
- app.py           — Streamlit app to explore topics (streamlit run app.py)
- requirements.txt — Python dependencies
- data/            — (generated) feedback_sample.csv
- outputs/         — (generated) topics_summary.csv, topic_freq.png, wordclouds/, lda.model, dictionary.dict, doc_topics.pkl

Quick setup (Linux/macOS / Windows PowerShell equivalent)
1. Create project folder and place files.
2. Create and activate virtualenv:
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\\Scripts\\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
4. Download spaCy model and NLTK data:
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


1. Generate synthetic data:
   python data_gen.py
   -> writes data/feedback_sample.csv (default N=1000). Edit N in file if you want fewer rows for speed.

2. Train model and produce outputs:
   python topic_model.py
   -> writes outputs/: topics_summary.csv, topic_freq.png, wordclouds/, lda.model, dictionary.dict, doc_topics.pkl

3. Visualize with Streamlit:
   streamlit run app.py
   -> open the local URL shown in terminal to explore topics, wordclouds, and example feedback.


Business impact / README summary
- This project groups customer feedback into topics (bugs, UX, pricing, feature requests, praise), enabling product teams to prioritize fixes without manual review.
- Deliverables: actionable topics_summary.csv with top words and sample feedback, visualizations (topic frequency, word clouds), and an interactive Streamlit app for stakeholders.
