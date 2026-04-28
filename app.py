import re
import json
import joblib
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lda = joblib.load('./model/topic_model_lda.pkl')
    topic_vectorizer = joblib.load('./model/topic_vectorizer.pkl')
    with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    topic_labels = {int(k): v for k, v in topic_labels.items()}
    sentiment_model = joblib.load('./model/sentiment_classifier.pkl')
    tfidf = joblib.load('./model/topic_vectorizer_using_tfidf.pkl')
    return lda, topic_vectorizer, topic_labels, sentiment_model, tfidf

lda, topic_vectorizer, topic_labels, sentiment_model, tfidf = load_models()

# ── NLP tools ─────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text_for_topic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

def get_topic(text):
    cleaned = clean_text_for_topic(text)
    X = topic_vectorizer.transform([cleaned])
    topic_probs = lda.transform(X)[0]
    topic_id = int(np.argmax(topic_probs))
    topic_label = topic_labels.get(topic_id, 'Unlabeled')
    topic_probability = float(topic_probs[topic_id])
    return topic_label, topic_probability

def get_sentiment(text):
    cleaned = clean_text_for_sentiment(text)
    vector = tfidf.transform([cleaned])
    pred = sentiment_model.predict(vector)[0]
    proba = sentiment_model.predict_proba(vector).max()
    return pred, round(float(proba), 3)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Review Analyser", page_icon="◈", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
}

.main-header {
    text-align: center;
    padding: 3rem 0 1rem 0;
}

.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    color: #f0ece4;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}

.main-header p {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b6b7a;
    font-weight: 300;
    margin-top: 0.75rem;
    letter-spacing: 0.02em;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a3a, transparent);
    margin: 2rem 0;
}

.result-card {
    background: #13131c;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin: 1rem 0;
}

.result-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a4a5e;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: #f0ece4;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}

.result-sub {
    font-size: 0.8rem;
    font-weight: 300;
    color: #4a4a5e;
}

.sentiment-positive { color: #4ade80; }
.sentiment-negative { color: #f87171; }
.sentiment-neutral  { color: #94a3b8; }

.confidence-bar-bg {
    height: 3px;
    background: #1e1e2e;
    border-radius: 2px;
    margin-top: 0.75rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

.stTextArea textarea {
    background: #13131c !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    padding: 1rem !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: #3a3a5e !important;
    box-shadow: 0 0 0 1px #3a3a5e !important;
}

.stButton button {
    background: #f0ece4 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s ease !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
}

label[data-testid="stWidgetLabel"] {
    color: #4a4a5e !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Review Analyser</h1>
    <p>Identify the topic and sentiment of any hotel review</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Review text",
    height=140,
    placeholder="The room was spotless and the front desk staff went above and beyond..."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyse = st.button("Analyse")

# ── Output ────────────────────────────────────────────────────────────────────
if analyse:
    if not user_input.strip():
        st.warning("Please enter some text before analysing.")
    else:
        topic_label, topic_prob = get_topic(user_input)
        sentiment, confidence = get_sentiment(user_input)

        sentiment_class = f"sentiment-{sentiment.lower()}"
        sentiment_color = {"positive": "#4ade80", "negative": "#f87171"}.get(sentiment.lower(), "#94a3b8")
        topic_bar_color = "#3a3a6e"
        confidence_pct = int(confidence * 100)
        topic_pct = int(topic_prob * 100)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Topic</div>
                <div class="result-value">{topic_label}</div>
                <div class="result-sub">Probability {topic_pct}%</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{topic_pct}%; background:{topic_bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Sentiment</div>
                <div class="result-value {sentiment_class}">{sentiment.capitalize()}</div>
                <div class="result-sub">Confidence {confidence_pct}%</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence_pct}%; background:{sentiment_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)