import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def load_trained_model():
    # تحميل النموذج المدرب مسبقًا
    model = load_model("sentiment_model.h5")
    return model

def preprocess_text(df):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['Review'])
    sequences = tokenizer.texts_to_sequences(df['Review'])
    return pad_sequences(sequences, maxlen=500), tokenizer

def predict_sentiment(model, processed_text):
    predictions = model.predict(processed_text)
    return (predictions > 0.5).astype(int).flatten()

def extract_top_negative_words(negative_reviews):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(negative_reviews['Review'].dropna())
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    def get_top_keywords(cluster, X, vectorizer, n_words=5):
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_texts = X[cluster_indices].toarray().sum(axis=0)
        top_indices = cluster_texts.argsort()[-n_words:][::-1]
        return [vectorizer.get_feature_names_out()[i] for i in top_indices]
    
    cluster_words = {i: get_top_keywords(i, X, vectorizer) for i in range(5)}
    return cluster_words

# Streamlit UI
st.title("تحليل المشاعر في التقييمات")
uploaded_file = st.file_uploader("قم برفع ملف CSV يحتوي على التقييمات", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Review' not in df.columns or 'label' not in df.columns:
        st.error("يجب أن يحتوي الملف على عمود 'Review' و 'label'")
    else:
        model = load_trained_model()
        processed_text, tokenizer = preprocess_text(df)
        df['Predicted Label'] = predict_sentiment(model, processed_text)
        
        positive_count = np.sum(df['Predicted Label'] == 1)
        negative_count = np.sum(df['Predicted Label'] == 0)
        
        st.subheader("نتائج التحليل:")
        st.write(f"عدد التقييمات الإيجابية: {positive_count}")
        st.write(f"عدد التقييمات السلبية: {negative_count}")
        
        df_negative = df[df['Predicted Label'] == 0]
        if not df_negative.empty:
            top_words = extract_top_negative_words(df_negative)
            st.subheader("أكثر الكلمات السلبية شيوعًا:")
            for cluster, words in top_words.items():
                st.write(f"🔹 Cluster {cluster}: {', '.join(words)}")
            
            st.subheader("أمثلة على التعليقات السلبية:")
            st.write(df_negative[['Review']].sample(5))
