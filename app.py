import streamlit as st
import joblib


@st.cache_resource
def load_model():
    return joblib.load('phishing_model.pkl'), joblib.load('vectorizer.pkl')

model, vectorizer = load_model()


st.title("E-posta Güvenlik Analizi")

email_text = st.text_area("Analiz edilecek e-posta içeriğini yapıştırın:", height=250)

if st.button("Analizi Başlat"):
    if len(email_text.strip()) > 10:
        probs = model.predict_proba(vectorizer.transform([email_text]))[0]
        if probs[1] > 0.5:
            st.error(f"KRİTİK: Oltalama Riski! (%{probs[1]*100:.1f})")
        else:
            st.success(f"GÜVENLİ: Tehdit saptanmadı. (%{probs[0]*100:.1f})")
    else:
        st.warning("Lütfen metin girin.")