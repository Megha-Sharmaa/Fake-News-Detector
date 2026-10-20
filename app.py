import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load your trained pipeline (make sure you saved it as model.pkl)
pipeline = joblib.load("model.pkl")

# App title and description
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♀️", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("""
Welcome to the **Fake News Detection App**!  
Paste a headline or short article below and let the model decide whether it's **REAL ✅** or **FAKE 🚨**.
""")

# Input box
user_input = st.text_area("✍️ Enter a news headline/article:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() != "":
        # Prediction
        pred = pipeline.predict([user_input])[0]
        prob = None
        try:
            prob = pipeline.predict_proba([user_input])[0]
        except:
            pass  # LinearSVC doesn't support predict_proba

        label = "REAL ✅" if pred == 1 else "FAKE 🚨"

        # Result card
        st.markdown(f"## Prediction: {label}")

        # Confidence bar chart (if probabilities available)
        if prob is not None:
            fig, ax = plt.subplots()
            categories = ["FAKE", "REAL"]
            ax.bar(categories, prob, color=["red", "green"])
            ax.set_ylabel("Confidence")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        # Progress bar (visual confidence)
        if prob is not None:
            st.progress(int(prob[1] * 100))  # confidence for REAL
        else:
            st.info("⚠️ Confidence scores not available (LinearSVC).")

    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
