import streamlit as st
from predict import predict_news
from news_fetcher import fetch_latest_news

st.set_page_config(page_title="📰 Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news headline is **Fake** or **Real** using a trained ML model.")

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Manual Input")
    news_text = st.text_area("Paste news content or headline here", height=150)
    if st.button("🔍 Predict"):
        if news_text.strip():
            result, prob = predict_news(news_text)
            st.success(f"**Prediction:** {result} ({prob:.2f}% confidence)")
            st.progress(int(prob))
            st.markdown("✅ Looks legit!" if result == "REAL" else "🚨 This might be misleading!")
        else:
            st.warning("Please enter some text to predict.")

with col2:
    st.subheader("🌐 Fetch Live News")
    if st.button("📰 Get Top 5 Headlines"):
        headlines = fetch_latest_news()
        if headlines:
            for i, headline in enumerate(headlines, start=1):
                result, prob = predict_news(headline)
                with st.expander(f"{i}. {headline}"):
                    st.write(f"**Prediction:** {result} ({prob:.2f}%)")
                    st.progress(int(prob))
        else:
            st.error("❌ Failed to fetch news. Please check your API key or internet connection.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn.")
