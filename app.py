"""DeepTrace Streamlit app: AI-text probability plus passage-level similarity."""
import streamlit as st

from deeptrace.chunk import make_windows, split_sentences
from deeptrace.detector import AITextDetector
from deeptrace.index import PassageIndex, load_embedder
from deeptrace.plagiarism import check_plagiarism
from deeptrace.scraper import fetch_many

st.set_page_config(page_title="DeepTrace", layout="wide")
st.title("DeepTrace")
st.caption(
    "Estimates how likely text is AI-generated and flags passages that closely "
    "match the reference pages you provide."
)


@st.cache_resource(show_spinner="Loading RoBERTa detector...")
def get_detector() -> AITextDetector:
    return AITextDetector()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return load_embedder()


text = st.text_area("Text to check", height=250)
urls_raw = st.text_area(
    "Reference URLs to compare against (one per line, optional)", height=100
)
threshold = st.slider("Similarity threshold for flagging", 0.5, 1.0, 0.85, 0.01)

if st.button("Analyze", type="primary") and text.strip():
    urls = [
        u.strip() for u in urls_raw.splitlines() if u.strip().startswith(("http://", "https://"))
    ]

    with st.spinner("Analyzing..."):
        ai_probability = get_detector().score(text)

        index = PassageIndex(get_embedder())
        pages = fetch_many(urls)
        for url, body in pages.items():
            passages = make_windows(split_sentences(body))
            index.add(passages, [url] * len(passages))

        flags = check_plagiarism(text, index, threshold=threshold)

    st.subheader("AI-generated probability")
    st.metric("RoBERTa estimate", f"{ai_probability:.0%}")
    st.caption("Based on the first 512 tokens. Treat this as a signal, not proof.")

    st.subheader("Matched passages")
    if urls and len(pages) < len(urls):
        st.warning(f"Could not fetch {len(urls) - len(pages)} of {len(urls)} URLs.")
    if not flags:
        st.info("No passages reached the similarity threshold.")
    for flag in flags:
        with st.container(border=True):
            st.markdown(f"**Score {flag['score']:.2f}** from {flag['source']}")
            st.write("Submitted:", flag["text"])
            st.write("Reference:", flag["matched"])
