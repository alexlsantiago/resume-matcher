import streamlit as st
import os
import fitz  # PyMuPDF
import zipfile
import pandas as pd
import io
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from synonyms import SYNONYM_MAP

st.set_page_config(page_title="AI Job Matcher", layout="centered")

st.markdown("""
    <style>
        h1 a, h2 a, h3 a, h4 a {
            display: none !important;
        }
        .matched-key {
            background:#d0ebff;
            color:#003366;
            border-radius:4px;
            padding:2px 6px;
            margin: 2px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='white-space: nowrap; font-size: 2.4rem; margin-bottom: 1rem;'>Find Your Next Job</h1>
<p style='font-size: 1.2rem; margin-bottom: 1rem;'>Upload your resume and job descriptions to find the best matches.</p>
""", unsafe_allow_html=True)

# ----- Functions -----

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_keywords(text, top_n=15):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_scores[:top_n]]

def expand_keywords(keywords, synonym_map):
    expanded = set()
    for kw in keywords:
        for key, syns in synonym_map.items():
            if kw.lower() in syns:
                expanded.update(syns)
                break
        else:
            expanded.add(kw.lower())
    return list(expanded)

def match_keywords_advanced(resume_text, keywords):
    ps = PorterStemmer()
    resume_tokens = word_tokenize(resume_text.lower())
    resume_stems = set(ps.stem(w) for w in resume_tokens)
    matches = [k for k in keywords if ps.stem(k.lower()) in resume_stems]
    match_percent = round(100 * len(matches) / len(keywords), 1) if keywords else 0.0
    return matches, match_percent

def extract_jobs_from_zip(zip_file):
    jobs = []
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith(".txt"):
                content = zip_ref.read(name).decode("utf-8")
                lines = content.strip().split("\n")
                if len(lines) >= 3:
                    title_line = lines[0]
                    match = re.match(r"^(.*?) at (.*?)$", title_line)
                    if match:
                        title = match.group(1)
                        company = match.group(2)
                    else:
                        title = title_line
                        company = "Unknown"
                    link = lines[1].strip()
                    desc = "\n".join(lines[2:])
                    jobs.append({
                        "title": title,
                        "company": company,
                        "link": link,
                        "description": desc
                    })
    return jobs

# ----- Upload Inputs -----

resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
zip_files = st.file_uploader("Upload One or More Job ZIP Files", type=["zip"], accept_multiple_files=True)

st.markdown("""
<p style='font-size: 0.95rem; margin-top: -0.5rem; margin-bottom: 1.5rem;'>
    Don’t have a ZIP of job descriptions? <a href="https://your-job-scraper-url.com" target="_blank" style="color: #4f8bf9; text-decoration: underline;">Click here to scrape jobs</a>.
</p>
""", unsafe_allow_html=True)

# ----- Analyze Button -----

if st.button("🔍 Analyze Fit"):
    if not resume_file or not zip_files:
        st.error("Please upload both a resume and at least one ZIP file of job descriptions.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file.read())
            all_jobs = []
            for zf in zip_files:
                all_jobs.extend(extract_jobs_from_zip(zf))

            model = SentenceTransformer('all-MiniLM-L6-v2')
            resume_embedding = model.encode(resume_text, convert_to_tensor=False)

            results = []
            for job in all_jobs:
                jd_embedding = model.encode(job["description"], convert_to_tensor=False)
                similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]

                base_keywords = extract_keywords(job["description"], top_n=20)
                expanded_keywords = expand_keywords(base_keywords, SYNONYM_MAP)
                matched, match_pct = match_keywords_advanced(resume_text, expanded_keywords)

                results.append({
                    "title": job["title"],
                    "company": job["company"],
                    "link": job["link"],
                    "score": round(similarity, 2),
                    "match_pct": match_pct,
                    "matched": matched
                })

            results.sort(key=lambda x: x["match_pct"], reverse=True)
            df = pd.DataFrame(results)

            st.session_state["results"] = results
            st.session_state["df"] = df

# ----- Display Results -----

if "results" in st.session_state:
    st.subheader("📄 Job Matches")

    st.download_button(
        label="📅 Download CSV of Results",
        data=st.session_state["df"].to_csv(index=False),
        file_name="resume_vs_jobs.csv",
        mime="text/csv"
    )

    for job in st.session_state["results"]:
        st.markdown(f"**{job['title']}** at *{job['company']}*")
        st.markdown(f"[Apply Here]({job['link']})")
        st.write(f"Relevance Score: `{job['score']:.2f}` | Keyword Match: `{job['match_pct']}%`")

        if job['matched']:
            st.markdown("Matched Keywords:")
            st.markdown("".join([f"<span class='matched-key'>{kw}</span>" for kw in job['matched']]), unsafe_allow_html=True)
        else:
            st.markdown("_No matched keywords._")

        st.markdown("---")
