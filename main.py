import os
import fitz  # PyMuPDF
import nltk
nltk.download("punkt")
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from synonyms import SYNONYM_MAP

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_resumes(folder_path="resumes"):
    resume_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(full_path)
            resume_texts[filename] = text
    return resume_texts

def load_job_description(path="job_description.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_keywords(text, top_n=15):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:top_n]]

def match_keywords_advanced(resume_text, keywords):
    ps = PorterStemmer()
    resume_tokens = word_tokenize(resume_text.lower())
    resume_stems = set(ps.stem(w) for w in resume_tokens)

    keyword_stems = set(ps.stem(k.lower()) for k in keywords)
    matches = [k for k in keywords if ps.stem(k.lower()) in resume_stems]

    match_percent = round(100 * len(matches) / len(keywords), 1) if keywords else 0.0
    return matches, match_percent

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

def rank_resumes(resume_texts, job_description_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode(job_description_text, convert_to_tensor=False)

    scored_resumes = []
    for filename, text in resume_texts.items():
        resume_embedding = model.encode(text, convert_to_tensor=False)
        score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        scored_resumes.append((filename, round(score, 4)))

    scored_resumes.sort(key=lambda x: x[1], reverse=True)
    return scored_resumes

# Standalone Test
if __name__ == "__main__":
    resumes = load_resumes()
    jd = load_job_description()

    base_keywords = extract_keywords(jd, top_n=15)
    jd_keywords = expand_keywords(base_keywords, SYNONYM_MAP)
    print(f"\nJob description keywords (expanded): {jd_keywords}")

    ranked = rank_resumes(resumes, jd)

    print("\nRanked resumes with keyword matches and match percentages:")
    for name, score in ranked:
        matches, percent = match_keywords_advanced(resumes[name], jd_keywords)
        print(f"{name}: {score:.4f} | Match: {percent}% | Keywords: {matches}")
