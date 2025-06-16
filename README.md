# AI Resume vs Job Descriptions Matcher

This web app intelligently compares your resume against one or more job descriptions to help you assess how well your experience matches specific roles.

### Features
- Upload your resume (PDF)
- Upload one or more job ZIP files (exported from our [Job Scraper Tool](https://github.com/alexlsantiago/job-scraper))
- Extracts relevant keywords from job descriptions
- Uses NLP and semantic similarity to analyze your resume
- Returns relevance score, keyword match %, and a CSV export

### Tech Stack
- Python
- Streamlit (UI)
- Sentence Transformers (Semantic Matching)
- TF-IDF (Keyword Extraction)
- NLTK (Stemming & Tokenization)

### Job ZIP Format

Job ZIP files should contain `.txt` files, each in this format:

    Job Title at Company
    https://apply-link.com

    Full job description...

You can generate these with the [Job Scraper](https://github.com/alexlsantiago/job-scraper) companion tool.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
