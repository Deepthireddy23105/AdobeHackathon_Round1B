import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load persona and job
with open("persona_job.json", "r", encoding="utf-8") as f:
    job_info = json.load(f)

persona = job_info["persona"]
job = job_info["job_to_be_done"]

# Get current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# PDF folder
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Collect and chunk PDF text
chunks = []
metadata = []

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(input_dir, filename)
        doc = fitz.open(path)
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if len(text.split()) > 20:  # filter small junk
                    chunks.append(text)
                    metadata.append({
                        "document": filename,
                        "page_number": page_num,
                        "section_title": text.split("\n")[0][:60]
                    })

# TF-IDF based relevance scoring
vectorizer = TfidfVectorizer(stop_words='english')
docs = [persona + " " + job] + chunks
X = vectorizer.fit_transform(docs)
cos_sim = cosine_similarity(X[0:1], X[1:]).flatten()

# Rank top relevant sections
top_n = 10
top_indices = cos_sim.argsort()[-top_n:][::-1]

extracted_sections = []
sub_section_analysis = []

for rank, idx in enumerate(top_indices, 1):
    meta = metadata[idx]
    extracted_sections.append({
        "document": meta["document"],
        "page_number": meta["page_number"],
        "section_title": meta["section_title"],
        "importance_rank": rank
    })
    sub_section_analysis.append({
        "document": meta["document"],
        "refined_text": chunks[idx],
        "page_number": meta["page_number"]
    })

# Final Output JSON
output = {
    "metadata": {
        "input_documents": sorted(os.listdir(input_dir)),
        "persona": persona,
        "job_to_be_done": job,
        "timestamp": timestamp
    },
    "extracted_sections": extracted_sections,
    "sub_section_analysis": sub_section_analysis
}

with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Output saved to output/output.json")
