import subprocess
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    companies = [
        "BYD", "CA", "CATL", "DSV", "FYG", "GWM"
    ]
    for company in companies:
        pdf_file = f"data/pdf/{company}.pdf"
        output_file = f"data/corpus/corpus_{company}.json"
        command = [
            "python", "src/ingest/parse_pdf.py",
            "--pdf", pdf_file,
            "--output", output_file,
            "--chunk-size", "500",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"{company} chunking successfully")
        except subprocess.CalledProcessError as e:
            print(f"{company} chunking failed: {e.stderr}")

    corpus_all = []
    merged_corpus = []
    chunk_counter = 0

    for company in companies:
        corpus_file = f"data/corpus/corpus_{company}.json"
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            for chunk in corpus:
                chunk["chunk_id"] = f"chunk_{chunk_counter}"
                merged_corpus.append(chunk)
                chunk_counter += 1
     
    with open("data/corpus/corpus_all.json", "w", encoding="utf-8") as f:
        json.dump(merged_corpus, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(companies)} companies into {len(merged_corpus)} chunks")