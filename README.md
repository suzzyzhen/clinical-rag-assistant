Project overview


Architecture diagram (placeholder)
                Clinical Guidelines
                Medical Literature
                Institutional Policies
                (Later: MIMIC-IV Notes)
                         │
                         ▼
                ETL / Document Pipeline
      (Cleaning, Chunking, Metadata, Versioning)
                         │
                         ▼
                 Embedding Generation
     (PubMedBERT / BioClinicalBERT / e5)
                         │
                         ▼
                 Vector Database (FAISS)
                         │
                         ▼
                 RAG Retrieval Layer
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Clinical Chatbot      AI Copilot API
              │                     │
              └──────────┬──────────┘
                         ▼
                     LLM Response
                         │
                         ▼
               LLM Evaluation Engine
                         │
       Faithfulness • Hallucination • Safety
       Bias • Completeness • Citation Quality
                         │
                         ▼
            Dashboard / Monitoring / Reports

Dataset
Latest Diagnostic guidelines in PDF format 
Raw Data Source:
Alzheimer's disease - https://pubmed.ncbi.nlm.nih.gov/38934362/
Asthma - https://ginasthma.org/wp-content/uploads/2026/05/GINA-2026-Strategy-Report-WMS.pdf
Colorectal Cancer - https://pubmed.ncbi.nlm.nih.gov/42200680/
Diabetes - https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes
Hypertension - https://www.ncbi.nlm.nih.gov/books/NBK547161/
Stroke - https://www.ahajournals.org/doi/10.1161/STR.0000000000000513





Setup


Roadmap


Current status


loader.py        # Read PDFs
parser.py        # Clean text
chunker.py       # Split documents
embedder.py      # Generate embeddings
index.py         # Build/search FAISS
generator.py     # Call the LLM
evaluator.py     # Score responses