import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import PyPDF2
import io

# Load saved models
job_model = joblib.load("logistic_model_job.pkl")
hire_model = joblib.load("naive_bayes_model_hire.pkl")

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Classifier (PDF Upload)")
st.write("Upload a resume in **PDF format** to classify the **Job Role** and **Recruiter Decision**.")

# PDF uploader
pdf_file = st.file_uploader("Upload your Resume PDF", type=["pdf"])

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()

if pdf_file is not None:
    resume_text = extract_text_from_pdf(pdf_file)

    if resume_text:
        st.subheader("📃 Extracted Resume Text Preview")
        st.write(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

        if st.button("🔍 Predict"):
            # BERT Embedding
            embedded_input = bert_model.encode([resume_text])

            # Predict
            job_pred = job_model.predict(embedded_input)[0]
            hire_pred = hire_model.predict(embedded_input)[0]

            # Results
            st.success(f"🧑‍💼 **Predicted Job Role**: {job_pred}")
            st.info(f"✅ **Recruiter Decision**: {'Hired' if hire_pred == 1 else 'Hired'}")
    else:
        st.warning("Could not extract any text from the PDF.")
