# 📄 AI-Based Resume Screening

This project is an **AI-powered Resume Classifier** built with **Streamlit**, **BERT embeddings**, and **Machine Learning models**.  
It automatically extracts text from uploaded PDF resumes and classifies:  

1. **Job Role Prediction** – Identifies the most suitable job role for the candidate.  
2. **Recruiter Decision** – Predicts whether the candidate would be **Hired / Not Hired**.  

---

## 🚀 Features
- Upload resume in **PDF format**
- Extracts text from PDF using `PyPDF2`
- Converts text into embeddings with **SentenceTransformer (BERT)**
- Classifies job role using a **Logistic Regression model**
- Predicts hiring decision using a **Naive Bayes model**
- Interactive UI powered by **Streamlit**

---

## 📂 Project Structure
```

.
├── app.py                     # Streamlit application
├── logistic_model_job.pkl      # Trained Logistic Regression model
├── naive_bayes_model_hire.pkl  # Trained Naive Bayes model
├── README.md                   # Project documentation
└── .vscode/                    # VS Code config

````

---

## 🛠️ Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/PrathyushaThulava/AI-Based-Resume-Screening.git
cd AI-Based-Resume-Screening

pip install -r requirements.txt
````

Example `requirements.txt`:

```
streamlit
joblib
sentence-transformers
PyPDF2
scikit-learn
```

---

## ▶️ Usage

Run the app locally:

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`) to use the web app.

---

## 📊 How it Works

1. Upload a **PDF Resume**
2. Resume text is extracted
3. Text is embedded using **BERT** (`all-MiniLM-L6-v2`)
4. Predictions are made:

   * **Logistic Regression** → Job Role
   * **Naive Bayes** → Hiring Decision

---

## ✨ Example Output

* **Predicted Job Role**: `Data Scientist`
* **Recruiter Decision**: `Hired` ✅

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).


