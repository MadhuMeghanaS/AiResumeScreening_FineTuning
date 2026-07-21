# AI Resume Screening — Fine-Tuning a Transformer (DistilBERT) to Classify Resumes by Job Category

an AI system
that automates part of *their own job* (resume screening). It's a genuinely
useful applied-NLP project — fine-tuning a pretrained Transformer on
real, messy resume text — rather than another generic sentiment classifier.

## Dataset

**Kaggle: "Resume Dataset"**
Link: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

- ~2,400 real resumes as raw text, each labeled with one of **24 job
  categories** (e.g. Data Science, HR, Sales, Java Developer, Health &
  Fitness, Business Analyst, Mechanical Engineer, etc.).
- Columns: `ID`, `Resume_str` , `Resume_html`, `Category`.
- Multi-class, moderately imbalanced across categories — realistic for a
  screening tool that needs to work across many job families.

### Download instructions
```bash
pip install kaggle
kaggle datasets download -d snehaanbhawal/resume-dataset -p data/ --unzip
```
You should end up with `data/Resume/Resume.csv` (path may vary slightly by
Kaggle's zip layout — check after unzipping and adjust `--csv_path` if needed).


## Folder structure
```
AIResume/
├── Resume/Resume.csv
├── data/data            
├── train.py              
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python train.py --epochs n
```

## The script will:
1. Load the resume CSV, clean whitespace/artifacts from the extracted text.
2. Encode the 24 category labels.
3. Split into train/val/test.
4. Tokenize resume text with a pretrained DistilBERT tokenizer
   (`distilbert-base-uncased`), truncating/padding to a fixed max length.
5. Fine-tune `DistilBertForSequenceClassification` end-to-end (all layers
   trainable) with a linear-warmup learning-rate schedule.
6. Evaluate on the held-out test set: accuracy, macro-F1, and a
   per-category classification report (useful since categories are
   imbalanced).
7. Save the fine-tuned model + tokenizer to `resume_classifier_model/` so it
   can be reloaded for inference without retraining.

## Future Enhancements
- Wrap the trained model in a small Streamlit/Gradio app: paste a resume,
  get the predicted category + confidence scores for all 24 classes.
- Add an interpretability layer (e.g. attention-weight or SHAP visualization)
  highlighting which phrases drove the prediction — very demo-friendly.
- Extend to a two-tower **resume-to-job-posting matching** model (encode
  resumes and job descriptions separately, score by embedding similarity) —
  the same architecture recommendation systems at LinkedIn/Indeed use.
- Compare fine-tuned DistilBERT against a classical TF-IDF + Logistic
  Regression baseline and report the accuracy/latency trade-off.
