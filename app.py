import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import spacy
import io

# ============================================================
# Page Configuration & Styling
# ============================================================
st.set_page_config(
    page_title="Transcript Action Item Extractor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Gradient Hero Title */
    .hero-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #8F94FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .hero-subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }
    /* Card design */
    .action-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 6px solid #6C63FF;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        background-color: #F8F9FA;
        transition: transform 0.2s ease-in-out;
    }
    /* Dark mode override for card */
    @media (prefers-color-scheme: dark) {
        .action-card {
            background-color: #1E1E2F;
            border-left: 6px solid #8F8FFF;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
    }
    .action-card:hover {
        transform: translateY(-2px);
    }
    .badge {
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
    }
    .badge-speaker {
        background-color: #E2E3F9;
        color: #3F51B5;
    }
    .badge-deadline {
        background-color: #FFEAEA;
        color: #FF4B4B;
    }
    .card-text {
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
    }
    .card-label {
        font-weight: bold;
        color: #6C63FF;
    }
    .card-sentence {
        font-size: 13px;
        color: #888;
        font-style: italic;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Cached Models & Data Loaders
# ============================================================
@st.cache_resource
def load_models():
    """Load SBERT and spaCy once and cache them."""
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        spacy_nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback if download failed
        spacy_nlp = spacy.load("en_core_web_sm")
    return sbert, spacy_nlp

@st.cache_data
def load_raw_dataset():
    """Load raw dataset CSV once."""
    return pd.read_csv("test_df.csv")

# Initialize models and load raw data
sbert_model, nlp = load_models()
raw_df = load_raw_dataset()

# ============================================================
# Preprocessing Helper Functions
# ============================================================
def extract_speaker_blocks(transcript):
    pattern = r'([^:\n]+):\s*(.*?)(?=\n[^:\n]+:|$)'
    matches = re.findall(pattern, transcript, re.DOTALL)
    blocks = []
    for speaker, text in matches:
        blocks.append((speaker.strip(), text.strip()))
    return blocks

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_into_sentences(blocks, meeting_id):
    records = []
    sentence_id = 0
    for speaker, text in blocks:
        sentences = split_sentences(text)
        for sentence in sentences:
            records.append({
                'meeting_id': meeting_id,
                'speaker': speaker,
                'sentence_id': sentence_id,
                'sentence': sentence
            })
            sentence_id += 1
    return records

# ============================================================
# Caching Model Training
# ============================================================
def train_and_cache_classifier(max_meetings):
    """Encodes dataset and trains Logistic Regression classifier."""
    # Slice the dataset
    df_slice = raw_df.head(max_meetings).copy()
    
    # Process into sentences
    all_records = []
    for meeting_id, transcript in enumerate(df_slice['Transcript']):
        blocks = extract_speaker_blocks(transcript)
        records = split_into_sentences(blocks, meeting_id)
        all_records.extend(records)
    sentence_df = pd.DataFrame(all_records)
    
    # SBERT Encode Summaries & Sentences
    summaries = df_slice["Summary"].fillna("").tolist()
    summary_embeddings = sbert_model.encode(summaries, batch_size=128, show_progress_bar=False)
    sentence_embeddings = sbert_model.encode(sentence_df["sentence"].tolist(), batch_size=128, show_progress_bar=False)
    
    # Generate labels
    TOP_K = 5
    retrieval_rows = []
    for meeting_id in sentence_df["meeting_id"].unique():
        summary = df_slice.loc[meeting_id, "Summary"]
        if pd.isna(summary):
            continue
        
        meeting_mask = sentence_df["meeting_id"] == meeting_id
        meeting_indices = sentence_df[meeting_mask].index.tolist()
        if len(meeting_indices) == 0:
            continue
            
        meeting_embs = sentence_embeddings[meeting_indices]
        sum_emb = summary_embeddings[meeting_id].reshape(1, -1)
        
        scores = cosine_similarity(sum_emb, meeting_embs).flatten()
        meeting = sentence_df.loc[meeting_indices].copy()
        meeting["similarity"] = scores
        topk = meeting.nlargest(TOP_K, "similarity")
        
        for rank, (_, row) in enumerate(topk.iterrows(), start=1):
            retrieval_rows.append({
                "meeting_id": meeting_id,
                "sentence_id": row["sentence_id"],
            })
            
    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df["key"] = retrieval_df["meeting_id"].astype(str) + "_" + retrieval_df["sentence_id"].astype(str)
    sentence_df["key"] = sentence_df["meeting_id"].astype(str) + "_" + sentence_df["sentence_id"].astype(str)
    
    positive_keys = set(retrieval_df["key"])
    sentence_df["label"] = sentence_df["key"].apply(lambda x: 1 if x in positive_keys else 0)
    
    # Train Logistic Regression
    X = sentence_embeddings
    y = sentence_df["label"].values
    
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    return clf

# ============================================================
# Action Parsing Helper (spaCy)
# ============================================================
def parse_action_details(sentence, speaker):
    doc = nlp(sentence)
    assignee = None
    action = None
    deadline = None

    # Deadline extraction
    for ent in doc.ents:
        if ent.label_ == "DATE":
            deadline = ent.text
            break

    # Assignee extraction
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            assignee = ent.text
            break

    if assignee is None:
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                assignee = token.text
                break

    # Pronoun resolution to Speaker / Team
    if assignee:
        assignee_lower = assignee.lower().strip()
        if assignee_lower in {"i", "me", "my", "myself"}:
            assignee = speaker
        elif assignee_lower in {"we", "us", "our", "ourselves"}:
            assignee = "Team / All"

    # Action extraction
    for token in doc:
        if token.pos_ == "VERB":
            phrase = token.lemma_
            for child in token.children:
                if child.dep_ in ("dobj", "obj", "attr"):
                    phrase += " " + " ".join(t.text for t in child.subtree)
            action = phrase
            break

    return {
        "assignee": assignee if assignee else "Unassigned",
        "action": action if action else sentence,
        "deadline": deadline if deadline else "No deadline specified"
    }

# ============================================================
# Sidebar Panel Controls
# ============================================================
st.sidebar.title("Configuration Settings")
st.sidebar.markdown("---")

# Slider: Dataset training size
max_meetings_slider = st.sidebar.slider(
    "Training Set Size (Meetings)",
    min_value=5,
    max_value=150,
    value=15,
    step=5,
    help="Select the number of meetings to train the model on. Lower sizes run instantly on CPU. Large sizes (~150) take about 10-15 minutes but yield higher accuracy."
)

# Button: Force retraining
retrain_button = st.sidebar.button("🔄 Retrain Classifier Model")

# Slider: Confidence Threshold
threshold_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Adjust classification threshold. Increase to reduce false positives (increase Precision). Decrease to catch more action items (increase Recall)."
)

# Initialize Session State
if "classifier" not in st.session_state or retrain_button or st.session_state.get("trained_size") != max_meetings_slider:
    with st.spinner(f"Training SBERT classifier on {max_meetings_slider} meetings... (Please wait a few seconds)"):
        st.session_state["classifier"] = train_and_cache_classifier(max_meetings_slider)
        st.session_state["trained_size"] = max_meetings_slider
    st.sidebar.success("🎉 Classifier Model Trained successfully!")

st.sidebar.markdown(f"""
**Current Model Stats:**
* **Training meetings:** `{st.session_state.trained_size}`
* **Model Type:** `LogisticRegression`
* **Features:** SBERT (`all-MiniLM-L6-v2`)
* **Embedding dimensions:** `384`
""")

# ============================================================
# Main Page Layout
# ============================================================
st.markdown('<div class="hero-title">🎯 Transcript Action Item Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Transform unstructured meeting transcripts into clean, structured action-item checklists in real-time.</div>', unsafe_allow_html=True)

# Sample Transcripts for quick-loading
sample_hospital = """Chief Doctor: Good morning.
Chief Doctor: We recommend that Nurse Emma prepare the operation theatre immediately.
Dr. Ryan: I recommend that we review the patient's reports by this evening.
Chief Doctor: We request Dr. Smith to contact the patient's family by 6 PM.
Receptionist: Okay."""

sample_council = """Speaker 2: And 14 is communication from the Office of the Council Member. I recommend to direct the City Clerk to place an advisory question on the April eight, 2014 ballot.
Speaker 1: Roslyn.
Speaker 9: Thank you, Mr. Mayor. Yes, I introduced this item for consideration of this body. We need the staff to prepare a comprehensive report within 30 days.
Speaker 3: I will second that motion."""

# Quick-Load Buttons in columns
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    load_hospital = st.button("🏥 Load Hospital Sample")
with col2:
    load_council = st.button("🏛️ Load City Council Sample")

# Session State for text area input
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if load_hospital:
    st.session_state.input_text = sample_hospital
if load_council:
    st.session_state.input_text = sample_council

# Big Text Area Input
transcript_input = st.text_area(
    "Paste your meeting transcript here:",
    value=st.session_state.input_text,
    height=250,
    placeholder="Format: \nSpeaker Name: Speech text...\nAnother Speaker: Speech text..."
)

# Extract Button
extract_button = st.button("Extract Action Items", type="primary")

# ============================================================
# Inference and Extraction Execution
# ============================================================
if extract_button and transcript_input.strip() != "":
    with st.spinner("Analyzing transcript & classifying action items..."):
        # Step 1: Preprocess and split into sentences
        speaker_blocks = extract_speaker_blocks(transcript_input)
        records = split_into_sentences(speaker_blocks, 0)
        sentence_texts = [r["sentence"] for r in records]
        
        if len(sentence_texts) == 0:
            st.warning("Could not extract sentences from the transcript. Make sure it contains Speaker tags (e.g., 'Speaker: text').")
        else:
            # Step 2: Encode sentences
            embeddings = sbert_model.encode(sentence_texts, show_progress_bar=False)
            
            # Step 3: Classify with probability threshold
            clf = st.session_state["classifier"]
            probabilities = clf.predict_proba(embeddings)[:, 1]
            
            # Filter by slider confidence threshold
            predictions = (probabilities >= threshold_slider).astype(int)
            
            # Step 4: Extract info for flagged sentences
            action_items = []
            for record, pred, prob in zip(records, predictions, probabilities):
                if pred == 1:
                    info = parse_action_details(record["sentence"], record["speaker"])
                    info["speaker"] = record["speaker"]
                    info["sentence"] = record["sentence"]
                    info["confidence"] = prob
                    action_items.append(info)
            
            # Step 5: Render results
            st.markdown("---")
            st.subheader(f"📊 Extraction Results (Confidence Threshold: {threshold_slider})")
            
            m1, m2 = st.columns(2)
            m1.metric("Total Sentences Analyzed", len(sentence_texts))
            m2.metric("Action Items Extracted", len(action_items))
            
            if len(action_items) == 0:
                st.info("No action items met the confidence threshold. Try lowering the threshold slider in the sidebar!")
            else:
                # Custom Card Grid
                for i, item in enumerate(action_items, 1):
                    # Badges
                    speaker_badge = f'<span class="badge badge-speaker">{item["speaker"]}</span>'
                    deadline_badge = ""
                    if item["deadline"] != "No deadline specified":
                        deadline_badge = f'<span class="badge badge-deadline">📅 {item["deadline"]}</span>'
                    
                    st.markdown(f"""
                    <div class="action-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>{speaker_badge}</div>
                            <div>{deadline_badge}</div>
                        </div>
                        <div class="card-text"><span class="card-label">🎯 Action:</span> {item["action"]}</div>
                        <div style="font-size:15px; margin-top:5px;"><span class="card-label">👤 Assignee:</span> {item["assignee"]}</div>
                        <div style="font-size:12px; color:#aaa; margin-top:5px;">Confidence Score: <b>{item["confidence"]:.2%}</b></div>
                        <div class="card-sentence">"{item["sentence"]}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export options
                export_df = pd.DataFrame(action_items)[["speaker", "assignee", "action", "deadline", "confidence", "sentence"]]
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Action Items as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="extracted_action_items.csv",
                    mime="text/csv"
                )
elif extract_button:
    st.warning("Please paste a transcript first.")
