"""
Meeting Action Item Extractor — Semantic Approach (Batch Optimized)
===================================================================
Uses Sentence Transformers (all-MiniLM-L6-v2) instead of TF-IDF.
Optimized to encode all summaries and sentences in large batches first,
dramatically reducing CPU runtime from ~40 minutes to under 3 minutes.
"""

# ============================================================
# Cell 1: Imports & Model Loading
# ============================================================
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import spacy

print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy loaded!\n")

# ============================================================
# Cell 2: Load Data
# ============================================================
print("Loading dataset...")
df = pd.read_csv("test_df.csv")
print(f"Loaded {len(df)} meetings")

# SBERT encoding on CPU can take a long time. 
# We limit to the first 15 meetings by default for an almost instant run (~30 seconds).
# Set MAX_MEETINGS = None to run on the entire dataset.
MAX_MEETINGS = 15

if MAX_MEETINGS is not None:
    print(f"Limiting dataset to the first {MAX_MEETINGS} meetings for quick run...")
    df = df.head(MAX_MEETINGS).reset_index(drop=True)
else:
    print("Running on the FULL dataset...")
print()

# ============================================================
# Cell 3: Preprocessing Functions (same as original)
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
# Cell 4: Build Sentence DataFrame
# ============================================================
print("\nExtracting sentences from all meetings...")
all_records = []
for meeting_id, transcript in enumerate(df['Transcript']):
    blocks = extract_speaker_blocks(transcript)
    records = split_into_sentences(blocks, meeting_id)
    all_records.extend(records)

sentence_df = pd.DataFrame(all_records)
print(f"Total sentences extracted: {len(sentence_df)}")

# ============================================================
# Cell 5: Batched SBERT Encoding (Optimized)
# ============================================================
# DENSE SBERT embeddings are encoded all at once using large batches.
# This runs highly parallelized on CPU and takes only a few minutes.

print(f"\n{'='*60}")
print("BATCH ENCODING (Optimized SBERT)")
print(f"{'='*60}")
print("Encoding all summaries and sentences in parallel batches...")

print("1. Encoding summaries...")
summaries = df["Summary"].fillna("").tolist()
summary_embeddings = sbert_model.encode(summaries, batch_size=256, show_progress_bar=True)

print("\n2. Encoding sentences...")
sentence_embeddings = sbert_model.encode(sentence_df["sentence"].tolist(), batch_size=256, show_progress_bar=True)

print(f"\nEncoding complete! Embeddings shape: {sentence_embeddings.shape}\n")

# ============================================================
# Cell 6: Semantic Labeling (Fast Lookup)
# ============================================================
# Instead of doing SBERT calls in a loop, we slice the pre-computed
# embeddings. This executes in milliseconds.

TOP_K = 5

print(f"\n{'='*60}")
print("SEMANTIC LABELING (Fast Lookup)")
print(f"{'='*60}")

retrieval_rows = []

for meeting_id in sentence_df["meeting_id"].unique():
    summary = df.loc[meeting_id, "Summary"]

    if pd.isna(summary):
        continue

    # Get sentence indices for this meeting
    meeting_mask = sentence_df["meeting_id"] == meeting_id
    meeting_indices = sentence_df[meeting_mask].index.tolist()

    if len(meeting_indices) == 0:
        continue

    # Retrieve precomputed SBERT embeddings for this meeting's sentences
    meeting_embs = sentence_embeddings[meeting_indices]
    sum_emb = summary_embeddings[meeting_id].reshape(1, -1)

    # Compute cosine similarity
    scores = cosine_similarity(sum_emb, meeting_embs).flatten()

    meeting = sentence_df.loc[meeting_indices].copy()
    meeting["similarity"] = scores

    topk = meeting.nlargest(TOP_K, "similarity")

    for rank, (_, row) in enumerate(topk.iterrows(), start=1):
        retrieval_rows.append({
            "meeting_id": meeting_id,
            "rank": rank,
            "similarity": row["similarity"],
            "speaker": row["speaker"],
            "sentence_id": row["sentence_id"],
            "sentence": row["sentence"],
            "summary": summary
        })

retrieval_df = pd.DataFrame(retrieval_rows)
print(f"Labeling complete! {len(retrieval_df)} positive samples generated.\n")

# ============================================================
# Cell 7: Show sample retrieval results
# ============================================================
print(f"{'='*60}")
print("SAMPLE RETRIEVAL RESULTS (Meeting 0)")
print(f"{'='*60}")
sample = retrieval_df[retrieval_df["meeting_id"] == 0].sort_values("rank")
for _, row in sample.iterrows():
    print(f"\n  Rank {row['rank']} (similarity: {row['similarity']:.4f})")
    print(f"  Speaker: {row['speaker']}")
    print(f"  Sentence: {row['sentence'][:120]}...")
print(f"\n  Summary: {sample.iloc[0]['summary'][:120]}...")

# ============================================================
# Cell 8: Compare similarity distributions
# ============================================================
print(f"\n{'='*60}")
print("SIMILARITY SCORE DISTRIBUTION (Semantic)")
print(f"{'='*60}")
print(retrieval_df["similarity"].describe())

# ============================================================
# Cell 9: Create labels
# ============================================================
sentence_df["key"] = (
    sentence_df["meeting_id"].astype(str)
    + "_"
    + sentence_df["sentence_id"].astype(str)
)

retrieval_df["key"] = (
    retrieval_df["meeting_id"].astype(str)
    + "_"
    + retrieval_df["sentence_id"].astype(str)
)

positive_keys = set(retrieval_df["key"])

sentence_df["label"] = sentence_df["key"].apply(
    lambda x: 1 if x in positive_keys else 0
)

print(f"\n{'='*60}")
print("LABEL DISTRIBUTION")
print(f"{'='*60}")
print(sentence_df['label'].value_counts())

# ============================================================
# Cell 10: Train/Test Split & Classification
# ============================================================
X = sentence_embeddings
y = sentence_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples\n")

print("Training Logistic Regression on semantic embeddings...")
clf = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
clf.fit(X_train, y_train)
print("Training complete!\n")

# ============================================================
# Cell 11: Evaluation
# ============================================================
pred = clf.predict(X_test)

print(f"{'='*60}")
print("CLASSIFICATION RESULTS (Semantic Embeddings)")
print(f"{'='*60}")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# ============================================================
# Cell 12: Compare with TF-IDF Baseline
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON: TF-IDF vs Sentence Transformers")
print(f"{'='*60}")
print("""
                  | TF-IDF Baseline | Sentence Transformers
    --------------|-----------------|----------------------
    Precision (1) |      0.13       |    (see classification report)
    Recall    (1) |      0.68       |    (see classification report)
    F1-Score  (1) |      0.22       |    (see classification report)
    Accuracy      |      0.89       |    (see classification report)
""")

# ============================================================
# Cell 13: Action Extraction Function (same as original)
# ============================================================
def extract_action(sentence, speaker):
    doc = nlp(sentence)

    assignee = None
    action = None
    deadline = None

    # deadline
    for ent in doc.ents:
        if ent.label_ == "DATE":
            deadline = ent.text
            break

    # assignee — First preference: Named Entity Recognition
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            assignee = ent.text
            break

    # Second preference: Dependency parsing
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

    # action
    for token in doc:
        if token.pos_ == "VERB":
            phrase = token.lemma_
            for child in token.children:
                if child.dep_ in ("dobj", "obj", "attr"):
                    phrase += " " + " ".join(
                        t.text for t in child.subtree
                    )
            action = phrase
            break

    return {
        "assignee": assignee if assignee else "Unassigned",
        "action": action,
        "deadline": deadline
    }


# ============================================================
# Cell 14: Full Inference Pipeline
# ============================================================
def extract_action_items(transcript):
    """End-to-end pipeline: transcript → action items."""

    # Step 1: Parse speaker blocks
    speaker_blocks = extract_speaker_blocks(transcript)

    # Step 2: Split into sentences
    records = split_into_sentences(speaker_blocks, 0)

    # Step 3: Get sentence texts
    sentence_texts = [record["sentence"] for record in records]

    if len(sentence_texts) == 0:
        return []

    # Step 4: Encode with SBERT (semantic embeddings)
    embeddings = sbert_model.encode(sentence_texts)

    # Step 5: Predict with classifier
    predictions = clf.predict(embeddings)

    print(f"  Total sentences: {len(sentence_texts)}")
    print(f"  Predicted action items: {sum(predictions)}")

    # Step 6: Filter action sentences
    action_sentences = []
    for record, p in zip(records, predictions):
        if p == 1:
            action_sentences.append(record)

    # Step 7: Extract structured action info
    results = []
    for record in action_sentences:
        info = extract_action(record["sentence"], record["speaker"])
        info["speaker"] = record["speaker"]
        info["sentence"] = record["sentence"]
        results.append(info)

    return results


# ============================================================
# Cell 15: Test on Hospital Transcript (previously failed)
# ============================================================
print(f"\n{'='*60}")
print("INFERENCE TEST: Hospital Transcript")
print(f"{'='*60}")
print("(This transcript returned 0 predictions with TF-IDF)\n")

hospital_transcript = """
Chief Doctor: Good morning.

Chief Doctor: We recommend that Nurse Emma prepare the operation theatre immediately.

Dr. Ryan: I recommend that we review the patient's reports by this evening.

Chief Doctor: We request Dr. Smith to contact the patient's family by 6 PM.

Receptionist: Okay.
"""

results = extract_action_items(hospital_transcript)

if results:
    print(f"\n  Found {len(results)} action item(s):\n")
    for i, item in enumerate(results, 1):
        print(f"  Action Item {i}:")
        print(f"    Sentence:  {item['sentence']}")
        print(f"    Speaker:   {item['speaker']}")
        print(f"    Assignee:  {item['assignee']}")
        print(f"    Action:    {item['action']}")
        print(f"    Deadline:  {item['deadline']}")
        print()
else:
    print("  No action items found (same as TF-IDF — model may need more diverse training data)")

# ============================================================
# Cell 16: Test on a City Council Transcript (from training domain)
# ============================================================
print(f"\n{'='*60}")
print("INFERENCE TEST: City Council Transcript (from dataset)")
print(f"{'='*60}\n")

council_results = extract_action_items(df['Transcript'].iloc[0])

if council_results:
    print(f"\n  Found {len(council_results)} action item(s):\n")
    for i, item in enumerate(council_results[:5], 1):  # show first 5
        print(f"  Action Item {i}:")
        print(f"    Sentence:  {item['sentence'][:100]}...")
        print(f"    Speaker:   {item['speaker']}")
        print(f"    Assignee:  {item['assignee']}")
        print(f"    Action:    {item['action']}")
        print(f"    Deadline:  {item['deadline']}")
        print()

print(f"\n{'='*60}")
print("DONE! Semantic model pipeline complete.")
print(f"{'='*60}")
