import fitz  # PyMuPDF
import os
import json
import re
import joblib
import numpy as np
import lightgbm as lgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
# Directory with PDFs and their corresponding ground truth JSON files
TRAINING_DATA_DIR = "/app/input"
# Path where the trained model will be saved
MODEL_OUTPUT_PATH = "/app/heading_classifier.joblib"

# --- Helper Functions ---

def clean_text(text):
    """Cleans text by normalizing whitespace and stripping."""
    return re.sub(r'\s+', ' ', text).strip()

def get_doc_body_size(doc):
    """Calculates the most common font size in the document."""
    sizes = Counter()
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    for span in line['spans']:
                        size_key = round(span['size'])
                        sizes[size_key] += len(span['text'].strip())
    if not sizes:
        return 12 # A reasonable default
    return sizes.most_common(1)[0][0]

def load_ground_truth(json_path):
    """Loads the ground truth JSON and prepares it for quick lookups."""
    with open(json_path, 'r', encoding='utf-8') as f:
        truth_data = json.load(f)
    
    # Create a dictionary for fast lookups: { "cleaned text": "LEVEL" }
    truth_map = {clean_text(item['text']): item['level'] for item in truth_data.get('outline', [])}
    
    # Add the title to the map
    title_text = clean_text(truth_data.get('title', ''))
    if title_text:
        truth_map[title_text] = 'Title'
        
    return truth_map

def extract_features_and_labels_from_truth(doc, truth_map):
    """
    Extracts features for each line and gets the label from the ground truth map.
    """
    features = []
    labels = []
    
    doc_body_size = get_doc_body_size(doc)
    page_height = doc[0].rect.height if doc.page_count > 0 else 1

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block['type'] == 0: # Text block
                for line in block['lines']:
                    if not line['spans']: continue
                    
                    line_text = clean_text(" ".join([s['text'] for s in line['spans']]))
                    if not line_text: continue

                    span = line['spans'][0]
                    font_size = round(span['size'])
                    is_bold = "bold" in span['font'].lower()
                    
                    # --- Feature Vector ---
                    feature_vector = {
                        'font_size': font_size,
                        'size_delta_body': font_size - doc_body_size,
                        'is_bold': 1 if is_bold else 0,
                        'char_count': len(line_text),
                        'word_count': len(line_text.split()),
                        'is_all_caps': 1 if line_text.isupper() and len(line_text) > 1 else 0,
                        'ends_with_period': 1 if line_text.endswith('.') else 0,
                        'page_num': page_num,
                        'rel_y_pos': line['bbox'][1] / page_height
                    }
                    features.append(feature_vector)

                    # --- Labeling from Ground Truth ---
                    # Check if the exact line text is in our truth map
                    label = truth_map.get(line_text, 'Body')
                    labels.append(label)

    return features, labels

# --- Main Training Logic ---

if __name__ == "__main__":
    all_features = []
    all_labels = []

    print("Starting feature extraction using ground truth data...")
    if not os.path.exists(TRAINING_DATA_DIR) or not os.listdir(TRAINING_DATA_DIR):
        print(f"Error: Training directory '{TRAINING_DATA_DIR}' is empty or not found.")
        exit(1)

    # Find PDF-JSON pairs for training
    pdf_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the input directory.")
        exit(1)

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(TRAINING_DATA_DIR, pdf_filename)
        base_name = os.path.splitext(pdf_filename)[0]
        json_path = os.path.join(TRAINING_DATA_DIR, base_name + ".json")

        if os.path.exists(json_path):
            print(f"... processing pair: {pdf_filename} and {base_name}.json")
            try:
                # Load the ground truth data first
                truth_map = load_ground_truth(json_path)
                
                # Open the PDF and extract features and labels
                doc = fitz.open(pdf_path)
                features, labels = extract_features_and_labels_from_truth(doc, truth_map)
                all_features.extend(features)
                all_labels.extend(labels)
                doc.close()
            except Exception as e:
                print(f"Could not process {pdf_filename}: {e}")
        else:
            print(f"Warning: Found {pdf_filename} but missing corresponding JSON file. Skipping.")

    if not all_features:
        print("No features could be extracted from the PDF/JSON pairs. Cannot train model.")
        exit(1)

    print(f"\nExtracted {len(all_features)} total text lines for training.")
    print(f"Label distribution: {Counter(all_labels)}")

    # Convert to numpy array for scikit-learn
    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining LightGBM classifier...")
    lgb_classifier = lgb.LGBMClassifier(objective='multiclass', random_state=42)
    lgb_classifier.fit(X_train, y_train)

    print("Training complete.")

    # Evaluate the model
    print("\nEvaluating model on test set:")
    y_pred = lgb_classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(lgb_classifier, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")
