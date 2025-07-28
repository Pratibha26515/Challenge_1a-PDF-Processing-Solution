import fitz  # PyMuPDF
import os
import json
import re
import joblib
import numpy as np
from collections import Counter

# --- Configuration ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MODEL_PATH = "/app/heading_classifier.joblib"

# --- Feature Extraction ---

def clean_text(text):
    """Cleans text."""
    return re.sub(r'\s+', ' ', text).strip()

def get_doc_body_size(doc):
    """Calculates the most common font size in the document, likely the body text size."""
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

def extract_features(doc):
    """
    Extracts features for each line of text in the document to be fed into the ML model.
    """
    features = []
    line_texts = []
    
    doc_body_size = get_doc_body_size(doc)
    page_height = doc[0].rect.height if doc.page_count > 0 else 1

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block['type'] == 0: # Text block
                for line in block['lines']:
                    if not line['spans']:
                        continue
                    
                    # Consolidate line text and get primary style
                    line_text = clean_text(" ".join([s['text'] for s in line['spans']]))
                    if not line_text:
                        continue

                    span = line['spans'][0]
                    font_size = round(span['size'])
                    is_bold = "bold" in span['font'].lower()
                    
                    # Feature set
                    feature_vector = {
                        'font_size': font_size,
                        'size_delta_body': font_size - doc_body_size,
                        'is_bold': 1 if is_bold else 0,
                        'char_count': len(line_text),
                        'word_count': len(line_text.split()),
                        'is_all_caps': 1 if line_text.isupper() and len(line_text) > 1 else 0,
                        'ends_with_period': 1 if line_text.endswith('.') else 0,
                        'page_num': page_num,
                        'rel_y_pos': line['bbox'][1] / page_height # Relative Y position
                    }
                    features.append(feature_vector)
                    line_texts.append({'text': line_text, 'page': page_num})

    # Convert list of dicts to a 2D numpy array for the model
    feature_names = list(features[0].keys()) if features else []
    feature_array = np.array([[f[name] for name in feature_names] for f in features])
    
    return feature_array, line_texts

# --- Main Execution Logic ---

def process_pdf(pdf_path, model):
    """
    Processes a single PDF using the loaded ML model.
    """
    print(f"Processing: {os.path.basename(pdf_path)}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return None

    # 1. Extract features from the PDF
    features, line_info = extract_features(doc)
    if features.shape[0] == 0:
        print(f"Could not extract any text lines from {pdf_path}")
        doc.close()
        return {"title": "Extraction Failed", "outline": []}

    # 2. Predict labels with the model
    predictions = model.predict(features)
    
    # 3. Format the output
    outline = []
    title = "Untitled Document"
    title_found = False

    for i, label in enumerate(predictions):
        if label != 'Body': # We only care about headings and titles
            # The first "Title" prediction is used as the document title
            if label == 'Title' and not title_found:
                title = line_info[i]['text']
                title_found = True
            elif label in ['H1', 'H2', 'H3']:
                 outline.append({
                    "level": label,
                    "text": line_info[i]['text'],
                    "page": line_info[i]['page']
                })

    # If no title was predicted, use metadata or the first heading
    if not title_found:
        metadata_title = doc.metadata.get('title', '')
        if clean_text(metadata_title):
            title = clean_text(metadata_title)
        elif outline:
            title = outline[0]['text']

    doc.close()
    return {"title": title, "outline": outline}

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run the training script first to generate the model file.")
        exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the pre-trained model
    print(f"Loading model from {MODEL_PATH}...")
    classifier = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            input_filepath = os.path.join(INPUT_DIR, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            
            result = process_pdf(input_filepath, classifier)
            
            if result:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Successfully generated outline for {filename} -> {output_filename}")

    print("\nProcessing complete.")