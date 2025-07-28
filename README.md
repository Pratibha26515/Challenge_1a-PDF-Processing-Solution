# PDF Heading Extraction Project

This project uses machine learning to extract headings and titles from PDF documents. It consists of two main scripts:
- `train_model.py`: Trains a LightGBM classifier using labeled PDF data.
- `process_pdfs.py`: Uses the trained model to extract headings/titles from new PDFs and outputs structured JSON outlines.

## Project Structure

```
project/
├── Dockerfile
├── process_pdfs.py
├── train_model.py
├── requirements.txt
├── input/           # Place PDFs (and for training, their .json ground truth) here
├── output/          # Inference results (JSON outlines) are written here
└── model_output/    # (Optional: for model artifacts)
```

## Requirements
- Docker (recommended)
- Or: Python 3.9+ and pip (for local development)

## Setup & Usage

### 1. Build the Docker Image
```bash
docker build -t pdf_heading_extractor .
```

### 2. Prepare Your Data
- For training: Place your training PDFs and their corresponding `.json` ground truth files in `input/`.
- For inference: Place the PDFs to be processed in `input/`.

### 3. Train the Model (in Docker)
```bash
docker run --rm \
  -v "$PWD/input":/app/input \
  -v "$PWD/model_output":/app/model_output \
  pdf_heading_extractor python train_model.py
```
- The trained model will be saved as `/app/heading_classifier.joblib` inside the container.
- Copy it to your host or mount a volume for persistence.

### 4. Run Inference (Process PDFs)
Make sure the trained model (`heading_classifier.joblib`) is present in your project directory (or copy it from `model_output`).

```bash
docker run --rm \
  -v "$PWD/input":/app/input \
  -v "$PWD/output":/app/output \
  -v "$PWD/heading_classifier.joblib":/app/heading_classifier.joblib \
  pdf_heading_extractor
```
- Output JSON files will be written to `output/`.

### 5. (Optional) Run Locally (Not Recommended for Submission)
If you want to run locally (not in Docker):
- Change all `/app/` paths in the scripts to local paths (`input`, `output`, etc.).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run:
  ```bash
  python train_model.py
  python process_pdfs.py
  ```

## Notes
- Page numbers in output start from 0 (zero-based).
- All dependencies are pinned in `requirements.txt` for reproducibility.
- Scripts check for missing files/directories and fail gracefully.
- For best results, use Docker as described above.

## License
MIT (or specify your license) 