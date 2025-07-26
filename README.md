
# Plant Health Assessment

A Python-based solution for evaluating the health of plants using leaf image analysis and machine learning techniques.

## Project Overview

This repository implements a pipeline to:
- Analyze plant health by processing leaf images
- Detect disease symptoms (e.g. blight, spots, chlorosis)
- Classify the condition of the plant using a trained ML model
- Provide visual and textual reports on plant status

## Key Features

- Image preprocessing (resizing, normalization, augmentation)  
- Machine learning classification (e.g. CNN-based)
- Generates interactive reports (charts, labels, visual overlays)
- Support for batch processing of multiple images  

## Technologies Used

- **Language**: Python  
- **Libraries**: `numpy`, `opencv-python`, `tensorflow` or `scikit-learn`  
- **Visualization**: `matplotlib`, possibly simple web frontend (Flask/Bokeh)  

## Project Structure

```
plant-health-assessment/
├── data/
│   ├── images/               # Sample leaf images organized by class
│   ├── labels.csv            # Metadata for labeled images
├── models/
│   └── trained_model.h5      # Pre-trained model artifact
├── notebooks/
│   └── training.ipynb        # Jupyter notebook for model training
├── src/
│   ├── preprocess.py         # Image processing pipeline
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation logic
│   ├── predict.py            # Inference script for new images
│   └── utils.py              # Shared utilities
├── README.md
└── requirements.txt
```

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

- Populate the `data/images/` folder with leaf images grouped by class or label.
- Ensure `labels.csv` contains mappings from filenames to labels.

### 3. Train the Model (optional)

```bash
python src/train.py --data_dir data/images --labels_csv data/labels.csv --output models/trained_model.h5
```

### 4. Evaluate the Model

```bash
python src/evaluate.py --model_path models/trained_model.h5 --data_dir data/images --labels_csv data/labels.csv
```

### 5. Predict on New Images

```bash
python src/predict.py --model models/trained_model.h5 --input path/to/leaf.jpg
```

Generates classification results and optionally overlayed visual output.

## Sample Output

```
Image: leaf_test.jpg
Prediction: Early Blight (Confidence: 92.3%)
Health Score: Unhealthy
```

## How It’s Structured

- `preprocess.py`: handles loading, resizing, normalization, and augmentation.
- `train.py`: builds and trains the model (CNN or transfer learning).
- `evaluate.py`: measures metrics like accuracy, precision, recall.
- `predict.py`: loads the saved model and predicts new images.

## Extensibility & To-Do

- Add support for more crop types and disease categories  
- Integrate explainable AI (e.g. Grad‑CAM saliency maps)  
- Build a LightWeight Flask or Streamlit app for web-based inference  
- Automate workflows or integrate with scheduling systems  

## Author

**Hemanth Kumar**  
GitHub: [@BL‑EN‑U4AIE22138‑HemanthKumar](https://github.com/BL-EN-U4AIE22138-HemanthKumar)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
