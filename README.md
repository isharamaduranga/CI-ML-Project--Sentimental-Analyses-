# CA_Sentiment_Analysis_Project - ICBT_CIS6005

## Project Structure

### 1. Notebooks Folder
This folder contains Jupyter notebooks used in the project:
- `download_dataset.ipynb` – Downloads the dataset.
- `model_building.ipynb` – Builds the machine learning model.
- `model_building_with_visualization.ipynb` – Builds the model with additional visualization.
- `prediction_pipeline.ipynb` – Implements the prediction pipeline.

### 2. Artifacts Folder
Contains important dataset files:
- `train.csv` – Training dataset file.
- `test.csv` – Test dataset file.

### 3. Environment Setup
The Python virtual environment (`.env`) file has been removed due to its large size.
To install dependencies, use the following command:
```bash
pip install -r requirements.txt
```

### 4. Static Folder
Contains essential files for model execution:
- `model.pkl` – Trained machine learning model file.
- `NLTK toolkit data` – Required for natural language processing.
- `vocabulary.txt` – Vocabulary data for the model.

### 5. Templates Folder
Contains the frontend HTML file:
- `index.html` – The main web interface file.

### 6. Backend Implementation
The backend logic and Flask implementation include:
- `helper.py` – Defines the prediction pipeline using `model.pkl`.
- `app.py` – Contains the main application logic.

### 7. Running the Application
Before running the application, ensure all required dependencies are installed.
To start the Flask application, use the command:
```bash
python app.py

