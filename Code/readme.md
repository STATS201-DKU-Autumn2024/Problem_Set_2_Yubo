# AI-Generated Fake Review Detection

This repository contains three Jupyter Notebook implementations for detecting AI-generated fake reviews using text, visual, and multimodal approaches. Each notebook demonstrates a machine learning pipeline tailored to its specific data modality. The models are designed to classify reviews as **authentic** or **fake** based on provided features.

---

## **Notebooks Overview**

### 1. `text_model.ipynb`
This notebook builds a machine learning pipeline to classify reviews based on textual data.

- **Input**: The `text` column from the dataset, which contains the review text.
- **Process**:
  - Text preprocessing using **TF-IDF** vectorization.
  - Training a **Random Forest Classifier** on the numerical text features.
  - Evaluation of the model using accuracy, classification report, and confusion matrix.
  - Analysis of important textual features contributing to the classification.
- **Use Case**: Ideal for scenarios where only the textual content of reviews is available.

---

### 2. `visual_model.ipynb`
This notebook focuses on classifying reviews based on visual features extracted from associated images.

- **Input**: Selected visual features from the dataset (e.g., `bright`, `cont`, `warm`, `colorf`, `sat`, `clar`).
- **Process**:
  - Preprocessing visual features using standardization.
  - Training a **Random Forest Classifier** on the visual features.
  - Evaluation of the model using accuracy, classification report, and confusion matrix.
  - Analysis of important visual features contributing to the classification.
- **Use Case**: Best suited for scenarios where image-related attributes are provided but no textual data is available.

---

### 3. `Multi_Modal_Model.ipynb`
This notebook combines textual and visual features to create a multimodal classification model.

- **Input**: Both the `text` column and visual features from the dataset.
- **Process**:
  - Separate pipelines for processing text (using **TF-IDF**) and visual features (using **StandardScaler**).
  - Integration of the text and visual pipelines using a **ColumnTransformer**.
  - Training a **Random Forest Classifier** on the combined features.
  - Evaluation of the model using accuracy, classification report, and confusion matrix.
  - Analysis of the combined feature importances.
- **Use Case**: Suitable for scenarios where both text and visual data are available, leveraging multimodal insights for better classification accuracy.

---

## **Dataset Requirements**
- All notebooks assume the dataset is provided as `train.csv` in the following format:
  - **Text Column**: Contains review texts.
  - **Visual Features**: Pre-extracted numerical features describing image attributes (e.g., brightness, contrast).
  - **Label**: A binary column (`0` for fake, `1` for real).

---

## **Usage Instructions**
1. Clone this repository.
2. Place the dataset (`train.csv`) in the root directory.
3. Open the desired notebook (`text_model.ipynb`, `visual_model.ipynb`, or `Multi_Modal_Model.ipynb`) in Jupyter Notebook or JupyterLab.
4. Follow the steps in the notebook to train and evaluate the model.

---

## **Model Comparison**
| Model Type      | Input Data     | Use Case                                      | Expected Accuracy |
|------------------|----------------|-----------------------------------------------|-------------------|
| **Text Model**   | Review text    | When only textual data is available.          | High              |
| **Visual Model** | Visual features| When only image attributes are provided.      | Moderate          |
| **Multimodal**   | Text + Visual  | When both text and visual data are available. | Highest           |

---

## **Dependencies**
- Python 3.x
- Jupyter Notebook or JupyterLab
- Required Python libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
