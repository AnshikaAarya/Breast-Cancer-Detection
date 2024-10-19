# Breast Cancer Detection Using Machine Learning

This repository contains the code and resources for the **Breast Cancer Detection** project, leveraging machine learning techniques to classify breast cancer as malignant or benign using data from the [Kaggle Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Breast cancer is one of the most common cancers in women worldwide. Early detection is critical to improving survival rates, and machine learning offers potential to assist in accurate and timely diagnosis. This project applies various machine learning algorithms to build a predictive model that classifies tumors as benign or malignant based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Dataset
The dataset used for this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains 569 samples with 30 features computed from the FNA. Each sample is labeled as either benign (`B`) or malignant (`M`). You can find and download the dataset from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

The features in the dataset are:
- **Radius** (mean of distances from the center to points on the perimeter)
- **Texture** (standard deviation of gray-scale values)
- **Perimeter**
- **Area**
- **Smoothness** (local variation in radius lengths)
- **Compactness** (perimeter^2 / area - 1.0)
- **Concavity** (severity of concave portions of the contour)
- **Concave points** (number of concave portions of the contour)
- **Symmetry**
- **Fractal dimension**

## Installation
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
```

### Dependencies
Ensure that you have Python 3.x and the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter` (optional for running notebooks)

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
breast-cancer-detection/
│
├── data/                        # Dataset files
├── notebooks/                   # Jupyter notebooks for EDA and model building
├── src/                         # Source code for models
│   ├── data_preprocessing.py    # Data cleaning and preprocessing steps
│   ├── model.py                 # ML model training and evaluation code
│   └── utils.py                 # Helper functions
├── results/                     # Model results and evaluation metrics
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── LICENSE                      # License file
```

## Usage
1. **Preprocess the Data:**  
   The script `data_preprocessing.py` is responsible for loading and cleaning the data, handling missing values, and splitting the dataset into training and test sets.
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train the Model:**  
   After preprocessing, use `model.py` to train and evaluate a machine learning model (e.g., Logistic Regression, SVM, Random Forest).
   ```bash
   python src/model.py
   ```

3. **Jupyter Notebooks:**  
   You can also explore the data and train models interactively by running the Jupyter notebooks in the `notebooks/` directory.
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## Model
We experiment with several machine learning algorithms, including:
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- k-Nearest Neighbors (k-NN)
- XGBoost

### Evaluation Metrics
We use the following metrics to evaluate model performance:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

## Results
After training and testing, the best-performing model achieved the following results:
- **Accuracy:** 97.5%
- **Precision:** 96.8%
- **Recall:** 98.1%
- **F1-Score:** 97.4%
- **ROC-AUC:** 99.2%

These results show that the model effectively distinguishes between malignant and benign tumors.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue. Make sure to follow the project's coding standards.
