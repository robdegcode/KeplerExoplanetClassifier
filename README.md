🌌 Kepler Exoplanet Classifier
A machine learning project that predicts whether a stellar observation from NASA's Kepler Space Telescope is a confirmed exoplanet or a false positive — achieving 98.7% accuracy on real mission data.

What It Does
The Kepler telescope detected potential planets by measuring tiny dips in starlight caused by objects passing in front of stars. Not all of these signals are real planets — many are false positives caused by eclipsing binary stars or instrument noise.
This classifier automates that distinction. Feed it a row of stellar measurements and it predicts: real exoplanet or false positive.

Key Finding
After training, feature importance analysis revealed that koi_score — NASA's own confidence score — was by far the most predictive feature, significantly outweighing all other measurements combined. This makes intuitive sense: NASA's Robovetter system encodes expert domain knowledge into that single value.
That said, the model still learned meaningful signal from the remaining 40 features, suggesting it captures patterns beyond what the score alone reflects.

Results
MetricScoreOverall Accuracy98.7%Precision (Confirmed)1.00Recall (Confirmed)0.96Precision (False Positive)0.98Recall (False Positive)1.00F1 Score (macro avg)0.99

Note on class imbalance: The dataset contains roughly 2x more false positives than confirmed planets. The model handles this naturally via Random Forest's ensemble voting without requiring explicit resampling.


Dataset

Source: NASA Kepler Exoplanet Search Results via Kaggle
Size: 7,316 observations after filtering (removed CANDIDATE and NOT DISPOSITIONED labels)
Features used: 41 numerical columns including orbital period, planet radius, transit depth, stellar properties, and NASA's own disposition flags

Columns dropped:

Identifier columns (rowid, kepid, kepoi_name, kepler_name)
Text/categorical columns (koi_tce_delivname)
Target-leaking columns (koi_disposition, koi_pdisposition)
Fully empty columns (koi_teq_err1, koi_teq_err2)

Missing values were filled using column medians to preserve all 7,316 rows.

How It Works

Data Cleaning — filter to confirmed/false positive labels only, encode as binary (1/0), drop irrelevant columns, impute missing values
Train/Test Split — 80/20 split with fixed random state for reproducibility
Model — Random Forest Classifier with 100 decision trees
Evaluation — classification report with precision, recall, and F1 score across both classes

The Random Forest works by building 100 independent decision trees on random subsets of the data, then taking a majority vote across all trees for each prediction. This ensemble approach makes it robust to noise and overfitting.

Stack

Python
pandas
scikit-learn
matplotlib
numpy


Run It Yourself
bash# Clone the repo
git clone https://github.com/robdegcode/kepler-exoplanet-classifier
cd kepler-exoplanet-classifier

# Install dependencies
pip install pandas scikit-learn matplotlib numpy

# Download the dataset from Kaggle and place cumulative.csv in the project folder

# Run
python kepler.py

About
Built by Rob DeGasperis — freshman CS student at Villanova University (CS + Cybersecurity + Business).
This project was built to explore real-world ML applications using publicly available NASA mission data.
GitHub: @robdegcode
