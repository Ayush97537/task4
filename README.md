# task4
##  Steps Followed
1. Load Dataset
   Used the built-in `load_breast_cancer()` dataset from Scikit-learn.
2. Data Preprocessing 
   - Split the dataset into training and test sets (80/20 split).
   - Standardized the features using `StandardScaler`.
3. Model Training 
   - Used `LogisticRegression` from Scikit-learn.
   - Fit the model to the training data.
4. Prediction and Evaluation
   - Made predictions on the test set.
   - Evaluated using:
     - Confusion Matrix
     - Precision, Recall, F1-Score (from classification report)
     - ROC-AUC Score
5. Plot ROC Curve
   - Plotted ROC Curve using `matplotlib`.


