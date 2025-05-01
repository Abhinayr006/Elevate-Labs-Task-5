# Task 5: Decision Trees and Random Forests

This project marks the fifth task of my internship at *Elevate Labs*, where I explored tree-based models for classification, focusing on **Decision Trees** and **Random Forests** to predict heart disease outcomes.

## Objective

The aim of this task was to master tree-based machine learning techniques for classification and regression, including training and evaluating a Decision Tree Classifier, analyzing overfitting, comparing it with a Random Forest, interpreting feature importance, and assessing performance using cross-validation.

##  Technologies and Libraries Used

- **Python**: Core programming language for implementation
- **Scikit-learn**: For building and evaluating Decision Tree and Random Forest models
- **Pandas**: For data handling and exploration
- **NumPy**: For numerical computations
- **Graphviz**: For visualizing the Decision Tree (if implemented in future iterations)

## Workflow and Implementation Steps

1. **Dataset Loading**  
   I utilized the *Heart Disease Dataset* (`Heart_Disease_Dataset.csv`) to predict the presence of heart disease. The dataset, stored at `/content/drive/MyDrive/Colab Notebooks/DataSets/5/Heart_Disease_Dataset.csv`, includes features like age, sex, chest pain type (cp), and target variable (presence of heart disease).

2. **Data Exploration**  
   - Loaded and inspected the dataset using `df.head()` to verify the structure and initial rows, which contain 14 columns including the target (`target`).

3. **Model Development**  
   - Trained a **Decision Tree Classifier** (with limited depth, e.g., `dt_limited`) to control overfitting.  
   - Trained a **Random Forest Classifier** (`rf`) with `random_state=42` for consistent results.  
   - Both models were fitted after preprocessing (assumed scaling of features, e.g., `X_test_scaled`, `y_test`).

4. **Model Evaluation and Comparison**  
   - Compared the accuracy of the Random Forest and Decision Tree models using `score()` on the test set, with Random Forest achieving ~98.5% and Decision Tree ~80% accuracy.  
   - Used **cross-validation** with 5 folds via `cross_val_score` to assess model robustness, yielding a Random Forest CV accuracy of ~99.7% and Decision Tree CV accuracy of ~83.4%.

5. **Feature Importance Analysis**  
   - Extracted and displayed feature importances from the Random Forest model, ranking features like `cp` (chest pain type), `ca` (number of major vessels), and `thalach` (maximum heart rate) as the top contributors.

## Project Files

- `ElevateLabsTask_5.ipynb`: Jupyter Notebook containing the full implementation, including data loading, model training, accuracy comparison, feature importance analysis, and cross-validation.  
- `Heart_Disease_Dataset.csv`: The dataset used for this task, located in the Colab environment.  
- `README.md`: Documentation summarizing the task and outcomes.

## Dataset Details

- **Source**: Custom *Heart Disease Dataset* uploaded to `/content/drive/MyDrive/Colab Notebooks/DataSets/5/Heart_Disease_Dataset.csv`.  
- **Description**: This dataset includes 14 features such as age, sex, resting blood pressure (`trestbps`), cholesterol (`chol`), and a binary target variable (`target`) indicating the presence (1) or absence (0) of heart disease.

## Performance Metrics

The following metrics were used to evaluate the models:  
- **Accuracy**: Measured using `score()` for both Random Forest (~98.5%) and Decision Tree (~80%) on the test set.  
- **Cross-Validation Accuracy**: Assessed with 5-fold cross-validation, showing Random Forest (~99.7%) outperforming Decision Tree (~83.4%).  
- **Feature Importance**: Quantified the contribution of each feature to the Random Forest predictions.

## Key Takeaways

- Gained practical experience in building and tuning Decision Tree and Random Forest models for classification.  
- Learned to address overfitting by limiting tree depth in Decision Trees.  
- Understood the power of ensemble methods like Random Forests, which provided higher accuracy and stability.  
- Explored feature importance to identify key predictors (e.g., chest pain type and maximum heart rate) in heart disease prediction.  
- Recognized the value of cross-validation for robust model evaluation.
