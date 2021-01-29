# ML_submission

[colab link] (https://colab.research.google.com/drive/1WlyDTQ9BOMiNVA-85z9k2eSrAkBHVGJA?usp=sharing

#### Dependencies
pandas 1.2.1 , numpy 1.19.5 , seaborn 0.11.1 , scikit-learn 0.24.1,  matplotlib 3.3.4 , xgboost 1.3.3 

#### Approach

  - Checking for datatype, null values, balancy of target variable
  - Standarisation
  - Looking for reducing dimentionality
  - Model Buidling and hyperparameter tuning

#### Final Performance
RandomForestClassifier(max_depth=None,max_features='log2',min_samples_leaf=1,min_samples_split=2,n_estimators=700,random_state=1)
##### training metrics
precision = 0.9688, recall = 0.9698, F1 = 0.9693, accuracy = 0.9691
##### validation metrics
precision = 0.9743, recall = 0.9710, F1 = 0.9726, accuracy = 0.9731
#### Result generation
[result_generation.py](https://github.com/karatuno/ML_submission/blob/main/result_generation.py)
#### Result generated on test data
[submission.csv](https://github.com/karatuno/ML_submission/blob/main/submissions.csv)
#### Final notebook with EDA and model generation
[arya.ai assignment.ipynb](https://github.com/karatuno/ML_submission/blob/main/arya_ai_assignment.ipynb)
