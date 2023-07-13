# ML_CAPE
Title:  A Text classification model to predict “Article Types”


Softwares Used:
  Python 3.9.11
  Pycharm Community Version 2020.3.5




GUI.py:- User interface code to select the analysis.

Run.py:- Main code starts here.

Pre_Process.py:- Pre-processing the dataset and extracting the labels from it.

Vectorization.py:- Vectorizing the contents of the Article. SentenceBERT tokenizer is used for the Vectorization purpose.

Random_Forest.py:- The Machine Learning Classifier used to predict the Article type is the Random Forest classifier and it is implemented in this file. The model is analysed by varying the Training data(%) and the results are displayed.

Random_Forest_KFold.py:- The implementation of the Random Forest model in order to analyse based on the Cross validation method called K-Fold analysis.

The folder named "Dataset" contains the dataset used.

The "Processed" folder consists of the Label extracted and the intermediary results.

rfc_model.pkl:- The pickle file containing the saved Random Forest model.
