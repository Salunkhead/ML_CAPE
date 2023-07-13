import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split


def Classify(data, label, folds):
    data = np.array(data)
    label = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    scores = cross_validate(rfc, data, label, cv=4,
                            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
    cr_val = np.array(
        [scores['test_accuracy'], scores['test_precision_macro'], scores['test_recall_macro'],
         scores['test_f1_macro']])
    # np.save('Processed/cross_validation', cr_val)

    cr_df = pd.DataFrame(cr_val, columns=['K-Fold1', 'K-Fold2', 'K-Fold3', 'K-Fold4'], index=['Accuracy', 'Precision', 'Recall', 'F1'])
    print('Cross validation Results')
    print(cr_df.to_markdown())