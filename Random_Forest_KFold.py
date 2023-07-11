import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def Classify(data, label):
    data = np.array(data)
    label = np.array(label)
    kf = KFold(n_splits=2)
    for train, test in kf.split(data):
        X_train, X_test, y_train, y_test = data[train], data[test], label[train], label[test]
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

    # Finding precision and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    F1_score = f1_score(y_test, y_pred, average='micro')

    cr = np.array(
        [accuracy, precision, recall,
         F1_score])
    # np.save('Processed/cross_validation', cr)

    cr_df = pd.DataFrame(cr, columns=['K-Fold Analysis'], index=['Accuracy', 'Precision', 'Recall', 'F1'])
    print('Cross validation Results')
    print(cr_df.to_markdown())
