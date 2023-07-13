import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


def Classify(data, label, tr_per):
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=tr_per)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    # joblib.dump(rfc, 'rfc_model.pkl')

    perf = np.array([accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'),
                   recall_score(y_test, y_pred, average='weighted'),
                   f1_score(y_test, y_pred, average='weighted')])
    # np.save('Processed/Performance_metrics', perf)

    tr_data = pd.DataFrame(perf, columns=['Performance metrics'], index=['Accuracy', 'Precision', 'Recall', 'F1'])
    print('performance metrics Results')
    print(tr_data.to_markdown())