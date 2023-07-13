import pandas as pd
import numpy as np
import Pre_Process, Vectorization, Random_Forest, Random_Forest_KFold


def callmain(Anal, tr_per, cv):
    df = pd.read_csv('Dataset/articles.csv', header=None, encoding='latin-1')  # Reading the dataset
    df1 = df.fillna(0)
    arr = np.array(df1)
    headers = arr[0]
    data = arr[1:]

    print("Pre-processing")
    Prcsd_data, label = Pre_Process.Process(data)  # Pre-Processing the dataset
    print("Vectorizing")
    Vec_data = Vectorization.Vectorize(Prcsd_data)  # Vectorization of the dataset
    Vec_data = np.array(Vec_data)

    if Anal=='Training data(%)':
        print("Random_Forest Classifier Running")
        Random_Forest.Classify(Vec_data, label, tr_per)
    else:
        print("Random_Forest Classifier Running for cross-validation")
        Random_Forest_KFold.Classify(Vec_data, label, cv)