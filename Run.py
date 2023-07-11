import pandas as pd
import numpy as np
from Main import Pre_Process, Vectorization, Random_Forest, Random_Forest_KFold


df = pd.read_csv('Dataset/articles.csv', header=None, encoding='latin-1')  # Reading the dataset
df1 = df.fillna(0)
arr = np.array(df1)
headers = arr[0]
data = arr[1:]

Prcsd_data, label = Pre_Process.Process(data)  # Pre-Processing the dataset
Vec_data = Vectorization.Vectorize(Prcsd_data)  # Vectorization of the dataset
print("Random_Forest Classifier Running")
Random_Forest.Classify(Vec_data, label)
Random_Forest_KFold.Classify(Vec_data, label)
