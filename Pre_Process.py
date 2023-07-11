import numpy as np


def Process(dta):
    lab = dta[:,6]
    uni = np.unique(lab)
    label = []
    for i in range(len(lab)):
        for j in range(len(uni)):
            if (lab[i]==uni[j]):
                label.append(j)
    # np.savetxt("Processed/Label.csv", label, delimiter=',', fmt='%s')


    content = []
    data = dta[:,5]
    for i in range(len(data)):
        sence = data[i].replace("<p>", "")
        sentence = sence.replace("</p>", "")
        content.append(sentence)

    return content, label