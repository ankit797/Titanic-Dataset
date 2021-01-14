

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing datasets
train = pd.read_csv('train.csv')                 
test = pd.read_csv('test.csv')                  
gender = pd.read_csv('gender_submission.csv')     


ak = {}
for i in range(len(model)-1):
    y_pred = model[i].predict(x_cv)
    ak[i] = y_pred
    
ak1 = pd.DataFrame(ak)
ak1 = ak1.T
ak2 = ak1.mode()
ak2 = ak2.T
ak2 = ak2.values
cm1 = confusion_matrix(y_cv , ak2)
