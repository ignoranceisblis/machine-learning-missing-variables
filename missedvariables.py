# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:16:00 2023

@author: Lenovo
"""



#kütüphaneler


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#kodlar

#veri önisleme

veriler = pd.read_csv("eksikveriler.csv")

print(veriler)


#Solution for missed variables
#taking mean


#sci-kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4]) 
#Fit for learning we want to learn 1 to 4 column in yas, learning mean
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
# nal values replace to 28.45 mean value
# fit learn transform apply
