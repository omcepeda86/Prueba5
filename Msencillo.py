#!/usr/bin/python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import sys
import os

def valores(Plot):

    clf = joblib.load(os.path.dirname(__file__) + '\\modelo.pkl') 
    vect = joblib.load(os.path.dirname(__file__) + '\\vector.pkl') 

    PlotTest = []
    PlotTest.append(Plot)
    PlotTest

    X_plott = vect.transform(PlotTest)
    y_pred = clf.predict_proba(X_plott)
       
    return y_pred

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingrese el plot')
        
    else:

        url = sys.argv[1]

        p1 = valores(Plot)
        
        print(url)
        print('Probabilidad de comprar: ', p1)
        