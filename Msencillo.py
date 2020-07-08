#!/usr/bin/python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
    
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family','p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance','p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    res = pd.DataFrame(y_pred, columns=cols)
       
    return res

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingrese el plot')
        
    else:

        Plot = sys.argv[1]

        res = valores(Plot)
        
        print(Plot)
        print('Probabilidades de pertenecer a un genero: ', res)
        