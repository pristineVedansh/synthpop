#!/usr/bin/env python
# coding: utf-8


import pandas as pd

print("Loading data...")
data = pd.read_csv("IHDS_cleaned.csv")

data['CasteReligion'] = data['ReligionLabel'].astype(str) +' '+ data['CasteLabel'].astype(str)


del data['ReligionLabel'], data['CasteLabel'], data['DISTRICT']

# Variables
STATE = "Maharashtra"

discrete_columns = ['Age', 'Height', 'Weight', 'PSUID', 'M_Fever', 'M_Cough',
       'M_Diarrhea', 'M_Cataract', 'M_TB', 'M_High_BP', 'M_Heart_disease',
       'M_Diabetes', 'M_Leprosy', 'M_Cancer', 'M_Asthma', 'M_Polio',
       'M_Paralysis', 'M_Epilepsy', 'SexLabel', 'StateLabel',
       'CasteReligion']


data_state = data[data['StateLabel']==STATE]

data_state = data_state.reset_index()



del data_state['index']
data_state.to_csv("Maharashtra.csv", index=False)

data_state['Age'].unique()

"""
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()

print("Training model...")
ctgan.fit(data_state, discrete_columns, epochs=10)


ctgan.sample(10)

print("Saving model...")

ctgan.save("Maharashtra.pkl")
"""




