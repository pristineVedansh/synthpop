import pandas as pd

data = pd.read_csv("jobs.csv")

discrete_columns = ['Job']

from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()
ctgan.fit(data, discrete_columns, epochs=10)
ctgan.save('job.pkl')