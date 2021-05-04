import pandas as pd

df = pd.read_csv("Pune_Synthetic_NewLatLong.csv")
del df['Unnamed: 0']

df.to_csv("Pune_Synthetic_NewLatLong_v2.csv", index=False)
df.sample(500).to_csv("Pune_Synthetic_Sample_v2.csv", index=False)
df[0:500].to_csv("Pune_Synthetic_Sample2_v2.csv", index=False)



#df.sample(500).to_csv("Pune_Synthetic_Sample.csv", index=False)

#file1 = open('pune_stats.txt', 'w')

#file1.write(str(df.describe(include='all')))
#df.describe(include='all').to_csv("pune_stats.csv")
#file1.close()
