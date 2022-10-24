import pandas as pd

names = ['Observation']
df = pd.read_csv('/home/berk/VS_Project/simglucose/examples/trajectories/DATA1.csv', 
                sep='\t')
print(df.columns)