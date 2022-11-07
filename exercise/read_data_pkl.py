import pickle as pkl
import pandas as pd

file = "/home/berk/VS_Project/simglucose/halfcheetah-expert-v2.pkl"


obj = pd.read_pickle(r'/home/berk/VS_Project/simglucose/halfcheetah-expert-v2.pkl')
#obj = pd.read_csv(r'/home/berk/VS_Project/simglucose/halfcheetah-expert-v2.pkl')

print(type(obj))
row=len(obj)
column=len(obj[0])
print(f'Rows:{row}, Column:{column}')
print("Shape of a list:",len(obj))

    
df = pd.DataFrame(obj)
csv = df.to_csv(r'data_halfcheet.csv')
print(df.columns)


print("Process has been completed..")
