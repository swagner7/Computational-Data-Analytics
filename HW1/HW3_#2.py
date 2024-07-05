import pandas as pd
import numpy as np

df = pd.read_csv('n90pol.csv')
amygdala = np.array(df.amygdala)
acc = np.array(df.acc)
orientation = np.array(df.orientation)

print(df)