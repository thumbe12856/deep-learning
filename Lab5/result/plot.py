import pandas as pd
import csv
import matplotlib.pyplot as plt

datas = pd.read_csv('T-state.csv')
datas.fillna(0, inplace=True)

plt.plot(datas['mean'])
plt.show()
