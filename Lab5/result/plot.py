import pandas as pd
import csv
import matplotlib.pyplot as plt

state_datas = pd.read_csv('TD-state.csv')
state_datas.fillna(0, inplace=True)

after_state_datas = pd.read_csv('TD-after-state.csv')
after_state_datas.fillna(0, inplace=True)

def plotFunction(column):
        d = {'state_' + column: state_datas[column], 'after_state_' + column: after_state_datas[column]};
        df = pd.DataFrame(d)
        df.index = df.index * 1000
        df.plot(fontsize=16)
        plt.legend(prop={'size': 16})
        plt.show()

column = ['mean', 'sum', '2048', '4096']
for c in column:
        plotFunction(c)
