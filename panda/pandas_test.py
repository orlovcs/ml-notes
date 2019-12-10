import pandas as pd #convention 
import matplotlib.pyplot as plt

#datagframe = df
#panel = 3d df

df = pd.read_csv('amazon.csv')
#this kept crashing when there was a space in the last line, deleted it and it worked
print(df['date'].head()) #print out forst 5 by default rows
print(df.head()) #print out forst 5 by default rows



acre = df[ df['state'] != 'Acre' ]
print(acre.head())

acre.set_index('date') #returns new df
acre = acre.set_index('date') #returns new df
# same as acre.set_index('date', inplace=True)
print(acre.head())

acre.plot()
plt.show() #graphs the data

