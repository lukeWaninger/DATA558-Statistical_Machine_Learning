import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting
import numpy as np

# 5 (a)
college = pd.read_csv("College.csv")

# 5 (b)
college.head(5)
college.rename(columns={'Unnamed: 0': 'School'}, inplace=True)
college.set_index('School')

# 5 (c) i.
round(college.describe(), 2)

# 5 (c) ii.
pd.scatter_matrix(college.iloc[:, 1::5])
plt.suptitle("5cii")

# 5 (c) iii.
college.loc[:, ['Room.Board', 'Private']].boxplot(by='Private')
plt.suptitle("5ciii")

# 5 (c) iv.
elite = np.array([False]*len(college))
elite[college['Top10perc'] > 50] = True
college['Elite'] = pd.Series(elite, index=college.index)

college.loc[:, 'Elite'].describe(include=['bool'])
college.loc[:, ['Room.Board', 'Elite']].boxplot(by='Elite')
plt.suptitle("5civ")

# 5 (c) v.
[college.loc[:, 'Room.Board'].hist(bins=x, alpha=0.7, stacked=False) for x in [10, 20, 30, 40]]
plt.suptitle("5cv")

# 5 (c) vi.
college['TotalCost.Student'] = college.loc[:, 'Room.Board'] + college.loc[:, 'Books'] + college.loc[:, 'Personal']

pd.scatter_matrix(college.loc[:, ['Expend', 'Grad.Rate', 'Outstate', 'Personal', 'Books']])
for col in college.columns:
    fig = plt.figure()
    plt.scatter(college[col], college['Grad.Rate'], alpha=.5)
    plt.xlabel(col)
    plt.ylabel('Graduation Rate')
    plt.title('%s vs. Graduation Rate' % col)
    plt.savefig('GradRate_%s.png' % col)
    plt.close()

fig = plt.figure()
plt.scatter(college.loc[:, 'School'], college.loc[:, 'Grad.Rate'],
            c=college.loc[:, 'Expend'],
            s=college.loc[:, 'perc.alumni'],
            alpha=.5)
plt.title("Higher expenditure per student bubble to the top")
plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
plt.xlabel("School")
plt.ylabel("Graduate rate")

# 6
auto = pd.read_csv('auto.csv')
auto.dropna(axis=0, how='any', inplace=True)

# 6 (a)
quant_cols = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration']


# 6 (b, c)
def six_sum(d):
    return pd.DataFrame({
        'range': d.loc['max', :] - d.loc['min', :],
        'mean' : d.loc['mean',:],
        'std'  : d.loc['std', :]
    })


six_sum(auto.loc[:, quant_cols].describe())

# 6 (d)
six_sum(auto.loc[0:(len(auto)-50), quant_cols].describe())

# 6 (e)
pd.plotting.boxplot(data=auto, column="mpg", by='year')
fig = plt.figure()
plt.scatter(auto.loc[:, 'year'], auto.loc[:, 'mpg'],
            c=auto.loc[:, 'weight'],
            s=(auto.loc[:, 'acceleration']/max(auto.loc[:, 'acceleration']))**-7, alpha=.7)
plt.title("mpg vs year (x), weight (color), and acceleration (size)")

# 6 (f)
pd.scatter_matrix(auto.loc[:, quant_cols])
plt.suptitle("6f")