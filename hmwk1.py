import pandas as pd
import numpy as np

# 5 (a)
college = pd.read_csv("College.csv")

# 5 (b)
college.head(5)
college.rename(columns={'Unnamed: 0': 'School'}, inplace=True)
college.set_index('School')

# 5 (c) i.
college.describe()

# 5 (c) ii.
pd.scatter_matrix(college.iloc[:, [3,5]])

# 5 (c) iii.
college.loc[:, ['Room.Board', 'Private']].boxplot(by='Private')

# 5 (c) iv.
elite = np.array([False]*len(college))
elite[college['Top10perc'] > 50] = True
college['Elite'] = pd.Series(elite, index=college.index)

college.loc[:, 'Elite'].describe(include=['bool'])
college.loc[:, ['Room.Board', 'Elite']].boxplot(by='Elite')

# 5 (c) v.
[college.loc[:, 'Room.Board'].hist(bins=x, alpha=0.7, stacked=False) for x in [10, 20, 30, 40]]

# 5 (c) vi.
pd.scatter_matrix(college.loc[:, ['Expend', 'Grad.Rate', 'S.F.Ratio']])


# 6
auto = pd.read_csv('auto.csv')
auto.dropna(axis=0, how='any', inplace=True)

# 6 (a)
quant_cols = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration']


# 6 (b, c)
def six_sum(d):
    return pd.DataFrame({
        'range': d.loc['max', :] - d.loc['min', :],
        'mean' : d.loc['mean', :],
        'std'  : d.loc['std', :]
    })


six_sum(auto.loc[:, quant_cols].describe())

# 6 (d)
six_sum(auto.loc[0:(len(auto)-50), quant_cols].describe())

