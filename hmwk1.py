import pandas as pd
import numpy as np
import scipy as sp

# (a)
college = pd.read_csv("College.csv")

# (b)
college.head(5)
college.rename(columns={'Unnamed: 0': 'School'}, inplace=True)
college.set_index('School')

# (c) i.
college.describe()

# (c) ii.
pd.scatter_matrix(college.iloc[:, [3,5]])

# (c) iii. TODO: FINISH
college.loc[:, ['Room.Board', 'Private']].boxplot()