import pandas as pd

# (a)
college = pd.read_csv("College.csv")

# (b)
college.head(5)
college.rename(columns={'Unnamed: 0': 'School'}, inplace=True)
college.set_index('School')

# (c)
