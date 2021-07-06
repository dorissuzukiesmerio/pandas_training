import pandas as pd
import numpy as np


df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)

print(df)

df2 = pd.DataFrame(
    {
        "A": pd.Series(np.random.randn(8), dtype="float16"),
        "B": pd.Series(np.random.randn(8)),
        "C": pd.Series(np.array(np.random.randn(8), dtype="uint8")),
    }
)

#>> Combining OVERLAPPING data sets

# A problem occasionally arising is the combination of two similar data sets where values in one are preferred over the other. 
# An example would be two data series representing a particular ECONOMIC INDICATOR where one is considered to be of “higher quality”. 

# However, the lower quality series might extend further back in history or have more complete data coverage. 
# As such, we would like to combine two DataFrame objects where missing values in one DataFrame are conditionally filled with like-labeled values from the other DataFrame. 
# The function implementing this operation is combine_first(), which we illustrate:

df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], 
    "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)


df2 = pd.DataFrame(
    {
        "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
    }
)


print(df1)
# 
#      A    B
# 0  1.0  NaN
# 1  NaN  2.0
# 2  3.0  3.0
# 3  5.0  NaN
# 4  NaN  6.0

print(df2)
# 
#      A    B
# 0  5.0  NaN
# 1  2.0  NaN
# 2  4.0  3.0
# 3  NaN  4.0
# 4  3.0  6.0
# 5  7.0  8.0

print(df1.combine_first(df2))
# 
#      A    B
# 0  1.0  NaN
# 1  2.0  2.0
# 2  3.0  3.0
# 3  5.0  4.0
# 4  3.0  6.0
# 5  7.0  8.0


#>>> General DataFrame combine
# The combine_first() method above calls the more general DataFrame.combine(). 
# This method takes another DataFrame and a combiner function, aligns the input DataFrame and then passes the combiner function pairs of Series (i.e., columns whose names are the same).

# So, for instance, to reproduce combine_first() as above:

def combiner(x, y):
    return np.where(pd.isna(x), y, x)

print("\nCombine df1 with df2")
print(df1.combine(df2, combiner))
# 
#      A    B
# 0  1.0  NaN
# 1  2.0  2.0
# 2  3.0  3.0
# 3  5.0  4.0
# 4  3.0  6.0
# 5  7.0  8.0


print("\nCombine df2 with df1")
print(df2.combine(df1, combiner))