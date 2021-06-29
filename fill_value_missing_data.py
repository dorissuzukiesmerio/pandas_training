import pandas as pd 
import numpy as np 

df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)


# Missing data / operations with fill values
# In Series and DataFrame, the arithmetic functions have the option of inputting a fill_value, namely a value to substitute when at most one of the values at a location are missing. For example, when adding two DataFrame objects, you may wish to treat NaN as 0 unless both DataFrames are missing that value, in which case the result will be NaN (you can later replace NaN with some other value using fillna if you wish).

print(df)
# 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print(df2)
# 
#         one       two     three
# a  1.394981  1.772517  1.000000
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print(df + df2)
# 
#         one       two     three
# a  2.789963  3.545034       NaN
# b  0.686107  3.824246 -0.100780
# c  1.390491  2.956737  2.454870
# d       NaN  0.558688 -1.226343

print(df.add(df2, fill_value=0))
# 
#         one       two     three
# a  2.789963  3.545034  1.000000
# b  0.686107  3.824246 -0.100780
# c  1.390491  2.956737  2.454870
# d       NaN  0.558688 -1.226343
# Flexible comparisons
# Series and DataFrame have the binary comparison methods eq, ne, lt, gt, le, and ge whose behavior is analogous to the binary arithmetic operations described above:

print(df.gt(df2))
# 
#      one    two  three
# a  False  False  False
# b  False  False  False
# c  False  False  False
# d  False  False  False

print(df2.ne(df)) 
# 
#      one    two  three
# a  False  False   True
# b  False  False  False
# c  False  False  False
# d   True  False  False
# These operations produce a pandas object of the same type as the left-hand-side input that is of dtype bool. These boolean objects can be used in indexing operations, see the section on Boolean indexing.
