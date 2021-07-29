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


# >> Index of min/max values
# The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the minimum and maximum corresponding values:

s1 = pd.Series(np.random.randn(5))
print(s1)
 # 
# 0    1.118076
# 1   -0.352051
# 2   -1.242883
# 3   -1.277155
# 4   -0.641184
# dtype: float64

print(s1.idxmin(), s1.idxmax())
 # (3, 0)

df1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])

print(df1)
 # 
#           A         B         C
# 0 -0.327863 -0.946180 -0.137570
# 1 -0.186235 -0.257213 -0.486567
# 2 -0.507027 -0.871259 -0.111110
# 3  2.000339 -2.430505  0.089759
# 4 -0.321434 -0.033695  0.096271

print(df1.idxmin(axis=0))
 # 
# A    2
# B    3
# C    1
# dtype: int64

print(df1.idxmax(axis=1))
 # 
# 0    C
# 1    A
# 2    C
# 3    A
# 4    C
# dtype: object
# When there are multiple rows (or columns) matching the minimum or maximum value, idxmin() and idxmax() return the first matching index:

df3 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=["A"], index=list("edcba"))

print(df3)
# 
#      A
# e  2.0
# d  1.0
# c  1.0
# b  3.0
# a  NaN

print( df3["A"].idxmin())
# 'd'
# Note

# idxmin and idxmax are called argmin and argmax in NumPy.

#>> Value counts (histogramming) / mode
# The value_counts() Series method and top-level function computes a histogram of a 1D array of values. 
# It can also be used as a function on regular arrays:

data = np.random.randint(0, 7, size=50)

print( data)
 # 
# array([6, 6, 2, 3, 5, 3, 2, 5, 4, 5, 4, 3, 4, 5, 0, 2, 0, 4, 2, 0, 3, 2,
#        2, 5, 6, 5, 3, 4, 6, 4, 3, 5, 6, 4, 3, 6, 2, 6, 6, 2, 3, 4, 2, 1,
#        6, 2, 6, 1, 5, 4])

s = pd.Series(data)

print( s.value_counts())
 # 
# 2    10
# 6    10
# 4     9
# 3     8
# 5     8
# 0     3
# 1     2
# dtype: int64

print( pd.value_counts(data))
 # 
# 2    10
# 6    10
# 4     9
# 3     8
# 5     8
# 0     3
# 1     2
# dtype: int64
# New in version 1.1.0.

# The value_counts() method can be used to count combinations across multiple columns. By default all columns are used but a subset can be selected using the subset argument.

data = {"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]}
frame = pd.DataFrame(data)

print( frame.value_counts())
 # 
# a  b
# 1  x    1
# 2  x    1
# 3  y    1
# 4  y    1
# dtype: int64
# Similarly, you can get the most frequently occurring value(s), i.e. the mode, of the values in a Series or DataFrame:

s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])

print( s5.mode())
 # 
# 0    3
# 1    7
# dtype: int64

df5 = pd.DataFrame(
    {
        "A": np.random.randint(0, 7, size=50),
        "B": np.random.randint(-10, 15, size=50),
    }
)


df5.mode()
 # 
#      A   B
# 0  1.0  -9
# 1  NaN  10
# 2  NaN  13

#>> Discretization and quantiling
# Continuous values can be discretized using the cut() (bins based on values) and qcut() (bins based on sample quantiles) functions:

arr = np.random.randn(20)

factor = pd.cut(arr, 4)

print( factor)
 # 
# [(-0.251, 0.464], (-0.968, -0.251], (0.464, 1.179], (-0.251, 0.464], (-0.968, -0.251], ..., (-0.251, 0.464], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251], (-0.968, -0.251]]
# Length: 20
# Categories (4, interval[float64]): [(-0.968, -0.251] < (-0.251, 0.464] < (0.464, 1.179] <
#                                     (1.179, 1.893]]

factor = pd.cut(arr, [-5, -1, 0, 1, 5])

print( factor)
 # 
# [(0, 1], (-1, 0], (0, 1], (0, 1], (-1, 0], ..., (-1, 0], (-1, 0], (-1, 0], (-1, 0], (-1, 0]]
# Length: 20
# Categories (4, interval[int64]): [(-5, -1] < (-1, 0] < (0, 1] < (1, 5]]
# qcut() computes sample quantiles. For example, we could slice up some normally distributed data into equal-size quartiles like so:

arr = np.random.randn(30)

factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])

print( factor)
 # 
# [(0.569, 1.184], (-2.278, -0.301], (-2.278, -0.301], (0.569, 1.184], (0.569, 1.184], ..., (-0.301, 0.569], (1.184, 2.346], (1.184, 2.346], (-0.301, 0.569], (-2.278, -0.301]]
# Length: 30
# Categories (4, interval[float64]): [(-2.278, -0.301] < (-0.301, 0.569] < (0.569, 1.184] <
#                                     (1.184, 2.346]]

print( pd.value_counts(factor))
 # 
# (-2.278, -0.301]    8
# (1.184, 2.346]      8
# (-0.301, 0.569]     7
# (0.569, 1.184]      7
# dtype: int64
# We can also pass infinite values to define the bins:

arr = np.random.randn(20)

factor = pd.cut(arr, [-np.inf, 0, np.inf])

print( factor)
 # 
# [(-inf, 0.0], (0.0, inf], (0.0, inf], (-inf, 0.0], (-inf, 0.0], ..., (-inf, 0.0], (-inf, 0.0], (-inf, 0.0], (0.0, inf], (0.0, inf]]
# Length: 20
# Categories (2, interval[float64]): [(-inf, 0.0] < (0.0, inf]]
# Function application
# To apply your own or another library’s functions to pandas objects, you should be aware of the three methods below. The appropriate method to use depends on whether your function expects to operate on an entire DataFrame or Series, row- or column-wise, or elementwise.

# Tablewise Function Application: pipe()

# Row or Column-wise Function Application: apply()

# Aggregation API: agg() and transform()

# Applying Elementwise Functions: applymap()
