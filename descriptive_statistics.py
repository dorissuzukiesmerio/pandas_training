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


# >>>Function Description

# Quick reference summary table of common functions. 
# Each also takes an optional level parameter which applies only if the object has a hierarchical index.

print("\nNumber of non-NA observations")
print(df.count("a"> "8")())

print("\nSum of values")
print(df.sum())

print("\nMean of values")
print(df.mean())

print("\nMean absolute deviation")
print(df.mad())

print("\nArithmetic median of values")
print(df.median())

print("\nMinimum")
print(df.min())

print("\nMaximum")
print(df.max())

print("\nMode")
print(df.mode())

print("\nAbsolute Value")
print(df.abs())

print("\nProduct of values")
print(df.prod())

print("\nBessel-corrected sample standard deviation")
print(df.std())

print("\nUnbiased variance")
print(df.var())

print("\nStandard error of the mean")
print(df.sem())

print("\nSample skewness (3rd moment)")
print(df.skew())

print("\nSample kurtosis (4th moment")
print(df.kurt())


print("\nSample quantile (value at %)")
print(df.quantile())

print("\nCumulative sum")
print(df.cumsum())

print("\nCumulative product")
print(df.cumprod())

print("\nCumulative maximun")
print(df.cummax())

print("\nCumulative minimun")
print(df.cummin())

# Combining 

#>> Descriptive statistics
# There exists a large number of methods for computing descriptive statistics and other related operations on Series, DataFrame. 
# Most of these are aggregations (hence producing a lower-dimensional result) like sum(), mean(), and quantile(), but some of them, like cumsum() and cumprod(), produce an object of the same size. Generally speaking, these methods take an axis argument, just like ndarray.{sum, std, …}, but the axis can be specified by name or integer:

# Series: no axis argument needed

# DataFrame: “index” (axis=0, default), “columns” (axis=1)

# For example:

print(df)
# 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print(df.mean(0))
# 
# one      0.811094
# two      1.360588
# three    0.187958
# dtype: float64

print(df.mean(1))
# 
# a    1.583749
# b    0.734929
# c    1.133683
# d   -0.166914
# dtype: float64
# All such methods have a skipna option signaling whether to exclude missing data (True by default):

print(df.sum(0, skipna=False))
# 
# one           NaN
# two      5.442353
# three         NaN
# dtype: float64

print(df.sum(axis=1, skipna=True))
# 
# a    3.167498
# b    2.204786
# c    3.401050
# d   -0.333828
# dtype: float64

#Create your own:
# Combined with the broadcasting / arithmetic behavior, one can describe various statistical procedures, 
#like standardization (rendering data zero mean and standard deviation of 1), very concisely:

ts_stand = (df - df.mean()) / df.std()

print(ts_stand.std())
# 
# one      1.0
# two      1.0
# three    1.0
# dtype: float64

xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

print(xs_stand.std(1))
# 
# a    1.0
# b    1.0
# c    1.0
# d    1.0
# dtype: float64
# Note that methods like cumsum() and cumprod() preserve the location of NaN values. This is somewhat different from expanding() and rolling() since NaN behavior is furthermore dictated by a min_periods parameter.

print(df.cumsum())
# 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  1.738035  3.684640 -0.050390
# c  2.433281  5.163008  1.177045
# d       NaN  5.442353  0.563873
