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

# >>> Filling while reindexing
# reindex() takes an optional parameter method which is a filling method chosen from the following table:

#>> Method / Action
# pad / ffill # Fill values forward
# bfill / backfill # Fill values backward
# nearest # Fill from the nearest index value

# We illustrate these fill methods on a simple Series:

rng = pd.date_range("8/3/2021", periods=8)

ts = pd.Series(np.random.randn(8), index=rng)

ts2 = ts[[0, 3, 6]]

print(ts)
# 
# 2000-01-03    0.183051
# 2000-01-04    0.400528
# 2000-01-05   -0.015083
# 2000-01-06    2.395489
# 2000-01-07    1.414806
# 2000-01-08    0.118428
# 2000-01-09    0.733639
# 2000-01-10   -0.936077
# Freq: D, dtype: float64

print(ts2)
# 
# 2000-01-03    0.183051
# 2000-01-06    2.395489
# 2000-01-09    0.733639
# Freq: 3D, dtype: float64

print(ts2.reindex(ts.index))
# 
# 2000-01-03    0.183051
# 2000-01-04         NaN
# 2000-01-05         NaN
# 2000-01-06    2.395489
# 2000-01-07         NaN
# 2000-01-08         NaN
# 2000-01-09    0.733639
# 2000-01-10         NaN
# Freq: D, dtype: float64

print(ts2.reindex(ts.index, method="ffill"))
# 
# 2000-01-03    0.183051
# 2000-01-04    0.183051
# 2000-01-05    0.183051
# 2000-01-06    2.395489
# 2000-01-07    2.395489
# 2000-01-08    2.395489
# 2000-01-09    0.733639
# 2000-01-10    0.733639
# Freq: D, dtype: float64

print( ts2.reindex(ts.index, method="bfill"))
 # 
# 2000-01-03    0.183051
# 2000-01-04    2.395489
# 2000-01-05    2.395489
# 2000-01-06    2.395489
# 2000-01-07    0.733639
# 2000-01-08    0.733639
# 2000-01-09    0.733639
# 2000-01-10         NaN
# Freq: D, dtype: float64

print( ts2.reindex(ts.index, method="nearest"))
 # 
# 2000-01-03    0.183051
# 2000-01-04    0.183051
# 2000-01-05    2.395489
# 2000-01-06    2.395489
# 2000-01-07    2.395489
# 2000-01-08    0.733639
# 2000-01-09    0.733639
# 2000-01-10    0.733639
# Freq: D, dtype: float64
# These methods require that the indexes are ordered increasing or decreasing.

# Note that the same result could have been achieved using fillna (except for method='nearest') or interpolate:

print( ts2.reindex(ts.index).fillna(method="ffill"))
 # 
# 2000-01-03    0.183051
# 2000-01-04    0.183051
# 2000-01-05    0.183051
# 2000-01-06    2.395489
# 2000-01-07    2.395489
# 2000-01-08    2.395489
# 2000-01-09    0.733639
# 2000-01-10    0.733639
# Freq: D, dtype: float64
# reindex() will raise a ValueError if the index is not monotonically increasing or decreasing. fillna() and interpolate() will not perform any checks on the order of the index.

# Limits on filling while reindexing
# The limit and tolerance arguments provide additional control over filling while reindexing. 
# Limit specifies the maximum count of consecutive matches:

print( ts2.reindex(ts.index, method="ffill", limit=1))
 # 
# 2000-01-03    0.183051
# 2000-01-04    0.183051
# 2000-01-05         NaN
# 2000-01-06    2.395489
# 2000-01-07    2.395489
# 2000-01-08         NaN
# 2000-01-09    0.733639
# 2000-01-10    0.733639
# Freq: D, dtype: float64
# In contrast, tolerance specifies the maximum distance between the index and indexer values:

print( ts2.reindex(ts.index, method="ffill", tolerance="1 day"))
 # 
# 2000-01-03    0.183051
# 2000-01-04    0.183051
# 2000-01-05         NaN
# 2000-01-06    2.395489
# 2000-01-07    2.395489
# 2000-01-08         NaN
# 2000-01-09    0.733639
# 2000-01-10    0.733639
# Freq: D, dtype: float64
# Notice that when used on a DatetimeIndex, TimedeltaIndex or PeriodIndex, tolerance will coerced into a Timedelta if possible. 
#This allows you to specify tolerance with appropriate strings.
