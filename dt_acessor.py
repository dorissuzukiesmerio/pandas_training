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

# .dt accessor
# Series has an accessor to succinctly return datetime like properties for the values of the Series, 
# if it is a datetime/period like Series. This will return a Series, indexed like the existing Series.

# # datetime
s = pd.Series(pd.date_range("20130101 09:10:12", periods=4))

print( s)
 # 
# 0   2013-01-01 09:10:12
# 1   2013-01-02 09:10:12
# 2   2013-01-03 09:10:12
# 3   2013-01-04 09:10:12
# dtype: datetime64[ns]

print( s.dt.hour)
 # 
# 0    9
# 1    9
# 2    9
# 3    9
# dtype: int64

print( s.dt.second)
 # 
# 0    12
# 1    12
# 2    12
# 3    12
# dtype: int64

print( s.dt.day)
 # 
# 0    1
# 1    2
# 2    3
# 3    4
# dtype: int64
# This enables nice expressions like this:

print( s[s.dt.day == 2])
 # 
# 1   2013-01-02 09:10:12
# dtype: datetime64[ns]
# You can easily produces tz aware transformations:

stz = s.dt.tz_localize("US/Eastern")

print( stz)
 # 
# 0   2013-01-01 09:10:12-05:00
# 1   2013-01-02 09:10:12-05:00
# 2   2013-01-03 09:10:12-05:00
# 3   2013-01-04 09:10:12-05:00
# dtype: datetime64[ns, US/Eastern]

print( stz.dt.tz)
 # <DstTzInfo 'US/Eastern' LMT-1 day, 19:04:00 STD>
# You can also chain these types of operations:

print( s.dt.tz_localize("UTC").dt.tz_convert("US/Eastern"))
 # 
# 0   2013-01-01 04:10:12-05:00
# 1   2013-01-02 04:10:12-05:00
# 2   2013-01-03 04:10:12-05:00
# 3   2013-01-04 04:10:12-05:00
# dtype: datetime64[ns, US/Eastern]
# You can also format datetime values as strings with Series.dt.strftime() which supports the same format as the standard strftime().

# # DatetimeIndex
s = pd.Series(pd.date_range("20130101", periods=4))

print( s)
 # 
# 0   2013-01-01
# 1   2013-01-02
# 2   2013-01-03
# 3   2013-01-04
# dtype: datetime64[ns]

print( s.dt.strftime("%Y/%m/%d"))
 # 
# 0    2013/01/01
# 1    2013/01/02
# 2    2013/01/03
# 3    2013/01/04
# dtype: object
# # PeriodIndex

s = pd.Series(pd.period_range("20130101", periods=4))

print( s)
 # 
# 0    2013-01-01
# 1    2013-01-02
# 2    2013-01-03
# 3    2013-01-04
# dtype: period[D]

print( s.dt.strftime("%Y/%m/%d"))
 # 
# 0    2013/01/01
# 1    2013/01/02
# 2    2013/01/03
# 3    2013/01/04
# dtype: object
# The .dt accessor works for period and timedelta dtypes.

# # period
s = pd.Series(pd.period_range("20130101", periods=4, freq="D"))

print( s)
 # 
# 0    2013-01-01
# 1    2013-01-02
# 2    2013-01-03
# 3    2013-01-04
# dtype: period[D]

print( s.dt.year)
 # 
# 0    2013
# 1    2013
# 2    2013
# 3    2013
# dtype: int64

print( s.dt.day)
 # 
# 0    1
# 1    2
# 2    3
# 3    4
# dtype: int64
# # timedelta
s = pd.Series(pd.timedelta_range("1 day 00:00:05", periods=4, freq="s"))

print( s)
 # 
# 0   1 days 00:00:05
# 1   1 days 00:00:06
# 2   1 days 00:00:07
# 3   1 days 00:00:08
# dtype: timedelta64[ns]

print( s.dt.days)
 # 
# 0    1
# 1    1
# 2    1
# 3    1
# dtype: int64

print( s.dt.seconds)
 # 
# 0    5
# 1    6
# 2    7
# 3    8
# dtype: int64

print( s.dt.components)
 # 
#    days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
# 0     1      0        0        5             0             0            0
# 1     1      0        0        6             0             0            0
# 2     1      0        0        7             0             0            0
# 3     1      0        0        8             0             0            0
# Note

# Series.dt will raise a TypeError if you access with a non-datetime-like values.

# Vectorized string methods
# Series is equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the Series’s str attribute and generally have names matching the equivalent (scalar) built-in string methods. For example:

print( s = pd.Series()
#    .....:     ["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"], dtype="string"
#    .....: )
#    .....: 

print( s.str.lower())
 # 
# 0       a
# 1       b
# 2       c
# 3    aaba
# 4    baca
# 5    <NA>
# 6    caba
# 7     dog
# 8     cat
# dtype: string
# Powerful pattern-matching methods are provided as well, but note that pattern-matching generally uses regular expressions by default (and in some cases always uses them).

# Note

# Prior to pandas 1.0, string methods were only available on object -dtype Series. pandas 1.0 added the StringDtype which is dedicated to strings. See Text data types for more.

# Please see Vectorized String Methods for a complete description.
