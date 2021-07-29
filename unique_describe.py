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


print(np.mean(df["one"]))
# 0.8110935116651192

print(np.mean(df["one"].to_numpy()))
# nan

#>> Series.nunique() will return the number of unique non-NA values in a Series:

series = pd.Series(np.random.randn(500))

print(series[20:500] = np.nan) # setting these values to nan

print(series[10:20] = 5) # setting these values to 5

print(series.nunique()) # checking number of unique values (non-NA , non-5)
# 11

#>> Summarizing data: describe
# There is a convenient describe() function which computes a variety of summary statistics about a Series or the columns of a DataFrame (excluding NAs of course):

series = pd.Series(np.random.randn(1000))

print(series[::2] = np.nan)

print(series.describe())
# 
# count    500.000000
# mean      -0.021292
# std        1.015906
# min       -2.683763
# 25%       -0.699070
# 50%       -0.069718
# 75%        0.714483
# max        3.160915
# dtype: float64

frame = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])

print(frame.iloc[::2] = np.nan)

print(frame.describe())
# 
#                 a           b           c           d           e
# count  500.000000  500.000000  500.000000  500.000000  500.000000
# mean     0.033387    0.030045   -0.043719   -0.051686    0.005979
# std      1.017152    0.978743    1.025270    1.015988    1.006695
# min     -3.000951   -2.637901   -3.303099   -3.159200   -3.188821
# 25%     -0.647623   -0.576449   -0.712369   -0.691338   -0.691115
# 50%      0.047578   -0.021499   -0.023888   -0.032652   -0.025363
# 75%      0.729907    0.775880    0.618896    0.670047    0.649748
# max      2.740139    2.752332    3.004229    2.728702    3.240991
# You can select specific percentiles to include in the output:

print( series.describe(percentiles=[0.05, 0.25, 0.75, 0.95]))
 # 
# count    500.000000
# mean      -0.021292
# std        1.015906
# min       -2.683763
# 5%        -1.645423
# 25%       -0.699070
# 50%       -0.069718
# 75%        0.714483
# 95%        1.711409
# max        3.160915
# dtype: float64
# By default, the median is always included.

# >> For a non-numerical Series object, describe() will give a simple summary of the number of unique values and most frequently occurring values:

s = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])

print( s.describe())
 # 
# count     9
# unique    4
# top       a
# freq      5
# dtype: object
# Note that on a mixed-type DataFrame object, describe() will restrict the summary to include only numerical columns or, if none are, only categorical columns:

frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})

print( frame.describe())
 # 
#               b
# count  4.000000
# mean   1.500000
# std    1.290994
# min    0.000000
# 25%    0.750000
# 50%    1.500000
# 75%    2.250000
# max    3.000000
# This behavior can be controlled by providing a list of types as include/exclude arguments. The special value all can also be used:

print( frame.describe(include=["object"]))
 # 
#          a
# count    4
# unique   2
# top     No
# freq     2

print( frame.describe(include=["number"]))
 # 
#               b
# count  4.000000
# mean   1.500000
# std    1.290994
# min    0.000000
# 25%    0.750000
# 50%    1.500000
# 75%    2.250000
# max    3.000000

print( frame.describe(include="all"))
 # 
#           a         b
# count     4  4.000000
# unique    2       NaN
# top      No       NaN
# freq      2       NaN
# mean    NaN  1.500000
# std     NaN  1.290994
# min     NaN  0.000000
# 25%     NaN  0.750000
# 50%     NaN  1.500000
# 75%     NaN  2.250000
# max     NaN  3.000000
# That feature relies on select_dtypes. Refer to there for details about accepted inputs.
