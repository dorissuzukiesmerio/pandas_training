# #https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html

import pandas as pd
import numpy as np

# Modify the examples
# "Predict" in your head : what you expect the result to be
#

#>>> Essential basic functionality
# Here we discuss a lot of the essential functionality common to the pandas data structures. 
# To begin, let’s create some example objects like we did in the 10 minutes to pandas section:

index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])

#>> Head and tail
# To view a small sample of a Series or DataFrame object, use the head() and tail() methods. 
# The default number of elements to display is five, but you may pass a custom number.

long_series = pd.Series(np.random.randn(1000))
print(long_series.head())
# 0   -1.157892
# 1   -1.344312
# 2    0.844885
# 3    1.075770
# 4   -0.109050
# dtype: float64

print(long_series.tail(3))
# 997   -0.289388
# 998   -1.020544
# 999    0.589993
# dtype: float64

#>> Attributes and underlying data
# pandas objects have a number of attributes enabling you to access the metadata
# shape: gives the axis dimensions of the object, consistent with ndarray

# >>Axis labels
# Series: index (only axis)

# DataFrame: index (rows) and columns

# Note, these attributes can be safely assigned to 

print(df[:2])
#                    A         B         C
# 2000-01-01 -0.173215  0.119209 -1.044236
# 2000-01-02 -0.861849 -2.104569 -0.494929

df.columns = [x.lower() for x in df.columns] #picks the names of columns and makes them lower case !! and them reasigns them to columns

print(df)
#                    a         b         c
# 2000-01-01 -0.173215  0.119209 -1.044236
# 2000-01-02 -0.861849 -2.104569 -0.494929
# 2000-01-03  1.071804  0.721555 -0.706771
# 2000-01-04 -1.039575  0.271860 -0.424972
# 2000-01-05  0.567020  0.276232 -1.087401
# 2000-01-06 -0.673690  0.113648 -1.478427
# 2000-01-07  0.524988  0.404705  0.577046
# 2000-01-08 -1.715002 -1.039268 -0.370647
# pandas objects (Index, Series, DataFrame) can be thought of as containers for arrays, which hold the actual data and do the actual computation. 
# For many types, the underlying array is a numpy.ndarray. 
# However, pandas and 3rd party libraries may extend NumPy’s type system to add support for custom arrays (see dtypes).

# To get the actual data inside a Index or Series, use the .array property

print(s.array)
# <PandasArray>
# [ 0.4691122999071863, -0.2828633443286633, -1.5090585031735124,
#  -1.1356323710171934,  1.2121120250208506]
# Length: 5, dtype: float64

print(s.index.array)
# <PandasArray>
# ['a', 'b', 'c', 'd', 'e']
# Length: 5, dtype: object
# array will always be an ExtensionArray. The exact details of what an ExtensionArray is and why pandas uses them are a bit beyond the scope of this introduction. See dtypes for more.

# If you know you need a NumPy array, use to_numpy() or numpy.asarray().

print(s.to_numpy())
# array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])

print(np.asarray(s))
# array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])
# When the Series or Index is backed by an ExtensionArray, to_numpy() may involve copying data and coercing values. See dtypes for more.

# to_numpy() gives some control over the dtype of the resulting numpy.ndarray. For example, consider datetimes with timezones. 
# NumPy doesn’t have a dtype to represent timezone-aware datetimes, so there are two possibly useful representations:

# 1) An object-dtype numpy.ndarray with Timestamp objects, each with the correct tz

# 2) A datetime64[ns] -dtype numpy.ndarray, where the values have been converted to UTC and the timezone discarded

# Timezones may be preserved with dtype=object

ser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
print(ser.to_numpy(dtype=object))
# array([Timestamp('2000-01-01 00:00:00+0100', tz='CET', freq='D'),
#        Timestamp('2000-01-02 00:00:00+0100', tz='CET', freq='D')],
#       dtype=object)
# Or thrown away with dtype='datetime64[ns]'

my_example = pd.Series(pd.date_range("2021", periods=30, tz= EST)) ########UNDERSTAND PROBLEM
print("\nMy Example Date time ")
print(my_example)

print(ser.to_numpy(dtype="datetime64[ns]")) 
# array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00.000000000'],
#       dtype='datetime64[ns]')
# Getting the “raw data” inside a DataFrame is possibly a bit more complex. 
# When your DataFrame only has a single data type for all the columns, DataFrame.to_numpy() will return the underlying data:

print(df.to_numpy())
# array([[-0.1732,  0.1192, -1.0442],
#        [-0.8618, -2.1046, -0.4949],
#        [ 1.0718,  0.7216, -0.7068],
#        [-1.0396,  0.2719, -0.425 ],
#        [ 0.567 ,  0.2762, -1.0874],
#        [-0.6737,  0.1136, -1.4784],
#        [ 0.525 ,  0.4047,  0.577 ],
#        [-1.715 , -1.0393, -0.3706]])

# If a DataFrame contains homogeneously-typed data, the ndarray can actually be modified in-place, and the changes will be reflected in the data structure. 
# For heterogeneous data (e.g. some of the DataFrame’s columns are not all the same dtype), this will not be the case. 
# The values attribute itself, unlike the axis labels, cannot be assigned to.

# Note
# When working with heterogeneous data, the dtype of the resulting ndarray will be chosen to accommodate all of the data involved. 
# For example, if strings are involved, the result will be of object dtype. If there are only floats and integers, the resulting array will be of float dtype.

# In the past, pandas recommended Series.values or DataFrame.values for extracting the data from a Series or DataFrame. 
# You’ll still find references to these in old code bases and online. Going forward, we recommend avoiding .values and using .array or .to_numpy(). 
# .values has the following drawbacks:


# When your Series contains an extension type, it’s unclear whether Series.values returns a NumPy array or the extension array. Series.array will always return an ExtensionArray, and will never copy data. Series.to_numpy() will always return a NumPy array, potentially at the cost of copying / coercing values.

# When your DataFrame contains a mixture of data types, DataFrame.values may involve copying data and coercing values to a common dtype, a relatively expensive operation. DataFrame.to_numpy(), being a method, makes it clearer that the returned NumPy array may not be a view on the same data in the DataFrame.

# Accelerated operations
# pandas has support for accelerating certain types of binary numerical and boolean operations using the numexpr library and the bottleneck libraries.

# These libraries are especially useful when dealing with large data sets, and provide large speedups. numexpr uses smart chunking, caching, and multiple cores. bottleneck is a set of specialized cython routines that are especially fast when dealing with arrays that have nans.

# Here is a sample (using 100 column x 100,000 row DataFrames):


# You are highly encouraged to install both libraries. See the section Recommended Dependencies for more installation info.

# These are both enabled to be used by default, you can control this by setting the options:

# pd.set_option("compute.use_bottleneck", False)
# pd.set_option("compute.use_numexpr", False)
# Flexible binary operations
# With binary operations between pandas data structures, there are two key points of interest:

# Broadcasting behavior between higher- (e.g. DataFrame) and lower-dimensional (e.g. Series) objects.

# Missing data in computations.

# We will demonstrate how to manage these issues independently, though they can be handled simultaneously.

# Matching / broadcasting behavior
# DataFrame has the methods add(), sub(), mul(), div() and related functions radd(), rsub(), … for carrying out binary operations. For broadcasting behavior, Series input is of primary interest. Using these functions, you can use to either match on the index or columns via the axis keyword:

df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)


print(df)
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

row = df.iloc[1]
column = df["two"]
print(df.sub(row, axis="columns")) ######UNDERSTAND 
#         one       two     three
# a  1.051928 -0.139606       NaN
# b  0.000000  0.000000  0.000000
# c  0.352192 -0.433754  1.277825
# d       NaN -1.632779 -0.562782

print(df.sub(row, axis=1))
#         one       two     three
# a  1.051928 -0.139606       NaN
# b  0.000000  0.000000  0.000000
# c  0.352192 -0.433754  1.277825
# d       NaN -1.632779 -0.562782

print(df.sub(column, axis="index"))
#         one  two     three
# a -0.377535  0.0       NaN
# b -1.569069  0.0 -1.962513
# c -0.783123  0.0 -0.250933
# d       NaN  0.0 -0.892516

print(df.sub(column, axis=0))
#         one  two     three
# a -0.377535  0.0       NaN
# b -1.569069  0.0 -1.962513
# c -0.783123  0.0 -0.250933
# d       NaN  0.0 -0.892516
# Furthermore you can align a level of a MultiIndexed DataFrame with a Series.

dfmi = df.copy()

dfmi.index = pd.MultiIndex.from_tuples(
    [(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"]
)

####MULTI -INDEX (like, two levels)

print(dfmi.sub(column, axis=0, level="second"))
#                    one       two     three
# first second                              
# 1     a      -0.377535  0.000000       NaN
#       b      -1.569069  0.000000 -1.962513
#       c      -0.783123  0.000000 -0.250933
# 2     a            NaN -1.493173 -2.385688
# Series and Index also support the divmod() builtin. This function takes the floor division and modulo operation at the same time returning a two-tuple of the same type as the left hand side. For example:

s = pd.Series(np.arange(10))
print(s)
# 0    0
# 1    1
# 2    2
# 3    3
# 4    4
# 5    5
# 6    6
# 7    7
# 8    8
# 9    9
# dtype: int64

div, rem = divmod(s, 3)

print(div) #DIVISION
# 0    0
# 1    0
# 2    0
# 3    1
# 4    1
# 5    1
# 6    2
# 7    2
# 8    2
# 9    3
# dtype: int64

print(rem) #remainder
# 0    0
# 1    1
# 2    2
# 3    0
# 4    1
# 5    2
# 6    0
# 7    1
# 8    2
# 9    0
# dtype: int64

idx = pd.Index(np.arange(10))
print(idx)
# Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')

div, rem = divmod(idx, 3)

print(div)
# Int64Index([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype='int64')

print(rem)
# Int64Index([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype='int64')
# We can also do elementwise divmod():

div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(div)
# 
# 0    0
# 1    0
# 2    0
# 3    1
# 4    1
# 5    1
# 6    1
# 7    1
# 8    1
# 9    1
# dtype: int64

print(rem)
# 
# 0    0
# 1    1
# 2    2
# 3    0
# 4    0
# 5    1
# 6    1
# 7    2
# 8    2
# 9    3
# dtype: int64
