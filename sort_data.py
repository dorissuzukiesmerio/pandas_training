# Sorting
# pandas supports three kinds of sorting: sorting by index labels, sorting by column values, and sorting by a combination of both.

# By index
# The Series.sort_index() and DataFrame.sort_index() methods are used to sort a pandas object by its index levels.

print( df = pd.DataFrame()
#    .....:     {
#    .....:         "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
#    .....:         "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
#    .....:         "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
#    .....:     }
#    .....: )
#    .....: 

print( unsorted_df = df.reindex()
#    .....:     index=["a", "d", "c", "b"], columns=["three", "two", "one"]
#    .....: )
#    .....: 

print( unsorted_df)
 # 
#       three       two       one
# a       NaN -1.152244  0.562973
# d -0.252916 -0.109597       NaN
# c  1.273388 -0.167123  0.640382
# b -0.098217  0.009797 -1.299504

# # DataFrame
print( unsorted_df.sort_index())
 # 
#       three       two       one
# a       NaN -1.152244  0.562973
# b -0.098217  0.009797 -1.299504
# c  1.273388 -0.167123  0.640382
# d -0.252916 -0.109597       NaN

print( unsorted_df.sort_index(ascending=False))
 # 
#       three       two       one
# d -0.252916 -0.109597       NaN
# c  1.273388 -0.167123  0.640382
# b -0.098217  0.009797 -1.299504
# a       NaN -1.152244  0.562973

print( unsorted_df.sort_index(axis=1))
 # 
#         one     three       two
# a  0.562973       NaN -1.152244
# d       NaN -0.252916 -0.109597
# c  0.640382  1.273388 -0.167123
# b -1.299504 -0.098217  0.009797

# # Series
print( unsorted_df["three"].sort_index())
 # 
# a         NaN
# b   -0.098217
# c    1.273388
# d   -0.252916
# Name: three, dtype: float64
# New in version 1.1.0.

# Sorting by index also supports a key parameter that takes a callable function to apply to the index being sorted. For MultiIndex objects, the key is applied per-level to the levels specified by level.

print( s1 = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3], "c": [2, 3, 4]}).set_index()
#    .....:     list("ab")
#    .....: )
#    .....: 

print( s1)
 # 
#      c
# a b   
# B 1  2
# a 2  3
# C 3  4
print( s1.sort_index(level="a"))
 # 
#      c
# a b   
# B 1  2
# C 3  4
# a 2  3

print( s1.sort_index(level="a", key=lambda idx: idx.str.lower()))
 # 
#      c
# a b   
# a 2  3
# B 1  2
# C 3  4
# For information on key sorting by value, see value sorting.

# By values
# The Series.sort_values() method is used to sort a Series by its values. The DataFrame.sort_values() method is used to sort a DataFrame by its column or row values. The optional by parameter to DataFrame.sort_values() may used to specify one or more columns to use to determine the sorted order.

print( df1 = pd.DataFrame()
#    .....:     {"one": [2, 1, 1, 1], "two": [1, 3, 2, 4], "three": [5, 4, 3, 2]}
#    .....: )
#    .....: 

print( df1.sort_values(by="two"))
 # 
#    one  two  three
# 0    2    1      5
# 2    1    2      3
# 1    1    3      4
# 3    1    4      2
# The by parameter can take a list of column names, e.g.:

print( df1[["one", "two", "three"]].sort_values(by=["one", "two"]))
 # 
#    one  two  three
# 2    1    2      3
# 1    1    3      4
# 3    1    4      2
# 0    2    1      5
# These methods have special treatment of NA values via the na_position argument:

print( s[2] = np.nan)

print( s.sort_values())
 # 
# 0       A
# 3    Aaba
# 1       B
# 4    Baca
# 6    CABA
# 8     cat
# 7     dog
# 2    <NA>
# 5    <NA>
# dtype: string

print( s.sort_values(na_position="first"))
 # 
# 2    <NA>
# 5    <NA>
# 0       A
# 3    Aaba
# 1       B
# 4    Baca
# 6    CABA
# 8     cat
# 7     dog
# dtype: string
# New in version 1.1.0.

# Sorting also supports a key parameter that takes a callable function to apply to the values being sorted.

print( s1 = pd.Series(["B", "a", "C"]))
print( s1.sort_values())
 # 
# 0    B
# 2    C
# 1    a
# dtype: object

print( s1.sort_values(key=lambda x: x.str.lower()))
 # 
# 1    a
# 0    B
# 2    C
# dtype: object
# key will be given the Series of values and should return a Series or array of the same shape with the transformed values. For DataFrame objects, the key is applied per column, so the key should still expect a Series and return a Series, e.g.

print( df = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3]}))
print( df.sort_values(by="a"))
 # 
#    a  b
# 0  B  1
# 2  C  3
# 1  a  2

print( df.sort_values(by="a", key=lambda col: col.str.lower()))
 # 
#    a  b
# 1  a  2
# 0  B  1
# 2  C  3
# The name or type of each column can be used to apply different functions to different columns.

# By indexes and values
# Strings passed as the by parameter to DataFrame.sort_values() may refer to either columns or index level names.

# # Build MultiIndex
print( idx = pd.MultiIndex.from_tuples()
#    .....:     [("a", 1), ("a", 2), ("a", 2), ("b", 2), ("b", 1), ("b", 1)]
#    .....: )
#    .....: 

print( idx.names = ["first", "second"])

# # Build DataFrame
print( df_multi = pd.DataFrame({"A": np.arange(6, 0, -1)}, index=idx))

print( df_multi)
 # 
#               A
# first second   
# a     1       6
#       2       5
#       2       4
# b     2       3
#       1       2
#       1       1
# Sort by ‘second’ (index) and ‘A’ (column)

print( df_multi.sort_values(by=["second", "A"]))
 # 
#               A
# first second   
# b     1       1
#       1       2
# a     1       6
# b     2       3
# a     2       4
#       2       5
# Note

# If a string matches both a column name and an index level name then a warning is issued and the column takes precedence. This will result in an ambiguity error in a future version.

# searchsorted
# Series has the searchsorted() method, which works similarly to numpy.ndarray.searchsorted().

print( ser = pd.Series([1, 2, 3]))

print( ser.searchsorted([0, 3]))
 # array([0, 2])

print( ser.searchsorted([0, 4]))
 # array([0, 3])

print( ser.searchsorted([1, 3], side="right"))
 # array([1, 3])

print( ser.searchsorted([1, 3], side="left"))
 # array([0, 2])

print( ser = pd.Series([3, 1, 2]))

print( ser.searchsorted([0, 3], sorter=np.argsort(ser)))
 # array([0, 2])
# smallest / largest values
# Series has the nsmallest() and nlargest() methods which return the smallest or largest n values. For a large Series this can be much faster than sorting the entire Series and calling head(n) on the result.

print( s = pd.Series(np.random.permutation(10)))

print( s)
 # 
# 0    2
# 1    0
# 2    3
# 3    7
# 4    1
# 5    5
# 6    9
# 7    6
# 8    8
# 9    4
# dtype: int64

print( s.sort_values())
 # 
# 1    0
# 4    1
# 0    2
# 2    3
# 9    4
# 5    5
# 7    6
# 3    7
# 8    8
# 6    9
# dtype: int64

print( s.nsmallest(3))
 # 
# 1    0
# 4    1
# 0    2
# dtype: int64

print( s.nlargest(3))
 # 
# 6    9
# 8    8
# 3    7
# dtype: int64
# DataFrame also has the nlargest and nsmallest methods.

print( df = pd.DataFrame()
#    .....:     {
#    .....:         "a": [-2, -1, 1, 10, 8, 11, -1],
#    .....:         "b": list("abdceff"),
#    .....:         "c": [1.0, 2.0, 4.0, 3.2, np.nan, 3.0, 4.0],
#    .....:     }
#    .....: )
#    .....: 

print( df.nlargest(3, "a"))
 # 
#     a  b    c
# 5  11  f  3.0
# 3  10  c  3.2
# 4   8  e  NaN

print( df.nlargest(5, ["a", "c"]))
 # 
#     a  b    c
# 5  11  f  3.0
# 3  10  c  3.2
# 4   8  e  NaN
# 2   1  d  4.0
# 6  -1  f  4.0

print( df.nsmallest(3, "a"))
 # 
#    a  b    c
# 0 -2  a  1.0
# 1 -1  b  2.0
# 6 -1  f  4.0

print( df.nsmallest(5, ["a", "c"]))
 # 
#    a  b    c
# 0 -2  a  1.0
# 1 -1  b  2.0
# 6 -1  f  4.0
# 2  1  d  4.0
# 4  8  e  NaN
# Sorting by a MultiIndex column
# You must be explicit about sorting when the column is a MultiIndex, and fully specify all levels to by.

print( df1.columns = pd.MultiIndex.from_tuples()
#    .....:     [("a", "one"), ("a", "two"), ("b", "three")]
#    .....: )
#    .....: 

print( df1.sort_values(by=("a", "two")))
 # 
#     a         b
#   one two three
# 0   2   1     5
# 2   1   2     3
# 1   1   3     4
# 3   1   4     2
