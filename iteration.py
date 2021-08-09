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
# Iteration
# The behavior of basic iteration over pandas objects depends on the type. 
#When iterating over a Series, it is regarded as array-like, and basic iteration produces the values. 
#DataFrames follow the dict-like convention of iterating over the “keys” of the objects.

# In short, basic iteration (for i in object) produces:

# Series: values

# DataFrame: column labels

# Thus, for example, iterating over a DataFrame gives you the column names:

df = pd.DataFrame(
    {"col1": np.random.randn(3), "col2": np.random.randn(3)}, index=["a", "b", "c"]
)

for col in df:
     print(col)
# col1
# col2
# pandas objects also have the dict-like items() method to iterate over the (key, value) pairs.

# To iterate over the rows of a DataFrame, you can use the following methods:

# iterrows(): Iterate over the rows of a DataFrame as (index, Series) pairs. 
#This converts the rows to Series objects, which can change the dtypes and has some performance implications.

# itertuples(): Iterate over the rows of a DataFrame as namedtuples of the values. 
#This is a lot faster than iterrows(), and is in most cases preferable to use to iterate over the values of a DataFrame.

# Warning

# Iterating through pandas objects is generally slow. In many cases, iterating manually over the rows is not needed and can be avoided with one of the following approaches:

# Look for a vectorized solution: many operations can be performed using built-in methods or NumPy functions, (boolean) indexing, …

# When you have a function that cannot work on the full DataFrame/Series at once, it is better to use apply() instead of iterating over the values. See the docs on function application.

# If you need to do iterative manipulations on the values but performance is important, consider writing the inner loop with cython or numba. See the enhancing performance section for some examples of this approach.

# Warning

# You should never modify something you are iterating over. This is not guaranteed to work in all cases. Depending on the data types, the iterator returns a copy and not a view, and writing to it will have no effect!

# For example, in the following case setting the value has no effect:

df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

for index, row in df.iterrows():
    row["a"] = 10 

print( df)
 # 
#    a  b
# 0  1  a
# 1  2  b
# 2  3  c
# items
# Consistent with the dict-like interface, items() iterates through key-value pairs:

# Series: (index, scalar value) pairs

# DataFrame: (column, Series) pairs

# For example:

for label, ser in df.items():
    print(label)
    print(ser)

# a
# 0    1
# 1    2
# 2    3
# Name: a, dtype: int64
# b
# 0    a
# 1    b
# 2    c
# Name: b, dtype: object
# iterrows
# iterrows() allows you to iterate through the rows of a DataFrame as Series objects. It returns an iterator yielding each index value along with a Series containing the data in each row:

for row_index, row in df.iterrows():
    print(row_index, row, sep="\n") 
# 0
# a    1
# b    a
# Name: 0, dtype: object
# 1
# a    2
# b    b
# Name: 1, dtype: object
# 2
# a    3
# b    c
# Name: 2, dtype: object
# Note

# Because iterrows() returns a Series for each row, it does not preserve dtypes across the rows (dtypes are preserved across columns for DataFrames). For example,

df_orig = pd.DataFrame([[1, 1.5]], columns=["int", "float"])

print( df_orig.dtypes)
 # 
# int        int64
# float    float64
# dtype: object

row = next(df_orig.iterrows())[1]

print( row)
 # 
# int      1.0
# float    1.5
# Name: 0, dtype: float64
# All values in row, returned as a Series, are now upcasted to floats, also the original integer value in column x:

print( row["int"].dtype)
 # dtype('float64')

print( df_orig["int"].dtype)
 # dtype('int64')
# To preserve dtypes while iterating over the rows, it is better to use itertuples() which returns namedtuples of the values and which is generally much faster than iterrows().

# For instance, a contrived way to transpose the DataFrame would be:

df2 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

print(df2)
#    x  y
# 0  1  4
# 1  2  5
# 2  3  6

print(df2.T)
#    0  1  2
# x  1  2  3
# y  4  5  6

df2_t = pd.DataFrame({idx: values for idx, values in df2.iterrows()})

print(df2_t)
#    0  1  2
# x  1  2  3
# y  4  5  6
# itertuples
# The itertuples() method will return an iterator yielding a namedtuple for each row in the DataFrame. The first element of the tuple will be the row’s corresponding index value, while the remaining values are the row values.

# For instance:

for row in df.itertuples():
     print(row) 
# Pandas(Index=0, a=1, b='a')
# Pandas(Index=1, a=2, b='b')
# Pandas(Index=2, a=3, b='c')
# This method does not convert the row to a Series object; it merely returns the values inside a namedtuple. Therefore, itertuples() preserves the data type of the values and is generally faster as iterrows().

# Note

# The column names will be renamed to positional names if they are invalid Python identifiers, repeated, or start with an underscore. With a large number of columns (>255), regular tuples are returned.
