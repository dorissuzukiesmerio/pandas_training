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

# Dropping labels from an axis
# A method closely related to reindex is the drop() function. It removes a set of labels from an axis:

print( df)
 # 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print( df.drop(["a", "d"], axis=0))
 # 
#         one       two     three
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435

print( df.drop(["one"], axis=1))
 # 
#         two     three
# a  1.772517       NaN
# b  1.912123 -0.050390
# c  1.478369  1.227435
# d  0.279344 -0.613172
# Note that the following also works, but is a bit less obvious / clean:

print( df.reindex(df.index.difference(["a", "d"])))
 # 
#         one       two     three
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# Renaming / mapping labels
# The rename() method allows you to relabel an axis based on some mapping (a dict or Series) or an arbitrary function.

print( s)
 # 
# a   -0.186646
# b   -1.692424
# c   -0.303893
# d   -1.425662
# e    1.114285
# dtype: float64

print( s.rename(str.upper))
 # 
# A   -0.186646
# B   -1.692424
# C   -0.303893
# D   -1.425662
# E    1.114285
# dtype: float64
# If you pass a function, it must return a value when called with any of the labels (and must produce a set of unique values). A dict or Series can also be used:

df.rename(
    columns={"one": "foo", "two": "bar"},
    index={"a": "apple", "b": "banana", "d": "durian"},
)

 # 
#              foo       bar     three
# apple   1.394981  1.772517       NaN
# banana  0.343054  1.912123 -0.050390
# c       0.695246  1.478369  1.227435
# durian       NaN  0.279344 -0.613172
# If the mapping doesn’t include a column/index label, it isn’t renamed. Note that extra labels in the mapping don’t throw an error.

# DataFrame.rename() also supports an “axis-style” calling convention, where you specify a single mapper and the axis to apply that mapping to.

print( df.rename({"one": "foo", "two": "bar"}, axis="columns"))
 # 
#         foo       bar     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print( df.rename({"a": "apple", "b": "banana", "d": "durian"}, axis="index"))
 # 
#              one       two     three
# apple   1.394981  1.772517       NaN
# banana  0.343054  1.912123 -0.050390
# c       0.695246  1.478369  1.227435
# durian       NaN  0.279344 -0.613172
# The rename() method also provides an inplace named parameter that is by default False and copies the underlying data. Pass inplace=True to rename the data in place.

# Finally, rename() also accepts a scalar or list-like for altering the Series.name attribute.

print( s.rename("scalar-name"))
 # 
# a   -0.186646
# b   -1.692424
# c   -0.303893
# d   -1.425662
# e    1.114285
# Name: scalar-name, dtype: float64
# New in version 0.24.0.

# The methods DataFrame.rename_axis() and Series.rename_axis() allow specific names of a MultiIndex to be changed (as opposed to the labels).

print( df = pd.DataFrame()
#    .....:     {"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]},
#    .....:     index=pd.MultiIndex.from_product(
#    .....:         [["a", "b", "c"], [1, 2]], names=["let", "num"]
#    .....:     ),
#    .....: )
#    .....: 

print( df)
 # 
#          x   y
# let num       
# a   1    1  10
#     2    2  20
# b   1    3  30
#     2    4  40
# c   1    5  50
#     2    6  60

print( df.rename_axis(index={"let": "abc"}))
 # 
#          x   y
# abc num       
# a   1    1  10
#     2    2  20
# b   1    3  30
#     2    4  40
# c   1    5  50
#     2    6  60

print( df.rename_axis(index=str.upper))
 # 
#          x   y
# LET NUM       
# a   1    1  10
#     2    2  20
# b   1    3  30
#     2    4  40
# c   1    5  50
#     2    6  60
