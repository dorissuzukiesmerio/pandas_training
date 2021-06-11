#https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#

import pandas as pd
import numpy as np 

####SERIES:

# s = pd.Series(data, index=index)
#data can be things like :
# i) a Python dict
# ii) an ndarray
# iii) a scalar value (like 5)

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
print(s.index)

#If not specified the indexes:
print(pd.Series(np.random.randn(5))) # indexes are 0,1,...

#i) FROM PYTHON DICTIONARIES:
d = {"b": 1, "a": 0, "c": 2}
print(pd.Series(d))
#Note: When the data is a dict, and an index is not passed, the Series index will be ordered by the dict’s insertion order

new = pd.Series(d, index=["b", "c", "d", "a"])
print(new) # assigns new index

scalar = pd.Series(5.0, index=["a", "b", "c", "d", "e"]) #must pass index; scalar will be repeated according to number of indexes assigned
print(scalar)

#SLICING SERIES:
print(s[0])
print(s[:3]) # it includes the 3rd row, unlike the numpy !
print(s[s>s.median()])
print(s[[4,3,1]])
print(np.exp(s)) # applying numpy function to exponentiate

print(s.dtype) # Data type (like numpy)
print(s.array) # retrieving the actual array
print(s.to_numpy())

print(s["a"])
s["e"] = 12.0 #Set value 
print(s)
print("e" in s) # returns boolean (True or False)
print("f" in s)
# print(s["f"]) # error
print(s.get("f")) # returns None
print(s.get("f", np.nan)) #returns np.Nan, or specified default

#VECTOR OPERATION
#Not necessary to loop through each obs

print(s+s)
print(s*2)
print(np.exp(s))
print(s[1:]) #prints b to e (index 1 to the end)
print(s[:-1]) # prints a to d (index beginner to -1)
print(s[1:] + s[:-1]) # adds common, respecting indexation / UNION

#NAME ATTRIBUTE:
s = pd.Series(np.random.randn(5), name="something")
print(s)
print(s.name) #retrieving the name 
s2 = s.rename("different")
print(s2.name)

##DATAFRAME:
# Accepts inputs such as : a series, other dataframes,
# Dict of 1D ndarrays, lists, dicts, or Series
# 2-D numpy.ndarray
# Structured or record ndarray

#>>>>>>> From dict of Series or dicts
#union of the indexes of the various Series. 
#If there are any nested dicts, these will first be converted to Series. 
#If no columns are passed, the columns will be the ordered list of dict keys.

#Dict of series
d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}

df = pd.DataFrame(d)

print(df)
#    one  two
# a  1.0  1.0
# b  2.0  2.0
# c  3.0  3.0
# d  NaN  4.0

print(pd.DataFrame(d, index=["d", "b", "a"])) # specify a different order
#    one  two
# d  NaN  4.0
# b  2.0  2.0
# a  1.0  1.0

print(pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"])) # specify other names for the columns; override the keys in the dict
#    two three
# d  4.0   NaN
# b  2.0   NaN
# a  1.0   NaN
# The row and column labels can be accessed respectively by accessing the index and columns attributes

print(df.index)
print(df.columns)

#>>>From dict of ndarrays / lists (not using pd.Series as in the previous examples)
#The ndarrays must all be the SAME LENGHT;
d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}

print(pd.DataFrame(d))
#    one  two
# 0  1.0  4.0
# 1  2.0  3.0
# 2  3.0  2.0
# 3  4.0  1.0

print(pd.DataFrame(d, index=["a", "b", "c", "d"]))
#    one  two
# a  1.0  4.0
# b  2.0  3.0
# c  3.0  2.0
# d  4.0  1.0

##>>>>From structured or record array
#This case is handled identically to a dict of arrays.

data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "a10")])
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
print(pd.DataFrame(data))
#    A    B         C
# 0  1  2.0  b'Hello'
# 1  2  3.0  b'World'

print(pd.DataFrame(data, index=["first", "second"]))
#         A    B         C
# first   1  2.0  b'Hello'
# second  2  3.0  b'World'

print(pd.DataFrame(data, columns=["C", "A", "B"]))
#           C  A    B
# 0  b'Hello'  1  2.0
# 1  b'World'  2  3.0

#################DataFrame is not intended to work exactly like a 2-dimensional NumPy ndarray.######### Q: what is then the difference?

#>>>>>From a list [] of dicts {"key":value, ... , ...}
data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]

print(pd.DataFrame(data2))
#    a   b     c
# 0  1   2   NaN
# 1  5  10  20.0

print(pd.DataFrame(data2, index=["first", "second"]))
#         a   b     c
# first   1   2   NaN
# second  5  10  20.0

print(pd.DataFrame(data2, columns=["a", "b"]))
#    a   b
# 0  1   2
# 1  5  10

## >> From a dict of tuples
# You can automatically create a MultiIndexed frame by passing a tuples dictionary.

turples_dict ={
     ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
     ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
     ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
     ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
     ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
 }
#        a              b      
#        b    a    c    a     b
# A B  1.0  4.0  5.0  8.0  10.0
#   C  2.0  3.0  6.0  7.0   NaN
#   D  NaN  NaN  NaN  NaN   9.0

########>>>From a Series ( , )

# The result will be a DataFrame with the same index as the input Series, 
# and with one column whose name is the original name of the Series (only if no other column name provided).

###>>> From a list of namedtuples
# The field names of the first namedtuple in the list determine the columns of the DataFrame. 
# The remaining namedtuples (or tuples) are simply unpacked and their values are fed into the rows of the DataFrame. 
# If any of those tuples is shorter than the first namedtuple then the later columns in the corresponding row are marked as missing values. 
# If any are longer than the first namedtuple, a ValueError is raised.

from collections import namedtuple

Point = namedtuple("Point", "x y")

print(pd.DataFrame([Point(0, 0), Point(0, 3), (2, 3)]))
#    x  y
# 0  0  0
# 1  0  3
# 2  2  3

Point3D = namedtuple("Point3D", "x y z")

print(pd.DataFrame([Point3D(0, 0, 0), Point3D(0, 3, 5), Point(2, 3)]))
#    x  y    z
# 0  0  0  0.0
# 1  0  3  5.0
# 2  2  3  NaN

###>>>From a list of dataclasses
# Data Classes as introduced in PEP557, can be passed into the DataFrame constructor. 
# Passing a list of dataclasses is equivalent to passing a list of dictionaries.
# Please be aware, that all values in the list should be dataclasses, mixing types in the list would result in a TypeError.

from dataclasses import make_dataclass

Point = make_dataclass("Point", [("x", int), ("y", int)])

print(pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)]))
#    x  y
# 0  0  0
# 1  0  3
# 2  2  3

# Missing data
# To construct a DataFrame with missing data: np.nan to represent missing values. 
# Alternatively, you may pass a numpy.MaskedArray as the data argument to the DataFrame constructor, and its masked entries will be considered missing.

#Alternate constructors
#>DataFrame.from_dict
print(pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))
#    A  B
# 0  1  4
# 1  2  5
# 2  3  6

# orient='index', the keys will be the row labels. In this case, you can also pass the desired column names:
pd.DataFrame.from_dict(
    dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]),
    orient="index",
    columns=["one", "two", "three"],
)

#    one  two  three
# A    1    2      3
# B    4    5      6

# >>DataFrame.from_records
print(data)

# array([(1, 2., b'Hello'), (2, 3., b'World')],
#       dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])

print(pd.DataFrame.from_records(data, index="C"))
#           A    B
# C               
# b'Hello'  1  2.0
# b'World'  2  3.0
# Column selection, addition, deletion
# You can treat a DataFrame semantically like a dict of like-indexed Series objects. Getting, setting, and deleting columns works with the same syntax as the analogous dict operations:

print(df["one"])
# a    1.0
# b    2.0
# c    3.0
# d    NaN
# Name: one, dtype: float64

df["three"] = df["one"] * df["two"]
df["flag"] = df["one"] > 2

print(df) 
#    one  two  three   flag
# a  1.0  1.0    1.0  False
# b  2.0  2.0    4.0  False
# c  3.0  3.0    9.0   True
# d  NaN  4.0    NaN  False
# Columns can be deleted or popped like with a dict:

del df["two"]
three = df.pop("three")

print(df) 
#    one   flag
# a  1.0  False
# b  2.0  False
# c  3.0   True
# d  NaN  False
# When inserting a scalar value, it will naturally be propagated to fill the column:

df["foo"] = "bar"
print(df)
#    one   flag  foo
# a  1.0  False  bar
# b  2.0  False  bar
# c  3.0   True  bar
# d  NaN  False  bar

# When inserting a Series that does not have the same index as the DataFrame, it will be conformed to the DataFrame’s index:
df["one_trunc"] = df["one"][:2]
print(df)
#    one   flag  foo  one_trunc
# a  1.0  False  bar        1.0
# b  2.0  False  bar        2.0
# c  3.0   True  bar        NaN
# d  NaN  False  bar        NaN

# You can insert raw ndarrays but their length must match the length of the DataFrame’s index.
# By default, columns get inserted at the end. The insert function is available to insert at a particular location in the columns:

df.insert(1, "bar", df["one"])
print(df) 
#    one  bar   flag  foo  one_trunc
# a  1.0  1.0  False  bar        1.0
# b  2.0  2.0  False  bar        2.0
# c  3.0  3.0   True  bar        NaN
# d  NaN  NaN  False  bar        NaN


# Assigning new columns in method chains
# Inspired by dplyr’s mutate verb, DataFrame has an assign() method that allows you to easily create new columns that are potentially derived from existing columns.
iris = pd.read_csv("data/iris.data") # https://archive.ics.uci.edu/ml/datasets/Iris 
# iris = datasets.load_iris() DIDN'T WORK
# print(iris)
print("Column names")
print(list(iris.columns.values)) #Identified mistake!! How to fix it ?
#['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']

# #Some ideas: save array, insert row in the beginning and then rename columns
# df.loc[-1] = ['45', 'Dean', 'male']  # adding a row
# df.index = df.index + 1  # shifting index
# df.sort_index(inplace=True) 

iris.loc[-1] = [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']  # adding a row
iris.index = iris.index + 1  # shifting index
iris.sort_index(inplace=True) 

iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']

print("Iris Dataset Head")
print(iris.head())

# ##PROBLEM: -> PROBLEM FIXED !
# Iris Dataset Head
#    5.1  3.5  1.4  0.2  Iris-setosa
# 0  4.9  3.0  1.4  0.2  Iris-setosa
# 1  4.7  3.2  1.3  0.2  Iris-setosa
# 2  4.6  3.1  1.5  0.2  Iris-setosa
# 3  5.0  3.6  1.4  0.2  Iris-setosa
# 4  5.4  3.9  1.7  0.4  Iris-setosa

#    SepalLength  SepalWidth  PetalLength  PetalWidth         Name
# 0          5.1         3.5          1.4         0.2  Iris-setosa
# 1          4.9         3.0          1.4         0.2  Iris-setosa
# 2          4.7         3.2          1.3         0.2  Iris-setosa
# 3          4.6         3.1          1.5         0.2  Iris-setosa
# 4          5.0         3.6          1.4         0.2  Iris-setosa

sepal_ratio = iris.assign(sepal_ratio = iris["SepalWidth"] / iris["SepalLength"]).head()
print(sepal_ratio)

#    SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
# 0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
# 1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
# 2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
# 3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
# 4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
# In the example above, we inserted a precomputed value. We can also pass in a function of one argument to be evaluated on the DataFrame being assigned to.

print(iris.assign(sepal_ratio=lambda x: (x["SepalWidth"] / x["SepalLength"])).head() )
#    SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
# 0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
# 1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
# 2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
# 3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
# 4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
# assign always returns a copy of the data, leaving the original DataFrame untouched.

# Passing a callable, as opposed to an actual value to be inserted, is useful when you don’t have a reference to the DataFrame at hand. 
# This is common when using assign in a chain of operations. 
# For example, we can limit the DataFrame to just those observations with a Sepal Length greater than 5, calculate the ratio, and plot:

print(
    iris.query("SepalLength > 5") #DataFrame that’s been filtered to those rows with sepal length greater than 5.
    .assign(
        SepalRatio=lambda x: x.SepalWidth / x.SepalLength,#The filtering happens first, and then the ratio calculations. 
        PetalRatio=lambda x: x.PetalWidth / x.PetalLength,
    )
    .plot(kind="scatter", x="SepalRatio", y="PetalRatio")
)

# # Since a function is passed in, the function is computed on the DataFrame being assigned to. 
# This is an example where we didn’t have a reference to the filtered DataFrame available.

# The function signature for assign is simply **kwargs. The keys are the column names for the new fields, and the values are either a value to be inserted (for example, a Series or NumPy array), or a function of one argument to be called on the DataFrame. A copy of the original DataFrame is returned, with the new values inserted.

# Starting with Python 3.6 the order of **kwargs is preserved. This allows for dependent assignment, where an expression later in **kwargs can refer to a column created earlier in the same assign().

dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"])) ########Question understand: why lambda?
#    A  B  C   D
# 0  1  4  5   6
# 1  2  5  7   9
# 2  3  6  9  12
# In the second expression, x['C'] will refer to the newly created column, that’s equal to dfa['A'] + dfa['B'].

# Indexing / selection examples:
#> Row selection -> returns a Series whose index is the columns of the DataFrame:

print(df.loc["b"])
# one            2.0
# bar            2.0
# flag         False
# foo            bar
# one_trunc      2.0
# Name: b, dtype: object

print(df.iloc[2])
# one           3.0
# bar           3.0
# flag         True
# foo           bar
# one_trunc     NaN
# Name: c, dtype: object

#>>>> Data alignment and arithmetic
# Data alignment between DataFrame objects automatically align on both the columns and the index (row labels). 
# Again, the resulting object will have the union of the column and row labels.
df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])
print(df + df2)
#           A         B         C   D
# 0  0.045691 -0.014138  1.380871 NaN
# 1 -0.955398 -1.501007  0.037181 NaN
# 2 -0.662690  1.534833 -0.859691 NaN
# 3 -2.452949  1.237274 -0.133712 NaN
# 4  1.414490  1.951676 -2.320422 NaN
# 5 -0.494922 -1.649727 -1.084601 NaN
# 6 -1.047551 -0.748572 -0.805479 NaN
# 7       NaN       NaN       NaN NaN
# 8       NaN       NaN       NaN NaN
# 9       NaN       NaN       NaN NaN
# When doing an operation between DataFrame and Series, the default behavior is to align the Series index on the DataFrame columns, 
# thus broadcasting row-wise. ############### UNDERSTAND HOW IT WORKED? 
print(df.iloc[0])
# A   -2.137747
# B    2.491047
# C    0.939359
# D    0.111989
print(df - df.iloc[0])
#           A         B         C         D
# 0  0.000000  0.000000  0.000000  0.000000
# 1 -1.359261 -0.248717 -0.453372 -1.754659
# 2  0.253128  0.829678  0.010026 -1.991234
# 3 -1.311128  0.054325 -1.724913 -1.620544
# 4  0.573025  1.500742 -0.676070  1.367331
# 5 -1.741248  0.781993 -1.241620 -2.053136
# 6 -1.240774 -0.869551 -0.153282  0.000430
# 7 -0.743894  0.411013 -0.929563 -0.282386
# 8 -1.194921  1.320690  0.238224 -1.482644
# 9  2.293786  1.856228  0.773289 -1.446531
# For explicit control over the matching and broadcasting behavior, see the section on flexible binary operations.

# >>>>Operations with scalars
print(df * 5 + 2)
#            A         B         C          D
# 0   3.359299 -0.124862  4.835102   3.381160
# 1  -3.437003 -1.368449  2.568242  -5.392133
# 2   4.624938  4.023526  4.885230  -6.575010
# 3  -3.196342  0.146766 -3.789461  -4.721559
# 4   6.224426  7.378849  1.454750  10.217815
# 5  -5.346940  3.785103 -1.373001  -6.884519
# 6  -2.844569 -4.472618  4.068691   3.383309
# 7  -0.360173  1.930201  0.187285   1.969232
# 8  -2.615303  6.478587  6.026220  -4.032059
# 9  14.828230  9.156280  8.701544  -3.851494

print(1 / df)
#           A          B         C           D
# 0  3.678365  -2.353094  1.763605    3.620145
# 1 -0.919624  -1.484363  8.799067   -0.676395
# 2  1.904807   2.470934  1.732964   -0.583090
# 3 -0.962215  -2.697986 -0.863638   -0.743875
# 4  1.183593   0.929567 -9.170108    0.608434
# 5 -0.680555   2.800959 -1.482360   -0.562777
# 6 -1.032084  -0.772485  2.416988    3.614523
# 7 -2.118489 -71.634509 -2.758294 -162.507295
# 8 -1.083352   1.116424  1.241860   -0.828904
# 9  0.389765   0.698687  0.746097   -0.854483

print(df ** 4)
#            A             B         C             D
# 0   0.005462  3.261689e-02  0.103370  5.822320e-03
# 1   1.398165  2.059869e-01  0.000167  4.777482e+00
# 2   0.075962  2.682596e-02  0.110877  8.650845e+00
# 3   1.166571  1.887302e-02  1.797515  3.265879e+00
# 4   0.509555  1.339298e+00  0.000141  7.297019e+00
# 5   4.661717  1.624699e-02  0.207103  9.969092e+00
# 6   0.881334  2.808277e+00  0.029302  5.858632e-03
# 7   0.049647  3.797614e-08  0.017276  1.433866e-09
# 8   0.725974  6.437005e-01  0.420446  2.118275e+00
# 9  43.329821  4.196326e+00  3.227153  1.875802e+00

#>>>>Boolean operators 
df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df1 & df2)
#        a      b
# 0  False  False
# 1  False   True
# 2   True  False
print(df1 | df2)
#       a     b
# 0  True  True
# 1  True  True
# 2  True  True

print(df1 ^ df2) ############SYNTAX UNDERSTAND
#        a      b
# 0   True   True
# 1   True  False
# 2  False   True

print(-df1) ############SYNTAX UNDERSTAND
#        a      b
# 0  False   True
# 1   True  False
# 2  False  False

# >>>Transposing
# To transpose, access the T attribute (also the transpose function), similar to an ndarray:

# # only show the first 5 rows
print(df[:5].T)
#           0         1         2         3         4
# A  0.271860 -1.087401  0.524988 -1.039268  0.844885
# B -0.424972 -0.673690  0.404705 -0.370647  1.075770
# C  0.567020  0.113648  0.577046 -1.157892 -0.109050
# D  0.276232 -1.478427 -1.715002 -1.344312  1.643563
# DataFrame interoperability with NumPy functions
# Elementwise NumPy ufuncs (log, exp, sqrt, …) and various other NumPy functions can be used with no issues on Series and DataFrame, assuming the data within are numeric:

print(np.exp(df))
#            A         B         C         D
# 0   1.312403  0.653788  1.763006  1.318154
# 1   0.337092  0.509824  1.120358  0.227996
# 2   1.690438  1.498861  1.780770  0.179963
# 3   0.353713  0.690288  0.314148  0.260719
# 4   2.327710  2.932249  0.896686  5.173571
# 5   0.230066  1.429065  0.509360  0.169161
# 6   0.379495  0.274028  1.512461  1.318720
# 7   0.623732  0.986137  0.695904  0.993865
# 8   0.397301  2.449092  2.237242  0.299269
# 9  13.009059  4.183951  3.820223  0.310274

print(np.asarray(df))
# array([[ 0.2719, -0.425 ,  0.567 ,  0.2762],
#        [-1.0874, -0.6737,  0.1136, -1.4784],
#        [ 0.525 ,  0.4047,  0.577 , -1.715 ],
#        [-1.0393, -0.3706, -1.1579, -1.3443],
#        [ 0.8449,  1.0758, -0.109 ,  1.6436],
#        [-1.4694,  0.357 , -0.6746, -1.7769],
#        [-0.9689, -1.2945,  0.4137,  0.2767],
#        [-0.472 , -0.014 , -0.3625, -0.0062],
#        [-0.9231,  0.8957,  0.8052, -1.2064],
#        [ 2.5656,  1.4313,  1.3403, -1.1703]])
# DataFrame is not intended to be a drop-in replacement for ndarray as its indexing semantics and data model are quite different in places from an n-dimensional array.

# Series implements __array_ufunc__, which allows it to work with NumPy’s universal functions.

# The ufunc is applied to the underlying array in a Series.

ser = pd.Series([1, 2, 3, 4])
print(np.exp(ser))
# 0     2.718282
# 1     7.389056
# 2    20.085537
# 3    54.598150
# dtype: float64
# Changed in version 0.25.0: When multiple Series are passed to a ufunc, they are aligned before performing the operation.

# Like other parts of the library, pandas will automatically align labeled inputs as part of a ufunc with multiple inputs. 
# For example, using numpy.remainder() on two Series with differently ordered labels will align before the operation.

ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])

print(ser1) 
# a    1
# b    2
# c    3
# dtype: int64

print(ser2) 
# b    1
# a    3
# c    5
# dtype: int64

print(np.remainder(ser1, ser2)) ################ UNDERSTAND
# a    1
# b    0
# c    3
# dtype: int64
# As usual, the union of the two indices is taken, and non-overlapping values are filled with missing values.

ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])
print(ser3)
# b    2
# c    4
# d    6
# dtype: int64

print(np.remainder(ser1, ser3))
# Out[116]: 
# a    NaN
# b    0.0
# c    3.0
# d    NaN
# dtype: float64

#>>>When a binary ufunc is applied to a Series and Index, the Series implementation takes precedence and a Series is returned.

ser = pd.Series([1, 2, 3])
idx = pd.Index([4, 5, 6])
print(np.maximum(ser, idx))
# 0    4
# 1    5
# 2    6
# dtype: int64
# NumPy ufuncs are safe to apply to Series backed by non-ndarray arrays, for example arrays.SparseArray (see Sparse calculation). 
# If possible, the ufunc is applied without converting the underlying data to an ndarray.

# Console display
# Very large DataFrames will be truncated to display them in the console. You can also get a summary using info(). 
# (Here I am reading a CSV version of the baseball dataset from the plyr R package): https://r-data.pmagunia.com/dataset/r-dataset-package-plyr-baseball

baseball = pd.read_csv("data/baseball.csv") #####LEARNED: DO NOT NEED TO NAME IT "baseball.csv" on the folder, just "baseball"
print("baseball_columns") ###########PROBLEM : SEEMS NOT TO BE SAME DATASET !!!
print(list(baseball.columns))

print("baseball")
print(baseball)
#        id     player  year  stint team  lg   g   ab   r    h  X2b  X3b  hr   rbi   sb   cs  bb    so  ibb  hbp   sh   sf  gidp
# 0   88641  womacto01  2006      2  CHN  NL  19   50   6   14    1    0   1   2.0  1.0  1.0   4   4.0  0.0  0.0  3.0  0.0   0.0
# 1   88643  schilcu01  2006      1  BOS  AL  31    2   0    1    0    0   0   0.0  0.0  0.0   0   1.0  0.0  0.0  0.0  0.0   0.0
# ..    ...        ...   ...    ...  ...  ..  ..  ...  ..  ...  ...  ...  ..   ...  ...  ...  ..   ...  ...  ...  ...  ...   ...
# 98  89533   aloumo01  2007      1  NYN  NL  87  328  51  112   19    1  13  49.0  3.0  0.0  27  30.0  5.0  2.0  0.0  3.0  13.0
# 99  89534  alomasa02  2007      1  NYN  NL   8   22   1    3    1    0   0   0.0  0.0  0.0   0   3.0  0.0  0.0  0.0  0.0   0.0

# [100 rows x 23 columns]

print("baseball info")
print(baseball.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 23 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   id      100 non-null    int64  
#  1   player  100 non-null    object 
#  2   year    100 non-null    int64  
#  3   stint   100 non-null    int64  
#  4   team    100 non-null    object 
#  5   lg      100 non-null    object 
#  6   g       100 non-null    int64  
#  7   ab      100 non-null    int64  
#  8   r       100 non-null    int64  
#  9   h       100 non-null    int64  
#  10  X2b     100 non-null    int64  
#  11  X3b     100 non-null    int64  
#  12  hr      100 non-null    int64  
#  13  rbi     100 non-null    float64
#  14  sb      100 non-null    float64
#  15  cs      100 non-null    float64
#  16  bb      100 non-null    int64  
#  17  so      100 non-null    float64
#  18  ibb     100 non-null    float64
#  19  hbp     100 non-null    float64
#  20  sh      100 non-null    float64
#  21  sf      100 non-null    float64
#  22  gidp    100 non-null    float64
# dtypes: float64(9), int64(11), object(3)
# memory usage: 18.1+ KB #########USED MORE MEMORY TOO !!!
# However, using to_string will return a string representation of the DataFrame in tabular form, though it won’t always fit the console width:

# print("baseball iloc convert to string")
print(baseball.iloc[-20:, :12].to_string())
#        id     player  year  stint team  lg    g   ab   r    h  X2b  X3b
# 80  89474  finlest01  2007      1  COL  NL   43   94   9   17    3    0
# 81  89480  embreal01  2007      1  OAK  AL    4    0   0    0    0    0
# 82  89481  edmonji01  2007      1  SLN  NL  117  365  39   92   15    2
# 83  89482  easleda01  2007      1  NYN  NL   76  193  24   54    6    0
# 84  89489  delgaca01  2007      1  NYN  NL  139  538  71  139   30    0
# 85  89493  cormirh01  2007      1  CIN  NL    6    0   0    0    0    0
# 86  89494  coninje01  2007      2  NYN  NL   21   41   2    8    2    0
# 87  89495  coninje01  2007      1  CIN  NL   80  215  23   57   11    1
# 88  89497  clemero02  2007      1  NYA  AL    2    2   0    1    0    0
# 89  89498  claytro01  2007      2  BOS  AL    8    6   1    0    0    0
# 90  89499  claytro01  2007      1  TOR  AL   69  189  23   48   14    0
# 91  89501  cirilje01  2007      2  ARI  NL   28   40   6    8    4    0
# 92  89502  cirilje01  2007      1  MIN  AL   50  153  18   40    9    2
# 93  89521  bondsba01  2007      1  SFN  NL  126  340  75   94   14    0
# 94  89523  biggicr01  2007      1  HOU  NL  141  517  68  130   31    3
# 95  89525  benitar01  2007      2  FLO  NL   34    0   0    0    0    0
# 96  89526  benitar01  2007      1  SFN  NL   19    0   0    0    0    0
# 97  89530  ausmubr01  2007      1  HOU  NL  117  349  38   82   16    3
# 98  89533   aloumo01  2007      1  NYN  NL   87  328  51  112   19    1
# 99  89534  alomasa02  2007      1  NYN  NL    8   22   1    3    1    0

# Wide DataFrames will be printed across multiple rows by default:

print(pd.DataFrame(np.random.randn(3, 12)))
#          0         1         2         3         4         5         6         7         8         9         10        11
# 0 -1.226825  0.769804 -1.281247 -0.727707 -0.121306 -0.097883  0.695775  0.341734  0.959726 -1.110336 -0.619976  0.149748
# 1 -0.732339  0.687738  0.176444  0.403310 -0.154951  0.301624 -2.179861 -1.369849 -0.954208  1.462696 -1.743161 -0.826591
# 2 -0.345352  1.314232  0.690579  0.995761  2.396780  0.014871  3.357427 -0.317441 -1.236269  0.896171 -0.487602 -0.082240
# You can change how much to print on a single row by setting the display.width option:

print(pd.set_option("display.width", 40))  # default is 80
print(pd.DataFrame(np.random.randn(3, 12))) ####UNDERSTAND :I saw no difference !
#          0         1         2         3         4         5         6         7         8         9         10        11
# 0 -2.182937  0.380396  0.084844  0.432390  1.519970 -0.493662  0.600178  0.274230  0.132885 -0.023688  2.410179  1.450520
# 1  0.206053 -0.251905 -2.213588  1.063327  1.266143  0.299368 -0.863838  0.408204 -1.048089 -0.025747 -0.988387  0.094055
# 2  1.262731  1.289997  0.082423 -0.055758  0.536580 -0.489682  0.369374 -0.034571 -2.484478 -0.281461  0.030711  0.109121
# You can adjust the max width of the individual columns by setting display.max_colwidth

datafile = {
    "filename": ["filename_01", "filename_02"],
    "path": [
        "media/user_name/storage/folder_01/filename_01",
        "media/user_name/storage/folder_02/filename_02",
    ],
}

print(pd.set_option("display.max_colwidth", 30))
print(pd.DataFrame(datafile))
#       filename                           path
# 0  filename_01  media/user_name/storage/fo...
# 1  filename_02  media/user_name/storage/fo...

print(pd.set_option("display.max_colwidth", 100))
print(pd.DataFrame(datafile))
#       filename                                           path
# 0  filename_01  media/user_name/storage/folder_01/filename_01
# 1  filename_02  media/user_name/storage/folder_02/filename_02
# You can also disable this feature via the expand_frame_repr option. This will print the table in one block.

# >>>>DataFrame column attribute access and IPython completion
# If a DataFrame column label is a valid Python variable name, the column can be accessed like an attribute:

df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})
print(df)
#        foo1      foo2
# 0  1.126203  0.781836
# 1 -0.977349 -1.071357
# 2  1.474071  0.441153
# 3 -0.064034  2.353925
# 4 -1.282782  0.583787

print(df.foo1)
# 0    1.126203
# 1   -0.977349
# 2    1.474071
# 3   -0.064034
# 4   -1.282782
# Name: foo1, dtype: float64

#>>>> The columns are also connected to the IPython completion mechanism so they can be tab-completed:

# In [5]: df.fo<TAB>  # noqa: E225, E999
# df.foo1  df.foo2


# ###Other questions/comments:
# # SQL
# # Notebooks (Jupyter, chrome, etc?'')

#forecasting exponential smoothing  
# Strata website with interview questions
