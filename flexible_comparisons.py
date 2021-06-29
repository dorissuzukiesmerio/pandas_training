import pandas as pd
import numpy as np


#Defining the datasets
df = pd.DataFrame(
	{
		"one": pd.Series(np.random.randn(3), index=["a","b","c"]),
		"two":pd.Series(np.random.randn(4), index=["a","b","c","d"]),
		"three":pd.Series(np.random.randn(3),index=["b","c","d"]),
	}
)

#Adjusting to fit the example
# pd.DataFrame.rename(columns={"one" : 1, "two": 2, "three": 3}) ###Understand why it didn't work
# df.rename({'one' : 1, "two": 2, "three": 3}, axis=1)
# df.rename({'one' : 1}, axis=1
# df.rename({'one' : 1}, axis=1

print("\ndf")
print(df)

df2 = pd.DataFrame(
    {
        "A": pd.Series(np.random.randn(8), dtype="float16"),
        "B": pd.Series(np.random.randn(8)),
        "C": pd.Series(np.array(np.random.randn(8), dtype="uint8")),
    }
)

df2.columns = [x.lower() for x in df2.columns] 
df2 = df2.T
print("\ndf2 transposed and lower case")
print(df2)

print("\ndf2")
print(df2)

print(pd.DataFrame({"a": [1, 2]}).dtypes)
print(pd.DataFrame({"a": 1}, index=list(range(2))).dtypes)

# Flexible comparisons
# Series and DataFrame have the binary comparison methods eq, ne, lt, gt, le, and ge whose behavior is analogous to the binary arithmetic operations described above:

print("\nIs df equal to df2")
print(df2.eq(df)) #equal to 

print("\nIs df non equal to df2")
print(df2.ne(df)) # non equal to 
# 
#      one    two  three
# a  False  False   True
# b  False  False  False
# c  False  False  False
# d   True  False  False
# These operations produce a pandas object of the same type as the left-hand-side input that is of dtype bool. These boolean objects can be used in indexing operations, see the section on Boolean indexing.

print("\nIs df lower than df2")
print(df2.lt(df)) #equal to 

print("\nIs df Greater than df2")
print(df.gt(df2)) # greater than 
# 
#      one    two  three
# a  False  False  False
# b  False  False  False
# c  False  False  False
# d  False  False  False

print("\nIs le Greater than df2")
print(df.gt(df2)) #  lower or equal to 

print("\nIs ge Greater than df2")
print(df.gt(df2)) # greater or equal to

#flexible wrappers (eq, ne, le, lt, ge, gt) to comparison operators.
# Equivalent to     ==, !=, <=, <, >=, > 