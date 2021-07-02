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

# Boolean reductions
# You can apply the reductions: empty, any(), all(), and bool() to provide a way to summarize a boolean result.
print("\nare ALL the elements in this column >0 ?")
print((df > 0).all())
# 
# one      False
# two       True
# three    False
# dtype: bool

print("\nIs ANY element in this column >0")
print((df > 0).any())
# 
# one      True
# two      True
# three    True
# dtype: bool

# You can reduce to a final boolean value.

print("\n Is any element on the DataFrame >0 ")
print((df > 0).any().any()) 
# True
# You can test if a pandas object is empty, via the empty property.

print("\n Is the DataFrame empty?")
print(df.empty)
# False

print("\nIs ")
print(pd.DataFrame(columns=list("ABC")).empty) ###UNDERSTAND WHAT IS THIS DOING ? 
# True
# To evaluate single-element pandas objects in a boolean context, use the method bool():


print(pd.Series([True]).bool())
# True

print(pd.Series([False]).bool())
# False

print(pd.DataFrame([[True]]).bool())
# True

print(pd.DataFrame([[False]]).bool())
# False
# Warning

# You might be tempted to do the following:

# >>> if df:
# ...     pass
# Or

# >>> df and df2
# These will both raise errors, as you are trying to compare multiple values.:

# ValueError: The truth value of an array is ambiguous. Use a.empty, a.any() or a.all().
# See gotchas for a more detailed discussion.

# Comparing if objects are equivalent
# Often you may find that there is more than one way to compute the same result. As a simple example, consider df + df and df * 2. 
# To test that these two computations produce the same result, given the tools shown above, you might imagine using (df + df == df * 2).all(). 
# But in fact, this expression is False:

print(df + df == df * 2)
# 
#      one   two  three
# a   True  True  False
# b   True  True   True
# c   True  True   True
# d  False  True   True

print("\nchecking to see whether this condition is true for all. == symbol")
print((df + df == df * 2).all())
# 
# one      False
# two       True
# three    False
# dtype: bool
# Notice that the boolean DataFrame df + df == df * 2 contains some False values! This is because NaNs do not compare as equals:

print(np.nan == np.nan)
# False
# So, NDFrames (such as Series and DataFrames) have an equals() method for testing equality, with NaNs in corresponding locations treated as equal.

print("\n Is (df + df) equal to (df * 2) ? USING THE EQUALS FUNCTION")
print((df + df).equals(df * 2))
# True
# Note that the Series or DataFrame index needs to be in the SAME ORDER for equality to be True:

df1 = pd.DataFrame({"col": ["foo", 0, np.nan]})

df2 = pd.DataFrame({"col": [np.nan, 0, "foo"]}, index=[2, 1, 0])

print(df1.equals(df2))
# False

print(df1.equals(df2.sort_index()))
# True

#>>> Comparing array-like objects
# You can conveniently perform element-wise comparisons when comparing a pandas data structure with a scalar value:

print(pd.Series(["foo", "bar", "baz"]) == "foo")
# 
# 0     True
# 1    False
# 2    False
# dtype: bool

print(pd.Index(["foo", "bar", "baz"]) == "foo")
# array([ True, False, False])
# pandas also handles element-wise comparisons between different array-like objects of the same length:

print(pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"]))
# 
# 0     True
# 1     True
# 2    False
# dtype: bool

print(pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"]))
# 
# 0     True
# 1     True
# 2    False
# dtype: bool
# Trying to compare Index or Series objects of different lengths will raise a ValueError:

# print(pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo', 'bar'])) #ERROR WHEN NOT SAME LENGHT
# ValueError: Series lengths must match to compare

# print(pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo'])) # ERROR WHEN NOT SAME LENGHT
# ValueError: Series lengths must match to compare


# Note that this is different from the NumPy behavior where a comparison can be broadcast:

print(np.array([1, 2, 3]) == np.array([2]))
# array([False,  True, False])
# or it can return False if broadcasting can not be done:

print(np.array([1, 2, 3]) == np.array([1, 2]))
# False
