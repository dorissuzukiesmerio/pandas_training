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


#>> Tablewise function application
# DataFrames and Series can be passed into functions. 
# However, if the function needs to be called in a chain, consider using the pipe() method.

# First some setup:

def extract_city_name(df):
"""
Chicago, IL -> Chicago for city_name column
"""
df["city_name"] = df["city_and_code"].str.split(",").str.get(0)
return df
#    .....: 

def add_country_name(df, country_name=None):
"""
Chicago -> Chicago-US for city_name column
"""
col = "city_name"
df["city_and_country"] = df[col] + country_name
return df
#    .....: 

df_p = pd.DataFrame({"city_and_code": ["Chicago, IL"]})
# extract_city_name and add_country_name are functions taking and returning DataFrames.

# Now compare the following:

print( add_country_name(extract_city_name(df_p), country_name="US"))
 # 
#   city_and_code city_name city_and_country
# 0   Chicago, IL   Chicago        ChicagoUS
# Is equivalent to:

print( df_p.pipe(extract_city_name).pipe(add_country_name, country_name="US"))
 # 
#   city_and_code city_name city_and_country
# 0   Chicago, IL   Chicago        ChicagoUS
# pandas encourages the second style, which is known as method chaining. pipe makes it easy to use your own or another library’s functions in method chains, alongside pandas’ methods.

# In the example above, the functions extract_city_name and add_country_name each expected a DataFrame as the first positional argument. What if the function you wish to apply takes its data as, say, the second argument? In this case, provide pipe with a tuple of (callable, data_keyword). .pipe will route the DataFrame to the argument specified in the tuple.

# For example, we can fit a regression using statsmodels. Their API expects a formula first and a DataFrame as the second argument, data. We pass in the function, keyword pair (sm.ols, 'data') to pipe:

import statsmodels.formula.api as sm

bb = pd.read_csv("data/baseball.csv", index_col="id")

print( ()
    bb.query("h > 0")
    .assign(ln_h=lambda df: np.log(df.h))
    .pipe((sm.ols, "data"), "hr ~ ln_h + year + g + C(lg)")
    .fit()
    .summary()
)

 # 
# <class 'statsmodels.iolib.summary.Summary'>
# """
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                     hr   R-squared:                       0.685
# Model:                            OLS   Adj. R-squared:                  0.665
# Method:                 Least Squares   F-statistic:                     34.28
# Date:                Mon, 12 Apr 2021   Prob (F-statistic):           3.48e-15
# Time:                        16:30:42   Log-Likelihood:                -205.92
# No. Observations:                  68   AIC:                             421.8
# Df Residuals:                      63   BIC:                             432.9
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept   -8484.7720   4664.146     -1.819      0.074   -1.78e+04     835.780
# C(lg)[T.NL]    -2.2736      1.325     -1.716      0.091      -4.922       0.375
# ln_h           -1.3542      0.875     -1.547      0.127      -3.103       0.395
# year            4.2277      2.324      1.819      0.074      -0.417       8.872
# g               0.1841      0.029      6.258      0.000       0.125       0.243
# ==============================================================================
# Omnibus:                       10.875   Durbin-Watson:                   1.999
# Prob(Omnibus):                  0.004   Jarque-Bera (JB):               17.298
# Skew:                           0.537   Prob(JB):                     0.000175
# Kurtosis:                       5.225   Cond. No.                     1.49e+07
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 1.49e+07. This might indicate that there are
# strong multicollinearity or other numerical problems.
# """
# The pipe method is inspired by unix pipes and more recently dplyr and magrittr, which have introduced the popular (%>%) (read pipe) operator for R. The implementation of pipe here is quite clean and feels right at home in Python. We encourage you to view the source code of pipe().

# Row or column-wise function application
# Arbitrary functions can be applied along the axes of a DataFrame using the apply() method, which, like the descriptive statistics methods, takes an optional axis argument:

print( df.apply(np.mean))
 # 
# one      0.811094
# two      1.360588
# three    0.187958
# dtype: float64

print( df.apply(np.mean, axis=1))
 # 
# a    1.583749
# b    0.734929
# c    1.133683
# d   -0.166914
# dtype: float64

print( df.apply(lambda x: x.max() - x.min()))
 # 
# one      1.051928
# two      1.632779
# three    1.840607
# dtype: float64

print( df.apply(np.cumsum))
 # 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  1.738035  3.684640 -0.050390
# c  2.433281  5.163008  1.177045
# d       NaN  5.442353  0.563873

print( df.apply(np.exp))
 # 
#         one       two     three
# a  4.034899  5.885648       NaN
# b  1.409244  6.767440  0.950858
# c  2.004201  4.385785  3.412466
# d       NaN  1.322262  0.541630
# The apply() method will also dispatch on a string method name.

print( df.apply("mean"))
 # 
# one      0.811094
# two      1.360588
# three    0.187958
# dtype: float64

print( df.apply("mean", axis=1))
 # 
# a    1.583749
# b    0.734929
# c    1.133683
# d   -0.166914
# dtype: float64
# The return type of the function passed to apply() affects the type of the final output from DataFrame.apply for the default behaviour:

# If the applied function returns a Series, the final output is a DataFrame. The columns match the index of the Series returned by the applied function.

# If the applied function returns any other type, the final output is a Series.

# This default behaviour can be overridden using the result_type, which accepts three options: reduce, broadcast, and expand. These will determine how list-likes return values expand (or not) to a DataFrame.

# apply() combined with some cleverness can be used to answer many questions about a data set. For example, suppose we wanted to extract the date where the maximum value for each column occurred:

tsdf = pd.DataFrame()
    np.random.randn(1000, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=1000),
)


print( tsdf.apply(lambda x: x.idxmax()))
 # 
# A   2000-08-06
# B   2001-01-18
# C   2001-07-18
# dtype: datetime64[ns]
# You may also pass additional arguments and keyword arguments to the apply() method. For instance, consider the following function you would like to apply:

# def subtract_and_divide(x, sub, divide=1):
#     return (x - sub) / divide
# You may then apply this function as follows:

# df.apply(subtract_and_divide, args=(5,), divide=3)
# Another useful feature is the ability to pass Series methods to carry out some Series operation on each column or row:

print( tsdf)
 # 
#                    A         B         C
# 2000-01-01 -0.158131 -0.232466  0.321604
# 2000-01-02 -1.810340 -3.105758  0.433834
# 2000-01-03 -1.209847 -1.156793 -0.136794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08 -0.653602  0.178875  1.008298
# 2000-01-09  1.007996  0.462824  0.254472
# 2000-01-10  0.307473  0.600337  1.643950

print( tsdf.apply(pd.Series.interpolate))
 # 
#                    A         B         C
# 2000-01-01 -0.158131 -0.232466  0.321604
# 2000-01-02 -1.810340 -3.105758  0.433834
# 2000-01-03 -1.209847 -1.156793 -0.136794
# 2000-01-04 -1.098598 -0.889659  0.092225
# 2000-01-05 -0.987349 -0.622526  0.321243
# 2000-01-06 -0.876100 -0.355392  0.550262
# 2000-01-07 -0.764851 -0.088259  0.779280
# 2000-01-08 -0.653602  0.178875  1.008298
# 2000-01-09  1.007996  0.462824  0.254472
# 2000-01-10  0.307473  0.600337  1.643950
# Finally, apply() takes an argument raw which is False by default, which converts each row or column into a Series before applying the function. When set to True, the passed function will instead receive an ndarray object, which has positive performance implications if you do not need the indexing functionality.

#>> Aggregation API
# The aggregation API allows one to express possibly multiple aggregation operations in a single concise way. This API is similar across pandas objects, see groupby API, the window API, and the resample API. The entry point for aggregation is DataFrame.aggregate(), or the alias DataFrame.agg().

# We will use a similar starting frame from above:

tsdf = pd.DataFrame()
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)


print( tsdf.iloc[3:7] = np.nan)

print( tsdf)
 # 
#                    A         B         C
# 2000-01-01  1.257606  1.004194  0.167574
# 2000-01-02 -0.749892  0.288112 -0.757304
# 2000-01-03 -0.207550 -0.298599  0.116018
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.814347 -0.257623  0.869226
# 2000-01-09 -0.250663 -1.206601  0.896839
# 2000-01-10  2.169758 -1.333363  0.283157
# Using a single function is equivalent to apply(). You can also pass named methods as strings. These will return a Series of the aggregated output:

print( tsdf.agg(np.sum))
 # 
# A    3.033606
# B   -1.803879
# C    1.575510
# dtype: float64

print( tsdf.agg("sum"))
 # 
# A    3.033606
# B   -1.803879
# C    1.575510
# dtype: float64

# # these are equivalent to a ``.sum()`` because we are aggregating
# # on a single function
print( tsdf.sum())
 # 
# A    3.033606
# B   -1.803879
# C    1.575510
# dtype: float64

# Single aggregations on a Series this will return a scalar value:

print( tsdf["A"].agg("sum"))
 # 3.033606102414146
# Aggregating with multiple functions
# You can pass multiple aggregation arguments as a list. The results of each of the passed functions will be a row in the resulting DataFrame. These are naturally named from the aggregation function.

print( tsdf.agg(["sum"]))
 # 
#             A         B        C
# sum  3.033606 -1.803879  1.57551
# Multiple functions yield multiple rows:

print( tsdf.agg(["sum", "mean"]))
 # 
#              A         B         C
# sum   3.033606 -1.803879  1.575510
# mean  0.505601 -0.300647  0.262585
# On a Series, multiple functions return a Series, indexed by the function names:

print( tsdf["A"].agg(["sum", "mean"]))
 # 
# sum     3.033606
# mean    0.505601
# Name: A, dtype: float64
# Passing a lambda function will yield a <lambda> named row:

print( tsdf["A"].agg(["sum", lambda x: x.mean()]))
 # 
# sum         3.033606
# <lambda>    0.505601
# Name: A, dtype: float64
# Passing a named function will yield that name for the row:

def mymean(x):
return x.mean()
#    .....: 

print( tsdf["A"].agg(["sum", mymean]))
 # 
# sum       3.033606
# mymean    0.505601
# Name: A, dtype: float64
# Aggregating with a dict
# Passing a dictionary of column names to a scalar or a list of scalars, to DataFrame.agg allows you to customize which functions are applied to which columns. Note that the results are not in any particular order, you can use an OrderedDict instead to guarantee ordering.

print( tsdf.agg({"A": "mean", "B": "sum"}))
 # 
# A    0.505601
# B   -1.803879
# dtype: float64
# Passing a list-like will generate a DataFrame output. You will get a matrix-like output of all of the aggregators. The output will consist of all unique functions. Those that are not noted for a particular column will be NaN:

print( tsdf.agg({"A": ["mean", "min"], "B": "sum"}))
 # 
#              A         B
# mean  0.505601       NaN
# min  -0.749892       NaN
# sum        NaN -1.803879
# Mixed dtypes
# When presented with mixed dtypes that cannot aggregate, .agg will only take the valid aggregations. This is similar to how .groupby.agg works.

print( mdf = pd.DataFrame()
{
    "A": [1, 2, 3],
    "B": [1.0, 2.0, 3.0],
    "C": ["foo", "bar", "baz"],
    "D": pd.date_range("20130101", periods=3),
}
)


print( mdf.dtypes)
 # 
# A             int64
# B           float64
# C            object
# D    datetime64[ns]
# dtype: object
print( mdf.agg(["min", "sum"]))
 # 
#      A    B          C          D
# min  1  1.0        bar 2013-01-01
# sum  6  6.0  foobarbaz        NaT
# Custom describe
# With .agg() it is possible to easily create a custom describe function, similar to the built in describe function.

from functools import partial

q_25 = partial(pd.Series.quantile, q=0.25)

print( q_25.__name__ = "25%")

q_75 = partial(pd.Series.quantile, q=0.75)

print( q_75.__name__ = "75%")

print( tsdf.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"]))
 # 
#                A         B         C
# count   6.000000  6.000000  6.000000
# mean    0.505601 -0.300647  0.262585
# std     1.103362  0.887508  0.606860
# min    -0.749892 -1.333363 -0.757304
# 25%    -0.239885 -0.979600  0.128907
# median  0.303398 -0.278111  0.225365
# 75%     1.146791  0.151678  0.722709
# max     2.169758  1.004194  0.896839
# Transform API
# The transform() method returns an object that is indexed the same (same size) as the original. This API allows you to provide multiple operations at the same time rather than one-by-one. Its API is quite similar to the .agg API.

# We create a frame similar to the one used in the above sections.

tsdf = pd.DataFrame()
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10)

print( tsdf.iloc[3:7] = np.nan)

print( tsdf)
 # 
#                    A         B         C
# 2000-01-01 -0.428759 -0.864890 -0.675341
# 2000-01-02 -0.168731  1.338144 -1.279321
# 2000-01-03 -1.621034  0.438107  0.903794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374 -1.240447 -0.201052
# 2000-01-09 -0.157795  0.791197 -1.144209
# 2000-01-10 -0.030876  0.371900  0.061932
# Transform the entire frame. .transform() allows input functions as: a NumPy function, a string function name or a user defined function.

print( tsdf.transform(np.abs))
 # 
#                    A         B         C
# 2000-01-01  0.428759  0.864890  0.675341
# 2000-01-02  0.168731  1.338144  1.279321
# 2000-01-03  1.621034  0.438107  0.903794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374  1.240447  0.201052
# 2000-01-09  0.157795  0.791197  1.144209
# 2000-01-10  0.030876  0.371900  0.061932

print( tsdf.transform("abs"))
 # 
#                    A         B         C
# 2000-01-01  0.428759  0.864890  0.675341
# 2000-01-02  0.168731  1.338144  1.279321
# 2000-01-03  1.621034  0.438107  0.903794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374  1.240447  0.201052
# 2000-01-09  0.157795  0.791197  1.144209
# 2000-01-10  0.030876  0.371900  0.061932

print( tsdf.transform(lambda x: x.abs()))
 # 
#                    A         B         C
# 2000-01-01  0.428759  0.864890  0.675341
# 2000-01-02  0.168731  1.338144  1.279321
# 2000-01-03  1.621034  0.438107  0.903794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374  1.240447  0.201052
# 2000-01-09  0.157795  0.791197  1.144209
# 2000-01-10  0.030876  0.371900  0.061932
# Here transform() received a single function; this is equivalent to a ufunc application.

print( np.abs(tsdf))
 # 
#                    A         B         C
# 2000-01-01  0.428759  0.864890  0.675341
# 2000-01-02  0.168731  1.338144  1.279321
# 2000-01-03  1.621034  0.438107  0.903794
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374  1.240447  0.201052
# 2000-01-09  0.157795  0.791197  1.144209
# 2000-01-10  0.030876  0.371900  0.061932
# Passing a single function to .transform() with a Series will yield a single Series in return.

print( tsdf["A"].transform(np.abs))
 # 
# 2000-01-01    0.428759
# 2000-01-02    0.168731
# 2000-01-03    1.621034
# 2000-01-04         NaN
# 2000-01-05         NaN
# 2000-01-06         NaN
# 2000-01-07         NaN
# 2000-01-08    0.254374
# 2000-01-09    0.157795
# 2000-01-10    0.030876
# Freq: D, Name: A, dtype: float64
# Transform with multiple functions
# Passing multiple functions will yield a column MultiIndexed DataFrame. The first level will be the original frame column names; the second level will be the names of the transforming functions.

print( tsdf.transform([np.abs, lambda x: x + 1]))
 # 
#                    A                   B                   C          
#             absolute  <lambda>  absolute  <lambda>  absolute  <lambda>
# 2000-01-01  0.428759  0.571241  0.864890  0.135110  0.675341  0.324659
# 2000-01-02  0.168731  0.831269  1.338144  2.338144  1.279321 -0.279321
# 2000-01-03  1.621034 -0.621034  0.438107  1.438107  0.903794  1.903794
# 2000-01-04       NaN       NaN       NaN       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN       NaN       NaN       NaN
# 2000-01-08  0.254374  1.254374  1.240447 -0.240447  0.201052  0.798948
# 2000-01-09  0.157795  0.842205  0.791197  1.791197  1.144209 -0.144209
# 2000-01-10  0.030876  0.969124  0.371900  1.371900  0.061932  1.061932
# Passing multiple functions to a Series will yield a DataFrame. The resulting column names will be the transforming functions.

print( tsdf["A"].transform([np.abs, lambda x: x + 1]))
 # 
#             absolute  <lambda>
# 2000-01-01  0.428759  0.571241
# 2000-01-02  0.168731  0.831269
# 2000-01-03  1.621034 -0.621034
# 2000-01-04       NaN       NaN
# 2000-01-05       NaN       NaN
# 2000-01-06       NaN       NaN
# 2000-01-07       NaN       NaN
# 2000-01-08  0.254374  1.254374
# 2000-01-09  0.157795  0.842205
# 2000-01-10  0.030876  0.969124
# Transforming with a dict
# Passing a dict of functions will allow selective transforming per column.

print( tsdf.transform({"A": np.abs, "B": lambda x: x + 1}))
 # 
#                    A         B
# 2000-01-01  0.428759  0.135110
# 2000-01-02  0.168731  2.338144
# 2000-01-03  1.621034  1.438107
# 2000-01-04       NaN       NaN
# 2000-01-05       NaN       NaN
# 2000-01-06       NaN       NaN
# 2000-01-07       NaN       NaN
# 2000-01-08  0.254374 -0.240447
# 2000-01-09  0.157795  1.791197
# 2000-01-10  0.030876  1.371900
# Passing a dict of lists will generate a MultiIndexed DataFrame with these selective transforms.

print( tsdf.transform({"A": np.abs, "B": [lambda x: x + 1, "sqrt"]}))
 # 
#                    A         B          
#             absolute  <lambda>      sqrt
# 2000-01-01  0.428759  0.135110       NaN
# 2000-01-02  0.168731  2.338144  1.156782
# 2000-01-03  1.621034  1.438107  0.661897
# 2000-01-04       NaN       NaN       NaN
# 2000-01-05       NaN       NaN       NaN
# 2000-01-06       NaN       NaN       NaN
# 2000-01-07       NaN       NaN       NaN
# 2000-01-08  0.254374 -0.240447       NaN
# 2000-01-09  0.157795  1.791197  0.889493
# 2000-01-10  0.030876  1.371900  0.609836
# Applying elementwise functions
# Since not all functions can be vectorized (accept NumPy arrays and return another array or value), the methods applymap() on DataFrame and analogously map() on Series accept any Python function taking a single value and returning a single value. For example:

print( df4)
 # 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

def f(x):
    return len(str(x))

print( df4["one"].map(f))
 # 
# a    18
# b    19
# c    18
# d     3
# Name: one, dtype: int64

print( df4.applymap(f))
 # 
#    one  two  three
# a   18   17      3
# b   19   18     20
# c   18   18     16
# d    3   19     19
# Series.map() has an additional feature; it can be used to easily “link” or “map” values defined by a secondary series. This is closely related to merging/joining functionality:

s = pd.Series()
["six", "seven", "six", "seven", "six"], index=["a", "b", "c", "d", "e"]
)


t = pd.Series({"six": 6.0, "seven": 7.0})

print( s)
 # 
# a      six
# b    seven
# c      six
# d    seven
# e      six
# dtype: object

print( s.map(t))
 # 
# a    6.0
# b    7.0
# c    6.0
# d    7.0
# e    6.0
# dtype: float64
# Reindexing and altering labels
# reindex() is the fundamental data alignment method in pandas. It is used to implement nearly all other features relying on label-alignment functionality. To reindex means to conform the data to match a given set of labels along a particular axis. This accomplishes several things:

# Reorders the existing data to match a new set of labels

# Inserts missing value (NA) markers in label locations where no data for that label existed

# If specified, fill data for missing labels using logic (highly relevant to working with time series data)

# Here is a simple example:

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

print( s)
 # 
# a    1.695148
# b    1.328614
# c    1.234686
# d   -0.385845
# e   -1.326508
# dtype: float64

print( s.reindex(["e", "b", "f", "d"]))
 # 
# e   -1.326508
# b    1.328614
# f         NaN
# d   -0.385845
# dtype: float64
# Here, the f label was not contained in the Series and hence appears as NaN in the result.

# With a DataFrame, you can simultaneously reindex the index and columns:

print( df)
 # 
#         one       two     three
# a  1.394981  1.772517       NaN
# b  0.343054  1.912123 -0.050390
# c  0.695246  1.478369  1.227435
# d       NaN  0.279344 -0.613172

print( df.reindex(index=["c", "f", "b"], columns=["three", "two", "one"]))
 # 
#       three       two       one
# c  1.227435  1.478369  0.695246
# f       NaN       NaN       NaN
# b -0.050390  1.912123  0.343054
# You may also use reindex with an axis keyword:

print( df.reindex(["c", "f", "b"], axis="index"))
 # 
#         one       two     three
# c  0.695246  1.478369  1.227435
# f       NaN       NaN       NaN
# b  0.343054  1.912123 -0.050390
# Note that the Index objects containing the actual axis labels can be shared between objects. So if we have a Series and a DataFrame, the following can be done:

rs = s.reindex(df.index)

print( rs)
 # 
# a    1.695148
# b    1.328614
# c    1.234686
# d   -0.385845
# dtype: float64

print( rs.index is df.index)
 # True
# This means that the reindexed Series’s index is the same Python object as the DataFrame’s index.

# DataFrame.reindex() also supports an “axis-style” calling convention, where you specify a single labels argument and the axis it applies to.

print( df.reindex(["c", "f", "b"], axis="index"))
 # 
#         one       two     three
# c  0.695246  1.478369  1.227435
# f       NaN       NaN       NaN
# b  0.343054  1.912123 -0.050390

print( df.reindex(["three", "two", "one"], axis="columns"))
 # 
#       three       two       one
# a       NaN  1.772517  1.394981
# b -0.050390  1.912123  0.343054
# c  1.227435  1.478369  0.695246
# d -0.613172  0.279344       NaN
# See also

# MultiIndex / Advanced Indexing is an even more concise way of doing reindexing.

# Note

# When writing performance-sensitive code, there is a good reason to spend some time becoming a reindexing ninja: many operations are faster on pre-aligned data. Adding two unaligned DataFrames internally triggers a reindexing step. For exploratory analysis you will hardly notice the difference (because reindex has been heavily optimized), but when CPU cycles matter sprinkling a few explicit reindex calls here and there can have an impact.

# Reindexing to align with another object
# You may wish to take an object and reindex its axes to be labeled the same as another object. While the syntax for this is straightforward albeit verbose, it is a common enough operation that the reindex_like() method is available to make this simpler:

print( df2)
 # 
#         one       two
# a  1.394981  1.772517
# b  0.343054  1.912123
# c  0.695246  1.478369

print( df3)
 # 
#         one       two
# a  0.583888  0.051514
# b -0.468040  0.191120
# c -0.115848 -0.242634

print( df.reindex_like(df2))
 # 
#         one       two
# a  1.394981  1.772517
# b  0.343054  1.912123
# c  0.695246  1.478369

#>> Aligning objects with each other with align
# The align() method is the fastest way to simultaneously align two objects. 
# It supports a join argument (related to joining and merging):

join='outer': take the union of the indexes (default)

join='left': use the calling object’s index

join='right': use the passed object’s index

join='inner': intersect the indexes

#It returns a tuple with both of the reindexed Series:

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

s1 = s[:4]

s2 = s[1:]

print( s1.align(s2))
 # 
# (a   -0.186646
#  b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  e         NaN
#  dtype: float64,
#  a         NaN
#  b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  e    1.114285
#  dtype: float64)

print( s1.align(s2, join="inner"))
 # 
# (b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  dtype: float64,
#  b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  dtype: float64)

print( s1.align(s2, join="left"))
 # 
# (a   -0.186646
#  b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  dtype: float64,
#  a         NaN
#  b   -1.692424
#  c   -0.303893
#  d   -1.425662
#  dtype: float64)
# For DataFrames, the join method will be applied to both the index and the columns by default:

print( df.align(df2, join="inner"))
 # 
# (        one       two
#  a  1.394981  1.772517
#  b  0.343054  1.912123
#  c  0.695246  1.478369,
#          one       two
#  a  1.394981  1.772517
#  b  0.343054  1.912123
#  c  0.695246  1.478369)
# You can also pass an axis option to only align on the specified axis:

print( df.align(df2, join="inner", axis=0))
 # 
# (        one       two     three
#  a  1.394981  1.772517       NaN
#  b  0.343054  1.912123 -0.050390
#  c  0.695246  1.478369  1.227435,
#          one       two
#  a  1.394981  1.772517
#  b  0.343054  1.912123
#  c  0.695246  1.478369)
# If you pass a Series to DataFrame.align(), you can choose to align both objects either on the DataFrame’s index or columns using the axis argument:

print( df.align(df2.iloc[0], axis=1))
 # 
# (        one     three       two
#  a  1.394981       NaN  1.772517
#  b  0.343054 -0.050390  1.912123
#  c  0.695246  1.227435  1.478369
#  d       NaN -0.613172  0.279344,
#  one      1.394981
#  three         NaN
#  two      1.772517
#  Name: a, dtype: float64)
