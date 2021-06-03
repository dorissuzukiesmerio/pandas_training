# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
import pandas as pd
import numpy as np 

#OBJECT CREATION
#Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range("20130101", periods=6) #6 dates starting at 2013-01-01, by day
print(dates) 

#DATAFRAME BY passing a NumPy array, with a datetime index and labeled columns:
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

#DATAFRAME BY DICTIONARY
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
print(df2)
print(df2.dtypes)

# df2.A                  df2.bool
# df2.abs                df2.boxplot
# df2.add                df2.C
# df2.add_prefix         df2.clip
# df2.add_suffix         df2.columns
# df2.align              df2.copy
# df2.all                df2.count
# df2.any                df2.combine
# df2.append             df2.D
# df2.apply              df2.describe
# df2.applymap           df2.diff
# df2.B                  df2.duplicated

#VIEWING DATA
print(df.head())
print(df.tail(3))
print(df.index)
print(df.columns)
#30 min until here...

#COPYING TO NUMPY -> stores all in one type of data for the whole dataset (while pandas allows one datatype per column )
print(df.to_numpy())
print(df2.to_numpy())
#doesn't include the column or index in the datatype

print(df.describe())
print(df.T) #transposing
print(df.sort_index(axis=1, ascending=False)) #######I DIDN'T UNDERSTAND THE SPECIFICATIONS
print(df.sort_index(axis=0, ascending=False))
print(df.sort_index(axis=1, ascending=True))
print(df.sort_index(axis=0, ascending=True))
print(df.sort_values(by="B"))

#SELECTING THE DATA
print(df["A"]) #COLUMN
print(df[0:3]) #ROWS
print(df.loc[dates[0]]) # cross selection           #              dates[0] # all columns  first row; format is diff
print(df.loc[:,["A","B"]])                          #                     :,["A","B"]]   all rows
print(df.loc["20130102":"20130104", ["A", "B"]])    #"20130102":"20130104", ["A", "B"]   selection of rows
print(df.loc["20130102", ["A", "B"]])               #           "20130102", ["A", "B"]   one row
print(df["20130102":"20130104"])                    #"20130102":"20130104"    
print(df.loc[dates[0], "A"])                        
print(df.at[dates[0], "A"])
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1, 2, 4], [0, 2]])
print(df.iloc[1:3, :])
print(df.iloc[:, 1:3])
print(df.iloc[1, 1])
print(df.iat[1, 1])
print(df[df["A"] > 0])
print(df[df > 0])

#FILTERING:
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print(df2)
print(df2[df2["E"].isin(["two", "four"])])

#SETTING NEW COLUMN #####################REVIEW THIS SECTION
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
print("s1")
print(s1)
df["F"] = s1
# print(df["F"])
df.at[dates[0], "A"] = 0
df.iat[0, 1] = 0
df.loc[:, "D"] = np.array([5] * len(df))
print(df)

df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
print(df1)

#MISSING DATA
df1.dropna(how="any") # drops any row with missing data
df1.fillna(value=5) # filling missing data
pd.isna(df1)        #True or False - missing data? 

#OPERATIONS:
#OBS: Operations in general exclude missing data.

print(df.mean())
print(df.mean(1))

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2) #DID NOT UNDERSTAND######################
print(df.sub(s, axis="index"))
print(df.apply(lambda x: x.max() - x.min()))

#HISTOGRAMMING:
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

#MERGE
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
pieces = [df[:3], df[3:7], df[7:]]
print("pieces")
print(pieces)
print(pd.concat(pieces))
''' Try to understand ##################
Adding a column to a DataFrame is relatively fast. However, adding a row requires a copy, and may be expensive. We recommend passing a pre-built list of records to the DataFrame constructor instead of building a DataFrame by iteratively appending records to it. See Appending to dataframe for more.'''


#JOIN
#Example 1:
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print(left)
print(right)

join = pd.merge(left, right, on = "key")
print(join)

#Example 2:
left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
print(left)
print(right)

join = pd.merge(left, right, on = "key")
print(join)

#GROUPING:
df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

print(df)
print(df.groupby("A").sum())
print(df.groupby(["A", "B"]).sum())

#APPLY FUNCTIONS TO DATA
# df.apply(np.cumsum)

#RESHAPPING:
tuples = list(
    zip(
        *[
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
    )
)

index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
print(df2)

stacked = df2.stack()
print(stacked)

print(stacked.unstack()) #unstacks LAST LEVEL
print(stacked.unstack(1))
print(stacked.unstack(0))

#PIVOT TABLES:
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)
print(df)

pivoted_table  = pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])
print(pivoted_table)

#TIME SERIES:
rng = pd.date_range("1/1/2012", periods=100, freq="S")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample("5Min").sum())

#Time zone representation:
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

ts_utc = ts.tz_localize("UTC")
print(ts_utc)
#Converting to another time zone:
print(ts_utc.tz_convert("US/Eastern"))

rng = pd.date_range("1/1/2012", periods=5, freq="M")
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())

prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9
print(ts.head())

#CATEGORICALS:
df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)

df["grade"] = df["raw_grade"].astype("category") # define as categories
print(df["grade"])
df["grade"].cat.categories = ["very good", "good", "very bad"] #RENAME categories to more meaningful

df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
print(df["grade"])
print(df.sort_values(by="grade"))
print(df.groupby("grade").size())

# PLOTTING:

import matplotlib.pyplot as plt 
plt.close("all")
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot() ############ FIND THE PLOT ON COMPUTER....

df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
plt.show() ##################################IMPORTANT TO ADD THIS WHEN RUNNING IN SUBLIME TEXT - check if there is any error . why blank ? 

#LOADING OR WRITING FILE:
#csv
# df.to_csv("foo.csv")
# pd.read_csv("foo.csv")

# #
# df.to_hdf("foo.h5", "df") ###ERROR HERE
# pd.read_hdf("foo.h5", "df")

# # excel
# df.to_excel("foo.xlsx", sheet_name="Sheet1")
# pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])
# ###Error: xlrd.biffh.XLRDError: Excel xlsx file; not supported


##TRY TO UNDERSTAND THE "GOTCHA" part
# if pd.Series([False, True, False]):
#  print("I was true")

#LESSONS LEARNED THROUGH MISTAKES:
#Do not name your file as pandas !