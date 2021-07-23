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



# >>>Function Description

print("\nNumber of non-NA observations")
print(df.count("a"> "8"))

print("\nSum of values")
print(df.sum)

print("\nMean of values")
print(df.mean)

print("\nMean absolute deviation")
print(df.mad)

print("\nArithmetic median of values")
print(df.median)

print("\nMinimum")
print(df.min)

print("\nMaximum")
print(df.max)

print("\nMode")
print(df.mode)

print("\nAbsolute Value")
print(df.abs)

print("\nProduct of values")
print(df.prod)

print("\nBessel-corrected sample standard deviation")
print(df.std)

print("\nUnbiased variance")
print(df.var)

print("\nStandard error of the mean")
print(df.sem)

print("\nSample skewness (3rd moment)")
print(df.skew)

print("\nSample kurtosis (4th moment")
print(df.kurt)


print("\nSample quantile (value at %)")
print(df.quantile)

print("\nCumulative sum")
print(df.cumsum)

print("\nCumulative product")
print(df.cumprod)

print("\nCumulative maximun")
print(df.cummax)

print("\nCumulative minimun")
print(df.cummin)