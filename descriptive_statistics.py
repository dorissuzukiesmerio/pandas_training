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

# Number of non-NA observations
df.count("a"> "8")


print(df.sum)
print("\nSum of values")


print(mean)
print("\nMean of values")


print(mad)
print("\nMean absolute deviation")


print(median)
print("\nArithmetic median of values")


print(min)
print("\nMinimum")


print(max(df))
print("\nMaximum")


print(mode)
print("\nMode")


print(abs(df))
print("\nAbsolute Value")


print(prod)
print("\nProduct of values")


print(std)
print("\nBessel-corrected sample standard deviation")


print(var(df))
print("\nUnbiased variance")


print(sem)
print("\nStandard error of the mean")


print(skew)
print("\nSample skewness (3rd moment)")

print("\nSample kurtosis (4th moment")
print(df.kurt)



print(quantile )
print("\nSample quantile (value at %)")


print(cumsum )
print(Cumulative sum)


print(cumprod #Cumulative product)


print(cummax #Cumulative maximum)


print(cummin #Cumulative minimum)