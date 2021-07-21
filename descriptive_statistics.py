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

sum(df) # Sum of values

mean(df) # Mean of values

mad(df) # Mean absolute deviation

median(df) # Arithmetic median of values

min(df) # Minimum

max(df)# Maximum

mode(df) # Mode

abs(df)# Absolute Value

prod(df) # Product of values

std(df) # Bessel-corrected sample standard deviation

var(df)# Unbiased variance

sem(df) # Standard error of the mean

skew(df) # Sample skewness (3rd moment)

kurt# # Sample kurtosis (4th moment)

quantile # # Sample quantile (value at %)

# # cumsum # # Cumulative sum

# # cumprod # # Cumulative product

# # cummax # # Cumulative maximum

# # cummin # # Cumulative minimum