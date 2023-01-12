import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# # import landsmark_1 csv and convert to numpy array
# df = pd.read_csv('landmarks_2.csv', header=None)
# # extract the data as a NumPy array
# # data = df[['x', 'y']].values

# # # convert the NumPy array to a list of tuples
# # data_tuples = [tuple(row) for row in data]
# values = df.values.tolist()

# ## drope first position of values
# values.pop(0)

# # Create the scaler
# scaler = MinMaxScaler(feature_range=(0, 1))

# # Fit and transform the data
# scaled_data = scaler.fit_transform(values)


# ## convert scaled data to data frame and save as csv
# df = pd.DataFrame(scaled_data)
# df.to_csv('scaled_landmarks_2.csv', index=False)


# print(scaled_data)

df = pd.read_csv("landmarks_2.csv", header=None)

# convert column values of x and y to float after dropping first row
df = df.drop(df.index[0])
df = df.astype(float)
## get min and max values of x and y of df
x_min = df[0].min()
x_max = df[0].max()
y_min = df[1].min()
y_max = df[1].max()


## generate a scale calculator function to scale the values between x_min and x_max to 0 and 1 and y_min and y_max to 0 and 1
def scale_calculator(n, n_min, n_max):
    if n <= n_min:
        return 0
    elif n >= n_max:
        return 1
    return (n - n_min) / (n_max - n_min)


# convert to scale 0 to 1 based on min and max values
# df[0] = (df[0] - x_min) / (x_max - x_min)
# df[1] = (df[1] - y_min) / (y_max - y_min)
print(x_min, x_max, y_min, y_max)
print(scale_calculator(179, x_min, x_max))
