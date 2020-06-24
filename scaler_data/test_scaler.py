import numpy as np
import pandas as pd
import scaler_data.scaler as sc
values = np.random.randint(0, 1000, 5)
column_name = "test_values"

df = pd.DataFrame(values, columns=[column_name])

original_values = df

scaler = sc.scaler(df)
scaler.clean_train(column_name)

print("Data Scaler:")
print(scaler.data)

scaler.clean_predict(column_name)

print("Check values:")
for index, row in original_values.iterrows():
    print(row[column_name] == scaler.data.at[index, column_name])