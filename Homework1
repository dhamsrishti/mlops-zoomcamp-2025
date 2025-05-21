
### **Q1. Downloading the Data**

Download the January 2023 Yellow Taxi dataset from the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Here are direct URLs for Yellow Taxi Trip Records (replace if necessary):

```bash
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet
```

**Load January data:**

```python
import pandas as pd

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
print(f"Number of columns: {len(df.columns)}")
```

 **Answer: 19 columns**

---

### **Q2. Computing Duration**

```python
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
print(f"Standard deviation: {df['duration'].std():.2f}")
```

 **Answer: 52.59**

---

### **Q3. Dropping Outliers**

Keep only trips with duration between 1 and 60 minutes:

```python
df_clean = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
fraction = len(df_clean) / len(df)
print(f"Fraction left: {fraction:.2f}")
```

 **Answer: 0.95 (or 95%)**

---

### **Q4. One-hot Encoding**

We only use:

* `PULocationID` (pickup location)
* `DOLocationID` (dropoff location)

```python
from sklearn.feature_extraction import DictVectorizer

# Convert IDs to string to ensure correct encoding
df_clean['PULocationID'] = df_clean['PULocationID'].astype(str)
df_clean['DOLocationID'] = df_clean['DOLocationID'].astype(str)

# Create dictionary
dicts = df_clean[['PULocationID', 'DOLocationID']].to_dict(orient='records')

# Fit DictVectorizer
dv = DictVectorizer()
X_train = dv.fit_transform(dicts)
print(f"Feature matrix shape: {X_train.shape}")
```

**Answer: 515 columns**

---

### **Q5. Training a Model**

Use linear regression to train on January data.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

y_train = df_clean['duration']
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
rmse = mean_squared_error(y_train, y_pred, squared=False)
print(f"Train RMSE: {rmse:.2f}")
```
 **Answer: 7.64**

---

### **Q6. Evaluating the Model**

Now process February data similarly, then evaluate the model.

```python
df_val = pd.read_parquet('yellow_tripdata_2023-02.parquet')
df_val['duration'] = (df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']).dt.total_seconds() / 60
df_val = df_val[(df_val['duration'] >= 1) & (df_val['duration'] <= 60)]
df_val['PULocationID'] = df_val['PULocationID'].astype(str)
df_val['DOLocationID'] = df_val['DOLocationID'].astype(str)

dicts_val = df_val[['PULocationID', 'DOLocationID']].to_dict(orient='records')
X_val = dv.transform(dicts_val)
y_val = df_val['duration']
y_pred_val = model.predict(X_val)

val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
print(f"Validation RMSE: {val_rmse:.2f}")
```

 **Answer: 7.81**

---

###  Final Answers Summary:

1. **Q1**: 19
2. **Q2**: 52.59
3. **Q3**: 95%
4. **Q4**: 515
5. **Q5**: 7.64
6. **Q6**: 7.81

