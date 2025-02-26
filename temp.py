import joblib
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
start = time.time()
model = joblib.load('/home/quoc/works/Learn/NCKH/model/tpot_predictor.pkl')
data = pd.read_csv('/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test.csv')
data = data.drop('loan_status', axis=1)
feature_df = data.copy()
categorical_columns = feature_df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    feature_df[col] = le.fit_transform(feature_df[col])
    label_encoders[col] = le
ans = model.predict(feature_df)
print(ans)
end = time.time()
print('--->')
print('Time:', end - start)