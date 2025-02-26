import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import time
print("Step 1: Import libraries")
h2o.init(verbose=True)
time.sleep(10)
print("Step 2: H2O initialized")
print('Everything must be ok')

# Tải dữ liệu mẫu
# data = pd.read_csv('/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test11.csv')
# dataset = h2o.H2OFrame(data)


# # Kiểm tra dữ liệu
# dataset.describe()

# # Xác định các cột
# dataset.columns = ['id', 'person_age', 'person_income', 'person_home_ownership',
#                    'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
#                    'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
#                    'cb_person_cred_hist_length', 'loan_status']

# # Chuyển cột target thành kiểu category
# dataset['loan_status'] = dataset['loan_status'].asfactor()

# # Chia tập dữ liệu thành train và test
# train, test = dataset.split_frame(ratios=[0.8], seed=42)

# # Xác định biến đầu vào và biến đầu ra
# x = ['person_age', 'person_income', 'person_home_ownership',
#      'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
#      'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
#      'cb_person_cred_hist_length']
# y = 'loan_status'

# # Tạo mô hình AutoML
# model = H2OAutoML(max_models=10, seed=42, max_runtime_secs=600)

# # Huấn luyện mô hình
# model.train(x=x, y=y, training_frame=train)
# print('Train Success')

# # Dự đoán trên tập test
# predictions = model.leader.predict(test)

# # Đánh giá mô hình
# performance = model.leader.model_performance(test)
# print(performance)

# # Xuất mô hình
# model_path = h2o.save_model(model=model.leader, path="./user_model_path", force=True)
# print(f'Model saved at: {model_path}')

# # Dừng H2O
h2o.shutdown(prompt=False)
