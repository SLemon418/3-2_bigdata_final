import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 데이터 로드
data_path = './flight2022.csv'
df = pd.read_csv(data_path)

# 범주형 변수 목록
categorical_cols = ['Airline', 'Origin', 'Dest']

# 레이블 인코더 생성 및 저장
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    # 레이블 인코더 저장
    joblib.dump(le, f'{col.lower()}_label_encoder.pkl')

print("레이블 인코더 저장 완료.") 