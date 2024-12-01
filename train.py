import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.font_manager as fm

# 데이터 로드
print("데이터 로드 중...")
data_path = './flight2022.csv'
df = pd.read_csv(data_path)
print("데이터 로드 완료.")

# 데이터 확인
print("데이터 확인 중...")
print(df.head())
print(df.info())

# 결측치 처리: 'DepDelayMinutes'와 'ArrDelayMinutes'의 결측치 제거
print("결측치 처리 중...")
df.dropna(subset=['DepDelayMinutes', 'ArrDelayMinutes'], inplace=True)
print("결측치 처리 완료.")

# 시간 변환 함수 정의
def convert_to_minutes(t):
    t = int(t)
    return (t // 100) * 60 + (t % 100)

# 시간 데이터 변환
print("시간 데이터 변환 중...")
df['CRSDepTime'] = df['CRSDepTime'].apply(convert_to_minutes)
df['DepTime'] = df['DepTime'].apply(lambda x: convert_to_minutes(x) if not np.isnan(x) else x)
df['CRSArrTime'] = df['CRSArrTime'].apply(convert_to_minutes)
df['ArrTime'] = df['ArrTime'].apply(lambda x: convert_to_minutes(x) if not np.isnan(x) else x)
print("시간 데이터 변환 완료.")

# 날짜 형식 변환 및 파생 변수 생성
print("날짜 형식 변환 및 파생 변수 생성 중...")
df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['Year'] = df['FlightDate'].dt.year
df['Month'] = df['FlightDate'].dt.month
df['DayofMonth'] = df['FlightDate'].dt.day
df['DayOfWeek'] = df['FlightDate'].dt.weekday
print("날짜 형식 변환 및 파생 변수 생성 완료.")

# 불필요한 열 제거
print("불필요한 열 제거 중...")
df.drop(columns=['Cancelled', 'Diverted'], inplace=True)
print("불필요한 열 제거 완료.")

# 범주형 변수 목록
categorical_cols = ['Airline', 'Origin', 'Dest', 'Tail_Number']

# 범주형 변수 레이블 인코딩
print("범주형 변수 레이블 인코딩 중...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("범주형 변수 레이블 인코딩 완료.")

# # 데이터 샘플링
# print("데이터 샘플링 중...")
# df_sampled = df.sample(frac=0.1, random_state=42)  # 데이터의 10%만 사용
# print("데이터 샘플링 완료.")
df_sampled = df

# 특징 변수와 타겟 변수 설정
X = df_sampled[['Airline', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime', 'Distance', 'Year', 'Month', 'DayofMonth', 'DayOfWeek']]
y = df_sampled['ArrDelayMinutes']
# 학습용과 테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 모델 초기화 및 학습
print("모델 학습 중...")
model = RandomForestRegressor(n_estimators=30, random_state=42, verbose=1)
model.fit(X_train, y_train)
print("모델 학습 완료.")
# 예측 수행
print("예측 수행 중...")
y_pred = model.predict(X_test)
print("예측 수행 완료.")

# 평균 절대 오차 계산
mae = mean_absolute_error(y_test, y_pred)
print(f'평균 절대 오차(MAE): {mae:.2f}분')
# 모델 저장
print("모델 저장 중...")
joblib.dump(model, 'flight_delay_predictor.pkl')
print("모델 저장 완료.")
# 모델 불러오기
loaded_model = joblib.load('flight_delay_predictor.pkl')

# 새로운 데이터 예시
new_data = pd.DataFrame({
    'Airline': [label_encoders['Airline'].transform(['American Airlines Inc.'])[0]],
    'Origin': [label_encoders['Origin'].transform(['GJT'])[0]],
    'Dest': [label_encoders['Dest'].transform(['DEN'])[0]],
    'CRSDepTime': [convert_to_minutes(900)],  # 09:00 AM
    'CRSArrTime': [convert_to_minutes(1200)],  # 12:00 PM
    'Distance': [4301],  # 예시 거리 (마일 단위)
    'Year': [2024],
    'Month': [12],
    'DayofMonth': [1],
    'DayOfWeek': [0]  # 월요일
})

# 지연 시간 예측
predicted_delay = loaded_model.predict(new_data)
print(f'예상 도착 지연 시간: {predicted_delay[0]:.2f}분')

# 폰트 경로 설정
font_path = './NanumBarunGothic.ttf'

# FontEntry를 사용하여 폰트 등록
fe = fm.FontEntry(
    fname=font_path,
    name='NanumGothic'
)
fm.fontManager.ttflist.insert(0, fe)

# 폰트 설정
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})

# 지연 시간 분포 시각화
print("지연 시간 분포 시각화 중...")
plt.figure(figsize=(10, 6))
sns.histplot(df['ArrDelayMinutes'], bins=50, kde=True)
plt.title('도착 지연 시간 분포')
plt.xlabel('지연 시간 (분)')
plt.ylabel('빈도')
plt.savefig('arrival_delay_distribution.png')  # 이미지 저장
plt.close()

# 항공사별 평균 지연 시간 시각화
print("항공사별 평균 지연 시간 시각화 중...")
plt.figure(figsize=(12, 6))
airline_delay = df.groupby('Airline')['ArrDelayMinutes'].mean().reset_index()
airline_delay['Airline'] = airline_delay['Airline'].apply(lambda x: label_encoders['Airline'].inverse_transform([x])[0])
sns.barplot(data=airline_delay, x='Airline', y='ArrDelayMinutes')
plt.title('항공사별 평균 도착 지연 시간')
plt.xlabel('항공사')
plt.ylabel('평균 지연 시간 (분)')
plt.xticks(rotation=90)
plt.savefig('airline_average_delay.png')  # 이미지 저장
plt.close()

# 출발 공항별 평균 지연 시간 계산
origin_delay = df.groupby('Origin')['DepDelayMinutes'].mean().reset_index()

# 인코딩된 공항 코드를 원래 값으로 변환
origin_delay['Origin'] = origin_delay['Origin'].apply(lambda x: label_encoders['Origin'].inverse_transform([x])[0])

# 출발 공항별 평균 지연 시간 시각화
print("출발 공항별 평균 지연 시간 시각화 중...")
plt.figure(figsize=(15, 8))
origin_delay = df.groupby('Origin')['DepDelayMinutes'].mean().reset_index()
origin_delay['Origin'] = origin_delay['Origin'].apply(lambda x: label_encoders['Origin'].inverse_transform([x])[0])
sns.barplot(data=origin_delay, x='Origin', y='DepDelayMinutes', palette='viridis')
plt.title('출발 공항별 평균 출발 지연 시간')
plt.xlabel('출발 공항 코드')
plt.ylabel('평균 출발 지연 시간 (분)')
plt.xticks(rotation=90)
plt.savefig('origin_average_delay.png')  # 이미지 저장
plt.close()