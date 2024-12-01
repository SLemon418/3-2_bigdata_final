import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 모델 및 레이블 인코더 로드
model = joblib.load('flight_delay_predictor.pkl')

# 레이블 인코더 로드
label_encoders = {
    'Airline': joblib.load('airline_label_encoder.pkl'),
    'Origin': joblib.load('origin_label_encoder.pkl'),
    'Dest': joblib.load('dest_label_encoder.pkl')
}

# 항공사 코드 매핑 (예시)
airline_codes = {
    "Commutair Aka Champlain Enterprises, Inc.": {"IATA": "C5", "ICAO": "UCA"},
    "GoJet Airlines, LLC d/b/a United Express": {"IATA": "G7", "ICAO": "GJS"},
    "Air Wisconsin Airlines Corp": {"IATA": "ZW", "ICAO": "AWI"},
    "Mesa Airlines Inc.": {"IATA": "YV", "ICAO": "ASH"},
    "Southwest Airlines Co.": {"IATA": "WN", "ICAO": "SWA"},
    "Republic Airlines": {"IATA": "YX", "ICAO": "RPA"},
    "Endeavor Air Inc.": {"IATA": "9E", "ICAO": "EDV"},
    "American Airlines Inc.": {"IATA": "AA", "ICAO": "AAL"},
    "Capital Cargo International": {"IATA": "PT", "ICAO": "CCI"},
    "SkyWest Airlines Inc.": {"IATA": "OO", "ICAO": "SKW"},
    "Alaska Airlines Inc.": {"IATA": "AS", "ICAO": "ASA"},
    "JetBlue Airways": {"IATA": "B6", "ICAO": "JBU"},
    "Delta Air Lines Inc.": {"IATA": "DL", "ICAO": "DAL"},
    "Frontier Airlines Inc.": {"IATA": "F9", "ICAO": "FFT"},
    "Allegiant Air": {"IATA": "G4", "ICAO": "AAY"},
    "Hawaiian Airlines Inc.": {"IATA": "HA", "ICAO": "HAL"},
    "Envoy Air": {"IATA": "MQ", "ICAO": "ENY"},
    "Spirit Air Lines": {"IATA": "NK", "ICAO": "NKS"},
    "Comair Inc.": {"IATA": "OH", "ICAO": "COM"},
    "Horizon Air": {"IATA": "QX", "ICAO": "QXE"},
    "United Air Lines Inc.": {"IATA": "UA", "ICAO": "UAL"},
}

# IATA 및 ICAO 코드로부터 항공사 이름을 찾는 함수
def find_airline_by_code(code):
    code = code.upper()
    for airline, codes in airline_codes.items():
        if code in codes.values():
            return airline
    return None

# 사용자 입력 받기
def get_user_input():
    while True:
        airline_input = input("항공사 이름 또는 IATA/ICAO 코드를 입력하세요: ").strip()
        airline = find_airline_by_code(airline_input) or airline_input.title()
        
        if airline not in label_encoders['Airline'].classes_:
            print("입력한 항공사가 데이터에 없습니다. 다시 시도하세요.")
            continue
        
        origin = input("출발 공항 코드를 입력하세요: ").strip().upper()
        if origin not in label_encoders['Origin'].classes_:
            print("입력한 출발 공항 코드가 데이터에 없습니다. 다시 시도하세요.")
            continue
        
        dest = input("도착 공항 코드를 입력하세요: ").strip().upper()
        if dest not in label_encoders['Dest'].classes_:
            print("입력한 도착 공항 코드가 데이터에 없습니다. 다시 시도하세요.")
            continue
        
        try:
            crs_dep_time = int(input("예정 출발 시간을 입력하세요 (HHMM 형식): ").strip())
            crs_arr_time = int(input("예정 도착 시간을 입력하세요 (HHMM 형식): ").strip())
            distance = int(input("비행 거리를 입력하세요 (마일 단위): ").strip())
            year = int(input("연도를 입력하세요: ").strip())
            month = int(input("월을 입력하세요: ").strip())
            day_of_month = int(input("일을 입력하세요: ").strip())
            day_of_week = int(input("요일을 입력하세요 (0=월요일, 6=일요일): ").strip())
        except ValueError:
            print("숫자를 입력해야 합니다. 다시 시도하세요.")
            continue
        
        return {
            'Airline': airline,
            'Origin': origin,
            'Dest': dest,
            'CRSDepTime': crs_dep_time,
            'CRSArrTime': crs_arr_time,
            'Distance': distance,
            'Year': year,
            'Month': month,
            'DayofMonth': day_of_month,
            'DayOfWeek': day_of_week
        }

# 입력 데이터 처리 및 예측
def predict_delay(input_data):
    # 입력 데이터 변환
    input_data['Airline'] = label_encoders['Airline'].transform([input_data['Airline']])[0]
    input_data['Origin'] = label_encoders['Origin'].transform([input_data['Origin']])[0]
    input_data['Dest'] = label_encoders['Dest'].transform([input_data['Dest']])[0]
    input_data['CRSDepTime'] = convert_to_minutes(input_data['CRSDepTime'])
    input_data['CRSArrTime'] = convert_to_minutes(input_data['CRSArrTime'])
    
    # 데이터프레임 생성
    input_df = pd.DataFrame([input_data])
    
    # 예측 수행
    predicted_delay = model.predict(input_df)
    return predicted_delay[0]

# 시간 변환 함수
def convert_to_minutes(t):
    t = int(t)
    return (t // 100) * 60 + (t % 100)

# 메인 함수
if __name__ == "__main__":
    user_input = get_user_input()
    predicted_delay = predict_delay(user_input)
    print(f'예상 도착 지연 시간: {predicted_delay:.2f}분')