# 🎬 영화 흥행 예측 프로젝트

**KOBIS API와 검색 트렌드 데이터를 활용한 영화 박스오피스 예측**

---

## 📋 프로젝트 개요

### 🎯 목표
영화의 최종 총 관객수를 예측하는 회귀 모델 구축

### 🔑 핵심 목표
1. **데이터셋 구축**: KOBIS API + Naver 검색 트렌드 데이터 결합
2. **핵심 요인 발견**: 흥행에 영향을 미치는 주요 변수 식별
3. **실무적 가치**: 영화 마케터와 투자자를 위한 의사결정 도구

### 📊 데이터
- **KOBIS 데이터**: 영화 제목, 개봉일, 장르, 감독, 주연배우, 배급사, 상영등급, 총 관객수
- **검색 트렌드**: 개봉 4주 전 ~ 1주 후 검색량 (주간/일간)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 압축 해제)
cd movie_box_office_prediction

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. API 키 발급

#### KOBIS API
1. [영화진흥위원회 오픈API](https://www.kobis.or.kr/kobisopenapi/homepg/main/main.do) 접속
2. 회원가입 및 로그인
3. API 키 발급

#### Naver DataLab (선택사항)
1. [Naver Developers](https://developers.naver.com/) 접속
2. 애플리케이션 등록
3. Client ID 및 Client Secret 발급

### 3. 노트북 실행

Jupyter Notebook을 실행하고 순서대로 진행하세요:

```bash
jupyter notebook
```

#### 노트북 실행 순서
1. `01_data_collection.ipynb` - 데이터 수집
2. `02_eda.ipynb` - 탐색적 데이터 분석
3. `03_preprocessing_feature_engineering.ipynb` - 전처리 및 Feature Engineering
4. `04_baseline_model.ipynb` - Baseline 모델 구축
5. `05_advanced_models.ipynb` - 고급 모델 비교 및 튜닝
6. `06_evaluation_interpretation.ipynb` - 최종 모델 평가 및 해석
7. `07_final_report.ipynb` - 프로젝트 종합 및 리포트

---

## 📁 프로젝트 구조

```
movie_box_office_prediction/
│
├── README.md                          # 프로젝트 개요
├── requirements.txt                   # 필요 라이브러리
├── movie_prediction_project_plan.md   # 상세 프로젝트 계획서
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_evaluation_interpretation.ipynb
│   └── 07_final_report.ipynb
│
├── data/                              # 데이터 디렉토리
│   ├── raw/                           # 원본 데이터
│   │   ├── kobis_boxoffice.csv
│   │   ├── kobis_movie_details.csv
│   │   └── naver_search_trends.csv
│   ├── processed/                     # 전처리된 데이터
│   │   ├── movie_features.csv
│   │   └── feature_description.txt
│   └── final/                         # 최종 학습 데이터
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       └── y_test.csv
│
├── models/                            # 저장된 모델
│   ├── baseline_lr.pkl
│   ├── best_model.pkl
│   ├── rf_grid_search.pkl
│   ├── gb_grid_search.pkl
│   └── scaler.pkl
│
├── utils/                             # 유틸리티 함수
│   ├── __init__.py
│   ├── data_collection.py             # API 호출 함수
│   ├── preprocessing.py               # 전처리 함수
│   └── evaluation.py                  # 평가 함수
│
└── results/                           # 결과물
    ├── figures/                       # 그래프 (노트북에서 생성)
    ├── tables/                        # 성능 비교표
    │   ├── baseline_results.csv
    │   └── model_comparison.csv
    └── feature_importance.csv
```

---

## 🔧 주요 기능

### 1. 데이터 수집 (`utils/data_collection.py`)
```python
from utils.data_collection import KOBISCollector, NaverTrendCollector

# KOBIS 데이터 수집
collector = KOBISCollector(api_key="YOUR_API_KEY")
boxoffice_data = collector.collect_boxoffice_data(start_date, end_date)

# Naver 검색 트렌드 수집
naver_collector = NaverTrendCollector(client_id="ID", client_secret="SECRET")
search_trends = naver_collector.collect_trends_for_movies(movies_df)
```

### 2. Feature Engineering (`utils/preprocessing.py`)
```python
from utils.preprocessing import (
    calculate_ticket_power,
    extract_search_features,
    encode_genres,
    extract_time_features
)

# Ticket Power 계산
director_power = calculate_ticket_power(df, 'director')
actor_power = calculate_ticket_power(df, 'actor')

# 검색 트렌드 파생 변수
search_features = extract_search_features(search_df, movie_df)

# 장르 인코딩
df = encode_genres(df)
```

### 3. 모델 평가 (`utils/evaluation.py`)
```python
from utils.evaluation import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_feature_importance
)

# 모델 평가
metrics = evaluate_model(y_true, y_pred, model_name="Random Forest")

# 시각화
plot_actual_vs_predicted(y_true, y_pred)
plot_feature_importance(model, feature_names)
```

---

## 📊 주요 결과

### 모델 성능
- **최종 모델**: [선정된 모델명]
- **Test R²**: [값]
- **Test RMSE**: [값] 만명
- **Test MAE**: [값] 만명

### 핵심 발견사항
1. **검색 트렌드의 중요성**: 개봉 2주 전 검색량이 최종 관객수의 강력한 예측 변수
2. **Ticket Power 효과**: 감독/배우의 과거 성적도 유의미한 영향
3. **장르 및 시즌 효과**: 특정 장르와 성수기가 관객수에 영향

### Top 5 중요 Feature
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]
4. [Feature 4]
5. [Feature 5]

---

## 💡 비즈니스 활용

### 1. 마케팅 예산 최적화
- 개봉 2주 전 검색량 모니터링
- 검색량이 낮을 경우 온라인 마케팅 강화
- 예상 관객수 기반 예산 배분

### 2. 투자 의사결정 지원
- 기획 단계에서 흥행 가능성 사전 평가
- 감독/배우 캐스팅의 영향력 정량화
- ROI 시뮬레이션

### 3. 리스크 관리
- 흥행 실패 위험 조기 감지
- 시나리오별 수익 예측
- 개봉 시기 최적화

---

## 🛠️ 기술 스택

### 데이터 처리
- `pandas` 2.0.3
- `numpy` 1.24.3

### 시각화
- `matplotlib` 3.7.2
- `seaborn` 0.12.2
- `plotly` 5.15.0

### 머신러닝
- `scikit-learn` 1.3.0

### API 및 크롤링
- `requests` 2.31.0
- `selenium` 4.11.2 (선택사항)
- `beautifulsoup4` 4.12.2 (선택사항)

### 기타
- `jupyter` 1.0.0
- `joblib` 1.3.1

---

## 📝 사용 예시

### 새로운 영화 예측

```python
import joblib
import pandas as pd

# 모델 로드
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# 새로운 영화 정보 (Feature Engineering 필요)
new_movie = {
    'search_2w_before': 25000,
    'search_growth_rate': 2.0,
    'ticket_power': 800000,
    'genre_Action': 1,
    'release_month': 7,
    # ... 모든 Feature
}

# DataFrame 변환 및 스케일링
new_movie_df = pd.DataFrame([new_movie])
new_movie_scaled = scaler.transform(new_movie_df)

# 예측
predicted_audience = model.predict(new_movie_scaled)[0]
print(f"예상 총 관객수: {predicted_audience:,.0f} 명")
```

---

## ⚠️ 주의사항

### API 키 관리
- API 키는 절대 Git에 커밋하지 마세요
- `.env` 파일이나 별도의 config 파일로 관리하세요
- `.gitignore`에 API 키 파일 추가

### 데이터 수집
- KOBIS API는 요청 제한이 있을 수 있습니다
- 적절한 delay를 설정하여 서버에 부담을 주지 않도록 하세요
- Naver 검색 트렌드는 수동 수집이 필요할 수 있습니다

### 모델 사용
- 이 모델은 한국 영화 시장 데이터로 학습되었습니다
- 다른 시장(해외)에는 적용이 제한적일 수 있습니다
- 예측값은 참고용이며, 실제 결과는 다를 수 있습니다

---

## 🔍 한계점 및 개선 방향

### 한계점
1. **데이터 크기**: 제한된 영화 데이터
2. **외부 요인 미반영**: 경쟁작, 사회적 이슈 등
3. **검색 트렌드 수집의 어려움**: API 제약
4. **장기 예측의 한계**: 개봉 임박 시점에만 정확

### 개선 방향
1. **데이터 확장**: 5년 이상 장기 데이터 수집
2. **추가 데이터**: SNS, 리뷰 감성분석, YouTube 조회수
3. **고급 모델링**: LSTM, Ensemble
4. **실시간 시스템**: 웹 대시보드, API 서비스화

---

## 👥 기여

이 프로젝트는 Data Science Practice 팀 프로젝트입니다.

기여 방법:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

## 🙏 감사의 말

- **KOBIS**: 영화진흥위원회 오픈API 제공
- **Naver**: DataLab 검색 트렌드 데이터
- **Scikit-learn**: 머신러닝 라이브러리
- **Kaggle Community**: 영감과 참고 자료

---

## 📚 참고 자료

- [KOBIS 영화진흥위원회 Open API](https://www.kobis.or.kr/kobisopenapi/homepg/main/main.do)
- [Naver Developers - DataLab](https://developers.naver.com/docs/datalab/search/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Kaggle: TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction)

---

**Last Updated**: 2025-01-01
**Version**: 1.0.0

**Happy Predicting! 🎬📊**
