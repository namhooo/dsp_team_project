"""
데이터 전처리 및 Feature Engineering 함수
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder


def merge_kobis_naver_data(kobis_df, naver_df):
    """
    KOBIS와 Naver 검색 트렌드 데이터 병합

    Parameters:
    -----------
    kobis_df : pd.DataFrame
        KOBIS 영화 데이터
    naver_df : pd.DataFrame
        Naver 검색 트렌드 데이터

    Returns:
    --------
    pd.DataFrame : 병합된 데이터프레임
    """
    merged_df = pd.merge(
        kobis_df,
        naver_df,
        on=['movieNm', 'openDt'],
        how='left'
    )

    return merged_df


def calculate_ticket_power(df, power_type='director'):
    """
    감독 또는 배우의 Ticket Power 계산

    Parameters:
    -----------
    df : pd.DataFrame
        영화 데이터프레임
    power_type : str
        'director' 또는 'actor'

    Returns:
    --------
    pd.DataFrame : Ticket Power가 추가된 데이터프레임
    """
    column = 'directors' if power_type == 'director' else 'actors'
    power_dict = {}

    # 각 영화별로 해당 인물의 과거 평균 관객수 계산
    for idx, row in df.iterrows():
        person = row[column]

        if pd.isna(person) or person == '':
            power_dict[idx] = df['audiAcc'].median()
            continue

        # 해당 인물의 과거 영화들 (현재 영화 제외)
        past_movies = df[
            (df[column].str.contains(str(person).split(',')[0], na=False)) &
            (df.index != idx) &
            (df['openDt'] < row['openDt'])
        ]

        if len(past_movies) > 0:
            power_dict[idx] = past_movies['audiAcc'].mean()
        else:
            # 신인인 경우 전체 중앙값 사용
            power_dict[idx] = df['audiAcc'].median()

    return power_dict


def extract_search_features(search_df, movie_info_df):
    """
    검색 트렌드 데이터에서 파생 변수 추출

    Parameters:
    -----------
    search_df : pd.DataFrame
        검색 트렌드 데이터 (movieNm, date, ratio 컬럼 필요)
    movie_info_df : pd.DataFrame
        영화 정보 데이터 (movieNm, openDt 컬럼 필요)

    Returns:
    --------
    pd.DataFrame : 검색 트렌드 파생 변수
    """
    search_features = []

    for _, movie in movie_info_df.iterrows():
        movie_name = movie['movieNm']

        # 날짜 파싱 시도, 실패 시 건너뛰기
        try:
            open_date = pd.to_datetime(movie['openDt'], errors='coerce')
            if pd.isna(open_date):
                # 유효하지 않은 날짜인 경우 건너뛰기
                search_features.append({
                    'movieNm': movie_name,
                    'search_4w_before': 0,
                    'search_2w_before': 0,
                    'search_1w_before': 0,
                    'search_opening': 0,
                    'search_1w_after': 0,
                    'max_search': 0,
                    'avg_search': 0,
                    'search_volatility': 0,
                    'search_growth_rate': 0
                })
                continue
        except Exception:
            # 파싱 실패 시 건너뛰기
            search_features.append({
                'movieNm': movie_name,
                'search_4w_before': 0,
                'search_2w_before': 0,
                'search_1w_before': 0,
                'search_opening': 0,
                'search_1w_after': 0,
                'max_search': 0,
                'avg_search': 0,
                'search_volatility': 0,
                'search_growth_rate': 0
            })
            continue

        # 해당 영화의 검색 트렌드 데이터
        movie_search = search_df[search_df['movieNm'] == movie_name].copy()

        if len(movie_search) == 0:
            # 검색 데이터가 없는 경우
            search_features.append({
                'movieNm': movie_name,
                'search_4w_before': 0,
                'search_2w_before': 0,
                'search_1w_before': 0,
                'search_opening': 0,
                'search_1w_after': 0,
                'max_search': 0,
                'avg_search': 0,
                'search_volatility': 0,
                'search_growth_rate': 0
            })
            continue

        movie_search['date'] = pd.to_datetime(movie_search['date'])

        # 시점별 검색량 추출
        def get_search_volume(target_date):
            """특정 날짜의 검색량 추출"""
            nearby = movie_search[
                (movie_search['date'] >= target_date - timedelta(days=3)) &
                (movie_search['date'] <= target_date + timedelta(days=3))
            ]
            return nearby['ratio'].mean() if len(nearby) > 0 else 0

        search_4w = get_search_volume(open_date - timedelta(weeks=4))
        search_2w = get_search_volume(open_date - timedelta(weeks=2))
        search_1w = get_search_volume(open_date - timedelta(weeks=1))
        search_opening = get_search_volume(open_date)
        search_1w_after = get_search_volume(open_date + timedelta(weeks=1))

        # 검색량 증가율
        growth_rate = 0
        if search_4w > 0:
            growth_rate = (search_1w - search_4w) / search_4w

        search_features.append({
            'movieNm': movie_name,
            'search_4w_before': search_4w,
            'search_2w_before': search_2w,
            'search_1w_before': search_1w,
            'search_opening': search_opening,
            'search_1w_after': search_1w_after,
            'max_search': movie_search['ratio'].max(),
            'avg_search': movie_search['ratio'].mean(),
            'search_volatility': movie_search['ratio'].std(),
            'search_growth_rate': growth_rate
        })

    return pd.DataFrame(search_features)


def encode_genres(df, genre_column='genres'):
    """
    장르 One-Hot Encoding

    Parameters:
    -----------
    df : pd.DataFrame
        영화 데이터프레임
    genre_column : str
        장르 컬럼명

    Returns:
    --------
    pd.DataFrame : One-Hot Encoding된 장르 컬럼이 추가된 데이터프레임
    """
    # 장르 분리 (콤마로 구분된 경우)
    genre_lists = df[genre_column].fillna('Unknown').str.split(',')

    # 모든 고유 장르 추출
    all_genres = set()
    for genres in genre_lists:
        all_genres.update([g.strip() for g in genres])

    # One-Hot Encoding
    for genre in all_genres:
        df[f'genre_{genre}'] = genre_lists.apply(
            lambda x: 1 if genre in [g.strip() for g in x] else 0
        )

    return df


def encode_categorical_features(df, method='frequency', columns=None):
    """
    범주형 변수 인코딩

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    method : str
        인코딩 방법 ('frequency', 'target', 'label')
    columns : list
        인코딩할 컬럼 리스트

    Returns:
    --------
    pd.DataFrame : 인코딩된 데이터프레임
    """
    if columns is None:
        columns = ['distributors', 'watchGradeNm']

    df_encoded = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'frequency':
            # Frequency Encoding
            freq = df[col].value_counts()
            df_encoded[f'{col}_freq'] = df[col].map(freq)

        elif method == 'target':
            # Target Encoding (평균 관객수)
            if 'audiAcc' in df.columns:
                target_mean = df.groupby(col)['audiAcc'].mean()
                df_encoded[f'{col}_target'] = df[col].map(target_mean)

        elif method == 'label':
            # Label Encoding
            le = LabelEncoder()
            df_encoded[f'{col}_label'] = le.fit_transform(df[col].fillna('Unknown'))

    return df_encoded


def extract_time_features(df, date_column='openDt'):
    """
    날짜 정보에서 시간 기반 변수 추출

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    date_column : str
        날짜 컬럼명

    Returns:
    --------
    pd.DataFrame : 시간 변수가 추가된 데이터프레임
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # 월, 분기
    df['release_month'] = df[date_column].dt.month
    df['release_quarter'] = df[date_column].dt.quarter
    df['release_year'] = df[date_column].dt.year

    # 요일 (0=월요일, 6=일요일)
    df['release_dayofweek'] = df[date_column].dt.dayofweek

    # 주말 개봉 여부
    df['is_weekend'] = df['release_dayofweek'].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)

    # 시즌 (성수기 구분)
    def get_season(month):
        if month in [7, 8]:
            return 'summer_peak'
        elif month in [12, 1, 2]:
            return 'winter_peak'
        elif month in [4, 5, 6]:
            return 'spring'
        else:
            return 'fall'

    df['season'] = df['release_month'].apply(get_season)

    # 시즌 One-Hot Encoding
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    return df


def handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode'):
    """
    결측치 처리

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    numeric_strategy : str
        수치형 변수 처리 전략 ('mean', 'median', 'zero')
    categorical_strategy : str
        범주형 변수 처리 전략 ('mode', 'unknown')

    Returns:
    --------
    pd.DataFrame : 결측치가 처리된 데이터프레임
    """
    df_filled = df.copy()

    # 수치형 변수
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_filled[col].isna().sum() > 0:
            if numeric_strategy == 'mean':
                df_filled[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df_filled[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == 'zero':
                df_filled[col].fillna(0, inplace=True)

    # 범주형 변수
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_filled[col].isna().sum() > 0:
            if categorical_strategy == 'mode':
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df_filled[col].fillna(mode_value, inplace=True)
            elif categorical_strategy == 'unknown':
                df_filled[col].fillna('Unknown', inplace=True)

    return df_filled


def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    이상치 제거

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    columns : list
        이상치를 확인할 컬럼 리스트
    method : str
        이상치 탐지 방법 ('iqr', 'zscore')
    threshold : float
        임계값 (IQR: 1.5, Z-score: 3)

    Returns:
    --------
    pd.DataFrame : 이상치가 제거된 데이터프레임
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_clean = df_clean[
                (df_clean[col] >= lower_bound) &
                (df_clean[col] <= upper_bound)
            ]

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df_clean = df_clean[z_scores < threshold]

    return df_clean


def scale_features(df, columns, scaler_type='standard'):
    """
    피처 스케일링

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    columns : list
        스케일링할 컬럼 리스트
    scaler_type : str
        스케일러 타입 ('standard', 'minmax', 'robust')

    Returns:
    --------
    tuple : (스케일링된 데이터프레임, 스케일러 객체)
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    df_scaled = df.copy()

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # 존재하는 컬럼만 필터링
    existing_columns = [col for col in columns if col in df.columns]

    if existing_columns:
        df_scaled[existing_columns] = scaler.fit_transform(df[existing_columns])

    return df_scaled, scaler


def create_feature_dataset(kobis_df, movie_details_df, search_trends_df):
    """
    전체 Feature 데이터셋 생성

    Parameters:
    -----------
    kobis_df : pd.DataFrame
        KOBIS 박스오피스 데이터
    movie_details_df : pd.DataFrame
        영화 상세 정보
    search_trends_df : pd.DataFrame
        검색 트렌드 데이터

    Returns:
    --------
    pd.DataFrame : 통합 Feature 데이터셋
    """
    # 1. 영화별 최종 관객수 계산
    final_audience = kobis_df.groupby('movieNm').agg({
        'audiAcc': 'max',
        'openDt': 'first',
        'movieCd': 'first'
    }).reset_index()

    # 2. 영화 상세 정보 병합
    df = pd.merge(final_audience, movie_details_df, on=['movieCd', 'movieNm'], how='left')

    # 3. 검색 트렌드 특성 추출 및 병합
    search_features = extract_search_features(search_trends_df, df)
    df = pd.merge(df, search_features, on='movieNm', how='left')

    # 4. Ticket Power 계산
    director_power = calculate_ticket_power(df, 'director')
    actor_power = calculate_ticket_power(df, 'actor')

    df['director_power'] = df.index.map(director_power)
    df['actor_power'] = df.index.map(actor_power)
    df['ticket_power'] = 0.4 * df['director_power'] + 0.6 * df['actor_power']

    # 5. 장르 인코딩
    df = encode_genres(df)

    # 6. 시간 기반 변수 추출
    df = extract_time_features(df)

    # 7. 기타 범주형 변수 인코딩
    df = encode_categorical_features(df, method='frequency')

    # 8. 결측치 처리
    df = handle_missing_values(df)

    return df


if __name__ == "__main__":
    print("Preprocessing and Feature Engineering Module")
    print("Usage:")
    print("  from utils.preprocessing import create_feature_dataset")
