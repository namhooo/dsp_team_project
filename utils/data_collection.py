"""
데이터 수집 관련 함수
KOBIS API와 Naver 검색 트렌드 데이터 수집
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time


class KOBISCollector:
    """KOBIS API 데이터 수집기"""

    def __init__(self, api_key):
        """
        Parameters:
        -----------
        api_key : str
            KOBIS API 키
        """
        self.api_key = api_key
        self.base_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest"

    def get_daily_boxoffice(self, target_date):
        """
        일별 박스오피스 조회

        Parameters:
        -----------
        target_date : str
            조회 날짜 (YYYYMMDD 형식)

        Returns:
        --------
        dict : API 응답 데이터
        """
        url = f"{self.base_url}/boxoffice/searchDailyBoxOfficeList.json"
        params = {
            'key': self.api_key,
            'targetDt': target_date
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {target_date}: {e}")
            return None

    def get_movie_info(self, movie_code):
        """
        영화 상세 정보 조회

        Parameters:
        -----------
        movie_code : str
            영화 코드

        Returns:
        --------
        dict : 영화 상세 정보
        """
        url = f"{self.base_url}/movie/searchMovieInfo.json"
        params = {
            'key': self.api_key,
            'movieCd': movie_code
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie info for {movie_code}: {e}")
            return None

    def collect_boxoffice_data(self, start_date, end_date, delay=0.1):
        """
        기간 내 박스오피스 데이터 수집

        Parameters:
        -----------
        start_date : datetime
            시작 날짜
        end_date : datetime
            종료 날짜
        delay : float
            API 호출 간 대기 시간 (초)

        Returns:
        --------
        list : 영화 데이터 리스트
        """
        movies_data = []
        current_date = start_date

        print(f"Collecting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        date_range = (end_date - start_date).days + 1

        with tqdm(total=date_range) as pbar:
            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                data = self.get_daily_boxoffice(date_str)

                if data and 'boxOfficeResult' in data:
                    daily_list = data['boxOfficeResult'].get('dailyBoxOfficeList', [])
                    for movie in daily_list:
                        movies_data.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'rank': movie.get('rank'),
                            'movieCd': movie.get('movieCd'),
                            'movieNm': movie.get('movieNm'),
                            'openDt': movie.get('openDt'),
                            'salesAmt': int(movie.get('salesAmt', 0)),
                            'audiCnt': int(movie.get('audiCnt', 0)),
                            'audiAcc': int(movie.get('audiAcc', 0)),
                            'scrnCnt': int(movie.get('scrnCnt', 0)),
                            'showCnt': int(movie.get('showCnt', 0))
                        })

                current_date += timedelta(days=1)
                pbar.update(1)
                time.sleep(delay)

        return movies_data

    def collect_movie_details(self, movie_codes, delay=0.1):
        """
        영화 상세 정보 수집

        Parameters:
        -----------
        movie_codes : list
            영화 코드 리스트
        delay : float
            API 호출 간 대기 시간 (초)

        Returns:
        --------
        list : 영화 상세 정보 리스트
        """
        movies_info = []

        print(f"Collecting details for {len(movie_codes)} movies")

        for movie_code in tqdm(movie_codes):
            data = self.get_movie_info(movie_code)

            if data and 'movieInfoResult' in data:
                movie_info = data['movieInfoResult'].get('movieInfo', {})

                # 감독 정보 추출
                directors = movie_info.get('directors', [])
                director_names = [d.get('peopleNm') for d in directors]

                # 배우 정보 추출
                actors = movie_info.get('actors', [])
                actor_names = [a.get('peopleNm') for a in actors[:5]]  # 상위 5명

                # 장르 정보 추출
                genres = movie_info.get('genres', [])
                genre_names = [g.get('genreNm') for g in genres]

                # 배급사 정보 추출
                companies = movie_info.get('companys', [])
                distributors = [c.get('companyNm') for c in companies
                              if c.get('companyPartNm') == '배급사']

                movies_info.append({
                    'movieCd': movie_info.get('movieCd'),
                    'movieNm': movie_info.get('movieNm'),
                    'movieNmEn': movie_info.get('movieNmEn'),
                    'prdtYear': movie_info.get('prdtYear'),
                    'openDt': movie_info.get('openDt'),
                    'typeNm': movie_info.get('typeNm'),
                    'nations': ','.join([n.get('nationNm') for n in movie_info.get('nations', [])]),
                    'genres': ','.join(genre_names),
                    'directors': ','.join(director_names),
                    'actors': ','.join(actor_names),
                    'watchGradeNm': movie_info.get('audits', [{}])[0].get('watchGradeNm') if movie_info.get('audits') else None,
                    'distributors': ','.join(distributors)
                })

            time.sleep(delay)

        return movies_info


def parse_boxoffice_data(raw_data):
    """
    원본 API 데이터를 파싱하여 DataFrame으로 변환

    Parameters:
    -----------
    raw_data : list
        수집된 박스오피스 데이터

    Returns:
    --------
    pd.DataFrame : 파싱된 데이터프레임
    """
    return pd.DataFrame(raw_data)


def save_data(df, filepath):
    """
    데이터프레임을 CSV 파일로 저장

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 데이터프레임
    filepath : str
        저장 경로
    """
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Data saved to {filepath}")


def load_data(filepath):
    """
    CSV 파일에서 데이터 로드

    Parameters:
    -----------
    filepath : str
        파일 경로

    Returns:
    --------
    pd.DataFrame : 로드된 데이터프레임
    """
    return pd.read_csv(filepath, encoding='utf-8-sig')


class NaverTrendCollector:
    """
    Naver 검색 트렌드 수집기

    Note:
    -----
    실제 구현을 위해서는 Naver DataLab API 키가 필요합니다.
    또는 수동으로 데이터를 수집하여 CSV로 저장할 수 있습니다.
    """

    def __init__(self, client_id=None, client_secret=None):
        """
        Parameters:
        -----------
        client_id : str
            Naver API Client ID
        client_secret : str
            Naver API Client Secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/datalab/search"

    def clean_keyword(self, keyword):
        """
        키워드 정리 (Naver API 호환성을 위해)

        Parameters:
        -----------
        keyword : str
            원본 키워드

        Returns:
        --------
        str : 정리된 키워드
        """
        # 특수문자를 공백으로 대체
        import re
        cleaned = re.sub(r'[:\-\–\—]', ' ', keyword)
        # 연속된 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # 양쪽 공백 제거
        cleaned = cleaned.strip()
        # 길이 제한 (50자)
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        return cleaned

    def get_search_trend(self, keyword, start_date, end_date, time_unit='date'):
        """
        검색 트렌드 조회

        Parameters:
        -----------
        keyword : str
            검색 키워드
        start_date : str
            시작 날짜 (YYYY-MM-DD)
        end_date : str
            종료 날짜 (YYYY-MM-DD)
        time_unit : str
            시간 단위 ('date', 'week', 'month')

        Returns:
        --------
        dict : 검색 트렌드 데이터
        """
        if not self.client_id or not self.client_secret:
            print("Naver API credentials not provided")
            return None

        # 키워드 정리
        cleaned_keyword = self.clean_keyword(keyword)

        headers = {
            'X-Naver-Client-Id': self.client_id,
            'X-Naver-Client-Secret': self.client_secret,
            'Content-Type': 'application/json'
        }

        body = {
            'startDate': start_date.replace('-', ''),
            'endDate': end_date.replace('-', ''),
            'timeUnit': time_unit,
            'keywordGroups': [
                {
                    'groupName': cleaned_keyword,
                    'keywords': [cleaned_keyword]
                }
            ]
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 더 자세한 에러 정보
            if response.status_code == 400:
                try:
                    error_detail = response.json()
                    print(f"Error for '{keyword}' (cleaned: '{cleaned_keyword}'): {error_detail}")
                except:
                    print(f"Error for '{keyword}': 400 Bad Request - Check keyword format and date range")
            else:
                print(f"Error fetching search trend for {keyword}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search trend for {keyword}: {e}")
            return None

    def collect_trends_for_movies(self, movies_df, weeks_before=4, weeks_after=1, time_unit='week'):
        """
        영화별 검색 트렌드 수집

        Parameters:
        -----------
        movies_df : pd.DataFrame
            영화 정보가 담긴 데이터프레임 (movieNm, openDt 컬럼 필요)
        weeks_before : int
            개봉 전 수집 주 수
        weeks_after : int
            개봉 후 수집 주 수
        time_unit : str
            시간 단위 ('date', 'week', 'month') - 기본값 'week' 권장

        Returns:
        --------
        list : 검색 트렌드 데이터 리스트
        """
        trends_data = []
        success_count = 0
        fail_count = 0

        for _, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
            movie_name = row['movieNm']

            # openDt 유효성 확인
            try:
                open_date = pd.to_datetime(row['openDt'])
                if pd.isna(open_date):
                    print(f"⚠️ '{movie_name}': 개봉일 누락, 스킵")
                    fail_count += 1
                    continue
            except:
                print(f"⚠️ '{movie_name}': 개봉일 형식 오류, 스킵")
                fail_count += 1
                continue

            # 수집 기간 설정
            start_date = (open_date - timedelta(weeks=weeks_before)).strftime('%Y-%m-%d')
            end_date = (open_date + timedelta(weeks=weeks_after)).strftime('%Y-%m-%d')

            # 검색 트렌드 조회 (time_unit 파라미터 추가)
            trend_data = self.get_search_trend(movie_name, start_date, end_date, time_unit=time_unit)

            if trend_data and 'results' in trend_data:
                for item in trend_data['results'][0]['data']:
                    trends_data.append({
                        'movieNm': movie_name,
                        'openDt': row['openDt'],
                        'date': item['period'],
                        'ratio': item['ratio']
                    })
                success_count += 1
            else:
                fail_count += 1

            time.sleep(0.1)  # API rate limit 고려

        print(f"\n수집 완료: 성공 {success_count}, 실패 {fail_count}")
        return trends_data


if __name__ == "__main__":
    # 사용 예시
    print("KOBIS Data Collector Module")
    print("Usage:")
    print("  from utils.data_collection import KOBISCollector, NaverTrendCollector")
