# Twitch Streamer Regression Analysis
<img src="https://user-images.githubusercontent.com/71831714/104867952-667e2e80-5985-11eb-8554-42e3952ae671.jpg"></img>
## 1. Intro

#### 1-1. Topic
- Twitch 스트리머의 팔로우 증가수를 회귀 모델로 분석 및 예측

#### 1-2. Contents
1. EDA
2. 데이터 전처리
3. 모델링
4. 성능 평가
5. 추가 데이터 예측

#### 1-3. Dataset
[kaggle] <https://www.kaggle.com/aayushmishra1512/twitchdata>
- twitchdata-update.csv

[twitchtracker] <https://twitchtracker.com/>
- BeautifulSoup을 활용한 크롤링으로 방송 시작일(Date) feature 추가

## 2. Process

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

```python
# 데이터 불러오기 
twitch_df = pd.read_csv("twitch.csv", parse_dates=['Date'])
twitch = twitch_df.copy()
```

#### 2-1. EDA
```python
plt.figure(figsize=(20,8))
sns.set_style("whitegrid")
sns.regplot(x='Peak viewers', y='Followers gained', data=twitch.drop(index=[13,14,25]), line_kws={"color": "red"});
```

```python
target = twitch['Followers gained']
feature = twitch.drop(columns = ['Channel','Language','Followers gained', 'Desc'])
feature = sm.add_constant(feature, has_constant='add')
scaler = MinMaxScaler()
scaler.fit(feature)
feature = pd.DataFrame(scaler.transform(feature), columns = feature.columns)
model = sm.OLS(target, feature).fit()
print(model.summary2())
```

```python
pd.DataFrame({
      "VIF Factor": [variance_inflation_factor(feature.values, idx) for idx in range(feature.shape[1])], 
      "features": feature.columns
})
```

#### 2-2. 데이터 전처리
```python
def get_outlier(df=None, column=None, weight=1.5):
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)
    
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    
    outlier_index = df[column][(df[column] < lowest_val) | (df[column] > highest_val)].index
    
    return outlier_index
```

#### 2-3. 모델링
```python

```

#### 2-4. 성능 평가
```python

```

#### 2-5. 추가 데이터 예측
```python

```

## 3. Built With

1. 김성준 : twitchtracker 크롤링, EDA, 모델링, 성능 평가, 추가 데이터 예측, README 작성
2. 김종찬 :   
3. 정하윤 :
