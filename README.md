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
# 패키지 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime, timedelta
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
# 문자 데이터를 숫자 데이터로 변환
twitch['English'] = twitch['Language'] == "English" # 1 if English otherwise 0
twitch['English'] = twitch['English'].astype('int')
twitch['Partnered'] = twitch['Partnered'].astype('int') # 1 if Partnered otherwise 0
twitch['Mature'] = twitch['Mature'].astype('int') # 1 if Mature otherwise 0
```

```python
# 방송 시작 일자 데이터를 방송 기간(일)로 변환
time = datetime(2020, 9, 1)
twitch['Date'] = time - twitch['Date']
twitch['Date'] = list(date.days for date in twitch['Date'])
twitch['Date'][twitch['Date'].isna()] = twitch['Date'].median()
```

```python
# Outlier 제거
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

```python
# 24시간 채널 
twitch_str_time = twitch.sort_values(by = "Stream time(minutes)", ascending=False)
twitch_str_time.reset_index(inplace=True, drop=True)
twitch_str_time['Desc'] = '24hr channels'
twitch_str_time.head()

index = [x for x in list(range(30)) if x not in [7,11,13,15,17,21,27,28]]
twitch_cln = twitch_str_time.drop(index=index)
twitch_cln = twitch_cln.sort_values(by="Watch time(Minutes)", ascending=False)
```

#### 2-3. 모델링
```python
df_2 = []

def lin_regr(data, drop_cols=[[], ['Date'], ['English'], ['Partnered'], ['Date','English','Partnered']]):

    df = []

    for column in drop_cols:
        X = data.drop(columns = ['Channel','Language','Desc', 'Followers gained'] + column)
        y = data['Followers gained']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        
        rgr_list = [LinearRegression(), ElasticNet(alpha=0.1, l1_ratio=0.5), SVR(kernel='poly', degree=2, C=100, epsilon=0.1), 
                    RandomForestRegressor(max_depth=2, random_state=10)]
        scaler_list = [StandardScaler(), MinMaxScaler()]
        
        for rgr in rgr_list:

            for  scaler in scaler_list:
            
                estimators = [('scaler', scaler),
                             ('rgr', rgr)]

                pipe = Pipeline(estimators)

                pipe.fit(X_train, y_train)

                y_pred_tr = pipe.predict(X_train)
                y_pred_test = pipe.predict(X_test)

                rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
                lin_mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

                df.append({'Dataset' : data.iloc[0]['Desc'], 'Drop Columns' : ', '.join(column), 'scaler' : scaler, 
                           'rgr' : rgr, 'RMSE' : int(round(rmse_test)), 'MAE' : int(round(lin_mae)), 
                           'MAPE' : int(round(mape)), 'R2_Score' : round(r2, 5)})
    df_2.extend(df)
    df = pd.DataFrame(df)
    return df.sort_values(by='R2_Score', ascending=False).head()
```

#### 2-4. 성능 평가
```python
# 모델 간 성능 비교
table_df.sort_values(by='R2_Score', ascending=False).head(20)

plt.figure(figsize=(16,9))
sns.set_style("whitegrid")
sns.boxplot(x='Drop Columns',y='R2_Score',data=table_df).set_title("R2 Score by Columns dropped");


plt.figure(figsize=(16,9))
sns.boxplot(x='Dataset',y='R2_Score',data=table_df).set_title("R2 Score by Dataset");

plt.figure(figsize=(16,9))
sns.boxplot(x='rgr',y='R2_Score',data=table_df).set_title("R2 Score by Rgr");

plt.figure(figsize=(16,9))
sns.boxplot(x='scaler',y='R2_Score',data=table_df).set_title("R2 Score by Scaler");
```


#### 2-5. 추가 데이터 예측
<img src="https://user-images.githubusercontent.com/71831714/104873394-bfed5a00-5993-11eb-8cc5-5bb17ae21ae3.png"></img>

```python
# 임의의  데이터 예측
def predict_followers_gained(channel):
    X = twitch_outlier_3.drop(columns = ['Channel','Language','Desc', 'Followers gained', 'Partnered'])
    y = twitch_outlier_3['Followers gained']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    estimators = [('scaler', StandardScaler()),
             ('rgr', LinearRegression())]

    pipe = Pipeline(estimators)

    pipe.fit(X_train, y_train)

    result = int(round(pipe.predict(channel)[0]))
    
    return result
    

dhtekkz = [[59378100,21180,18364,2821,179851,1626543,0,1235,1]]
predict_followers_gained(dhtekkz)

# Real Followers gained : 179,851 Predict Followers gained : 182,147
# In fact, there are too many external variables that affects prediction.
# But prediction for this channel was pretty good :)
```

## 3. Built With

1. 김성준 : twitchtracker 크롤링, EDA, 모델링, 성능 평가, 추가 데이터 예측, README 작성
2. 김종찬 :   
3. 정하윤 :
