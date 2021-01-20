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

## 2. Preview

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
<img src="https://user-images.githubusercontent.com/71831714/104876380-ec58a480-599a-11eb-8953-7d8754bb02ae.png" width='600'></img>

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
<img src="https://user-images.githubusercontent.com/71831714/104876541-3fcaf280-599b-11eb-80b9-14692b9ef6df.png" width='600'></img>

```python
pd.DataFrame({
      "VIF Factor": [variance_inflation_factor(feature.values, idx) for idx in range(feature.shape[1])], 
      "features": feature.columns
})
```
<img src="https://user-images.githubusercontent.com/71831714/105168264-3c7f6480-5b5d-11eb-94f1-90f631874346.png" width='300'></img>

```python
mask = np.zeros_like(twitch.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(twitch.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
```
<img src="https://user-images.githubusercontent.com/71831714/104876166-76ecd400-599a-11eb-9889-1dbca18a7aad.png" width='600'></img>

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
    
lin_regr(twitch)
```
<img src="https://user-images.githubusercontent.com/71831714/104876812-c384df00-599b-11eb-9041-ee7de97ab6fb.png" width='600'></img>

#### 2-4. 성능 평가
```python
# 모델 간 성능 비교
table_df.sort_values(by='R2_Score', ascending=False).head(20)
```
<img src="https://user-images.githubusercontent.com/71831714/104876990-270f0c80-599c-11eb-83d1-617dd6ba649a.png" width='800'></img>
```
# 제거한 컬럼별 R2 Score
plt.figure(figsize=(16,9))
sns.set_style("whitegrid")
sns.boxplot(x='Drop Columns',y='R2_Score',data=table_df).set_title("R2 Score by Columns dropped");
```
<img src="https://user-images.githubusercontent.com/71831714/104876993-28403980-599c-11eb-99b4-03636c0f0fb1.png"></img>
```
# 데이터셋에 따른 R2 Score
plt.figure(figsize=(16,9))
sns.boxplot(x='Dataset',y='R2_Score',data=table_df).set_title("R2 Score by Dataset");
```
<img src="https://user-images.githubusercontent.com/71831714/104876996-29716680-599c-11eb-87cc-13f3a2e2baba.png"></img>
```
# 회귀 모델별 R2 Score
plt.figure(figsize=(16,9))
sns.boxplot(x='rgr',y='R2_Score',data=table_df).set_title("R2 Score by Rgr");
```
<img src="https://user-images.githubusercontent.com/71831714/104876998-2aa29380-599c-11eb-97fb-397c10ef272e.png"></img>
```
# Scaler별 R2 Score
plt.figure(figsize=(16,9))
sns.boxplot(x='scaler',y='R2_Score',data=table_df).set_title("R2 Score by Scaler");
```
<img src="https://user-images.githubusercontent.com/71831714/104876999-2bd3c080-599c-11eb-85cb-607f9217cc5e.png"></img>

#### 2-5. 추가 데이터 예측
<img src="https://user-images.githubusercontent.com/71831714/104873394-bfed5a00-5993-11eb-8cc5-5bb17ae21ae3.png" width='600'></img>

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
#### 2-6. 파라미터 값 조정

```python
# RandomForestRegressor의 max_depth를 4로 조정
# Tuning max_depth and dropping Date column makes the best performance
X = twitch_outlier_3.drop(columns = ['Channel','Language','Desc', 'Followers gained', 'Date'])
y = twitch_outlier_3['Followers gained']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

estimators = [('scaler', StandardScaler()),
     ('rgr', RandomForestRegressor(max_depth=4, random_state=10))]

pipe = Pipeline(estimators)

pipe.fit(X_train, y_train)

y_pred_test = pipe.predict(X_test)

rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
lin_mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print('RMSE : {}'.format(int(round(rmse_test))), 'MAE : {}'.format(int(round(lin_mae))),
      'MAPE : {}'.format(int(round(mape))), 'R2_Score : {}'.format(round(r2, 5)))
      
### RMSE : 200823 MAE : 111958 MAPE : 172 R2_Score : 0.71019
```

## 3. Review
      1. Linear Regression에서 StandardScaler와 MinMaxScaler의 성능 차이가 없다는 의문점 -> 추가 검색 및 학습 예정
      2. 회귀 모델들의 parameter값에 따라 성능이 크게 개선됨 -> 추가 학습 후 적용 예정
      3. Support Vector Classifier가 가장 좋은 성능을 보였음
      4. 주어진 features만으로 종속 변수를 예측하기에는 종속 변수에 영향을 끼치는 외부 변수가 너무 많다는 한계점 존재

## 4. Built With

1. 김성준 : twitchtracker 크롤링, EDA, 모델링, 성능 평가, 새로운 데이터 예측, README 작성
2. 김종찬 : EDA, 모델링, 성능 평가, 새로운 데이터 예측 
3. 정하윤 :
