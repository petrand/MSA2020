import pandas as pd
import math 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from heatmap import heatmap, corrplot
from sklearn.model_selection import train_test_split

#some code snippets were adapted from https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning


data = pd.read_csv('property_with_travelTimes.csv')

corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



out_col = ['Bedrooms','Bathrooms', 'Distance To CBD', '30-39 years']
fig = plt.figure(figsize=(20,5))
for index,col in enumerate(out_col):
    plt.subplot(1,5,index+1)
    sns.boxplot(y=col, data=data)
fig.tight_layout(pad=1.5)

data = data.drop(data[(data['Bedrooms'] > 10) & (data['Bathrooms'] > 6)].index)

# dropping test column
X = data.drop('CV', axis=1)
y = data['CV']
print(y.head())
print(X.dtypes)
print(X.shape, y.shape)


# converting population to percentages
def population_to_percentage(age_gap, population):
    try:
        age_gap = age_gap/population
    except (ZeroDivisionError):
        pass
    return age_gap

#converting population to percentage 
X['0-19 years'] = X.apply(lambda row: population_to_percentage(row['0-19 years'], 
                row['Population']), axis=1)
X['20-29 years'] = X.apply(lambda row: population_to_percentage(row['20-29 years'], 
                row['Population']), axis=1)
X['30-39 years'] = X.apply(lambda row: population_to_percentage(row['30-39 years'],
                row['Population']), axis=1)
X['40-49 years'] = X.apply(lambda row: population_to_percentage(row['40-49 years'],
                row['Population']), axis=1)
X['50-59 years'] = X.apply(lambda row: population_to_percentage(row['50-59 years'],
                row['Population']), axis=1)
X['60+ years'] = X.apply(lambda row: population_to_percentage(row['60+ years'],
                row['Population']), axis=1)

numeric_col = X.select_dtypes(exclude=['object']).copy()

fig = plt.figure(figsize=(18,16))
for index,col in enumerate(numeric_col.columns):
    plt.subplot(6,4,index+1)
    sns.distplot(numeric_col.loc[:,col].dropna(), kde=False)
fig.tight_layout(pad=1.0)

plt.savefig("distributions age Normalised")
plt.show()

fig = plt.figure(figsize=(18,20))
for index in range(1,2):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_train.iloc[:,index], data=cat_train.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)

cat_train = X.select_dtypes(include=['object']).copy()

plt.figure(figsize=(30,8))
ax = sns.countplot(x=cat_train.iloc[:,1], data=cat_train.dropna(),  
                    order = cat_train['Suburbs'].value_counts().index)
plt.xticks(rotation=90, fontsize=7)
plt.savefig("Suburbs")

correlation = data.corr()
print(correlation[['CV']].sort_values(['CV'], ascending=False))

plt.figure(figsize=(14,12))
correlation = numeric_col.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')



X.drop(['NZDep2018_Score', 'Longitude', 'Address', 'Population'], axis=1, inplace=True)

plt.show()
# checking for outliers 

X = pd.get_dummies(X)

plt.figure(figsize=(10,6))
plt.title("Before transformation of CV")
dist = sns.distplot(data['CV'],norm_hist=False)

plt.figure(figsize=(10,6))
plt.title("After transformation of CV")
dist = sns.distplot(np.log(data['CV']),norm_hist=False)

y = np.log(y)
print(X.shape)

from sklearn.preprocessing import RobustScaler

cols = X.select_dtypes(np.number).columns
transformer = RobustScaler().fit(X[cols])
X[cols] = transformer.transform(X[cols])


#y[cols] = transformer.transform(y[cols])


#train test split 


train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from joblib import dump
from joblib import load



from sklearn.model_selection import RandomizedSearchCV

param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [100, 500, 1000, 2000, 3000],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10, 20],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1]
}

xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')

xgb_reg = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst, 
                            n_iter = 100,  cv = 5)
       
xgb_search = xgb_reg.fit(train_x, train_y)
best_param = xgb_search.best_params_
xgb = XGBRegressor(**best_param)
xgb.fit(train_x, train_y)   
dump(xgb, "pima_xgb.joblib.dat")
print('Saved model to: pima_xgb.joblib.dat')


cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')

param_lst = {
    'n_estimators' : [100, 300, 500, 1000, 1300, 1600],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
    'l2_leaf_reg' : [0.001, 0.01, 0.1],
    'random_strength' : [0.25, 0.5 ,1],
    'max_depth' : [3, 6, 9],
    'min_child_samples' : [2, 5, 10, 15, 20],
    'rsm' : [0.5, 0.7, 0.9],    
}

catboost = RandomizedSearchCV(estimator = cb, param_distributions = param_lst,
                              n_iter = 100, 
                              cv = 5)

catboost_search = catboost.fit(train_x, train_y)

best_param = catboost_search.best_params_
cb = CatBoostRegressor(logging_level='Silent', **best_param)
cb.fit(train_x, train_y)   
dump(cb, "pima.joblib.dat")
print("Saved model to: pima.joblib.dat")

from sklearn.ensemble import RandomForestRegressor

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the parameter list
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_x, train_y)

print(rf_random.best_params_)

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [60, 70, 80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [400, 600, 800, 1000]
}
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_x, train_y)
print(grid_search.best_params_)













from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# load model from file
exptest_y = test_y.apply(math.exp)

'''

'''
import math 
#predictions = [math.exp(i) for i in predictions]
predictions = pd.Series(predictions)
exptest_y = test_y.apply(math.exp)
print(predictions)
print(exptest_y)

fig, ax = plt.subplots()
ax.scatter(predictions, test_y)
plt.xlabel('predictions')
plt.ylabel('actualval')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.show()


from sklearn.ensemble import RandomForestRegressor
# {'bootstrap': False, 'max_depth': 90, 
# 'max_features': 3, 'min_samples_leaf': 1, 

# 'min_samples_split': 5, 'n_estimators': 1000}

def mean_cross_val(model, X, y):
    score = cross_val_score(model, X, y, cv=5)
    mean = score.mean()
    return mean


rf = RandomForestRegressor(bootstrap = False, max_depth = 90, 
                            max_features=3, min_samples_leaf=1, 
                            min_samples_split=5, n_estimators=1000)
rf.fit(train_x, train_y)
preds = rf.predict(test_x)
preds_test_rf = rf.predict(test_x)
mae_rf = mean_absolute_error(test_y, preds)
rmse_rf = np.sqrt(mean_squared_error(test_y, preds))
score_rf = rf.score(test_x, test_y)
cv_rf = mean_cross_val(rf, X, y)
r_rf = r2_score(test_y, preds)

cb = load("pima.joblib.dat")
preds = cb.predict(test_x) 
preds_test_cb = cb.predict(test_x)
mae_cb = mean_absolute_error(test_y, preds)
rmse_cb = np.sqrt(mean_squared_error(test_y, preds))
score_cb = cb.score(test_x, test_y)
cv_cb = mean_cross_val(cb, X, y)
r_cb = r2_score(test_y, preds)

xgb = load("pima_xgb.joblib.dat")
preds = xgb.predict(test_x) 
preds_test_xgb = xgb.predict(test_x)
mae_xgb = mean_absolute_error(test_y, preds)
rmse_xgb = np.sqrt(mean_squared_error(test_y, preds))
score_xgb = xgb.score(test_x, test_y)
cv_xgb = mean_cross_val(xgb, X, y)
r_xgb = r2_score(test_y, preds)



#training the model
model_performances = pd.DataFrame({
    "Model" : ["CatBoost", "XGBoost", "RFG"],
    "CV(5)" : [str(cv_cb)[0:5], str(cv_xgb)[0:5], str(cv_rf)[0:5]],
    "MAE" : [str(mae_cb)[0:5], str(mae_xgb)[0:5], str(mae_rf)[0:5]],
    "RMSE" : [str(rmse_cb)[0:5], str(rmse_xgb)[0:5], str(rmse_rf)[0:5]],
    "Score" : [str(score_cb)[0:5], str(score_xgb)[0:5], str(score_rf)[0:5]],
    "R^2" : [str(r_cb)[0:5], str(r_xgb)[0:5], str(r_rf)[0:5]]
})

print("Sorted by Score:")
print(model_performances.sort_values(by="Score", ascending=False))