import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

train=pd.read_csv(r"train.csv")
test_data=pd.read_csv(r"test.csv")

train.head()
test_data.head()

train.info()
test_data.info()
train.describe()
test_data.describe()

train.apply(lambda x:len(x.unique()))
test_data.apply(lambda x: len(x.unique()))

plt.style.use('fivethirtyeight')
plt.figure(figsize=(13,7))
sns.distplot(train['Purchase'],bins=25)

sns.countplot(train['Gender'])
sns.countplot(train['Age'])
sns.countplot(train['Occupation'])
sns.countplot(train['City_Category'])
sns.countplot(train['Stay_In_Current_City_Years'])
sns.countplot(train['Marital_Status'])
sns.countplot(train['Product_Category_1'])
sns.countplot(train['Product_Category_2'])
sns.countplot(train['Product_Category_3'])
sns.countplot(train['Purchase'])

occupation_plot = pd.pivot_table(train,index='Occupation', values='Purchase', aggfunc=np.mean)
occupation_plot.plot(kind='bar', figsize=(8, 8),color='g')
plt.xlabel('Occupation')
plt.ylabel("Purchase")
plt.title("Occupation and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()

age_plot=pd.pivot_table(train,index='Gender',values='Purchase',aggfunc=np.mean)
age_plot.plot(kind='bar',figsize=(5,9))
plt.xticks(rotation=0)
plt.show()

age_plot=pd.pivot_table(train,index='Age',values='Purchase',aggfunc=np.mean)
age_plot.plot(kind='bar',figsize=(14,9))
plt.show()

train.isnull().sum()
test_data.isnull().sum()

train['Product_Category_2']=train['Product_Category_2'].fillna(-0.2).astype('int')
train['Product_Category_3']=train['Product_Category_3'].fillna(-0.2).astype('int')
test_data['Product_Category_2']=test_data['Product_Category_2'].fillna(-0.2).astype('int')
test_data['Product_Category_3']=test_data['Product_Category_3'].fillna(-0.2).astype('int')
train.isnull().sum()

#check missing values are cleared
test_data.isnull().sum()

#categorical value to numerical value
#using dict
gender_dict={'F':0,'M':1}

train['Gender']=train['Gender'].apply(lambda x:gender_dict[x])
train.head()

#label encoding
cols=['Age','City_Category','Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cols:
    train[col]=le.fit_transform(train[col])
train.head()

test_data=test_data.drop(columns=['User_ID','Product_ID'])
test_data.head()

cols=['Age','City_Category','Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cols:
    test_data[col]=le.fit_transform(test_data[col])
test_data.head()

#correlation matrix
corr=train.corr()
plt.figure(figsize=(17,17))
sns.heatmap(corr,annot=True,cmap='coolwarm')

#values of x and y
train.head()
x=train.drop(columns=['User_ID','Product_ID','Purchase'])
y=train['Purchase']
y

#Model Training
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    model.fit(x_train, y_train)
    
    # predict the results
    pred = model.predict(x_test)
    
    # cross validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Results")
    print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("CV Score:", np.sqrt(cv_score))

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
train(model, x, y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

pred = model.predict(x_test)
pred
submission=pd.read_csv(r"sample_submission.csv")
predict=model.predict(test_data)
submission['Purchase'] = predict
submission.to_csv('Black friday sales prediction sample.csv', index=False)
