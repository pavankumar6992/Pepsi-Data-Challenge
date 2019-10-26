import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#pip install eli5
import eli5


data = pd.read_csv("D:\Pavan_Research\Datasets\Pepsi\Data.csv")
describe = data.describe()
data.head()
data.isnull().sum()
data.notnull().sum()
data.dtypes


#data.loc[:,'Product Type'].unique()
#data['Product Type'].nunique()
#data['Base Ingredient'].unique()
#data['Process Type'].unique()
#data['Storage Conditions'].unique()
#data['Packaging Stabilizer Added'].unique()
#data['Transparent Window in Package'].unique()
#data['Preservative Added'].unique()


data.isnull().sum()
X = data.iloc[:,[2,3,4,5,7,8,10,12,13]]
y = data['Difference From Fresh']
X.isnull().sum()


bin_cols = data.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    data[i] = le.fit_transform(data[i].astype(str))
    
    
#for col in encodecolumns:
#    le.fit_transform(X[col].astype(str))

sns.distplot(data.iloc[:,2])
sns.distplot(data.iloc[:,3])
sns.distplot(data.iloc[:,4])
sns.distplot(data.iloc[:,5])
sns.distplot(data.iloc[:,6])
sns.distplot(data.iloc[:,7])
sns.distplot(data.iloc[:,8])
sns.distplot(data.iloc[:,9])
sns.distplot(data.iloc[:,10])
sns.distplot(data.iloc[:,11])


# Using Random Forest to impute the missing columns
# Using RAndom Forest to identity the variable importance

# Splitting the data into Traning and Testing sets.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 50, shuffle = True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 100, random_state = 1)
model = RandomForestRegressor(n_estimators= 100, random_state = 1)
model.fit(X_train, y_train)
model.predict(X_test)
importance = model.feature_importances_
indices = np.argsort(importance)
features = list(X.columns)


plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()



# Imputing Base Ingredient(3) **********************


Xwith = X[pd.isnull(X['Base Ingredient']) == False]
Xwithout = X[pd.isnull(X['Base Ingredient'])]
columns = ['Product Type','Process Type','Sample Age (Weeks)','Storage Conditions','Packaging Stabilizer Added','Processing Agent Stability Index']

bin_cols = Xwith.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwith[i] = le.fit_transform(Xwith[i].astype(str))


bin_cols = Xwithout.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwithout[i] = le.fit_transform(Xwithout[i].astype(str))


model = RandomForestClassifier()
model.fit(Xwith[columns], Xwith['Base Ingredient'])
generated_values = model.predict(X = Xwithout[columns])
Xwithout['Base Ingredient'] = generated_values
data_new = Xwith.append(Xwithout)
X['Base Ingredient'] = data_new['Base Ingredient']


### ************************ Imputing Storage Conditions ******************

Xwith = X[pd.isnull(X['Storage Conditions']) == False]
Xwithout = X[pd.isnull(X['Storage Conditions'])]
columns = ['Product Type','Process Type','Sample Age (Weeks)','Base Ingredient','Packaging Stabilizer Added','Processing Agent Stability Index']

bin_cols = Xwith.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwith[i] = le.fit_transform(Xwith[i].astype(str))


bin_cols = Xwithout.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwithout[i] = le.fit_transform(Xwithout[i].astype(str))


model = RandomForestClassifier()
model.fit(Xwith[columns], Xwith['Storage Conditions'])
generated_values = model.predict(X = Xwithout[columns])
Xwithout['Storage Conditions'] = generated_values
data_new = Xwith.append(Xwithout)
X['Storage Conditions'] = data_new['Storage Conditions']

### ************************ Packaging Stabilizer Added  ******************


Xwith = X[pd.isnull(X['Packaging Stabilizer Added']) == False]
Xwithout = X[pd.isnull(X['Packaging Stabilizer Added'])]
columns = ['Product Type','Process Type','Sample Age (Weeks)','Base Ingredient','Storage Conditions','Processing Agent Stability Index']

bin_cols = Xwith.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwith[i] = le.fit_transform(Xwith[i].astype(str))


bin_cols = Xwithout.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwithout[i] = le.fit_transform(Xwithout[i].astype(str))


model = RandomForestClassifier()
model.fit(Xwith[columns], Xwith['Packaging Stabilizer Added'])
generated_values = model.predict(X = Xwithout[columns])
Xwithout['Packaging Stabilizer Added'] = generated_values
data_new = Xwith.append(Xwithout)
X['Packaging Stabilizer Added'] = data_new['Packaging Stabilizer Added']

### ************************ Moisture (%)  ******************

Xwith = X[pd.isnull(X['Moisture (%)']) == False]
Xwithout = X[pd.isnull(X['Moisture (%)'])]
columns = ['Product Type','Process Type','Sample Age (Weeks)','Base Ingredient','Storage Conditions','Processing Agent Stability Index','Packaging Stabilizer Added']

bin_cols = Xwith.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwith[i] = le.fit_transform(Xwith[i].astype(str))


bin_cols = Xwithout.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwithout[i] = le.fit_transform(Xwithout[i].astype(str))


model = RandomForestRegressor()
model.fit(Xwith[columns], Xwith['Moisture (%)'])
generated_values = model.predict(X = Xwithout[columns])
Xwithout['Moisture (%)'] = generated_values
data_new = Xwith.append(Xwithout)
X['Moisture (%)'] = data_new['Moisture (%)']


### ************************ Residual Oxygen (%)  ******************

Xwith = X[pd.isnull(X['Residual Oxygen (%)']) == False]
Xwithout = X[pd.isnull(X['Residual Oxygen (%)'])]
columns = ['Product Type','Process Type','Sample Age (Weeks)','Base Ingredient','Storage Conditions','Processing Agent Stability Index','Packaging Stabilizer Added','Moisture (%)']

bin_cols = Xwith.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwith[i] = le.fit_transform(Xwith[i].astype(str))


bin_cols = Xwithout.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
#X["code"] = le.fit_transform(X['Packaging Stabilizer Added'].astype(str))
for i in bin_cols :
    Xwithout[i] = le.fit_transform(Xwithout[i].astype(str))


model = RandomForestRegressor()
model.fit(Xwith[columns], Xwith['Residual Oxygen (%)'])
generated_values = model.predict(X = Xwithout[columns])
Xwithout['Residual Oxygen (%)'] = generated_values
data_new = Xwith.append(Xwithout)
X['Residual Oxygen (%)'] = data_new['Residual Oxygen (%)']

X.isnull().sum()
### ************************ Imputation End ******************

# **** Model Developement Start ****


data_new.reset_index(inplace=True)
data_new.drop('index',inplace=True,axis=1)
data_new['Difference From Fresh'] = data['Difference From Fresh']


# **** Model Developement End ****




