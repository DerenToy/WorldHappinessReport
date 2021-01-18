#
# KIYMET DEREN TOY  170709012
# GÖRKEM SAVRAN     180709010
#
#
#                   DATA MINING - FINAL PROJECT
#
#


# LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression






# TASK 1 -> En çok etki eden değişkeni bulma
"""
import statsmodels.api as sm

X = np.append(arr = np.ones((156,1)).astype(int), values= X, axis=1)
X_opt = X[:,[0,2,3,4,5,6,7]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# genoristy elendiii... 

X_opt = X[:,[0,2,3,4,5,7]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# corruption eliminated..

X_opt = X[:,[0,2,3,4,5]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# -> GDP, Healthy life, Social Support, Freedom to make life choices kaldı..

"""






# TASK 2 -> 2018 verisinden 2019 verisini tahmin etme ve R2 hesaplama
"""
# Veri setini yükleme
dataset_2019 = pd.read_csv('2019.csv')

# Boş veri kontrolü 
# #print(dataset.isnull().sum()) -> Boş veri yok..

# # Veri setinin 'features'ını oluşturma
# X_one = dataset_2019.iloc[:, 0].values
# X_one = pd.DataFrame(data = X_one, columns = ['Overall rank'])

# X_two = dataset_2019.iloc[:, 3:9].values
# X_two = pd.DataFrame(data = X_two, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
# X_ikiBinOnDokuz = pd.concat([X_one, X_two], axis=1)

# features_2019 = X_ikiBinOnDokuz.iloc[:,1:7]
# features_2019 = features_2019.to_numpy()

# # Label'ları ayırma
# y_2019 = dataset.iloc[:,2].values
# y_2019 = pd.DataFrame(data= y_2019, columns=['Score'])


# dataset_2018 = pd.read_csv('2018.csv')
# X_one_2018 = dataset_2018.iloc[:, 0].values
# X_one_2018 = pd.DataFrame(data = X_one_2018, columns = ['Overall rank'])

# X_two_2018 = dataset_2018.iloc[:, 3:8].values
# X_two_2018 = pd.DataFrame(data = X_two_2018, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity'])
# x_2018 = pd.concat([X_one_2018, X_two_2018], axis=1)

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# corruption =dataset_2018.iloc[:, 8].values
# imputer = imputer.fit(corruption.reshape(-1, 1))
# corruption= imputer.transform(corruption.reshape(-1, 1))
# corruption = pd.DataFrame(corruption, columns = ['Perceptions of corruption'])

# X_ikiBinOnSekiz = pd.concat([x_2018, corruption], axis = 1)


# features_2018 = X_ikiBinOnSekiz.iloc[:,1:7]
# features_2018 = features_2018.to_numpy()

# #X_2018 = pd.DataFrame(data = X_2018)

# # Label'ları ayırma
# y_2018 = dataset_2018.iloc[:,2].values
# y_2018 = pd.DataFrame(data= y_2018, columns=['Score'])

y_2018 = y_2018.to_numpy()

lin_reg = LinearRegression()
lin_reg.fit(features_2018, y_2018)

predict = lin_reg.predict(features_2019)

r2 = r2_score(y_2019, predict)
mean_squarred = mean_squared_error(y_2019, predict)
score = lin_reg.score(features_2019, y_2019)
print("Lin:" ,score)
print ( "R2: ", r2)
print("Mean Squarred Error: ", mean_squarred)

"""









# TASK 3 -> Yıllara göre değişkenlerin nasıl etkilediğini gösterme
"""
coefofGDPAccordingToYears = []

# 2015
dataset_2015 = pd.read_csv('2015.csv')
#print(dataset_2015.isnull().sum()) -> boş veri yok
features_2015 = dataset_2015.iloc[:, 5:12].values

y_2015 = dataset_2015.iloc[:,3].values

lin_reg2015 = LinearRegression()
lin_reg2015.fit(features_2015, y_2015)
coef_2015 = lin_reg2015.coef_
 
coefofGDPAccordingToYears.append(coef_2015[0])

# 2016
dataset_2016 = pd.read_csv('2016.csv')
#print(dataset_2016.isnull().sum()) -> boş veri yok
features_2016 = dataset_2016.iloc[:, 6:14].values

y_2016 = dataset_2016.iloc[:,3].values

lin_reg2016 = LinearRegression()
lin_reg2016.fit(features_2016, y_2016)
coef_2016 = lin_reg2016.coef_

coefofGDPAccordingToYears.append(coef_2016[0])

# 2017
dataset_2017 = pd.read_csv('2017.csv')
#print(dataset_2017.isnull().sum()) -> Boş veri yok
features_2017 = dataset_2017.iloc[:, 5:12].values
y_2017 = dataset_2017.iloc[:,2].values

lin_reg2017 = LinearRegression()
lin_reg2017.fit(features_2017, y_2017)
coef_2017= lin_reg2017.coef_

coefofGDPAccordingToYears.append(coef_2017[0])


# 2018 
dataset_2018 = pd.read_csv('2018.csv')

X_one_2018 = dataset_2018.iloc[:, 3:8].values
X_one_2018 = pd.DataFrame(X_one_2018)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

corruption =dataset_2018.iloc[:, 8].values
imputer = imputer.fit(corruption.reshape(-1, 1))
corruption= imputer.transform(corruption.reshape(-1, 1))
corruption = pd.DataFrame(corruption, columns = ['5'])

features_2018 = pd.concat([X_one_2018, corruption], axis = 1)
y_2018 = dataset_2018.iloc[:,2].values

lin_reg2018 = LinearRegression()
lin_reg2018.fit(features_2018, y_2018)
coef_2018= lin_reg2018.coef_

coefofGDPAccordingToYears.append(coef_2018[0])


# 2019
dataset_2019 = pd.read_csv('2019.csv')
#print(dataset_2019.isnull().sum())-> Boş veri yok..

features_2019 = dataset_2019.iloc[:, 3:9].values
y_2019 = dataset_2019.iloc[:,2].values

lin_reg2019 = LinearRegression()
lin_reg2019.fit(features_2019, y_2019)
coef_2019 = lin_reg2019.coef_

coefofGDPAccordingToYears.append(coef_2019[0])


listOfYears = [2015,2016,2017,2018,2019]

plt.plot(listOfYears,coefofGDPAccordingToYears, linewidth=4)
plt.xlabel("Years")
plt.ylabel("GDP coefficient")
plt.title("GDP Coefficient According To Years")
plt.show()
"""































