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

# VERİ KÜMESİNİ HAZIRLAMA

# Veri setini yükleme
dataset = pd.read_csv('2019.csv')

# Boş veri kontrolü 
#print(dataset.isnull().sum())
# -> Boş veri yok..

# Veri setinin 'features'ını oluşturma
X_one = dataset.iloc[:, 0].values
X_one = pd.DataFrame(data = X_one, columns = ['Overall rank'])

X_two = dataset.iloc[:, 3:9].values
X_two = pd.DataFrame(data = X_two, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
X_ikiBinOnDokuz = pd.concat([X_one, X_two], axis=1)

features_2019 = X_ikiBinOnDokuz.iloc[:,1:7]
features_2019 = features_2019.to_numpy()

# Label'ları ayırma
y_2019 = dataset.iloc[:,2].values
y_2019 = pd.DataFrame(data= y_2019, columns=['Score'])



"""

import statsmodels.api as sm

X = np.append(arr = np.ones((156,1)).astype(int), values= X, axis=1)
X_opt = X[:,[0,2,3,4,5,6,7]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

#  genoristy elendiii... 

X_opt = X[:,[0,2,3,4,5,7]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# corruption eliminated..

X_opt = X[:,[0,2,3,4,5]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# -> GDP, Healthy life, Social Support, Freedom to make life choices kaldı..

"""

# plt.scatter(y, X['GDP per capita'], s=50, c='yellow',  label ='GDP per capita' )
# plt.scatter(y, X['Social support'], s=50, c='orange', marker='x', label ="Social Support" )
# plt.scatter(y, X['Healthy life expectancy'], s=50, c='green', marker='^', label="Healthy life expectancy")

# plt.xlabel("Score")
# plt.title("Effected Features 2019")
# plt.legend()
# plt.show()


# plt.scatter(y, X['Generosity'], s=50, c='black', marker='o', label ='Generosity' )
# plt.scatter(y, X['Perceptions of corruption'], s=50, c='blue', marker='D', label ='Perceptions of corruption' )
# plt.scatter(y, X['Freedom to make life choices'], s=50, c='red', marker='+', label ='Freedom to make life choices' )
# plt.xlabel("Score")
# plt.title("Non-effected Features 2019")
# plt.ylim((0, 1.75))

# plt.legend()
# plt.show()



dataset_2018 = pd.read_csv('2018.csv')
X_one_2018 = dataset_2018.iloc[:, 0].values
X_one_2018 = pd.DataFrame(data = X_one_2018, columns = ['Overall rank'])

X_two_2018 = dataset_2018.iloc[:, 3:8].values
X_two_2018 = pd.DataFrame(data = X_two_2018, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity'])
x_2018 = pd.concat([X_one_2018, X_two_2018], axis=1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
corruption =dataset_2018.iloc[:, 8].values
imputer = imputer.fit(corruption.reshape(-1, 1))
corruption= imputer.transform(corruption.reshape(-1, 1))
corruption = pd.DataFrame(corruption, columns = ['Perceptions of corruption'])

X_ikiBinOnSekiz = pd.concat([x_2018, corruption], axis = 1)


features_2018 = X_ikiBinOnSekiz.iloc[:,1:7]
features_2018 = features_2018.to_numpy()

#X_2018 = pd.DataFrame(data = X_2018)

# Label'ları ayırma
y_2018 = dataset_2018.iloc[:,2].values
y_2018 = pd.DataFrame(data= y_2018, columns=['Score'])
y_2018.to_numpy()

# plt.scatter(y_2018, x_2018['GDP per capita'], s=50, c='yellow',  label ='GDP per capita' )
# plt.scatter(y_2018, x_2018['Social support'], s=50, c='orange', marker='x', label ="Social Support" )
# plt.scatter(y_2018, x_2018['Healthy life expectancy'], s=50, c='green', marker='^', label="Healthy life expectancy")

# plt.xlabel("Score")
# plt.title("Effected Features 2018")
# plt.legend()
# plt.show()

# plt.scatter(y_2018, x_2018['Generosity'], s=50, c='black', marker='o', label ='Generosity' )
# plt.scatter(y_2018, x_2018['Perceptions of corruption'], s=50, c='blue', marker='D', label ='Perceptions of corruption' )
# plt.scatter(y_2018, x_2018['Freedom to make life choices'], s=50, c='red', marker='+', label ='Freedom to make life choices' )
# plt.xlabel("Score")
# plt.ylim((0, 1.75))
# plt.title("Non-effected Features 2018")
# plt.legend()
# plt.show()



"""
features = x_2018.iloc[:,1:7]
features = features.to_numpy()

from sklearn.cluster import KMeans
km=KMeans(n_clusters =6)
ymeans= km.fit_predict(features)


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[ymeans==0,0],features[ymeans==0,1],features[ymeans==0,2],c = 'r', marker = 'o')
ax.scatter(features[ymeans==1,0],features[ymeans==1,1],features[ymeans==1,2],c = 'b', marker = '^')
ax.scatter(features[ymeans==2,0],features[ymeans==2,1],features[ymeans==2,2],c = 'g', marker = 'x')
ax.scatter(features[ymeans==3,0],features[ymeans==3,1],features[ymeans==3,2],c = 'g', marker = 'x')
ax.scatter(features[ymeans==4,0],features[ymeans==4,1],features[ymeans==4,2],c = 'g', marker = 'x')

ax.set_xlabel('GDP per capita')
ax.set_ylabel('Social support')
ax.set_zlabel('Healthy life expectancy')
plt.show()

"""

# TASK 1 
y_2018 = y_2018.to_numpy()
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(features_2018, y_2018)

predict = lin_reg.predict(features_2019)

r2 = r2_score(y_2019, predict)
mean_squarred = mean_squared_error(y_2019, predict)
score = lin_reg.score(features_2019, y_2019)
print("Lin:" ,score)
print ( "R2: ", r2)
print("Mean Squarred Error: ", mean_squarred)


