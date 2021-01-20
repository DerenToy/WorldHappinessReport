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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as  sns
import plotly as py
import plotly.graph_objs as go
from sklearn import cluster
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


# DATA VISUALIZATION
dataset_2019 = pd.read_csv('2019.csv')
# NULL DATA
#print(dataset_2019.isnull().sum()) #-> Boş veri yok..

# HEATMAP
cor =dataset_2019.corr()
sns.heatmap(cor, square = True)
plt.show()

# PAIRPLOT
sns.pairplot(dataset_2019)
plt.show()



new_dataset = dataset_2019[(dataset_2019['Country or region'].isin(['Finland','Denmark','Norway', 'Moldova','Turkey', 'Tanzania','Afghanistan']))]

ax = new_dataset.plot(y="Social support", x="Country or region", kind="bar",color='C3')
new_dataset.plot(y="GDP per capita", x="Country or region", kind="bar", ax=ax, color="C1")
new_dataset.plot(y="Healthy life expectancy", x="Country or region", kind="bar", ax=ax, color="C2")
plt.show()

ax = new_dataset.plot(y="Freedom to make life choices", x="Country or region", kind="bar",color='C3')
new_dataset.plot(y="Generosity", x="Country or region", kind="bar", ax=ax, color="C1",)
new_dataset.plot(y="Perceptions of corruption", x="Country or region", kind="bar", ax=ax, color="C2",)
plt.show()




dataset_2019 = pd.read_csv('2019.csv')


X_one = dataset_2019.iloc[:, 0].values
X_one = pd.DataFrame(data = X_one, columns = ['Overall rank'])

X_two = dataset_2019.iloc[:, 3:9].values
X_two = pd.DataFrame(data = X_two, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
X_ikiBinOnDokuz = pd.concat([X_one, X_two], axis=1)

features_2019 = X_ikiBinOnDokuz.iloc[:,1:7]
features_2019 = features_2019.to_numpy()


y_2019 = dataset_2019.iloc[:,2].values
y_2019 = pd.DataFrame(data= y_2019, columns=['Score'])



# TASK 1 -> Finding most significant features
X = np.append(arr = np.ones((156,1)).astype(int), values= features_2019, axis=1)
X_opt = X[:,[0,1,2,3,4,5,6]]
model = sm.OLS(y_2019, X_opt).fit()
print(model.summary())


X_opt = X[:,[0,1,2,3,4,6]]
model = sm.OLS(y_2019, X_opt).fit()
print(model.summary())


X_opt = X[:,[0,1,2,3,4]]
model = sm.OLS(y_2019, X_opt).fit()
print(model.summary())

# -> Most Significant Features: GDP, Healthy life, Social Support, Freedom to make life choices

# Global happiness of 2019
data = dict(type = 'choropleth', 
           locations = dataset_2019['Country or region'],
           locationmode = 'country names',
           z = dataset_2019['Overall rank'], 
           text = dataset_2019['Country or region'],
           colorbar = {'title':'Happiness ranking'})
layout = dict(title = 'Global Happiness in 2019', 
             geo = dict(showframe = False, 
                       projection = {'type': 'mercator'}))
cluster_map = go.Figure(data = [data], layout=layout)
py.offline.plot(cluster_map)


# Visualizing Features
fig , axs = plt.subplots(1,2, figsize = (20,10))
fig.set_size_inches(30, 10.5, forward=True)

features_2019 = pd.DataFrame(features_2019)
axs[0].scatter(y_2019,features_2019.loc[:,0],c = 'r', marker = 'o', label = 'GDP per capita')
axs[0].scatter(y_2019,features_2019.loc[:,1],c = 'b', marker = '^', label ='Social support' )
axs[0].scatter(y_2019,features_2019.loc[:,2],c = 'g', marker = 'x', label = 'Healthy life expectancy')
axs[0].set_xlabel('Score')
axs[0].title.set_text('Visualizing Most Significant Features')
axs[0].legend()

axs[1].set_ylim([0,1.75])
axs[1].scatter(y_2019,features_2019.loc[:,3],c = 'black', marker = 'o', label = 'Freedom to make life choices')
axs[1].scatter(y_2019,features_2019.loc[:,4],c = 'yellow', marker = '^', label ='Generosity' )
axs[1].scatter(y_2019,features_2019.loc[:,5],c = 'orange', marker = 'x', label = 'Perceptions of corruption')
axs[1].set_xlabel('Score')
axs[1].title.set_text('Visualizing Least Significant Features')
axs[1].legend()
plt.show()


# TASK 2 -> Predict 2019 data from 2018 data and calculate R2
dataset_2018 = pd.read_csv('2018.csv')
X_one_2018 = dataset_2018.iloc[:, 0].values
X_one_2018 = pd.DataFrame(data = X_one_2018, columns = ['Overall rank'])

X_two_2018 = dataset_2018.iloc[:, 3:8].values
X_two_2018 = pd.DataFrame(data = X_two_2018, columns = ['GDP per capita', 'Social support','Healthy life expectancy', 'Freedom to make life choices', 'Generosity'])
x_2018 = pd.concat([X_one_2018, X_two_2018], axis=1)


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
corruption =dataset_2018.iloc[:, 8].values
imputer = imputer.fit(corruption.reshape(-1, 1))
corruption= imputer.transform(corruption.reshape(-1, 1))
corruption = pd.DataFrame(corruption, columns = ['Perceptions of corruption'])

X_ikiBinOnSekiz = pd.concat([x_2018, corruption], axis = 1)

features_2018 = X_ikiBinOnSekiz.iloc[:,1:7]
features_2018 = features_2018.to_numpy()

# Label'ları ayırma
y_2018 = dataset_2018.iloc[:,2].values
y_2018 = pd.DataFrame(data= y_2018, columns=['Score'])

y_2018 = y_2018.to_numpy()

lin_reg = LinearRegression()
lin_reg.fit(features_2018, y_2018)

predict = lin_reg.predict(features_2019)

r2 = r2_score(y_2019, predict)
#mean_squarred = mean_squared_error(y_2019, predict)

# score = lin_reg.score(features_2019, y_2019)
# print("Lin:" ,score)

print ( "R2 score: ", r2)
#print("Mean Squarred Error: ", mean_squarred)



# TASK 3 -> Showing how GDP impacts by years

coefofGDPAccordingToYears = []

dataset_2015 = pd.read_csv('2015.csv')
#print(dataset_2015.isnull().sum()) -> boş veri yok
features_2015 = dataset_2015.iloc[:, 5:12].values

y_2015 = dataset_2015.iloc[:,3].values

lin_reg2015 = LinearRegression()
lin_reg2015.fit(features_2015, y_2015)
coef_2015 = lin_reg2015.coef_
 
coefofGDPAccordingToYears.append(coef_2015[0])


dataset_2016 = pd.read_csv('2016.csv')
#print(dataset_2016.isnull().sum()) -> boş veri yok
features_2016 = dataset_2016.iloc[:, 6:14].values

y_2016 = dataset_2016.iloc[:,3].values

lin_reg2016 = LinearRegression()
lin_reg2016.fit(features_2016, y_2016)
coef_2016 = lin_reg2016.coef_

coefofGDPAccordingToYears.append(coef_2016[0])


dataset_2017 = pd.read_csv('2017.csv')
#print(dataset_2017.isnull().sum()) -> Boş veri yok
features_2017 = dataset_2017.iloc[:, 5:12].values
y_2017 = dataset_2017.iloc[:,2].values

lin_reg2017 = LinearRegression()
lin_reg2017.fit(features_2017, y_2017)
coef_2017= lin_reg2017.coef_

coefofGDPAccordingToYears.append(coef_2017[0])


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


# Change in happiness score by years
plt.plot(dataset_2015['Happiness Score'], 'b', label='2015')
plt.plot(dataset_2016['Happiness Score'], 'g', label='2016')
plt.plot(dataset_2017['Happiness.Score'], 'r', label='2017')
plt.plot(dataset_2018['Score'], 'yellow', label='2018')
plt.plot(dataset_2019['Score'], 'black', label='2019')
plt.title('Happiness Score of 2015 & 2016 & 2017 & 2018 & 2019 ', fontsize=18)
plt.xlabel('Rank of Country', fontsize=16)
plt.ylabel('Happiness Score', fontsize=16)
plt.legend()
plt.show()





# CLUSTERING 
country=dataset_2019[dataset_2019.columns[1]]
data= dataset_2019.iloc[:,2:]

def normalizedData(x):
    normalised = StandardScaler()
    normalised.fit_transform(x)
    return(x)
    
data = normalizedData(data)    

n_clusters=3
def Kmeans(x, y):
    km= cluster.KMeans(x)
    ymeans=km.fit_predict(y)
    return(ymeans)
   
ymeans = Kmeans(3,data)
data['Kmeans'] = pd.DataFrame(ymeans)

dataset=pd.concat([data,country],axis=1)
dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country or region'],
           locationmode = 'country names',
           z = dataset['Kmeans'], 
           text = dataset['Country or region'],
           colorbar = {'title':'Cluster Group'})
layout = dict(title = 'Kmeans Clustering 2019', 
           geo = dict(showframe = False, 
           projection = {'type': 'mercator'}))
cluster_map = go.Figure(data = [dataPlot], layout=layout) 
py.offline.plot(cluster_map)


#%% Clustering 

dataset_2019 = pd.read_csv('2019.csv')
x_2019 = dataset_2019.iloc[:,3:9].values
y_2019 = dataset_2019["Score"].values

kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_2019)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax1 = fig.add_subplot(111, proje)
ax.scatter(x_2019[kmeans.labels_==0,0],x_2019[kmeans.labels_==0,1],x_2019[kmeans.labels_==0,2],c = 'r', marker = 'o')
ax.scatter(x_2019[kmeans.labels_==1,0],x_2019[kmeans.labels_==1,1],x_2019[kmeans.labels_==1,2],c = 'b', marker = '^')

ax.set_xlabel('GDP per capita')
ax.set_ylabel('Social support')
ax.set_zlabel('Healthy life expectancy')

ax.scatter(x_2019[kmeans.labels_==0,3],x_2019[kmeans.labels_==0,4],x_2019[kmeans.labels_==0,5],c = 'r', marker = 'o')
ax.scatter(x_2019[kmeans.labels_==0,3],x_2019[kmeans.labels_==0,4],x_2019[kmeans.labels_==0,5],c = 'b', marker = '^')

plt.show()



























