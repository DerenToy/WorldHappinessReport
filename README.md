# WORLD HAPPINESS REPORT
# Description: 
Our project includes how the happiness scores of countries
 change and which factors affect these scores the most. 
By looking at our project, you can find answers to questions
 such as which factors are in the foreground in the happiest
 countries and which continents are more common in happiness.

# Team & Roles: 
Kıymet Deren Toy 
- Model training with linear regression algorithm and prediction of 2018 data for 2019 data (pre-processing of data sets)

- A linear model is created and the weights of the GDP per capita feature are compared

- Visualization of the 2019 data set on the map

- Finding the most significant features with backward elimination method

- Visualization of each year's happiness score in a single graphic

- Dividing the 2019 data set into 3 clusters by clustering and visualizing them on the map

Görkem Savran
- Visualization of relationships between heatmap and pairplot and features

- Visualization of the features that have the most and least effect on the happiness score

- Selecting 7 countries and comparing the effects of features on these countries

- Creating models using ensemble learning methods and comparing these models

# Structure: 
/WorldHappinessReport/<br />
├── datasets<br />
│   ├── 2015.csv<br />
│   ├── 2016.csv<br />
│   ├── 2017.csv<br />
│   ├── 2018.csv<br />
│   └── 2019.csv<br />
├── README.md<br />
├── report_source<br />
│   ├── clustering3.png<br />
│   ├── clusteringmap.png<br />
│   ├── country.png<br />
│   ├── ensembleLearning.png<br />
│   ├── features.png<br />
│   ├── gdp.png<br />
│   ├── globalmap.png<br />
│   ├── happines2019.png<br />
│   ├── heatmap.png<br />
│   ├── Report.tex<br />
│   ├── ols.png<br />
│   ├── pairplot.png<br />
│   ├── r2.png<br />
│   └── score.png<br />
├── requirements.txt<br />
├── WorldHappinessReport.pdf<br />
└── WorldHappinessReport.py<br />



# Language, version, and main file: 
Language: Python<br />
Version: 3.8.5<br />
Main File: WorldHappinessReport.py<br />
