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
/WorldHappinessReport/\n
├── 3FeaturesLeastSignificantClustering.pdf
├── 3FeaturesMostSignificantClustering.pdf
├── ComparisonOf7CountriesAccordingToLeastSignificantFeatures.pdf
├── ComparisonOf7CountriesAccordingToMostSignificantFeatures.pdf
├── datasets
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   └── 2019.csv
├── GDPCoefficientAccordingToYears.pdf
├── HappinessScoreAccordingToYears.pdf
├── Heatmap.pdf
├── Pairplot.pdf
├── README.md
├── report_source
│   ├── clustering3.png
│   ├── clusteringmap.png
│   ├── country.png
│   ├── ensembleLearning.png
│   ├── features.png
│   ├── gdp.png
│   ├── globalmap.png
│   ├── happines2019.png
│   ├── heatmap.png
│   ├── main.tex
│   ├── ols.png
│   ├── pairplot.png
│   ├── r2.png
│   └── score.png
├── requirements.txt
├── temp-plot.html
├── VisualizingFeatures.pdf
├── WorldHappinessReport.pdf
└── WorldHappinessReport.py


# Language, version, and main file: 
Python 3.8.5
WorldHappinessReport.py
