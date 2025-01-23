![image 15](https://github.com/user-attachments/assets/5653148c-a0a8-4316-a94d-57fd457f5b5f)
![image 14](https://github.com/user-attachments/assets/2d52991a-6ea6-4064-8b5a-c3c05719cb69)
![image 13](https://github.com/user-attachments/assets/e245b2f0-d5ab-4c19-9b3d-685182d8b424)
![image 12](https://github.com/user-attachments/assets/e0b12857-5737-4fa0-a02e-7878ddec1db3)
![image 11](https://github.com/user-attachments/assets/9a568d8e-3dd5-42cd-9048-246f15fe919e)
![image 10](https://github.com/user-attachments/assets/b00d3575-3241-4958-8f4f-7b44dfeb0708)
![image 9](https://github.com/user-attachments/assets/b2c510d2-114f-4cdd-a6b3-56e16a9cf30a)
![image 8](https://github.com/user-attachments/assets/9d59a7cb-00e9-4c0c-93b7-9b115efef901)
![image 7](https://github.com/user-attachments/assets/640110db-f439-4888-88d5-09d6912d6c21)
![image 6](https://github.com/user-attachments/assets/2377a32a-d391-46ae-a243-66cd2b0ffcec)
![image 5](https://github.com/user-attachments/assets/fde5d8bb-76af-4054-8753-87f6d0a35c9e)
![image 4](https://github.com/user-attachments/assets/6aa1ceb0-443f-46ef-bad8-6f1186ab3d3f)
![image 3](https://github.com/user-attachments/assets/23a1459a-7eea-41df-8dc3-ff100b69899d)
![image 2](https://github.com/user-attachments/assets/fc22b7b6-e323-4d39-8b27-f8ac4522f7d8)
![Image 1](https://github.com/user-attachments/assets/82f4830d-e2e4-4abb-aabb-579cc83d8a06)
# Heart-and-Disease-Prediction..
Using machine learning for disease prediction involves teaching computers to study lots of medical information to guess if someone might get sick. For example, with heart disease prediction using machine learning, computers can look at factors like age, blood pressure, and cholesterol levels to guess who might have heart problems in the future. This helps doctors catch issues early and keep people healthy.

Importing Necessary Libraries Data Loading Plotting Librariesimport pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import cufflinks as cf %matplotlib inline

Metrics for Classification techniquefrom sklearn.metrics import classification_report,confusion_matrix,accuracy_score

Scalerfrom sklearn.preprocessing import StandardScaler from sklearn.model_selection import RandomizedSearchCV, train_test_split

Model buildingfrom xgboost import XGBClassifier from catboost import CatBoostClassifier from sklearn.ensemble import RandomForestClassifier from sklearn.neighbors import KNeighborsClassifier from sklearn.svm import SVC

Data Loading Here we will be using the pandas read_csv function to read the dataset. Specify the location of the dataset and import them.

Importing Datadata = pd.read_csv(“heart.csv”) data.head(6) # Mention no of rows to be displayed from the top in the argument

Output:

Exploratory Data Analysis Now, let’s see the size of the datasetdata.shape

Output:(303, 14)

Inference: We have a dataset with 303 rows which indicates a smaller set of data.

As above we saw the size of our dataset now let’s see the type of each feature that our dataset holds.

Python Code:

Inference: The inference we can derive from the above output is:

Out of 14 features, we have 13 int types and only one with the float data types. Woah! Fortunately, this dataset doesn’t hold any missing values. As we are getting some information from each feature so let’s see how statistically the dataset is spread.data.describe()

Output:

Exploratory Data Analysis It is always better to check the correlation between the features so that we can analyze that which feature is negatively correlated and which is positively correlated so, Let’s check the correlation between various features.plt.figure(figsize=(20,12)) sns.set_context(‘notebook’,font_scale = 1.3) sns.heatmap(data.corr(),annot=True,linewidth =2) plt.tight_layout()

Output:

output , heart disease prediction using Machine learning By far we have checked the correlation between the features but it is also a good practice to check the correlation of the target variable.

So, let’s do this!sns.set_context(‘notebook’,font_scale = 2.3) data.drop(‘target’, axis=1).corrwith(data.target).plot(kind=’bar’, grid=True, figsize=(20, 10), title=”Correlation with the target feature”) plt.tight_layout()

Output:

Correlation with the Target Feature , Inference: Insights from the above graph are:

Four feature( “cp”, “restecg”, “thalach”, “slope” ) are positively correlated with the target feature. Other features are negatively correlated with the target feature. So, we have done enough collective analysis now let’s go for the analysis of the individual features which comprises both univariate and bivariate analysis.

Age(“age”) Analysis Here we will be checking the 10 ages and their counts.plt.figure(figsize=(25,12)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values) plt.tight_layout()

Output:

Age Analysis| Heart Disease Prediction Inference: Here we can see that the 58 age column has the highest frequency.

Let’s check the range of age in the dataset.minAge=min(data.age) maxAge=max(data.age) meanAge=data.age.mean() print(‘Min Age :’,minAge) print(‘Max Age :’,maxAge) print(‘Mean Age :’,meanAge)

Output:

Output | Heart Disease Prediction Min Age : 29 Max Age : 77 Mean Age : 54.366336633663366

We should divide the Age feature into three parts – “Young”, “Middle” and “Elder”Young = data[(data.age>=29)&(data.age<40)] Middle = data[(data.age>=40)&(data.age<55)] Elder = data[(data.age>55)] plt.figure(figsize=(23,10)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=[‘young ages’,’middle ages’,’elderly ages’],y=[len(Young),len(Middle),len(Elder)]) plt.tight_layout()

Output:

Heart Disease Prediction Inference: Here we can see that elder people are the most affected by heart disease and young ones are the least affected.

To prove the above inference we will plot the pie chart.colors = [‘blue’,’green’,’yellow’] explode = [0,0,0.1] plt.figure(figsize=(10,10)) sns.set_context(‘notebook’,font_scale = 1.2) plt.pie([len(Young),len(Middle),len(Elder)],labels=[‘young ages’,’middle ages’,’elderly ages’],explode=explode,colors=colors, autopct=’%1.1f%%’) plt.tight_layout()

Output:

Sex(“sex”) Feature Analysis Sex feature analysis | Heart Disease Prediction plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’]) plt.tight_layout()

Output:

Inference: Here it is clearly visible that, Ratio of Male to Female is approx 2:1.

Now let’s plot the relation between sex and slope.plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’],hue=data[“slope”]) plt.tight_layout()

Output:

Output of Sex Analysis, Inference: Here it is clearly visible that the slope value is higher in the case of males(1).

Chest Pain Type(“cp”) Analysis plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘cp’]) plt.tight_layout()

Output:

Chest Pain Inference: As seen, there are 4 types of chest pain

status at least condition slightly distressed condition medium problem condition too bad Analyzing cp vs target column

Heart Disease Prediction Inference: From the above graph we can make some inferences,

People having the least chest pain are not likely to have heart disease. People having severe chest pain are likely to have heart disease. Elderly people are more likely to have chest pain.

Thal Analysis plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘thal’]) plt.tight_layout()

Output:

Thal Analysis Target plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘target’]) plt.tight_layout()

Output:

Target | Heart Disease Prediction Inference: The ratio between 1 and 0 is much less than 1.5 which indicates that the target feature is not imbalanced. So for a balanced dataset, we can use accuracy_score as evaluation metrics for our model.

Feature Engineering Now we will see the complete description of the continuous data as well as the categorical datacategorical_val = [] continous_val = [] for column in data.columns: print(“——————–“) print(f”{column} : {data[column].unique()}”) if len(data[column].unique()) <= 10: categorical_val.append(column) else: continous_val.append(column)

Output:

Feature Engineering Output | Heart Disease Prediction Now here first we will be removing the target column from our set of features then we will categorize all the categorical variables using the get dummies method which will create a separate column for each category suppose X variable contains 2 types of unique values then it will create 2 different columns for the X variable.categorical_val.remove(‘target’) dfs = pd.get_dummies(data, columns = categorical_val) dfs.head(6)

Output:

Output | Heart Disease Prediction Now we will be using the standard scaler method to scale down the data so that it won’t raise the outliers also dataset which is scaled to general units leads to having better accuracy.sc = StandardScaler() col_to_scale = [‘age’, ‘trestbps’, ‘chol’, ‘thalach’, ‘oldpeak’] dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale]) dfs.head(6)

Output:

Output | Heart Disease Prediction Modeling Splitting our DatasetX = dfs.drop(‘target’, axis=1) y = dfs.target X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

The KNN Machine Learning Algorithm knn = KNeighborsClassifier(n_neighbors = 10) knn.fit(X_train,y_train) y_pred1 = knn.predict(X_test) print(accuracy_score(y_test,y_pred1))

Output:0.8571428571428571

Conclusion on Heart Disease Prediction

We did data visualization and data analysis of the target variable, age features, and whatnot along with its univariate analysis and bivariate analysis.

We also did a complete feature engineering part in this article which summons all the valid steps needed for further steps i.e. model building.

From the above model accuracy, KNN is giving us the accuracy which is 89%.
Conclusion Heart disease prediction using machine learning utilizes algorithms to analyze medical data like age, blood pressure, and cholesterol levels, aiding in early detection and prevention. Machine learning greatly enhances disease prediction by analyzing large datasets, identifying patterns, and making accurate forecasts, ultimately improving healthcare outcomes and saving lives.
