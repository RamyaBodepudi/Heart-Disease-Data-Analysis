# Heart Disease Data Analysis

## Introduction

### Project Title: Heart Disease Data Analysis

### Objective: 
The objective of this project is to analyze a dataset related to heart disease to uncover insights into various factors affecting heart health. By exploring and visualizing the data, we aim to understand the relationships between different health indicators such as age, cholesterol levels, chest pain types, and the presence of heart disease. The findings from this analysis can help in identifying patterns and trends that are significant for diagnosing and preventing heart disease.

### Source
The dataset used in this analysis was downloaded from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease/data).

### Attributes
List and describe the attributes in the dataset:

- **Age**: The age of the patient
- **Sex**: The gender of the patient (M/F)
- **ChestPainType**: Type of chest pain experienced by the patient
- **RestingBP**: Resting blood pressure
- **Cholesterol**: Serum cholesterol level
- **FastingBS**: Fasting blood sugar
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment
- **HeartDisease**: Diagnosis of heart disease (1=yes, 0=no)

## Data Preprocessing

### Loading Data

The first step in our data analysis process is to load the dataset. We will use the `pandas` library in Python to read the dataset from a CSV file. Below is the code snippet used to load the dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

#Upload the dataset
uploaded = files.upload()

# Read the dataset into a DataFrame
df = pd.read_csv('heart_disease.csv')

# Display the first few rows of the DataFrame
df.head()
```
### Handling Missing Values
After loading the dataset, we checked for missing values to ensure data integrity. Here's how we handled missing values:

```python
# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)
```
### Remove Duplicates
After handling missing values, we checked for and removed any duplicate rows from the dataset:
```python
# Remove duplicate rows
duplicate_rows = df.duplicated().any()
print(duplicate_rows)
```
### Correcting Data Types
To ensure that all columns have the correct data type, we converted the relevant columns to the appropriate data type. For example:
```python
# Correcting the Data Types
df = df.astype({'Sex': 'string', 'ChestPainType': 'string', 'RestingECG': 'string', 'ExerciseAngina': 'string',  'ST_Slope': 'string'})
df.dtypes
```
## Exploratory Data Analysis (EDA)

### 5. Distribution of Ages in the Dataset
The distribution of ages in the dataset to understand the age demographics of the patients.
```python
# # Visualizing the distribution of patient ages with a KDE curve and customized aesthetics
plt.figure(figsize=(10, 6))
sns.displot(df['Age'], bins=20, kde=True, color= 'green')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Distribution')
plt.show()
```
### 6. Distribution of Cholesterol Levels Among Patients
The distribution of cholesterol levels among the patients was analyzed to understand how cholesterol is distributed.
```python
# Plotting the distribution of cholesterol levels with histogram and KDE curve, including customized figure size and color
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Cholesterol', bins=20, kde=True, color='blue')
plt.title('Distribution of Cholesterol Levels')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```
### 7. Count of People with and without Heart Disease
The count of people who have heart disease versus those who do not was visualized.
```python
# Counting the occurrences of heart disease cases
count_heart_diseases = df['HeartDisease'].value_counts()
print(count_heart_diseases)

# Visualizing the count of heart disease cases with a countplot, using custom colors
plt.figure(figsize=(10, 6))
sns.countplot(x='HeartDisease', data=df, palette=['#00aaff', '#aa00ff'])
plt.title('Count of Heart Disease Cases')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()
```
### 8. Count of Male & Female in the Dataset
The gender distribution in the dataset was analyzed.
```python
# Counting the occurrences of male and female patients
Count_of_male_female = df['Sex'].value_counts()
print(Count_of_male_female)

# Visualizing the gender distribution by heart disease status using a countplot with custom labels and legend
sns.countplot(x='Sex', hue='HeartDisease', data= df)
plt.xticks(['M',"F"], ['Male', "Female"])
plt.legend(labels=['No-Disease', 'Disease'])
plt.title('Gender Distribution by Heart Disease Status')
plt.xlabel("Sex")
plt.ylabel('Count')
plt.show()
```
### 9. Gender Distribution According to Heart Diseases
The gender distribution by heart disease status was visualized.
```python
# Counting the occurrences of different chest pain types
chest_pain_counts = df['ChestPainType'].value_counts()
print(chest_pain_counts)

# Visualizing the frequency distribution of chest pain types using a barplot with a muted color palette
sns.barplot(x=chest_pain_counts.values, y=chest_pain_counts.index, palette='muted')
plt.title('Frequency Distribution of Chest Pain Types')
plt.xlabel('Count')
plt.ylabel('Chest Pain Type')
plt.show()
```
### 10. Frequency Distribution of Chest Pain Types
The frequency distribution of different chest pain types in the dataset was explored.
```python
# Counting the occurrences of different chest pain types
chest_pain_counts = df['ChestPainType'].value_counts()
print(chest_pain_counts)

# Visualizing the frequency distribution of chest pain types using a barplot with a muted color palette
sns.barplot(x=chest_pain_counts.values, y=chest_pain_counts.index, palette='muted')
plt.title('Frequency Distribution of Chest Pain Types')
plt.xlabel('Count')
plt.ylabel('Chest Pain Type')
plt.show()
```
### 11. Relationship Between Age and Cholesterol Levels
The Pearson correlation coefficient to understand the relationship between age and cholesterol levels.
```python
# Calculating the Pearson correlation coefficient between age and cholesterol levels
correlation = df['Age'].corr(df['Cholesterol'])
print(correlation)

# Visualizing the relationship between age and cholesterol levels with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Cholesterol', data = df)
plt.title('Relation between age and cholesterol levels')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.grid(True)
plt.show()
```
### 12. Variation of Maximum Heart Rate (MaxHR) with Age
The variation of maximum heart rate with age was analyzed.
```python
# Define bins and labels for age groups
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Box plot for age groups
plt.figure(figsize=(12, 7))
sns.boxplot(x='AgeGroup', y='MaxHR', data=df, palette='Set3')
plt.title('Box Plot of MaxHR by Age Group')
plt.xlabel('Age Group')
plt.ylabel('MaxHR')
plt.xticks(rotation=45)
plt.show()
```
### 13. Variation of Resting Blood Pressure Across Different Chest Pain Types
The variation of resting blood pressure across different chest pain types was examined.
```python
# Visualizing the variation of resting blood pressure across different chest pain types using a violin plot
plt.figure(figsize=(12, 7))
sns.violinplot(x='RestingBP', y='ChestPainType', data = df, palette='Set2')
plt.title('Resting Blood Pressure by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Resting Blood Pressure')
plt.show()
```
### 14. Percentage of Patients Experiencing Exercise-Induced Angina
The percentage of patients who experience exercise-induced angina was calculated.
```python
# Calclate total Number of patients
total_patients = len(df)
print(total_patients)

# Calculate the number of patients with exercise-induced angina
angina_patients = df['ExerciseAngina'].value_counts()['Y']
print(angina_patients)

# Calculate the percentage 
percentage_angina = (angina_patients / total_patients) * 100
print(percentage_angina)
```
### 15. Distribution of ST Depression Induced by Exercise (Oldpeak)
The distribution of ST depression induced by exercise relative to rest was analyzed.
```python
# Calculating the mean Oldpeak for different chest pain types
mean_oldpeak = df.groupby('ChestPainType')['Oldpeak'].mean().reset_index()
plt.figure(figsize=(10, 6))

# Visualizing the mean Oldpeak for different chest pain types using a bar plot
sns.barplot(x='ChestPainType', y='Oldpeak', data=mean_oldpeak, palette='Set2')
plt.title('Mean Oldpeak for Different Chest Pain Types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Mean Oldpeak')
plt.show()
```
### 16. Relationship Between Heart Disease and Other Risk Factors
The relationship between heart disease and other risk factors such as age, cholesterol, and resting blood pressure was explored.
```python
# Selecting columns of interest
columns_of_interest = ['Age', 'Cholesterol', 'RestingBP', 'HeartDisease']

# Creating a pair plot of heart disease and risk factors
sns.pairplot(df[columns_of_interest], hue='HeartDisease', palette='Set1')
plt.suptitle("PPair Plot of Heart Disease and Risk Factors")
plt.show()
```
## Interpretation and Insights

The analysis of the heart disease dataset revealed several key insights. There is a mild positive correlation between age and cholesterol levels, suggesting a potential health risk with increasing age. The distribution of chest pain types varies, indicating diverse symptomatology among patients. Notably, a percentage of patients experience exercise-induced angina, warranting careful monitoring during physical activities. The prevalence of heart disease underscores the importance of early detection and preventive measures. Regular monitoring of vital signs such as cholesterol levels and resting blood pressure is essential for at-risk individuals. Lifestyle modifications, including regular exercise and a healthy diet, are recommended to reduce the risk of heart disease. Overall, these insights emphasize the need for proactive measures and personalized interventions to promote cardiovascular health and mitigate the burden of heart disease.

## Conclusion
The analysis of the heart disease dataset provided valuable insights into the factors affecting heart health. By examining various health indicators such as age, cholesterol levels, and chest pain types, we identified significant patterns and relationships. These findings can be utilized to enhance diagnostic processes and preventive measures for heart disease.

## References
1. Heart Disease Data: [Kaggle Dataset](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease/data)
2. Python: [Python Software Foundation](https://www.python.org/)
3. Pandas: [Pandas Documentation](https://pandas.pydata.org/docs/)
4. Matplotlib: [Matplotlib Documentation](https://matplotlib.org/contents.html)
5. Seaborn: [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
6. NumPy: [NumPy Documentation](https://numpy.org/doc/)
7. Google Colab Notebook: [Heart Disease Analysis Notebook](https://colab.research.google.com/drive/1ItMG7WDv2XrGCajl6zq9P6yi5LuY_dun?authuser=1#scrollTo=zpf5CBn4AaJj)
