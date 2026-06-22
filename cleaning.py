import pandas as pd #using pd for reading the csv file and cleaning the data
import numpy as np #using np for for numerical operations
df = pd.read_csv('enhanced_health_insurance_claims.csv')

# 1- DATA UNDERSTANDING

df.head() # top 5 rows
df.tail() # bottom 5 rows
df.shape # number of rows and columns -- and is a tuple thats why we are not using parentheses
df.columns # displays all column names
df.info() # gives us information about the data types and non-null values
df.describe() # gives us statistical summary of the numerical columns
df.isnull().sum() # checking for missing values in each column

#Handling missing values
missing_pert = df.isnull().sum() / len(df) * 100 # percentage of missing values in each column
missing_pert.sort_values(ascending=False) # sorting the missing values in descending order

#Checking duplicates
df.duplicated().sum() # counting the number of duplicate rows

df['ClaimID'].duplicated().sum()
df['ProviderID'].unique()

df['ProviderSpecialty'].value_counts().head(10)

df.dtypes

df['ClaimDate'] = pd.to_datetime(df['ClaimDate'], errors='coerce') # converting ClaimDate to datetime format, and coercing errors to NaT
df['ClaimAmount']= pd.to_numeric(df['ClaimAmount'], errors='coerce') # converting ClaimAmount to numeric format, and coercing errors to NaN
df['PatientIncome'] = pd.to_numeric(df['PatientIncome'], errors='coerce') # converting PatientIncome to numeric format, and coercing errors to NaN
df['PatientAge'] = pd.to_numeric(df['PatientAge'], errors='coerce') # converting PatientAge to numeric format, and coercing errors to NaN

#2- DATA QUALITY CHECKS

df['ClaimStatus'].value_counts() # checking the unique values in ClaimStatus column
df['ClaimType'].value_counts() # checking the unique values in ClaimType column
df['ProviderSpecialty'].value_counts() # checking the unique values in ProviderSpecialty column
df['ClaimSubmissionMethod'].value_counts() # checking the unique values in ClaimSubmissionMethod column
df['PatientGender'].value_counts() # checking the unique values in PatientGender column
df['PatientAge'].describe()
df.nlargest(5,'PatientAge') # checking the top 5 largest values in PatientAge column
df['ClaimAmount'].describe()

# 3- Outlier Detection and Handling
Q1 = df['ClaimAmount'].quantile(0.25)
Q3 = df['ClaimAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['ClaimAmount'] < lower_bound) | (df['ClaimAmount'] > upper_bound)]
print("Outliers :", len(outliers))

#4- Date Analysis 
print(df['ClaimDate'].min()) # earliest claim date
print(df['ClaimDate'].max()) # latest claim date

df['ClaimDate'].dt.year.value_counts().sort_index()
df['ClaimDate'].dt.month.value_counts().sort_index()

#5- Feature Engineering

df['Year'] = df['ClaimDate'].dt.year
df['Quater']= df['ClaimDate'].dt.quarter
df['MonthNo'] = df['ClaimDate'].dt.month
df['Month'] = df['ClaimDate'].dt.month_name()

#---age groups
df['AgeGroup']= pd.cut(
    df['PatientAge'],
    bins = [0,18,35,50,65,100],
    labels=[
        '0-18','19-35','36-50','51-65','65+'
    ]
)

#--- Income bands
df['IncomeBand'] = pd.qcut(
    df['PatientIncome'],
    q=4,
    labels =[
        'Low','Medium','High','Very High'
    ]
)

#6- KPI

total_claims = len(df)
total_claim_amount = df['ClaimAmount'].sum()
average_claim_amount = df['ClaimAmount'].mean()
max_claim_amount = df['ClaimAmount'].max()
approval_rate = ((df['ClaimStatus']=='Approved').mean())*100

df.to_csv('cleaned_health_insurance_claims.csv', index=False) # saving the cleaned data to a new csv file