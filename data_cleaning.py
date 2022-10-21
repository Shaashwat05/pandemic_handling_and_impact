import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics

'''
3. Isolation - 1000 NA values
'''
'''
KEY OBSERVATIONS
1. Risk group column highly biased - 77k yes, 27k no, 9k not sure ( normally bias in labels affect model, what about bias in columns?)
2. 66 percent accuracy of the ridge classifier for marital status missing values (tried with employment, gender, education)
3. isolation adults has 19000 missing values - remove column
4. isolation children has 20000 missing values - remove column (might need)
'''

def clean(df):
    # Dropping uninformative columns
    df = df.drop(columns=['Dem_edu_mom', 'UserLanguage', 'Dem_state', "Dem_isolation_kids"])

    # Removing rows with more than or equal to 50% values missing 
    df = df.loc[df.isna().mean(axis=1)<0.50]

    # Augmentnig or Dropping Null values
    df.loc[df["Dem_gender"].isna(), "Dem_gender"] = "Other/would rather not say"
    df.loc[df["Dem_dependents"].isna(), "Dem_dependents"] = 0
    df = df[df['Country'].notna()] # 492 rows
    df = df[df['Dem_Expat'].notna()] # 536 rows
    df = df[df['Dem_riskgroup'].notna()] # 455 rows
    df = df[df['Dem_riskgroup'].notna()] # 455 rows

    # Modifying education column values. If employed then education = Up to 12 years of School
    df.loc[df["Dem_edu"] == 'Uninformative response', "Dem_edu"] = np.nan
    df.loc[df["Dem_edu"].isna() & ~df["Dem_employment"].isin(["Not employed", np.nan]), "Dem_edu"] = "Up to 12 years of school"
    df.loc[df["Dem_edu"].isna() & df["Dem_employment"].isin(["Not employed", np.nan]), "Dem_edu"] = "None"

    # For employment NA values, if age > avg global retirement age(62, acording to CNBC) then retired, else, drop data (1000 rows)
    df.loc[df["Dem_employment"].isna() & df["Dem_age"] >= 62, "Dem_employment"] = "Retired"
    df = df[df['Dem_employment'].notna()]
    #print(df.loc[df['Dem_gender'].isna()])

    # Missing value handling for isolation - whenever isolation_adults in NA, isolation - life carries on
    df.loc[df["Dem_islolation"].isna() & df["Dem_isolation_adults"].isna(), "Dem_islolation"] = "Life carries on with minor changes"
    df = df[df['Dem_islolation'].notna()] # approx 1000 rows

    # Creating a Ridge Classifier for marital status missing values
    y = pd.Categorical(df['Dem_maritalstatus']).codes
    lr = RidgeClassifier()
    lr.fit(df[['Dem_age', 'Dem_dependents']],y)
    df.loc[df['Dem_maritalstatus'].isna(), 'Dem_maritalstatus'] = lr.predict(df[['Dem_age', 'Dem_dependents']])[df['Dem_maritalstatus'].isna()]
    #pred = lr.predict(dummy_df)
    #print(metrics.accuracy_score(y, pred))

    return df

df = pd.read_csv("COVIDiSTRESS_April_27_clean.csv", encoding= 'unicode_escape')
df = clean(df)
print(df['Dem_edu'].head(50))