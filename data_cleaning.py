import pandas as pd
import numpy as np

def clean(df):
    pass





df = pd.read_csv("COVIDiSTRESS_April_27_clean.csv", encoding= 'unicode_escape')

# Removing rows with more than or equal to 50% values missing 
df = df.loc[df.isna().mean(axis=1)<0.50]

# Augmentnig or Dropping Null values
df.loc[df["Dem_gender"].isna(), "Dem_gender"] = "Other/would rather not say"
df.loc[df["Dem_dependents"].isna(), "Dem_dependents"] = 0
df = df[df['Country'].notna()] # 492 rows
df = df[df['Dem_Expat'].notna()] # 536 rows

# Modifying education column values. If employed then education = Up to 12 years of School
df.loc[df["Dem_edu"] == 'Uninformative response', "Dem_edu"] = np.nan
df["Dem_edu"] = np.where(df["Dem_edu"].isna() and ~df["Dem_employment"].isin(["Not employed", np.nan]), "Up to 12 years of school", "None")

# For employment NA values, if age > avg global retirement age(62, acording to CNBC) then retired, else, drop data (1000 rows)
df.loc[df["Dem_employment"].isna() and df["Dem_age"] >= 62, "Dem_employment"] = "Retired"
df = df[df['Dem_employment'].notna()]
#print(df.loc[df['Dem_gender'].isna()])
