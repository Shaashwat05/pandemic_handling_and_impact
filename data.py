import requests

with open("COVIDiSTRESS_May_30_cleaned_final.csv", "wb") as f:
    f.write(requests.get("https://osf.io/download/cjxua/").content)

with open("COVIDiSTRESS_April_27_clean.csv", "wb") as f:
    f.write(requests.get("https://osf.io/download/yp4rv/").content)

with open("COVIDiSTRESS June 17.csv", "wb") as f:
    f.write(requests.get("https://osf.io/download/m5s8d/").content)