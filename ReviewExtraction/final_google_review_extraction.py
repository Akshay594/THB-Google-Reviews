# %%
import pandas as pd
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import locationtagger



import nltk
import spacy
 
# essential entity models downloads
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %%
 
# initializing sample text
 
# extracting entities.

text = "Shop No 1 Kaveri Apartments Ramghat Rd opposite Meenakshi Cinema Sudhama Puri Aligarh Uttar Pradesh 202001"

place_entity = locationtagger.find_locations(text=text)
# getting all countries
print("The countries in text : ")
print(place_entity.countries)
 
# getting all states
print("The states in text : ")
print(place_entity.regions)
 
# getting all cities
print("The cities in text : ")
print(place_entity.cities)

# %%
df = pd.read_csv("google_review_29_08_v3.csv")
df.head()

# %%
df = df[['SVAAS_ID','firstName', 'middleName', 'lastName', 'name', 'baseCity']]
df.head()

# %%
df.isna().sum()

# %%
df.describe()

# %%
df.info()

# %% [markdown]
# ### Let's calculate the % of missing values

# %%
(df.isna().sum())/len(df) * 100

# %% [markdown]
# It's apparent that middleName columns has the most number of missing values here. 

# %%
plt.figure(figsize=(16, 16))
df['middleName'].value_counts().plot.bar()

# %% [markdown]
# Kumar is the most common middle name we can observe here.

# %%
df[df['firstName'].isna() == False]

# %% [markdown]
# ### I think, it will be safe to drop the middleName column, and we can proceed with the firstName only here.

# %%
df.drop('middleName', axis=1, inplace=True)

# %% [markdown]
# ## Let's seggregate the data based on missing vs non-missing row values

# %%
df_not_missing = df.dropna()

# %%
df_not_missing.head()

# %%
df_not_missing.isna().sum()

# %%
df_not_missing.info()

# %%
print(len(df_not_missing))

# %%
len(df_not_missing)/len(df)

# %%
df_not_missing['query'] = df_not_missing['firstName'].apply(lambda x:x.lower()) +"+" + df_not_missing['lastName'].apply(lambda x:x.lower()) \
                          +"+"+ df_not_missing['name'].apply(lambda x:x.lower()) + "+" + df_not_missing["baseCity"].apply(lambda x:x.lower())

# %%
df_not_missing['query']

# %% [markdown]
# ## For missing data

# %%
import preprocess_kgptalkie as ps
import re

def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x




# %%
df_not_missing['query'] = df_not_missing['query'].apply(lambda x:ps.remove_accented_chars(x))
df_not_missing['query'].head()

# %%
len(df_not_missing['query'].values)

# %%
from collections import defaultdict

final_df = defaultdict(list)

# %%
df_not_missing

# %%
# df_ch = df_not_missing[df_not_missing['baseCity'] == 'Chennai']
# df_ch.isna().sum()

# %%
def isDigit(char):
    pattern = "^[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$"
    reg = re.compile(pattern)
    if reg.match(char):
        return True
    return False

# %%
from bs4 import BeautifulSoup
import urllib.request
import numpy as np

def extractReviews(df_not_missing, limit=100):
    queries = df_not_missing['query'].values[:limit]

    for i in tqdm(range(len(queries))):
        query = queries[i]
        query = "+".join(query.split(" "))
        url = 'https://google.com/search?q='+query
        print(url)

        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36')
        raw_response = urllib.request.urlopen(request).read()

        # Read the repsonse as a utf-8 string
        html = raw_response.decode("utf-8")

        soup = BeautifulSoup(html, 'html.parser')

        try:
            rating = soup.find("span", class_="Aq14fc")
            review = soup.find("span", class_="hqzQac")
            address = soup.find("span", class_="LrzXr").get_text()
            phone = soup.find("span", class_="LrzXr zdqRlf kno-fv").get_text()
            speciality = soup.find("span", class_="YhemCb").get_text()
            working_hours = soup.find("table", class_="WgFkxc").get_text()
            
            print("address: ", address)
            print("phone: ", phone)
            print("spec: ", speciality)

            rating = rating.get_text()
            greviews = review.get_text()
            print(greviews)
            greviews = [ps.remove_special_chars(i) for i in greviews.split(" ")]
            greviews = [i for i in greviews if isDigit(i)][0]
            
            
            final_df['firstName'].append(df_not_missing.iloc[i]['firstName'])
            final_df['lastName'].append(df_not_missing.iloc[i]['lastName'])
            final_df['clinicName'].append(df_not_missing.iloc[i]['name'])
            final_df['baseCity'].append(df_not_missing.iloc[i]['baseCity'])
            final_df['SVAAS_ID'].append(df_not_missing.iloc[i]['SVAAS_ID'])

            if greviews:
                final_df['GoogleNRs'].append(float(greviews))
                print("GNumber of google reviews: ", greviews)
            else:
                final_df['GoogleNRs'].append("NA")
                print("Google reviews are not available.")
            
            if working_hours:
                final_df['workingHours'].append(working_hours)
                print("working hours: ", working_hours)
            else:
                print("Working hours are not available.")
                final_df['workingHours'].append("NA")
            
            

            if rating:
                final_df['rating'].append(rating)
                print("Rating is :", rating)
            else:
                final_df['rating'].append("NA")
                print("Rating is not available.")

            if address:
                final_df['address'].append(address)
                print("Address: ", address)
            else:
                final_df['address'].append("NA")
                print("Address is not available.")

            if phone:
                final_df['phoneNo'].append(phone)
                print("Phone number: ", phone)
            else:
                final_df['phoneNo'].append("NA")
                print("Phone number is not availble.")


            if speciality:
                final_df['speciality'].append(speciality)
                print("Speciality: ", speciality)
            else:
                final_df['speciality'].append("NA")
                print("Speciality not available")


            text = ps.remove_accented_chars(address)
            text = ps.remove_special_chars(text)
            text = ps.remove_html_tags(text)

            text = text.split(" ")
            print(text)
            print(text[-4:][:-1])
            city = " ".join(text[-4:][:-1])


            
            if city:
                print("Extracted city: ", city)
                final_df['extractedCity'].append(city)
            else:
                print("Couldn't extract the city.")
                final_df["extractedCity"].append("NA")
            
        except Exception as e:
            print("Error was: ", e)
                

    return pd.DataFrame(final_df)

    #             print(div.get_text())


# %%
# final_df = extractReviews(df_not_missing, limit=len(df_not_missing))

# %%
# pd.DataFrame(final_df).to_csv("completed_google_review_7_sep_2022.csv", index=False)

# %%
df1 = pd.read_csv("completed_google_review_7_sep_2022.csv")
df1.head()


# %% [markdown]
# ```
# dt[,baseCity_Raw:=baseCity]
# dt[baseCity=='Gurgaon',baseCity:='Gurgaon|Gurugram|Delhi|Manesar|Ghaziabad|Faridabad']
# dt[baseCity=='Hyderabad',baseCity:='Hyderabad|Telangana']
# dt[baseCity=='Bangalore',baseCity:='Bangalore|Bengaluru']
# dt[baseCity=='New Delhi',baseCity:='Delhi|Manesar|Ghaziabad|Gurgaon|Gurugram|Faridabad']
# dt[baseCity=='Mumbai',baseCity:='Mumbai|Pune|Maharashtra|Thane']
# dt[baseCity=='Pune',baseCity:='Mumbai|Pune|Maharashtra|Thane']
# dt[baseCity=='Thane',baseCity:='Mumbai|Pune|Maharashtra|Thane']
# dt[baseCity=='Navi Mumbai',baseCity:='Mumbai|Pune|Maharashtra|Thane']
# ```

# %%
df1['extractedCityMod'] = df1['extractedCity'].apply(lambda x: x.lower()).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("gurgaon", "gurugram")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("manesar", "gurugram")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("ghaziabad", "gurugram")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("faridabad", "gurugram")).values

df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("telangana", "hyderabad")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("sangareddy", "hyderabad")).values

df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("tamil nadu", "chennai")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("thoraipakkam", "chennai")).values

df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("bengaluru", "bangalore")).values


df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("bengaluru", "bangalore")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("kasavanahalli", "bangalore")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("karnataka", "bangalore")).values

df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("gurugram", "delhi")).values

df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("pune", "mumbai")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("maharashtra", "mumbai")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("thane", "mumbai")).values
df1['extractedCityMod'] = df1['extractedCityMod'].apply(lambda x: x.replace("navi mumbai", "mumbai")).values

# %%
df1['baseCityMod'] = df1['baseCity'].apply(lambda x: x.lower()).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("gurgaon", "gurugram")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("manesar", "gurugram")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("ghaziabad", "gurugram")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("faridabad", "gurugram")).values

df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("telangana", "hyderabad")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("sangareddy", "hyderabad")).values

#thoraipakkam tamil nadu  

df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("tamil nadu", "chennai")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("thoraipakkam", "chennai")).values

df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("bengaluru", "bangalore")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("kasavanahalli", "bangalore")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("karnataka", "bangalore")).values

df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("gurugram", "delhi")).values

df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("pune", "mumbai")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("maharashtra", "mumbai")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("thane", "mumbai")).values
df1['baseCityMod'] = df1['baseCityMod'].apply(lambda x: x.replace("navi mumbai", "mumbai")).values

# %%
baseCity = df1['baseCityMod'].values
exCity = df1['extractedCityMod'].values

res = []
for i in range(len(df1)):
    l1 = baseCity[i].split()
    l2 = exCity[i].split()
    
    if set(l1).intersection(set(l2)):
        res.append(True)
    else:
        res.append(False)

# %%
df1['flag'] = res

# %%
df1

# %%
df1[df1['flag'] == False].groupby("extractedCityMod")['baseCityMod'].value_counts()

# %%
len(df1[df1['flag'] == False])

# %%
"SVAAS_ID	firstName	lastName	name	baseCity".split()

# %%
pd.merge(df1, df, how="outer").to_csv("complete_google_review_with_flag_08_09_22_v1.csv", index=False)

# %%



