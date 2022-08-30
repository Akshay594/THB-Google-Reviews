import pandas as pd
import re
from tqdm import tqdm
import preprocess_kgptalkie as ps

import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup
import urllib.request


final_df = defaultdict(list)

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

def isDigit(char):
    pattern = "^[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$"
    reg = re.compile(pattern)
    if reg.match(char):
        return True
    return False

df = pd.read_csv("google_review_29_08_v3.csv")
df = df[['firstName', 'middleName', 'lastName', 'name', 'baseCity']]
df.drop('middleName', axis=1, inplace=True)

df_not_missing = df.dropna()
df_not_missing['query'] = df_not_missing['firstName'].apply(lambda x:x.lower()) +"+" + df_not_missing['lastName'].apply(lambda x:x.lower()) \
                          +"+"+ df_not_missing['name'].apply(lambda x:x.lower()) + "+" + df_not_missing["baseCity"].apply(lambda x:x.lower())



df_not_missing['query'] = df_not_missing['query'].apply(lambda x:ps.remove_accented_chars(x))
df_not_missing = df_not_missing[['firstName', 'lastName', 'name', 'baseCity','query']]


queries = df_not_missing['query'].values

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

    divs = soup.find_all("span", class_="Aq14fc")
    spans = soup.find_all("span", class_="hqzQac")

    if len(divs) > 0:
        for div, span in zip(divs, spans):
            # Search for a h3 tag
            rating = div.get_text()
            greviews = span.get_text()
            print(greviews)
            greviews = [ps.remove_special_chars(i) for i in greviews.split(" ")]
            greviews = [i for i in greviews if isDigit(i)][0]
            print("Rating: ", rating, "Number of Reviews: ", greviews)
            final_df['firstName'].append(df_not_missing.iloc[i]['firstName'])
            final_df['lastName'].append(df_not_missing.iloc[i]['lastName'])
            final_df['clinicName'].append(df_not_missing.iloc[i]['name'])
            final_df['baseCity'].append(df_not_missing.iloc[i]['baseCity'])
            print(df_not_missing.iloc[i]['baseCity'])
            final_df['GoogleNRs'].append(float(greviews))
            if rating is None:
                print("Appending None!")
                final_df['rating'].append("NA")
            else:
                final_df['rating'].append(rating)
            print(div.get_text())


df_final = pd.DataFrame(final_df)
print(df_final)

pd.merge(df, df_final, how='outer').drop(["clinicName", "query"], axis=1).to_csv("full_output_30_aug_2022.csv", index=False)

print(pd.read_csv("full_output_30_aug_2022.csv"))