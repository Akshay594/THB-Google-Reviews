#!/usr/bin/env python
# coding: utf-8

# ## Agenda: we have to merge couple of these files based on product description/ medicine name so we can add more parameters in main file

# In[1]:


import numpy as np
np.set_printoptions(precision=2)

import pandas as pd


# In[2]:


main = pd.read_csv("med_master_files/csquare-medicine(check).csv", low_memory=False)
main.head(20)


# In[3]:


main.info()


# In[4]:


print("Missing values are: \n\n\n", (main.isna().sum().sort_values(ascending=False)/len(main))*100)


# In[5]:


df_ims = pd.read_csv("med_master_files/IMS Data.csv")
df_ims.head(20)


# In[6]:


df_ims.info()


# ```
# ims2<-unique(ims[,.(BRANDS,SUPERGROUP,`RPM Name`,`eRPM Name`,ACUTE_CHRONIC)])
# ims2<-ims2[BRANDS %in% ims2[,.N,BRANDS][N==1]$BRANDS]
# dt2<-merge(ims2,dt,by.x='BRANDS',by.y='c_brand_name',all.y = T)
# ```

# In[7]:


from collections import Counter

cols = ['BRANDS', 'SUPERGROUP', 'RPM Name', 'eRPM Name', 'ACUTE_CHRONIC']
df_req = df_ims[cols]

df_req = df_req.drop_duplicates()


# In[8]:


df_req


# In[9]:


from collections import Counter

d_dict = Counter(df_req['BRANDS'].values)

print(d_dict)


# In[10]:


# Taking only unique brand values

unique_brands = []

for brand, count in d_dict.items():
    if count == 1:
        unique_brands.append(brand)


# In[11]:


print("Total number of unique brands: ", len(unique_brands))


# In[12]:


# unique_brands_2 = []

# for brand, count in di_req.items():
#     if count == 2:
#         unique_brands_2.append(brand)


# In[13]:


df_final = df_req[df_req['BRANDS'].isin(unique_brands)]
df_final


# ### Extraction of unique brands

# In[14]:


print("Total unique brands: ", len((df_final['BRANDS'].values)))


# In[15]:


print("Total brands in data: ", len(df_req['BRANDS']))


# In[16]:


print("Duplicated brands: ", -len(set(df_final['BRANDS'])) + len(df_req['BRANDS']))


# In[17]:


df_final.head(20)


# In[18]:


df_final.rename(columns={
    'BRANDS':'c_brand_name'
}, inplace=True)


# In[19]:


merged_main = main.merge(df_final, on='c_brand_name', how='outer')
merged_main.head()


# In[20]:


merged_main['ACUTE_CHRONIC'].fillna("NA", inplace=True)


# In[21]:


merged_main_non_null = merged_main[merged_main['ACUTE_CHRONIC'] != "NA"]


# In[22]:


merged_main_non_null.to_csv("merged_main_non_null_09_09_2022.csv", index=False)


# In[23]:


merged_main_na = merged_main[merged_main['ACUTE_CHRONIC'] == "NA"]


# In[26]:


merged_main_na.to_csv("main_after_step_1_v2.csv", index=False)


# ### Removal of the data

# In[26]:


df_final


# In[29]:


df_ims


# In[30]:


cols = ['BRANDS', 'SUPERGROUP', 'RPM Name', 'eRPM Name', 'ACUTE_CHRONIC']
df_req = df_ims[cols]

df_req = df_req.drop_duplicates()


# In[31]:


df_req


# In[32]:


df_final


# In[33]:


brands_full = df_req['BRANDS'].values
brands_removed = df_final['c_brand_name'].values

req_values = set(brands_full) - set(brands_removed)


# In[56]:


df_req = df_ims[df_ims['BRANDS'].isin(req_values)][cols].drop_duplicates()


# In[28]:


df_req.to_csv("ims_after_step_1_v2.csv", index=False)


# In[ ]:




