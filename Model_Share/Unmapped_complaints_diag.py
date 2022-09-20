# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:27:11 2022

@author: RahulKumarSisodia
"""


import pandas as pd
import numpy as np
import os
import re

os.chdir(r'D:\SVAAS_2\SVAAS_healthcare_dictionary')
#Emr
EMR_complaints=pd.read_excel('complaints_diagnosis_cnt_EMR.xlsx',sheet_name='EMR_Compliant')
EMR_diagnosis=pd.read_excel('complaints_diagnosis_cnt_EMR.xlsx',sheet_name='EMR_Diagnosis')
Emr_complaints_diag= EMR_complaints.append(EMR_diagnosis)

#SVAAS
Svaas_complaints=pd.read_excel('complaints_diagnosis_cnt_EMR.xlsx',sheet_name='complaints1_cnt')
Svaas_diagnosis=pd.read_excel('complaints_diagnosis_cnt_EMR.xlsx',sheet_name='diagnosis_cnt')

#################################### Is in #########################################################

#Checking savvas data in EMR data
EMR_complt=list(Emr_complaints_diag['value'].unique())
SVAAS_unmapped_complaints=Svaas_complaints[~Svaas_complaints['complaints'].isin(EMR_complt)]
SVAAS_unmapped_diagnosis=Svaas_diagnosis[~Svaas_diagnosis['diagnosis'].isin(EMR_complt)]

#sending file to csv
#SVAAS_unmapped_complaints.to_csv('SVAAS_unmapped_complaints.csv',index=False)
#SVAAS_unmapped_diagnosis.to_csv('SVAAS_unmapped_diagnosis.csv',index=False)


###################### 28-07-2022 Diagnosis and complanits ###########################################################

################################ compalints+diagnosis ##########################################################
#
#diagnosis=Emr_complaints_diag[Emr_complaints_diag["status"]=="Diagnosis"].set_index('value').to_dict()['unified_name']
complaint1=Emr_complaints_diag.set_index('value').to_dict()['unified_name']

complaint1 = dict(sorted(complaint1.items(), key = lambda t: t[1]))

def complain(string_name):
    final_string=set()
    for i in complaint1.keys():
        if re.search(i,string_name):
            final_string.add(complaint1[i])
            string_name=re.sub(i,"",string_name)
    return ' , '.join(final_string)


Svaas_complaints["mapped_complaints_1"]=Svaas_complaints["complaints"].apply(lambda x: complain(x))
Svaas_diagnosis["mapped_diagnosis_1"]=Svaas_diagnosis["diagnosis"].apply(lambda x: complain(x))




######################################list+dict##############################################################


complaint1_diag=Emr_complaints_diag.set_index('value').to_dict()['unified_name']

complaint1_diag = dict(sorted(list(complaint1_diag.items()), key=lambda t: len(t[0]),reverse=True))

def complain(string_name):
    final_string=set()
    for i in complaint1_diag.keys():
        if re.search(i,string_name):
            final_string.add(complaint1_diag[i])
            string_name=re.sub(i,"",string_name)
    return ' , '.join(final_string)


Svaas_complaints["mapped_complaints_list"]=Svaas_complaints["complaints"].apply(lambda x: complain(x))
Svaas_diagnosis["mapped_diagnosis_list"]=Svaas_diagnosis["diagnosis"].apply(lambda x: complain(x))


Svaas_diagnosis.to_csv('Svaas_diagnosis_mapped_v1.csv',index=False)

Svaas_complaints.to_csv('Svaas_complaints_mapped_v1.csv',index=False)



###################### 02/08/2022 adding compalints and diagnosis data #################
entity_match=pd.read_csv(r'D:\SVAAS_2\SVAAS_healthcare_dictionary\entity_matching_model1_lakh_output_test2.csv')
entity_match=entity_match[['COMPLAINT','DIAGNOSIS']]


entity_match['']=entity_match['COMPLAINT'].str.lower()
entity_match['DIAGNOSIS']=entity_match['DIAGNOSIS'].str.lower()

entity_match['COMPLAINT']=entity_match['COMPLAINT'].str.split(',')
compt_entity_match = entity_match.explode('COMPLAINT')

compt_entity_match1=list(compt_entity_match['COMPLAINT'].unique())
compt_entity_match1 = pd.DataFrame(compt_entity_match1)
compt_entity_match1.columns=['complaints']

compt_entity_match1.to_csv('compt_entity_match1.csv',index=False)



entity_match['DIAGNOSIS']=entity_match['DIAGNOSIS'].str.split(',')
diag_entity_match= entity_match.explode('DIAGNOSIS')

diag_entity_match1=list(diag_entity_match['DIAGNOSIS'].unique())
diag_entity_match1 = pd.DataFrame(diag_entity_match1)
diag_entity_match1.columns=['diagnosis']
diag_entity_match1.to_csv('diag_entity_match1.csv',index=False)

################### Opening file complaints and diagnosis count ##################df=
diag_all=pd.read_excel('complaints_diagnosis_cnt.xlsx',sheet_name='EMR_Diagnosis')
diag_all.sort_values(by='value',inplace=True)
diag_all.drop_duplicates('value',keep='first',inplace=True)
#diag_all.to_csv('diag_all.csv',index=False)

comp_all=pd.read_excel('complaints_diagnosis_cnt.xlsx',sheet_name='EMR_Compliant')
comp_all.sort_values(by='value',inplace=True)
comp_all.drop_duplicates('value',keep='first',inplace=True)
#comp_all.to_csv('comp_all.csv',index=False)



##############################04-08-20222##############################################################
#complanits full list
full_list_comp=pd.read_excel('Full_list.xlsx',sheet_name='Complaint')
full_list_comp.drop('Unnamed: 0',axis=1,inplace=True)
full_list_comp['Words']=full_list_comp['Words'].str.split(',')
full_list_comp= full_list_comp.explode('Words')
full_list_comp.sort_values(by='Words',inplace=True)
full_list_comp.drop_duplicates('Words',keep="first",inplace=True)
full_list_comp['Words']=full_list_comp['Words'].str.strip()
full_list_comp=full_list_comp[['Words','Label']]

#EMR complaints and diagnosis
comp_all.columns=['Words','Label','status']
diag_all .columns=['Words','Label','status']       

#Diagnosis_dict
diagnosis_dict=pd.read_csv('Diagnosis_dict.csv')       
diagnosis_dict.drop('Unnamed: 0',axis=1,inplace=True)
diagnosis_dict['Words']=diagnosis_dict['Words'].str.split(',')
diagnosis_dict= diagnosis_dict.explode('Words')

diagnosis_dict.sort_values(by='Words',inplace=True)
diagnosis_dict.drop_duplicates('Words',keep="first",inplace=True)
diagnosis_dict['Words']=diagnosis_dict['Words'].str.strip()
diagnosis_dict=diagnosis_dict[['Words','Label']]

#Appending in complaints
complaints_list=full_list_comp.append(comp_all)
complaints_list.sort_values(by=['Words','Label'],ascending=False)
complaints_list.drop_duplicates('Words',keep="first",inplace=True)
#cleaning complaints
#removing number 
complaints_list['Words'] = complaints_list['Words'].str.replace('\d+', '')
complaints_list['Words'] = complaints_list['Words'].str.lower()
complaints_list['Words'] = complaints_list['Words'].str.strip()
#removing stopswords
complaints_list=complaints_list[complaints_list['Words'].notnull()]

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

complaints_list['Words'] = complaints_list['Words'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
complaints_list['Words'] = complaints_list['Words'].str.strip()
complaints_list.sort_values(by=['Words','Label'], inplace=True)
complaints_list.drop_duplicates(['Words','Label'],keep="first",inplace=True)
complaints_list.drop_duplicates('Words', keep="first",inplace=True)

### diagnosis
diagnosis_list=diagnosis_dict.append(diag_all)
diagnosis_list.drop_duplicates('Words',keep="first",inplace=True)
diagnosis_list['Words'] = diagnosis_list['Words'].str.replace('\d+', '')
diagnosis_list['Words'] = diagnosis_list['Words'].str.lower()
diagnosis_list['Words'] = diagnosis_list['Words'].str.strip()
#removing stopswords
diagnosis_list=diagnosis_list[diagnosis_list['Words'].notnull()]
diagnosis_list['Words'] = diagnosis_list['Words'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
diagnosis_list['Words'] = diagnosis_list['Words'].str.strip()
diagnosis_list.sort_values(by=['Words','Label'], inplace=True)
diagnosis_list.drop_duplicates(['Words','Label'],keep="first",inplace=True)
diagnosis_list.drop_duplicates('Words',keep="first",inplace=True)


# sending files to csv
complaints_list.to_csv('complaints_list.csv',index=False)
diagnosis_list.to_csv('diagnosis_list.csv',index=False)







