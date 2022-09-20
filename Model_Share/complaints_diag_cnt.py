# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:11:04 2022

@author: RahulKumarSisodia
"""

import pandas as pd
import numpy as np
import os

os.chdir(r'D:\SVAAS_2\SVAAS_healthcare_dictionary')
complaints_diag= pd.read_csv('brdg_appt_conslt_mapping.csv',sep="|")

compt_diag=complaints_diag.copy()

#complaints
compt_diag['complaints']=compt_diag['complaints'].str.lower()
compt_diag['complaints']=compt_diag['complaints'].str.replace(',','&')

# split each & field and create a new row per entry
compt_diag['complaints']=compt_diag['complaints'].str.split('&')
compt_diag1 = compt_diag.explode('complaints')


compt_diag['diagnosis']=compt_diag['diagnosis'].str.lower()


#diagnosis
compt_diag['diagnosis']=compt_diag['diagnosis'].str.lower()
compt_diag['diagnosis']=compt_diag['diagnosis'].str.replace(',','&')

compt_diag['diagnosis']=compt_diag['diagnosis'].str.split('&')
compt_diag2 = compt_diag.explode('diagnosis')



complaints1=compt_diag1.groupby('complaints').agg({'complaints':'count'}).rename(columns={'complaints':'complaints_counts'}).sort_values(by='complaints_counts',ascending=False)

diagnosis=compt_diag2.groupby('diagnosis').agg({'diagnosis':'count'}).rename(columns={'diagnosis':'diagnosis_counts'}).sort_values(by='diagnosis_counts',ascending=False)
complaints1=complaints1.reset_index()
diagnosis=diagnosis.reset_index()

#complaints1.to_csv('complaints1_cnt.csv')
#diagnosis.to_csv('diagnosis.csv')

######### Date 27-07-2022 #########
#complaints count by insurance partner name
compt_diag1.columns
complaints_ins=compt_diag1.groupby(['complaints','insurance_company_name']).agg({'complaints':'count'}).rename(columns={'complaints':'complaints_counts'}).sort_values(by='complaints_counts',ascending=False)
diagnosis_ins=compt_diag2.groupby(['diagnosis','insurance_company_name']).agg({'diagnosis':'count'}).rename(columns={'diagnosis':'diagnosis_counts'}).sort_values(by='diagnosis_counts',ascending=False)

complaints_ins.to_csv('complaints_ins.csv')
diagnosis_ins.to_csv('diagnosis_ins.csv')

########### Policy name #######################################################
#According policy holder complaint and diagnosis list
policy=pd.read_csv('dim_policy.csv')
#complaints
brdg_appt_cnslt_mapping=compt_diag1[['brdg_appt_conslt_sid','policy_sid','patient_sid','complaints','diagnosis']]
merge1= pd.merge(brdg_appt_cnslt_mapping,policy,on='policy_sid',how='outer')
#diagnosis
brdg_appt_cnslt_mapping1=compt_diag2[['brdg_appt_conslt_sid','policy_sid','patient_sid','complaints','diagnosis']]
merge2= pd.merge(brdg_appt_cnslt_mapping1,policy,on='policy_sid',how='outer')



#According policy holder complaint and diagnosis list

policy_name_com=merge1.groupby(['policy_name','complaints']).agg({'complaints':'count'}).rename(columns={'complaints':'complaints_counts'}).sort_values(by='complaints_counts',ascending=False)
policy_name_diag=merge2.groupby(['policy_name','diagnosis']).agg({'diagnosis':'count'}).rename(columns={'diagnosis':'diagnosis_counts'}).sort_values(by='diagnosis_counts',ascending=False)

policy_name_com.to_csv('policy_name_com.csv')
policy_name_diag.to_csv('policy_name_diag.csv')
