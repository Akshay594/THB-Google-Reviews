#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install scispacy negspacy memory_profiler\n')


# In[ ]:


import spacy
import scispacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from negspacy.negation import Negex
from scispacy.linking import EntityLinker
import pandas as pd  
import numpy as np
from spacy.lang.en import English
import re
from memory_profiler import profile
import time
import sys
import psutil 
import warnings
from pandas.core.common import SettingWithCopyWarning


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


pd.read_csv("dt.csv").head(10).columns


# In[ ]:


pd.read_excel('Validation_Dataset_10.xlsx',engine='openpyxl',usecols='A,B,C,D,E').columns


# In[ ]:


extractor_to_use = 5
# Functions needed for entity extraction models
f=open('acronyms.txt','r')
acronyms_dict={}
l=f.readline()
l=f.readline()
while l:
    l1=l.split("\t")
    acronyms_dict[l1[0].lower()]=l1[1].lower()
    l=f.readline()
f.close()


# In[ ]:


#lemmatizing the notes to capture all forms of negation(e.g., deny: denies, denying)
def lemmatize(note, nlp):
    doc = nlp(note)
    lemNote = [wd.lemma_ for wd in doc]
    return " ".join(lemNote)

def acronym_to_full_text(row,acronyms_dict):
    final_text = ""
  
    for word in row.split(" "):
        if word.strip() in acronyms_dict.keys():
            final_text=final_text+acronyms_dict[word.strip()]+" "
        else:
            final_text=final_text+word.strip()+" "
    return final_text.strip()


# In[ ]:


Full_data=pd.ExcelFile(r'Diagnosis N Gram MAX 3 12April22upd2.xlsx',engine='openpyxl')


# In[ ]:


pd.read_excel(Full_data, sheet_name="Unigram")


# In[ ]:


def preprocessing_matching_algorithm():
   # Input file is this diagnosis one where we are taking one, bi, and ngram values.
   # combination helps in making newer words for each word/s and combination/s.
    Full_data=pd.ExcelFile(r'Diagnosis N Gram MAX 3 12April22upd2.xlsx',engine='openpyxl')

    one_gram=pd.read_excel(Full_data,sheet_name='Unigram')

    one_gram=one_gram.dropna()
    one_gram.rename(columns={'Category(Investigation/Complaint/Diagnosis)':'Category(Investigation/Complaint/Diagnosis/Procedure)'},inplace=True)
    one_gram.dropna(subset=['Label'],inplace=True)
    one_gram=one_gram[['word','Category(Investigation/Complaint/Diagnosis/Procedure)','Label']]


    Bigram=pd.read_excel(Full_data,sheet_name='Bigram')
    Bigram['Label']=np.where(Bigram['Label2'].isna(),Bigram['Label1'],Bigram['Label1']+" "+Bigram['Label2'])
    Bigram.rename(columns={'Category(Investigation/Complaint/Diagnosis)':'Category(Investigation/Complaint/Diagnosis/Procedure)'},inplace=True)
    Bigram.dropna(subset=['Label'],inplace=True)
    Bigram=Bigram[['word','Category(Investigation/Complaint/Diagnosis/Procedure)','Label']]

    
    Trigram=pd.read_excel(Full_data,sheet_name='Trigram')
    Trigram.fillna("",inplace=True)
    Trigram['Label']=Trigram['Label1']+" "+Trigram['Label2']+" "+Trigram['Label3']
    Trigram['Label']=Trigram['Label'].str.strip()
    Trigram['len']=Trigram.Label.str.len()
    Trigram=Trigram[Trigram['len']>0]
    Trigram.dropna(subset=['Label'],inplace=True)
    Trigram=Trigram[['word','Category(Investigation/Complaint/Diagnosis/Procedure)','Label']]
    
    n_gram_sheet=one_gram.append(Bigram)
    n_gram_sheet=n_gram_sheet.append(Trigram)
    n_gram_sheet.fillna("",inplace=True)
    n_gram_sheet=n_gram_sheet.replace('nan',"")
    n_gram_sheet['len']=n_gram_sheet['Category(Investigation/Complaint/Diagnosis/Procedure)'].str.len()
    n_gram_sheet=n_gram_sheet[n_gram_sheet['len']>0]
    n_gram_sheet['label']=n_gram_sheet['Category(Investigation/Complaint/Diagnosis/Procedure)']+":"+n_gram_sheet['Label']
    n_gram_sheet.rename(columns={'word':'pattern'},inplace=True)
    
    investigation=pd.read_excel('UnifiedBiomarkerMapping_1.xlsx',engine='openpyxl',sheet_name='Top TestNames')
    investigation.dropna()
    investigation.rename(columns={'lab test names':'pattern'},inplace=True)
    investigation['pattern']=investigation['pattern'].astype(str)

    investigation['label']='Investigation:'+(investigation['pattern'])
    n_gram_sheet=n_gram_sheet.append(investigation)

    drugs=pd.read_csv('final_drug_list.csv')
    drugs.dropna()
    drugs.rename(columns={'value_upd_3':'pattern'},inplace=True)
    drugs['label']='Drugs:'+drugs['pattern']
    print(drugs)
    n_gram_sheet=n_gram_sheet.append(drugs)
    n_gram_sheet=n_gram_sheet[['label','pattern']]

    gram_dict=n_gram_sheet.to_dict("record")
  
    length_gram_dict = len(gram_dict)
    for i in range(0,length_gram_dict):
        pattern = gram_dict[i]['pattern']
        pattern = acronym_to_full_text(pattern,acronyms_dict)
        pattern_list = pattern.split()
        l=[]
        l1 = []
        for j in range(0,len(pattern_list)):
            d={}
            d["LOWER"]=pattern_list[j]
            l.append(d)
            l1.append(d)
            if j!=len(pattern_list)-1:
                l1.append({"IS_PUNCT": True})
        gram_dict.append({}) 
        gram_dict[i]['pattern'] = l
        
        gram_dict[i+length_gram_dict]['pattern'] = l1
        gram_dict[i+length_gram_dict]['label'] = gram_dict[i]['label']
    #print(gram_dict)
    return gram_dict


# In[ ]:


#Reference https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
valid_semantic_types = ['T109','T116','T020','T190','T017','T195','T053','T038','T030','T019','T200','T060','T047','T129','T037','T059','T034','T048','T046','T121','T061','T184','T033','T191' ]
complaint_semantic_types = ['T184','T033']
diagnosis_semantic_types = ['T020','T190','T053','T038','T029','T023','T030','T019','T047','T129','T037','T048','T046','T191']
investigation_semantic_types=['T017','T034','T060','T059','T061']
procedure_semantic_types=[]
drugs_semantic_types = ['T195','T200','T116','T109','T122']


# In[ ]:


patterns = preprocessing_matching_algorithm()

if extractor_to_use==1 or extractor_to_use == 3:
    nlp1 = spacy.load("en_core_sci_lg")

    nlp1.add_pipe("scispacy_linker", config={"linker_name": "umls","max_entities_per_mention": 1,"threshold": 0.9})
    nlp1.add_pipe("negex",config={"chunk_prefix": ["no"]})
if extractor_to_use == 2 or extractor_to_use == 3:
    nlp2 = spacy.load("en_ner_bc5cdr_md")
    nlp2.add_pipe("negex",config={"chunk_prefix": ["no"]})
if extractor_to_use == 4:
    nlp3 = English()
    ruler = nlp3.add_pipe("entity_ruler", config={"overwrite_ents":True})
    ruler.add_patterns(patterns)
    nlp3.add_pipe("negex",config={"chunk_prefix": ["no"]})


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz\n')


# In[ ]:


# RG Files -> SVAAS -> HC Folder -> Complain List


# In[ ]:


if extractor_to_use == 5 or extractor_to_use == 7:
    nlp4 = spacy.load("en_core_sci_lg")
    nlp4.add_pipe("scispacy_linker", config={"linker_name": "umls","max_entities_per_mention": 1,"threshold": 0.9})
    ruler1 = nlp4.add_pipe("entity_ruler", config={"overwrite_ents":True})
    ruler1.add_patterns(patterns)
    print("Ruler 1: ", ruler1)
    nlp4.add_pipe("negex",config={"chunk_prefix": ["no"]})
  

if extractor_to_use == 6 or extractor_to_use == 7:
    nlp5 = spacy.load("en_ner_bc5cdr_md")
    ruler2 = nlp5.add_pipe("entity_ruler", config={"overwrite_ents":True})
    ruler2.add_patterns(patterns)
    nlp5.add_pipe("negex",config={"chunk_prefix": ["no"]})


# In[ ]:


def entity_extraction(row,value):
 #   print(row[value])
    val_list =row[value].split("##")
   # print(val_list)
    complaint = []
    diagnosis = []
    investigation=[]
    procedure = []
    drugs=[]
        
    for val in val_list:
        regexpattern = r'\.\.+'
        val = re.sub(regexpattern, ' ', val)
        val  = ' '.join(acronym_to_full_text(val.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split())

        #print("val is ",val)
            
        if extractor_to_use==1 or extractor_to_use == 3:
            doc1 = nlp1(val)
        if extractor_to_use==2 or extractor_to_use ==3 :
            doc2 = nlp2(val)
        if extractor_to_use==4:
            doc3 = nlp3(val)
        if extractor_to_use==5 or extractor_to_use==7:
            doc4 = nlp4(val)
        if extractor_to_use==6 or extractor_to_use==7:
            doc5=nlp5(val)
        
       
        if extractor_to_use == 4 or extractor_to_use==5 or extractor_to_use==6 or extractor_to_use==7:
            
            label=[]
            text=[]
            ent_list=[]
            if extractor_to_use ==4:
                ent_list.append(doc3.ents)
            if extractor_to_use == 5 or extractor_to_use==7:
                ent_list.append(doc4.ents)
            if extractor_to_use==6 or extractor_to_use == 7:
                ent_list.append(doc5.ents)
            
            for ent_tuple in ent_list:
                for ent in ent_tuple:
                   # print(ent,ent.label_,ent.text,ent._.negex)
                    if ent._.negex == 0 and ":" in ent.label_:
                        label.append(str(ent.label_))
                        text.append(ent.text)
            
            if len(label) > 0:
                all_labels=pd.DataFrame(label,columns =['Names'])
                all_labels[['Category', 'insight']] = all_labels['Names'].str.split(':', 1, expand=True)
                all_labels['Category']=all_labels.Category.str.strip()
                all_labels['insight']=all_labels.insight.str.strip()
                final_label=all_labels[['Category','insight']].groupby(['Category'])['insight'].apply(','.join).reset_index()
                #print(final_label)
                for i in range(len(final_label)):    
                    if final_label.iloc[i]['Category']=='Complaint':
                        if final_label.iloc[i]['insight'] not in complaint:
                            complaint.append(final_label.iloc[i]['insight'])
                    elif final_label.iloc[i]['Category']=='Investigation':
                        if final_label.iloc[i]['insight'] not in investigation:
                            investigation.append(final_label.iloc[i]['insight'])
                    elif final_label.iloc[i]['Category']=='Diagnosis':
                        if final_label.iloc[i]['insight'] not in diagnosis:
                            diagnosis.append(final_label.iloc[i]['insight'])
                    elif final_label.iloc[i]['Category']=='Drugs':
                        if final_label.iloc[i]['insight'] not in drugs:
                            drugs.append(final_label.iloc[i]['insight'])
                    elif final_label.iloc[i]['Category']=='Procedure':
                        if final_label.iloc[i]['insight'] not in procedure:
                            procedure.append(final_label.iloc[i]['insight']) 
          
        if extractor_to_use ==1 or extractor_to_use == 3 or extractor_to_use == 5 or extractor_to_use==7:
            if extractor_to_use ==1 or extractor_to_use == 3:
                ent_list = doc1.ents
                linker = nlp1.get_pipe("scispacy_linker")
            else:
                ent_list = doc4.ents
                linker = nlp4.get_pipe("scispacy_linker")
            #print(ent_list)
            for entity in ent_list:
                #print(entity.text,entity.label_,entity._.negex,entity._.kb_ents)
                if entity._.negex == 0 and entity.label_=='ENTITY':
                #Link to UMLS knowledge baseT048
                
             #   print(entity._.kb_ents)
                    for kb_entry in entity._.kb_ents:
                        sem_type=linker.kb.cui_to_entity[kb_entry[0]].types[0]
                        #print(entity,sem_type,linker.kb.cui_to_entity[kb_entry[0]].types)
                        #print(linker.kb.cui_to_entity[kb_entry[0]].canonical_name)
                        if sem_type in valid_semantic_types:
                            if sem_type in complaint_semantic_types:
                                if str(entity.text) not in complaint:
                                    complaint.append(str(entity.text))
                            elif sem_type in diagnosis_semantic_types:
                                if str(entity.text) not in diagnosis:
                                    diagnosis.append(str(entity.text))
                            elif sem_type in investigation_semantic_types:
                                if str(entity.text) not in investigation:
                                    investigation.append(str(entity.text))
                            elif sem_type in procedure_semantic_types:
                                if str(entity.text) not in procedure:
                                    procedure.append(str(entity.text))
                            elif sem_type in drugs_semantic_types:
                                if str(entity.text) not in drugs:
                                    drugs.append(str(entity.text))
            
                
        if extractor_to_use==2 or extractor_to_use ==3 or extractor_to_use==6 or extractor_to_use==7:
            if extractor_to_use == 2 or extractor_to_use == 3:
                ent_list = doc2.ents
            else:
                ent_list = doc5.ents
            for entity in ent_list:
                if entity._.negex == 0:
                    if entity.label_ == 'DISEASE':
                        if str(entity.text) not in diagnosis and str(entity.text) not in complaint:
                            diagnosis.append(str(entity.text))
                    elif entity.label_ == 'CHEMICAL':
                        if str(entity.text) not in drugs:
                            drugs.append(str(entity.text))
        
       
                    
     
       
    row['COMPLAINT'] = ",".join(complaint)
    row['DIAGNOSIS'] = ",".join(diagnosis)
    row['INVESTIGATION'] = ",".join(investigation)
    row['PROCEDURE'] = ",".join(procedure)
    row['DRUGS'] = ','.join(drugs)
    
        
    return row


# In[ ]:


def clean_data(row):
   rowlist = row.split(",")
   modrowlist=[]
   for r in rowlist:
       if r.strip() not in modrowlist and r.strip()!="":
           modrowlist.append(r.strip())
   return ",".join(modrowlist)
     
# def len_distribution(df_ext_Remarks,df_ext_diagnosis,df_ext_Investigation):
#    df_ext=df_ext_Remarks.merge(df_ext_diagnosis,on=['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation','Prescription_Medicine_Advised', 'Prescription_Remarks'],how='left')
#    print(df_ext.shape)
#    df_ext=df_ext.merge(df_ext_Investigation,on=['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation', 'Prescription_Medicine_Advised','Prescription_Remarks'],how='left')
#    print(df_ext.shape)
#    df_ext=df_ext.replace('nan',"")
#    df_ext['COMPLAINT_upd']=df_ext['COMPLAINT']+","+df_ext['COMPLAINT_x']+","+df_ext['COMPLAINT_y']
#    df_ext['DIAGNOSIS_upd']=df_ext['DIAGNOSIS']+","+df_ext['DIAGNOSIS_x']+","+df_ext['DIAGNOSIS_y']
#    df_ext['INVESTIGATION_upd']=df_ext['INVESTIGATION']+","+df_ext['INVESTIGATION_x']+","+df_ext['INVESTIGATION_y']
#    df_ext['PROCEDURE_upd']=df_ext['PROCEDURE']+","+df_ext['PROCEDURE_x']+","+df_ext['PROCEDURE_y']
#    df_ext['DRUGS_upd']=df_ext['DRUGS']+","+df_ext['DRUGS_x']+","+df_ext['DRUGS_y']
  
#    df_ext['COMPLAINT_upd'] = df_ext.apply(lambda row : clean_data(row['COMPLAINT_upd'].lower()), axis = 1)
#    df_ext['DIAGNOSIS_upd'] = df_ext.apply(lambda row : clean_data(row['DIAGNOSIS_upd'].lower()), axis = 1)
#    df_ext['INVESTIGATION_upd'] = df_ext.apply(lambda row : clean_data(row['INVESTIGATION_upd'].lower()), axis = 1)
#    df_ext['PROCEDURE_upd'] = df_ext.apply(lambda row : clean_data(row['PROCEDURE_upd'].lower()), axis = 1)
#    df_ext['DRUGS_upd'] = df_ext.apply(lambda row : clean_data(row['DRUGS_upd'].lower()), axis = 1)
   
#    df_ext.fillna("",inplace=True)
#    df_ext=df_ext.replace('nan',"")


#    x=df_ext[['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation', 'Prescription_Medicine_Advised','Prescription_Remarks','COMPLAINT_upd', 'DIAGNOSIS_upd', 'INVESTIGATION_upd','DRUGS_upd','PROCEDURE_upd']]   
#    x['length']=x['COMPLAINT_upd'].apply(lambda y:len(str(y))) 
#    x['length1']=x['DIAGNOSIS_upd'].apply(lambda y:len(str(y)))
#    x['length2']=x['DRUGS_upd'].apply(lambda y:len(str(y)))
#    x['length3']=x['INVESTIGATION_upd'].apply(lambda y:len(str(y)))
#    x['length4']=x['PROCEDURE_upd'].apply(lambda y:len(str(y)))
  
#    x['len']=x['length']+x['length1']+x['length2']+x['length3']+x['length4']
#    return x


# In[ ]:


data=pd.read_excel('Validation_Dataset_10.xlsx',engine='openpyxl',usecols='A,B,C,D,E')
data.head()


# In[ ]:


(data.iloc[3]['Prescription_Diagnosis'])


# In[ ]:


(data.iloc[3]['Prescription_Remarks'])


# In[ ]:


#brdg_appt_consult_id -> Visit ID
# 'complaints','diagnosis', 'advice'

data2 = pd.read_csv("dt.csv")
data2.fillna("NA", inplace=True)

combined_values = data2['complaints'].values + "," + data2['advice'].values + "," + data2['diagnosis'].values

data_req = pd.DataFrame(np.c_[data2['brdg_appt_conslt_sid'].values, combined_values], 
                          columns=['brdg_appt_consult_id', 'combined_diagnosis']
                        )

data_req.head()


# In[ ]:


data_req['combined_diagnosis'] = data_req['combined_diagnosis'].apply(lambda x: x.lower())
data_req.head()


# In[ ]:


data = data_req.copy()


# In[ ]:


def generateResults():
  # data=pd.read_excel('Validation_Dataset_10.xlsx',engine='openpyxl',usecols='A,B,C,D,E')
  # print(data.shape)
  # data.dropna(how='all',axis=0,inplace=True)
  #data.dropna(how='all', axis=1, inplace=True)
  # print(data.shape)
  data['combined_diagnosis']=data['combined_diagnosis'].astype(str)
  # data['Prescription_Diagnosis']=data['Prescription_Diagnosis'].astype(str)
  # data['Prescription_Investigation']=data['Prescription_Investigation'].astype(str)
  # data['Prescription_Medicine_Advised']=data['Prescription_Medicine_Advised'].astype(str)
  # nan_value = float("NaN")
  data.replace("", "na", inplace=True)
  print(data.shape)
  data.dropna(how='all',axis=0,inplace=True)
  data.dropna(how='all', axis=1, inplace=True)
  print(data.shape)
  df_ext_Remarks=data.apply(lambda row:entity_extraction(row,'combined_diagnosis'),axis=1)
  # print("here1")
  # df_ext_diagnosis=data.apply(lambda row:entity_extraction(row,'Prescription_Diagnosis'),axis=1)
  # print("here2")
  # df_ext_Investigation=data.apply(lambda row:entity_extraction(row,'Prescription_Investigation'),axis=1)

  # print("here3")
  # df_ext_medicine=data.apply(lambda row:entity_extraction(row,'Prescription_Medicine_Advised'),axis=1)


  # print("here4")
  df_ext_Remarks.to_csv("remarks.csv", index=False)
  # df_ext_diagnosis.to_csv("2.csv")
  # df_ext_Investigation.to_csv("3.csv")
  print(df_ext_Remarks.shape)
  # print(df_ext_diagnosis.shape)
  # print(df_ext_Investigation.shape)

  # df_extractor_output=len_distribution(df_ext_Remarks)

  # df_extractor_output.to_csv('entity_extractor_model_'+str(extractor_to_use)+'_output_test4.csv', index=False)
  return df_ext_Remarks


# In[ ]:


final_df = generateResults()


# In[26]:


def generateMappedComplainDataFrame(complain_data, original_data):
    df_result = generateResults()
    df_result.fillna("NaN", inplace=True)
    df_result['DIAGNOSIS'] = df_result['DIAGNOSIS'].apply(lambda x: x.lower())
    
    df_result["DIAGNOSIS"] = df_result["DIAGNOSIS"].str.split(",")
    df_result = df_result.explode("DIAGNOSIS")
    
    assert ".csv" in complain_data, "Make sure to insert a csv file."
    assert ".csv" in original_data, "Make sure to insert a csv file."
    data_req = pd.read_csv(complain_data)
    
    brdg = pd.read_csv(original_data)
    df_result.rename(columns={'brdg_appt_consult_id':'brdg_appt_conslt_sid'}, inplace=True)
    
    
    cols = ["brdg_appt_conslt_sid", "DIAGNOSIS", "COMPLAINT"]

    df_result = df_result[cols]
    df_result.columns = ['brdg_appt_conslt_sid', 'mapped_DIAGNOSIS', 'mapped_COMPLAINT']
    
    brdg.merge(df_result, on="brdg_appt_conslt_sid", how="left").to_csv("merged_csv_complain_v2.csv", index=False)

    df_final = pd.read_csv("merged_csv_complain_v2.csv")
    print("Final dataframe has been saved with the name of output.csv")
    df_final.replace("nan", "").fillna(" ").to_csv("output.csv", index=False)
    
    return df_final.replace("nan", "").fillna(" ")
    
    


# In[ ]:


final_df.head(20)


# In[ ]:


output_csv = pd.read_csv("entity_extractor_model_5_output_test4.csv")
output_csv.head(10)


# ## Part 2

# In[1]:


import pandas as pd  
import numpy as np


# In[2]:


df_result = pd.read_csv("remarks.csv")
df_result.head()


# In[3]:


# df_complain_list = pd.read_csv("complaints_list.csv")
# df_complain_list.head(20)


# In[4]:


# df_complain_list.info()


# In[5]:


# df_complain_list.fillna("NaN", inplace=True)


# In[6]:


# print("Unique complains in the complain list: ", len(set(df_complain_list['Label'].values)) - 1)


# In[7]:


df_result.fillna("NaN", inplace=True)


# In[8]:


# print("Found unique complain in the input file: ", len(set(df_result['DIAGNOSIS'].values)) - 1)


# In[9]:


# df_complain_list['Label'] = df_complain_list['Label'].apply(lambda x: x.lower())
# df_complain_list['Label'].head(10)


# In[10]:


df_result['DIAGNOSIS'] = df_result['DIAGNOSIS'].apply(lambda x: x.lower())
df_result['DIAGNOSIS'].head(10)


# In[11]:


df_result["DIAGNOSIS"] = df_result["DIAGNOSIS"].str.split(",")
df_result = df_result.explode("DIAGNOSIS")
df_result.head(10)


# In[12]:


# df_result.to_csv("exploded_result.csv", index=False)


# ```Python
# diagnosis_list = pd.read_csv("diagnosis_list.csv")
# diagnosis_list.head(20)
# 
# df_req = diagnosis_list.append(df_complain_list)
# df_req.drop("status", axis=1, inplace=True)
# 
# df_req = df_req[df_req['Label'].isna() == False]
# df_req = df_req[df_req['Words'].isna() == False]
# df_req = df_req.drop_duplicates()
# 
# ## Removing duplicated words
# df_req = df_req[df_req["Words"].duplicated() == False]
# df_req['Label'] = df_req['Label'].apply(lambda x: x.lower())
# 
# df_giri = pd.read_csv("Copy of drug_master2.csv")
# df_giri.columns = ["Words", "Label"]
# 
# 
# df_req = df_req.append(df_giri)
# df_req = df_req[df_req['Label'].isna() == False]
# df_req = df_req[df_req['Words'].isna() == False]
# df_req = df_req.drop_duplicates()
# df_req["Label"] = df_req["Label"].apply(lambda x: x.lower())
# 
# df_req.to_csv("full_complain_list.csv", index=False)
# ```

# ### Loading req data

# In[13]:


data_req = pd.read_csv("full_complain_list.csv")


# In[14]:


df_result = pd.read_csv("exploded_result.csv")


# In[15]:


brdg = pd.read_csv("dt.csv")


# In[16]:


df_result.info()


# In[17]:


df_result.rename(columns={'brdg_appt_consult_id':'brdg_appt_conslt_sid'}, inplace=True)


# In[18]:


cols = ["brdg_appt_conslt_sid", "DIAGNOSIS", "COMPLAINT"]

df_result = df_result[cols]


# In[19]:


df_result


# In[20]:


df_result.columns = ['brdg_appt_conslt_sid', 'mapped_DIAGNOSIS', 'mapped_COMPLAINT']


# In[21]:


brdg.merge(df_result, on="brdg_appt_conslt_sid", how="left").to_csv("merged_csv_complain_v2.csv", index=False)

df_final = pd.read_csv("merged_csv_complain_v2.csv")


# In[22]:


df_final.head()


# In[23]:


df_final.replace("nan", "").fillna(" ").to_csv("merged_csv_complain_v3.csv", index=False)


# In[25]:


df_final = pd.read_csv("merged_csv_complain_v3.csv")
df_final.head()


# In[ ]:




