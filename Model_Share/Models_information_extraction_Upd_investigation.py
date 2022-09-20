#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:

#extractor_to_use = 1 Use en_core_sci_lg model - Link to UMLS to extract complaint,disease, drugs etc
#extractor_to_use = 2 Use en_ner_bc5cdr_md model
#extractor_to_use = 3 Use a combination of model 1 and 2
#extractor_to_use = 4 Use unigram,bigram,trigram based matching models
#extractor_to_use = 5 Use a combination of extractors 1 and 4
#extractor_to_use = 6 Use a combination of extractors 2 and 4
#extractor_to_use = 7 Use a combination of extractors 3 and 4
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


# In[3]:


def preprocessing_matching_algorithm():
    Full_data=pd.ExcelFile(r'Diagnosis N Gram MAX 3 12April22upd2.xlsx',engine='openpyxl')

    one_gram=pd.read_excel(Full_data,sheet_name='Unigram')

    #del one_gram['Unnamed: 5']
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
    
    # investigation=pd.read_excel('UnifiedBiomarkerMapping_1.xlsx',sheet_name='Top TestNames')
    # investigation.dropna()
    # investigation['x'] = pd.to_numeric(investigation['lab test names'], errors='coerce')
    # investigation = investigation[investigation['x'].isnull()]
    # investigation=investigation[['lab test names']]
    # investigation['lab test names']= investigation['lab test names'].apply(lambda x:''.join(re.sub(r'\W+',' ',x)))
    # investigation['lab test names']= investigation['lab test names'].apply(lambda x:re.sub(' {2,}', ' ',x))
    # investigation['lab test names']=investigation['lab test names'].astype(str)
    # investigation['len']=investigation['lab test names'].apply(lambda x:len(x))
    # investigation.drop_duplicates(inplace=True)
    # investigation.rename(columns={'lab test names':'pattern'},inplace=True)
    # investigation['pattern']=investigation['pattern'].astype(str)
    # investigation['Label']='Investigation:'+investigation['pattern']

    
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
   # print(n_gram_sheet)
    n_gram_sheet=n_gram_sheet.append(drugs)
    n_gram_sheet=n_gram_sheet[['label','pattern']]

    gram_dict=n_gram_sheet.to_dict("record")
    #print(gram_dict)
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


# In[4]:


#Reference https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
valid_semantic_types = ['T109','T116','T020','T190','T017','T195','T053','T038','T030','T019','T200','T060','T047','T129','T037','T059','T034','T048','T046','T121','T061','T184','T033','T191' ]
complaint_semantic_types = ['T184','T033']
diagnosis_semantic_types = ['T020','T190','T053','T038','T029','T023','T030','T019','T047','T129','T037','T048','T046','T191']
investigation_semantic_types=['T017','T034','T060','T059','T061']
procedure_semantic_types=[]
drugs_semantic_types = ['T195','T200','T116','T109','T122']


# In[5]:


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
    



# In[6]:

if extractor_to_use == 5 or extractor_to_use == 7:
    nlp4 = spacy.load("en_core_sci_lg")
    nlp4.add_pipe("scispacy_linker", config={"linker_name": "umls","max_entities_per_mention": 1,"threshold": 0.9})
    ruler1 = nlp4.add_pipe("entity_ruler", config={"overwrite_ents":True})
    ruler1.add_patterns(patterns)
    nlp4.add_pipe("negex",config={"chunk_prefix": ["no"]})
  

if extractor_to_use == 6 or extractor_to_use == 7:
    nlp5 = spacy.load("en_ner_bc5cdr_md")
    ruler2 = nlp5.add_pipe("entity_ruler", config={"overwrite_ents":True})
    ruler2.add_patterns(patterns)
    nlp5.add_pipe("negex",config={"chunk_prefix": ["no"]})
    


# In[7]:



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
    


# In[8]:



def clean_data(row):
   rowlist = row.split(",")
   modrowlist=[]
   for r in rowlist:
       if r.strip() not in modrowlist and r.strip()!="":
           modrowlist.append(r.strip())
   return ",".join(modrowlist)
     
def len_distribution(df_ext_Remarks,df_ext_diagnosis,df_ext_Investigation):
   df_ext=df_ext_Remarks.merge(df_ext_diagnosis,on=['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation','Prescription_Medicine_Advised', 'Prescription_Remarks'],how='left')
   print(df_ext.shape)
   df_ext=df_ext.merge(df_ext_Investigation,on=['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation', 'Prescription_Medicine_Advised','Prescription_Remarks'],how='left')
   print(df_ext.shape)
   df_ext=df_ext.replace('nan',"")
   df_ext['COMPLAINT_upd']=df_ext['COMPLAINT']+","+df_ext['COMPLAINT_x']+","+df_ext['COMPLAINT_y']
   df_ext['DIAGNOSIS_upd']=df_ext['DIAGNOSIS']+","+df_ext['DIAGNOSIS_x']+","+df_ext['DIAGNOSIS_y']
   df_ext['INVESTIGATION_upd']=df_ext['INVESTIGATION']+","+df_ext['INVESTIGATION_x']+","+df_ext['INVESTIGATION_y']
   df_ext['PROCEDURE_upd']=df_ext['PROCEDURE']+","+df_ext['PROCEDURE_x']+","+df_ext['PROCEDURE_y']
   df_ext['DRUGS_upd']=df_ext['DRUGS']+","+df_ext['DRUGS_x']+","+df_ext['DRUGS_y']
  
   df_ext['COMPLAINT_upd'] = df_ext.apply(lambda row : clean_data(row['COMPLAINT_upd'].lower()), axis = 1)
   df_ext['DIAGNOSIS_upd'] = df_ext.apply(lambda row : clean_data(row['DIAGNOSIS_upd'].lower()), axis = 1)
   df_ext['INVESTIGATION_upd'] = df_ext.apply(lambda row : clean_data(row['INVESTIGATION_upd'].lower()), axis = 1)
   df_ext['PROCEDURE_upd'] = df_ext.apply(lambda row : clean_data(row['PROCEDURE_upd'].lower()), axis = 1)
   df_ext['DRUGS_upd'] = df_ext.apply(lambda row : clean_data(row['DRUGS_upd'].lower()), axis = 1)
   
   df_ext.fillna("",inplace=True)
   df_ext=df_ext.replace('nan',"")


   x=df_ext[['Prescription id', 'Prescription_Diagnosis','Prescription_Investigation', 'Prescription_Medicine_Advised','Prescription_Remarks','COMPLAINT_upd', 'DIAGNOSIS_upd', 'INVESTIGATION_upd','DRUGS_upd','PROCEDURE_upd']]   
   x['length']=x['COMPLAINT_upd'].apply(lambda y:len(str(y))) 
   x['length1']=x['DIAGNOSIS_upd'].apply(lambda y:len(str(y)))
   x['length2']=x['DRUGS_upd'].apply(lambda y:len(str(y)))
   x['length3']=x['INVESTIGATION_upd'].apply(lambda y:len(str(y)))
   x['length4']=x['PROCEDURE_upd'].apply(lambda y:len(str(y)))
  
   x['len']=x['length']+x['length1']+x['length2']+x['length3']+x['length4']
   return x


# In[ ]:


data=pd.read_excel('Validation_Dataset_10.xlsx',engine='openpyxl',usecols='A,B,C,D,E')
print(data.shape)
data.dropna(how='all',axis=0,inplace=True)
#data.dropna(how='all', axis=1, inplace=True)
print(data.shape)
data['Prescription_Remarks']=data['Prescription_Remarks'].astype(str)
data['Prescription_Diagnosis']=data['Prescription_Diagnosis'].astype(str)
data['Prescription_Investigation']=data['Prescription_Investigation'].astype(str)
data['Prescription_Medicine_Advised']=data['Prescription_Medicine_Advised'].astype(str)
nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
print(data.shape)
data.dropna(how='all',axis=0,inplace=True)
data.dropna(how='all', axis=1, inplace=True)
print(data.shape)
df_ext_Remarks=data.apply(lambda row:entity_extraction(row,'Prescription_Remarks'),axis=1)
print("here1")
df_ext_diagnosis=data.apply(lambda row:entity_extraction(row,'Prescription_Diagnosis'),axis=1)
print("here2")
df_ext_Investigation=data.apply(lambda row:entity_extraction(row,'Prescription_Investigation'),axis=1)

print("here3")
df_ext_medicine=data.apply(lambda row:entity_extraction(row,'Prescription_Medicine_Advised'),axis=1)


print("here4")
df_ext_Remarks.to_csv("1.csv")
df_ext_diagnosis.to_csv("2.csv")
df_ext_Investigation.to_csv("3.csv")
print(df_ext_Remarks.shape)
print(df_ext_diagnosis.shape)
print(df_ext_Investigation.shape)

df_extractor_output=len_distribution(df_ext_Remarks,df_ext_diagnosis,df_ext_Investigation)

df_extractor_output.to_csv('entity_extractor_model_'+str(extractor_to_use)+'_output_test4.csv')


# In[ ]:


# Generate Results:

def generate_result_complaint(row):
    fully_matched_complaint = 0
    false_positive_complaint = 0
    false_negative_complaint = 0
    partial_overlap_complaint = 0
    NUM_COMPLAINTS = 5
    #print(row['COMPLAINT_upd'])
    complaint_list = row['COMPLAINT_upd'].split(",")
    complaint_list = [i for i in complaint_list if i]
    complaint1_tokens = acronym_to_full_text(row['Complaint1'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    complaint2_tokens = acronym_to_full_text(row['Complaint2'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    complaint3_tokens = acronym_to_full_text(row['Complaint3'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    complaint4_tokens = acronym_to_full_text(row['Complaint4'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    complaint5_tokens = acronym_to_full_text(row['Complaint5'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    complaint1_tokens = [i for i in complaint1_tokens if i]
    complaint2_tokens = [i for i in complaint2_tokens if i]
    complaint3_tokens = [i for i in complaint3_tokens if i]
    complaint4_tokens = [i for i in complaint4_tokens if i]
    complaint5_tokens = [i for i in complaint5_tokens if i]
    complaint_tokens = []
    complaint_tokens.append(complaint1_tokens)
    complaint_tokens.append(complaint2_tokens)
    complaint_tokens.append(complaint3_tokens)
    complaint_tokens.append(complaint4_tokens)
    complaint_tokens.append(complaint5_tokens)
    complaints_matched=[]
   
    for i in range(NUM_COMPLAINTS):
        if len(complaint_tokens[i]) > 0 and complaint_tokens[i][0]!="":
            complaints_matched.append(0)
        else:
            complaints_matched.append(2)
    
    for comp in complaint_list:
        comp_tokens = acronym_to_full_text(comp.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
        tot_tokens_comp_alg = len(comp_tokens)
        #Change this line if NUM_COMPLAINTS change
        tot_tokens_comp_doc= [len(complaint1_tokens),len(complaint2_tokens),len(complaint3_tokens),len(complaint4_tokens),len(complaint5_tokens)]
        count = 0
        for i in range(NUM_COMPLAINTS):
            for x in comp_tokens:
                if complaints_matched[i] == 0:
                    if x in complaint_tokens[i]:
                        count+=1
            if count > 0 and count == tot_tokens_comp_doc[i] and count==tot_tokens_comp_alg:
                fully_matched_complaint+=1
                complaints_matched[i] = 1
                break
            elif count > 0:
                complaints_matched[i] = 1
                partial_overlap_complaint+=1;
                break
        if count==0:
            false_positive_complaint+=1
                 
    for i in range(NUM_COMPLAINTS):
        if complaints_matched[i] == 0:
            false_negative_complaint+=1
        
    row['Complaint_TP']=fully_matched_complaint
    row['Complaint_FP']=false_positive_complaint
    row['Complaint_FN']=false_negative_complaint
    row['Complaint_Paritial_Matched']=partial_overlap_complaint
    return row

def generate_result_diagnosis(row):
    fully_matched_diagnosis = 0
    false_positive_diagnosis = 0
    false_negative_diagnosis = 0
    partial_overlap_diagnosis = 0
    NUM_DIAGNOSIS = 5
    #print(row['DIAGNOSIS_upd'])
    diagnosis_list = row['DIAGNOSIS_upd'].split(",")
    diagnosis_list = [i for i in diagnosis_list if i]
    diagnosis1_tokens = acronym_to_full_text(row['Diagnosis1'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    diagnosis2_tokens = acronym_to_full_text(row['Diagnosis2'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    diagnosis3_tokens = acronym_to_full_text(row['Diagnosis3'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    diagnosis4_tokens = acronym_to_full_text(row['Diagnosis4'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    diagnosis5_tokens = acronym_to_full_text(row['Diagnosis5'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    diagnosis1_tokens = [i for i in diagnosis1_tokens if i]
    diagnosis2_tokens = [i for i in diagnosis2_tokens if i]
    diagnosis3_tokens = [i for i in diagnosis3_tokens if i]
    diagnosis4_tokens = [i for i in diagnosis4_tokens if i]
    diagnosis5_tokens = [i for i in diagnosis5_tokens if i]
    diagnosis_tokens = []
    diagnosis_tokens.append(diagnosis1_tokens)
    diagnosis_tokens.append(diagnosis2_tokens)
    diagnosis_tokens.append(diagnosis3_tokens)
    diagnosis_tokens.append(diagnosis4_tokens)
    diagnosis_tokens.append(diagnosis5_tokens)
    diagnosis_matched=[]
   
    for i in range(NUM_DIAGNOSIS):
        if len(diagnosis_tokens[i]) > 0 and diagnosis_tokens[i][0]!="":
            diagnosis_matched.append(0)
        else:
            diagnosis_matched.append(2)
    
    for diag in diagnosis_list:
        diag_tokens = acronym_to_full_text(diag.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
        tot_tokens_diag_alg = len(diag_tokens)
        #Change this line if NUM_DIAGNOSIS change
        tot_tokens_diag_doc = [len(diagnosis1_tokens),len(diagnosis2_tokens),len(diagnosis3_tokens),len(diagnosis4_tokens),len(diagnosis5_tokens)]
        count = 0
        for i in range(NUM_DIAGNOSIS):
            for x in diag_tokens:
                if diagnosis_matched[i] == 0:
                    if x in diagnosis_tokens[i]:
                        count+=1
            if count > 0 and count == tot_tokens_diag_doc[i] and count==tot_tokens_diag_alg:
                fully_matched_diagnosis+=1
                diagnosis_matched[i] = 1
                break
            elif count > 0:
                diagnosis_matched[i] = 1
                partial_overlap_diagnosis+=1;
                break
        if count==0:
            false_positive_diagnosis+=1
                 
    for i in range(NUM_DIAGNOSIS):
        if diagnosis_matched[i] == 0:
            false_negative_diagnosis+=1
        
    row['Diagnosis_TP']=fully_matched_diagnosis
    row['Diagnosis_FP']=false_positive_diagnosis
    row['Diagnosis_FN']=false_negative_diagnosis
    row['Diagnosis_Paritial_Matched']=partial_overlap_diagnosis
    return row

def generate_result_investigation(row):
    fully_matched_investigation = 0
    false_positive_investigation = 0
    false_negative_investigation = 0
    partial_overlap_investigation = 0
    NUM_INVESTIGATION = 5
   # print(row['INVESTIGATION_upd'])
    investigation_list = row['INVESTIGATION_upd'].split(",")
    investigation_list = [i for i in investigation_list if i]
    investigation1_tokens = acronym_to_full_text(row['Investigation1'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    investigation2_tokens = acronym_to_full_text(row['Investigation2'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    investigation3_tokens = acronym_to_full_text(row['Investigation3'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    investigation4_tokens = acronym_to_full_text(row['Investigation4'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    investigation5_tokens = acronym_to_full_text(row['Investigation5'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    investigation1_tokens = [i for i in investigation1_tokens if i]
    investigation2_tokens = [i for i in investigation2_tokens if i]
    investigation3_tokens = [i for i in investigation3_tokens if i]
    investigation4_tokens = [i for i in investigation4_tokens if i]
    investigation5_tokens = [i for i in investigation5_tokens if i]
    investigation_tokens = []
    investigation_tokens.append(investigation1_tokens)
    investigation_tokens.append(investigation2_tokens)
    investigation_tokens.append(investigation3_tokens)
    investigation_tokens.append(investigation4_tokens)
    investigation_tokens.append(investigation5_tokens)
    investigation_matched=[]
   
    for i in range(NUM_INVESTIGATION):
        if len(investigation_tokens[i]) > 0 and investigation_tokens[i][0]!="":
            investigation_matched.append(0)
        else:
            investigation_matched.append(2)
    
    for invest in investigation_list:
        invest_tokens = acronym_to_full_text(invest.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
        tot_tokens_invest_alg = len(invest_tokens)
        #Change this line if NUM_INVESTIGATION change
        tot_tokens_invest_doc = [len(investigation1_tokens),len(investigation2_tokens),len(investigation3_tokens),len(investigation4_tokens),len(investigation5_tokens)]
        count = 0
        for i in range(NUM_INVESTIGATION):
            for x in invest_tokens:
                if investigation_matched[i] == 0:
                    if x in investigation_tokens[i]:
                        count+=1
            if count > 0 and count == tot_tokens_invest_doc[i] and count==tot_tokens_invest_alg:
                fully_matched_investigation+=1
                investigation_matched[i] = 1
                break
            elif count > 0:
                investigation_matched[i] = 1
                partial_overlap_investigation+=1;
                break
        if count==0:
            false_positive_investigation+=1
                 
    for i in range(NUM_INVESTIGATION):
        if investigation_matched[i] == 0:
            false_negative_investigation+=1
        
    row['Investigation_TP']=fully_matched_investigation
    row['Investigation_FP']=false_positive_investigation
    row['Investigation_FN']=false_negative_investigation
    row['Investigation_Paritial_Matched']=partial_overlap_investigation
    return row

def generate_result_procedure(row):
    fully_matched_procedure = 0
    false_positive_procedure = 0
    false_negative_procedure = 0
    partial_overlap_procedure = 0
    NUM_PROCEDURE = 2
  #  print(row['PROCEDURE_upd'])
    procedure_list = row['PROCEDURE_upd'].split(",")
    procedure_list = [i for i in procedure_list if i]
    procedure1_tokens = acronym_to_full_text(row['Procedure1'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    procedure2_tokens = acronym_to_full_text(row['Procedure2'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    # procedure3_tokens = acronym_to_full_text(row['Procedure3'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    procedure1_tokens = [i for i in procedure1_tokens if i]
    procedure2_tokens = [i for i in procedure2_tokens if i]
    # procedure3_tokens = [i for i in procedure3_tokens if i]
    procedure_tokens = []
    procedure_tokens.append(procedure1_tokens)
    procedure_tokens.append(procedure2_tokens)
    # procedure_tokens.append(procedure3_tokens)
    procedure_matched=[]
   
    for i in range(NUM_PROCEDURE):
        if len(procedure_tokens[i]) > 0 and procedure_tokens[i][0]!="":
            procedure_matched.append(0)
        else:
            procedure_matched.append(2)
    
    for proc in procedure_list:
        proc_tokens = acronym_to_full_text(proc.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
        tot_tokens_proc_alg = len(proc_tokens)
        #Change this line if NUM_PROCEDURE change
        # tot_tokens_proc_doc = [len(procedure1_tokens),len(procedure2_tokens),len(procedure3_tokens)]
        tot_tokens_proc_doc = [len(procedure1_tokens),len(procedure2_tokens)]

        count = 0
        for i in range(NUM_PROCEDURE):
            for x in proc_tokens:
                if procedure_matched[i] == 0:
                    if x in procedure_tokens[i]:
                        count+=1
            if count > 0 and count == tot_tokens_proc_doc[i] and count==tot_tokens_proc_alg:
                fully_matched_procedure+=1
                procedure_matched[i] = 1
                break
            elif count > 0:
                procedure_matched[i] = 1
                partial_overlap_procedure+=1;
                break
        if count==0:
            false_positive_procedure+=1
                 
    for i in range(NUM_PROCEDURE):
        if procedure_matched[i] == 0:
            false_negative_procedure+=1
        
    row['Procedure_TP']=fully_matched_procedure
    row['Procedure_FP']=false_positive_procedure
    row['Procedure_FN']=false_negative_procedure
    row['Procedure_Paritial_Matched']=partial_overlap_procedure
    return row

def generate_result_drugs(row):
    fully_matched_drug = 0
    false_positive_drug = 0
    false_negative_drug = 0
    partial_overlap_drug = 0
    NUM_DRUG = 4
    #print(row['DRUGS_upd'])
    drug_list = row['DRUGS_upd'].split(",")
    drug_list = [i for i in drug_list if i]
    drug1_tokens = acronym_to_full_text(row['Drug1'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    drug2_tokens = acronym_to_full_text(row['Drug2'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    drug3_tokens = acronym_to_full_text(row['Drug3'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    drug4_tokens = acronym_to_full_text(row['Drug4'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    #drug5_tokens = acronym_to_full_text(row['Drug5'].lower().strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
    drug1_tokens = [i for i in drug1_tokens if i]
    drug2_tokens = [i for i in drug2_tokens if i]
    drug3_tokens = [i for i in drug3_tokens if i]
    drug4_tokens = [i for i in drug4_tokens if i]
    #drug5_tokens = [i for i in drug5_tokens if i]
    drug_tokens = []
    drug_tokens.append(drug1_tokens)
    drug_tokens.append(drug2_tokens)
    drug_tokens.append(drug3_tokens)
    drug_tokens.append(drug4_tokens)
    #drug_tokens.append(drug5_tokens)
    drug_matched=[]
   
    for i in range(NUM_DRUG):
        if len(drug_tokens[i]) > 0 and drug_tokens[i][0]!="":
            drug_matched.append(0)
        else:
            drug_matched.append(2)
    
    for drug in drug_list:
        dr_tokens = acronym_to_full_text(drug.strip().replace('-',' ').replace('/',' ').replace('+',' ').strip(),acronyms_dict).split(" ")
        tot_tokens_drug_alg = len(dr_tokens)
        #Change this line if NUM_DRUG change
        tot_tokens_drug_doc = [len(drug1_tokens),len(drug2_tokens),len(drug3_tokens),len(drug4_tokens)]#,len(drug5_tokens)]
        count = 0
        for i in range(NUM_DRUG):
            for x in dr_tokens:
                if drug_matched[i] == 0:
                    if x in drug_tokens[i]:
                        count+=1
            if count > 0 and count == tot_tokens_drug_doc[i] and count==tot_tokens_drug_alg:
                fully_matched_drug+=1
                drug_matched[i] = 1
                break
            elif count > 0:
                drug_matched[i] = 1
                partial_overlap_drug+=1;
                break
        if count==0:
            false_positive_drug+=1
                 
    for i in range(NUM_DRUG):
        if drug_matched[i] == 0:
            false_negative_drug+=1
        
    row['Drug_TP']=fully_matched_drug
    row['Drug_FP']=false_positive_drug
    row['Drug_FN']=false_negative_drug
    row['Drug_Paritial_Matched']=partial_overlap_drug
    return row
  


"""data_ent_out =pd.read_csv('entity_extractor_model_'+str(extractor_to_use)+'_output_test4.csv')

data_ent_out.dropna(how='all',inplace=True)
#data_ent_out.dropna(how='all', axis=1, inplace=True)
data_doc_out=pd.read_excel('Validation Dataset_filled.xlsx',engine='openpyxl')

data_doc_out.dropna(how='all',inplace=True)
#data_doc_out.dropna(how='all', axis=1, inplace=True)
df_results = data_ent_out.merge(data_doc_out,on='Prescription id')

df_results.fillna("",inplace=True)
df_results=df_results.apply(lambda row:generate_result_complaint(row),axis=1)
df_results=df_results.apply(lambda row:generate_result_diagnosis(row),axis=1)
df_results=df_results.apply(lambda row:generate_result_investigation(row),axis=1)
df_results=df_results.apply(lambda row:generate_result_procedure(row),axis=1)
df_results=df_results.apply(lambda row:generate_result_drugs(row),axis=1)


df_results.to_csv('entity_extractor_model_'+str(extractor_to_use)+'_results_output_test4.csv')"""


# In[ ]:



#whole_data =pd.read_csv('Max_diagnosis_final.csv')
#whole_data.dropna(how='all',inplace=True)
#whole_data.dropna(how='all', axis=1, inplace=True)
#whole_data.rename(columns = {'Unnamed: 0':'Prescription id'}, inplace = True)
#data_ent_out =pd.read_csv('entity_extractor_model_'+str(extractor_to_use)+'_output_test.csv')
#data_ent_out.dropna(how='all',inplace=True)
#data_ent_out.dropna(how='all', axis=1, inplace=True)
#data_ent_out=data_ent_out.merge(whole_data, on='Prescription id')
#data_ent_out.to_csv('examples_to_send.csv')


# In[ ]:


#whole_data =pd.read_csv('bookiniddate.csv')
#whole_data.dropna(how='all',inplace=True)
#whole_data.dropna(how='all', axis=1, inplace=True)
#examples =pd.read_csv('examples_to_send.csv')
#examples.dropna(how='all',inplace=True)
#examples.dropna(how='all', axis=1, inplace=True)
#examples=examples.merge(whole_data, on='bookingid')
#examples.to_csv('examples_to_send1.csv')


# In[ ]:




