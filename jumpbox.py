import pyodbc
import pymysql
import pandas as pd
import datetime 

Server='synapsewpprod.sql.azuresynapse.net'
db1='ddksqlpoolprod'
connection = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=synapsewpprod.sql.azuresynapse.net;DATABASE=ddksqlpoolprod;UID=powerbiuser;PWD=Azurepbu@921')

import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
#Setting directory of data
#os.chdir(r'D:\SVAAS_2\No_show_analysis\svaas_04_07_2022')
diag_Test= pd.read_sql("select * from dbo.dim_diagnostic_test",connection)
appt= pd.read_sql("select * from dbo.brdg_appt_conslt_mapping",connection)
appt_status=pd.read_sql("select * from dbo.fact_appt_status_hist",connection)
doct_jo=pd.read_sql("select * from dboumd.SVAAS_Doctor_Appointment_Journey_vw",connection)
doct_jo2=doct_jo[['dr_id','dr_Name','dr_gender','dr_specialization','dr_exp','tenant_id','consultation_fee','Service_Type','payable_by_user','payable_by_insurance','brdg_appt_conslt_sid','Booked_schedule','re_scheduled','in_progress_schedule','declined_schedule','noshow_schedule','missed_scheduled','completed_scheduled','appointment_start_scheduled','appointment_end_scheduled','Final_Status']]
doct_jo3=doct_jo2[['brdg_appt_conslt_sid','Final_Status']]
doct_jo3=doct_jo3.drop_duplicates(subset=['brdg_appt_conslt_sid','Final_Status'], keep='last')
doct_jo3.rename(columns = {'Final_Status':'doc_final_status'}, inplace = True)
diag_Test= pd.read_sql("select * from dbo.dim_diagnostic_test",connection)


#Changing to date time
appt_status.info()
appt_status['created_date']=pd.to_datetime(appt_status['created_date'])
appt_status['schedule_date']=pd.to_datetime(appt_status['schedule_date'])

appt_status['created_date']=pd.to_datetime(appt_status['created_date'])
appt_status.columns

# Sorting the data
appt_status.sort_values(by=['created_date'], ascending=True, inplace=True)
# Dropping duplicates
appt_status2=appt_status.drop_duplicates(subset=['brdg_appt_conslt_sid','schedule_status'], keep='last')
appt_status2.drop('row_insert_dt',axis=1,inplace=True)

# Here we are considering only the newest time.

#Pivoting
pivoted= appt_status2.pivot(index=['brdg_appt_conslt_sid',"appointment_id",'start_time', 'end_time'], 
                            columns='schedule_status', values="created_date").reset_index()

#for final status
appt_status3=appt_status2[['brdg_appt_conslt_sid','schedule_status']]
appt_status3=appt_status3.drop_duplicates(subset=['brdg_appt_conslt_sid'], keep='last')

pivoted1= pd.merge(pivoted,appt_status3,how='left',on='brdg_appt_conslt_sid')
# Adding columns complted, and consulation time
pivoted1['complted_time']=pivoted1['confirmed']-pivoted1['booked']
pivoted1['conslutation_time']=pivoted1['completed']-pivoted1['in progress']
# copy the data

# Extracting hours from complted_time
pivoted1['complted_time_hours']=pivoted1['complted_time'].dt.total_seconds()/3600
# Extracting hours from complted_time
pivoted1['conslutation_time_hours']=pivoted1['conslutation_time'].dt.total_seconds()/3600

appt_status_final=pivoted1.drop(['complted_time','conslutation_time'],axis=1)
# merging with brdg appt conslut mapping
appt_status_final.rename(columns = {'schedule_status':'final_status'}, inplace = True)

appt2=appt.drop(['row_update_dt','row_insert_dt','clinic_id','policy_id','policy_sid','insurance_partner_sid',
                 'cancellation_reason','svaas_appointment_id','prescription','next_visit','appointment_dt_sid',
                 'appointment_id','appointment_datetime','consultation_id'],axis=1)

dt=pd.merge(appt2,appt_status_final,how='outer',on='brdg_appt_conslt_sid')

dt=pd.merge(dt,doct_jo3,how='outer',on='brdg_appt_conslt_sid')
# =============================================================================
# Prescribed medicine and prescribe test
# =============================================================================
med=pd.read_sql("select * from dbo.prescribed_medicine_mapping",connection)
test=pd.read_sql("select * from dbo.prescribed_test_mapping",connection)

pres_med=med.groupby('brdg_appt_conslt_sid').agg({'prescribed_medicine':'count'}).rename(columns={"prescribed_medicine":"pres_med_cnt"}).sort_values(by='pres_med_cnt',ascending=False)
# Mapped medicine count
mapped_med=med.groupby('brdg_appt_conslt_sid').agg({'mapped_medicine_name':'count'}).rename(columns={"mapped_medicine_name":"mapped_medicine_count"}).sort_values(by='mapped_medicine_count',ascending=False)

pres_med=pd.merge(pres_med,mapped_med,how='outer',on='brdg_appt_conslt_sid')

#test
pres_test=test.groupby('brdg_appt_conslt_sid').agg({'prescribed_test':'count'}).rename(columns={"prescribed_test":"pres_test_cnt"}).sort_values(by='pres_test_cnt',ascending=False)
mapped_test=test.groupby('brdg_appt_conslt_sid').agg({'mapped_test_name':'count'}).rename(columns={"mapped_test_name":"mapped_test_count"}).sort_values(by='mapped_test_count',ascending=False)

pres_test=pd.merge(pres_test,mapped_test,how='outer',on='brdg_appt_conslt_sid')

#merging medicine and test
med_test=pd.merge(pres_med,pres_test,how='outer',on='brdg_appt_conslt_sid')

#merging with dt and med_test
dt2=pd.merge(dt,med_test,how='outer',on='brdg_appt_conslt_sid')

# =============================================================================
# Bill summary
# =============================================================================
bill=pd.read_sql("select * from dbo.fact_bill_summary",connection)
bill_order_Status_hist=pd.read_sql("select * from dbo.fact_bill_order_status_hist",connection)
bill_order_Status_hist=bill_order_Status_hist[['order_status','brdg_bill_sid']]
bill_order_Status_hist.columns=['order_status','bill_sid']
bill=pd.merge(bill,bill_order_Status_hist,how='left',on='bill_sid')



doc_bill=bill[bill['vendor_type']=="Doctor"]
pharmacy_bill=bill[bill['vendor_type']=="Pharmacy services"]
lab_bill=bill[bill['vendor_type']=="Diagnostic service process"]

doc_bill=doc_bill[['insurance_partner_name','bill_id','service_type','brdg_appt_conslt_sid','bill_sid','doctor_sid','doctor_clinic_addr_sid',
                   'patient_sid','vendor_sid','transaction_dt','consultation_amount','paid_by_customer',
                   'paid_by_insurer','pg_charges','transaction_amount','payment_type','overall_transaction_status',
                   'refund_status','total_gst_amount','convenience_charge','delivery_charge','total_transaction_amount',
                   'platform_fee_amount','is_pay_out','is_settle_vendor','is_settle_vendor_datetime','payout_datetime','source_payment_type',
                   'marketing_fees_amount_settle','platform_fee_amount_settle','source_pg_charges','order_status']]

pharmacy_bill=pharmacy_bill[['insurance_partner_name','bill_id','service_type','pharmacy_sid','pharmacy_branch_id','pharmacy_order_type','pharmacy_flag','bill_sid','doctor_sid','doctor_clinic_addr_sid','brdg_appt_conslt_sid',
                   'patient_sid','vendor_sid','transaction_dt','consultation_amount','paid_by_customer',
                   'paid_by_insurer','pg_charges','transaction_amount','payment_type','overall_transaction_status',
                   'refund_status','total_gst_amount','convenience_charge','delivery_charge','total_transaction_amount',
                   'platform_fee_amount','is_pay_out','is_settle_vendor','is_settle_vendor_datetime','payout_datetime','source_payment_type',
                   'marketing_fees_amount_settle','platform_fee_amount_settle','source_pg_charges','order_status']]

lab_bill=lab_bill[['insurance_partner_name','bill_id','service_type','test_type','lab_id','diagnostic_lab_sid','brdg_appt_conslt_sid','bill_sid','doctor_sid','doctor_clinic_addr_sid',
                   'patient_sid','vendor_sid','transaction_dt','consultation_amount','paid_by_customer',
                   'paid_by_insurer','pg_charges','transaction_amount','payment_type','overall_transaction_status',
                   'refund_status','total_gst_amount','convenience_charge','delivery_charge','total_transaction_amount',
                   'platform_fee_amount','is_pay_out','is_settle_vendor','is_settle_vendor_datetime','payout_datetime','source_payment_type',
                   'marketing_fees_amount_settle','platform_fee_amount_settle','source_pg_charges','order_status']]

#Merging doc_bill with dt2
dt3=pd.merge(dt2,doc_bill,how='outer',on='brdg_appt_conslt_sid')

#dt3.to_csv('first_file_py.csv',index=False)
#pharmacy_bill.to_csv('pharmacy_bill_py.csv',index=False)
#lab_bill.to_csv('lab_bill_py.csv',index=False)

#####################################################################################################################

dt4=dt3[(dt3['completed'].notnull()) & (dt3['in progress'].notnull())]

dt4=dt4.groupby('patient_sid_y').agg({'patient_sid_y':'count'})
dt4.columns=['no_repeated_patients']
dt4=dt4.reset_index()
dt4.to_csv('repeated_patients_py.csv',index=False)

traffic=dt3[dt3['completed'].notnull()][['patient_sid_y','doctor_sid_y','completed']].reset_index(drop=True)
traffic['last_date']=traffic['completed'].max()
traffic['diff']=traffic['last_date']-traffic['completed']
traffic['last_week']=traffic['diff']<='7 days'
traffic['last_month']=traffic['diff']<='28 days'

traffic['last_week']=traffic.last_week.replace({True: 1, False: 0})
traffic['last_month']=traffic.last_month.replace({True: 1, False: 0})

traffic.loc[(traffic['last_week']==0) &  (traffic['diff']<='14 days'), 'last_week2'] = 1
traffic['last_week2'] = traffic['last_week2'].replace(np.nan, 0)

traffic.loc[(traffic['last_week']==0) & (traffic['last_week2']==0) & (traffic['diff']<='21 days'),'last_week3']=1
traffic['last_week3'] = traffic['last_week3'].replace(np.nan, 0)

traffic.loc[(traffic['last_week']==0) & (traffic['last_week2']==0) & (traffic['last_week3']==0)&(traffic['diff']<='28 days'),'last_week4']=1
traffic['last_week4'] = traffic['last_week4'].replace(np.nan, 0)


t1=traffic.groupby('doctor_sid_y').agg({'last_week':'max'}).reset_index()
t2=traffic.groupby('doctor_sid_y').agg({'last_week2':'max'}).reset_index()
t3=traffic.groupby('doctor_sid_y').agg({'last_week3':'max'}).reset_index()
t4=traffic.groupby('doctor_sid_y').agg({'last_week4':'max'}).reset_index()

t5=pd.merge(t1,t2,on='doctor_sid_y',how='inner')
t6=pd.merge(t3,t4,on='doctor_sid_y',how='inner')
traffic1=pd.merge(t5,t6,on='doctor_sid_y',how='inner')

traffic1['sum_weeks']=traffic1['last_week']+traffic1['last_week2']+traffic1['last_week3']+traffic1['last_week4']

t7=traffic1.groupby('doctor_sid_y').agg({'sum_weeks':'max'}).reset_index()
#to make column sum_month
t8=traffic.groupby('doctor_sid_y').agg({'last_month':'sum'}).rename(columns={'last_month':'sum_month'}).reset_index()
traffic2=pd.merge(t7,t8,on='doctor_sid_y',how='inner')

traffic2=traffic2.drop_duplicates(['doctor_sid_y','sum_weeks','sum_month'])

traffic2.loc[(traffic2['sum_weeks']==4), 'traffic_doc'] = 4
traffic2['traffic_doc'] = traffic2['traffic_doc'].replace(np.nan, 0)

traffic2.loc[(traffic2['traffic_doc']==0) & (traffic2['sum_month']>=4), 'traffic_doc']=3
traffic2.loc[(traffic2['traffic_doc']==0) & (traffic2['sum_month']>=1) & (traffic2['sum_month']<=3), 'traffic_doc']=2
traffic2.loc[(traffic2['traffic_doc']==0) & (traffic2['sum_month']==0), 'traffic_doc']=1

traffic2.to_csv('traffic_doc_py.csv',index=False)

###################################################################################################
#dim medicine
med_mas= pd.read_sql("select * from dbo.dim_medicine",connection)
med_mas=med_mas[['medicine_sid','medicine_name','medicine_brand_name','manufacturer','medicine_form_name','is_brand_formulary','accute_or_chronic','therapeutic_class']]
med_mas=med_mas.replace('#', '', regex=True)
med_mas.to_csv('med_mas_py.csv',index=False)

#########################################################################################
dt5=dt3[['brdg_appt_conslt_sid','patient_sid_y','booked','doctor_sid_y']]
dt5.info()
dt5=dt5[(dt5['patient_sid_y'].notnull()) & (dt5['patient_sid_y']!= -1) &(dt5['doctor_sid_y'].notnull())&(dt5['booked'].notnull())]
dt5=dt5.sort_values(['patient_sid_y','booked'])

d1=dt5.groupby(['doctor_sid_y','patient_sid_y']).agg({'booked':'min'}).rename(columns={'booked':'start_booked_doc'}).reset_index()
d2=dt5.groupby('patient_sid_y').agg({'booked':'min'}).rename(columns={'booked':'start_booked'}).reset_index()
dt6=pd.merge(d1,d2,on='patient_sid_y',how='inner')

dt7= pd.merge(dt5,dt6,on=['doctor_sid_y','patient_sid_y'],how='inner')

dt7.loc[dt7['start_booked']==dt7['booked'], 'is_new_patient']=1
dt7['is_new_patient'] = dt7['is_new_patient'].replace(np.nan, 0)

dt7.loc[dt7['start_booked_doc']==dt7['booked'], 'is_new_patient_doc']=1
dt7['is_new_patient_doc'] = dt7['is_new_patient'].replace(np.nan, 0)
dt7=dt7[['brdg_appt_conslt_sid','is_new_patient','is_new_patient_doc']]
dt7.to_csv('is_new_patient_py.csv',index=False)

################################## Lab bill##############################################################
bill_lab_convert=pd.merge(doc_bill[['brdg_appt_conslt_sid','transaction_dt','patient_sid']],
                          lab_bill[['transaction_dt','is_settle_vendor_datetime','is_pay_out','is_settle_vendor','patient_sid','doctor_sid','paid_by_customer','paid_by_insurer','bill_sid','bill_id','insurance_partner_name','refund_status','transaction_amount','overall_transaction_status','payment_type','total_transaction_amount','order_status','marketing_fees_amount_settle','platform_fee_amount_settle']],
                          how='outer',on='patient_sid')

bill_lab_convert=bill_lab_convert[bill_lab_convert['transaction_dt_y'].notnull()]
bill_lab_convert['transaction_dt_y']=pd.to_datetime(bill_lab_convert['transaction_dt_y'])
bill_lab_convert['transaction_dt_x']=pd.to_datetime(bill_lab_convert['transaction_dt_x'])
bill_lab_convert['diff']=bill_lab_convert['transaction_dt_y']-bill_lab_convert['transaction_dt_x']

bill_lab_convert=bill_lab_convert[(bill_lab_convert['diff']>'-1 days')&(bill_lab_convert['diff']>'11 days')]

bill_lab_convert.sort_values(['patient_sid','brdg_appt_conslt_sid','transaction_dt_x','transaction_dt_y'],ascending=True,inplace=True)
bill_lab_convert=bill_lab_convert.drop_duplicates(subset=['patient_sid','brdg_appt_conslt_sid','transaction_dt_x'], keep='first')

bill_lab_convert.sort_values(['patient_sid','brdg_appt_conslt_sid','diff'],ascending=True,inplace=True)
bill_lab_convert=bill_lab_convert.drop_duplicates(subset=['patient_sid','bill_sid','insurance_partner_name'], keep='first')
#necessary columns
bill_lab_convert=bill_lab_convert[['brdg_appt_conslt_sid','is_pay_out','is_settle_vendor_datetime','is_settle_vendor','paid_by_customer','paid_by_insurer','overall_transaction_status','refund_status','transaction_amount','payment_type','total_transaction_amount','order_status','marketing_fees_amount_settle','platform_fee_amount_settle']]
bill_lab_convert.columns=['brdg_appt_conslt_sid','is_pay_out_lab','is_settle_vendor_datetime_lab','is_settle_vendor_lab','paid_by_customer_lab','paid_by_insurer_lab','overall_transaction_status_lab','refund_status_lab','transaction_amount_lab','payment_type_lab','total_transaction_amount_lab','order_status_lab','marketing_fees_amount_settle_lab','platform_fee_amount_settle_lab']
dt22=pd.merge(dt3,bill_lab_convert,on='brdg_appt_conslt_sid',how='outer')

############################### Pharma medicine ###################################################################

bill_med_convert=pd.merge(doc_bill[['brdg_appt_conslt_sid','transaction_dt','patient_sid']],
                          pharmacy_bill[['transaction_dt','is_settle_vendor_datetime','is_pay_out','is_settle_vendor','patient_sid','doctor_sid','paid_by_customer','paid_by_insurer','bill_sid','bill_id','insurance_partner_name','refund_status','transaction_amount','overall_transaction_status','payment_type','total_transaction_amount','pharmacy_sid','order_status','marketing_fees_amount_settle','platform_fee_amount_settle']],
                          how='outer',on='patient_sid')


bill_med_convert=bill_med_convert[bill_med_convert['transaction_dt_y'].notnull()]
bill_med_convert['transaction_dt_y']=pd.to_datetime(bill_med_convert['transaction_dt_y'])
bill_med_convert['transaction_dt_x']=pd.to_datetime(bill_med_convert['transaction_dt_x'])
bill_med_convert['diff']=bill_med_convert['transaction_dt_y']-bill_med_convert['transaction_dt_x']

bill_med_convert=bill_med_convert[(bill_med_convert['diff']>'-1 days')&(bill_med_convert['diff']>'11 days')]

bill_med_convert.sort_values(['patient_sid','brdg_appt_conslt_sid','transaction_dt_x','transaction_dt_y'],ascending=True,inplace=True)
bill_med_convert=bill_med_convert.drop_duplicates(subset=['patient_sid','brdg_appt_conslt_sid','transaction_dt_x'], keep='first')

bill_med_convert.sort_values(['patient_sid','brdg_appt_conslt_sid','diff'],ascending=True,inplace=True)
bill_med_convert=bill_med_convert.drop_duplicates(subset=['patient_sid','bill_sid','insurance_partner_name'], keep='first')
#nexessary columns
bill_med_convert=bill_med_convert[['brdg_appt_conslt_sid','is_pay_out','is_settle_vendor_datetime','is_settle_vendor','paid_by_customer','paid_by_insurer','overall_transaction_status','refund_status','transaction_amount','payment_type','total_transaction_amount','pharmacy_sid','order_status','marketing_fees_amount_settle','platform_fee_amount_settle']]
bill_med_convert.columns=['brdg_appt_conslt_sid','is_pay_out_med','is_settle_vendor_datetime_med','is_settle_vendor_med','paid_by_customer_med','paid_by_insurer_med','overall_transaction_status_med','refund_status_med','transaction_amount_med','payment_type_med','total_transaction_amount_med','pharmacy_sid','order_status_med','marketing_fees_amount_settle_med','platform_fee_amount_settle_med']

dt22=pd.merge(dt22,bill_med_convert,on='brdg_appt_conslt_sid',how='outer')
dt22.columns

# Removing the useless columns
dt23=dt22.drop(['doctor_sid_x','vendor_sid_y','patient_id','patient_sid_x','appointment_id','insurance_partner_name','bill_id','doctor_clinic_addr_sid'],axis=1)

dt23.columns
dt23.columns=['brdg_appt_conslt_sid', 'appointment_status', 'appointment_type',
       'vendor_sid_x', 'doctor_id', 'insurance_company_name', 'complaints',
       'diagnosis', 'advice', 'bp_dia', 'bp_sys', 'pulse', 'weight', 'height',
       'bmi','consultation_doc_ref' ,'start_time', 'end_time', 'booked', 'cancelled', 'completed',
       'confirmed', 'declined', 'in progress', 'missed', 'no_show', 'pending',
       're-scheduled', 'final_status', 'complted_time_hours',
       'conslutation_time_hours', 'doc_final_status', 'pres_med_cnt',
       'mapped_medicine_count', 'pres_test_cnt', 'mapped_test_count',
       'service_type','bill_sid', 'doctor_sid_y', 'patient_sid_y_doc', 'transaction_dt_doc',
       'consultation_amount_doc', 'paid_by_customer_doc', 'paid_by_insurer_doc',
       'pg_charges_doc', 'transaction_amount_doc', 'payment_type_doc',
       'overall_transaction_status_doc', 'refund_status_doc', 'total_gst_amount_doc',
       'convenience_charge_doc', 'delivery_charge_doc', 'total_transaction_amount_doc',
       'platform_fee_amount_doc', 'is_pay_out_doc', 'is_settle_vendor_doc',
       'is_settle_vendor_datetime_doc', 'payout_datetime_doc', 'source_payment_type_doc',
       'marketing_fees_amount_settle_doc', 'platform_fee_amount_settle_doc',
       'source_pg_charges','order_status', 'is_pay_out_lab', 'is_settle_vendor_datetime_lab',
       'is_settle_vendor_lab', 'paid_by_customer_lab', 'paid_by_insurer_lab',
       'overall_transaction_status_lab', 'refund_status_lab',
       'transaction_amount_lab', 'payment_type_lab',
       'total_transaction_amount_lab','order_status_lab','marketing_fees_amount_settle_lab','platform_fee_amount_settle_lab', 'is_pay_out_med',
       'is_settle_vendor_datetime_med', 'is_settle_vendor_med',
       'paid_by_customer_med', 'paid_by_insurer_med',
       'overall_transaction_status_med', 'refund_status_med',
       'transaction_amount_med', 'payment_type_med',
       'total_transaction_amount_med','pharmacy_sid','ordder_status_med','marketing_fees_amount_settle_med','platform_fee_amount_settle_med']

dt23.to_csv('final_first_File_py.csv',index=False)


######################### dim_doctor #######################################################

del(t1)
del(t2)
del(t3)
del(t4)
del(t5)
del(t7)
del(t8)

del(t6)

