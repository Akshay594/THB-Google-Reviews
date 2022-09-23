import os
import smtplib
from email.message import EmailMessage
import pandas as pd

email_id = 'gopalsingh@thb.co.in'
email_pass = 'passwordHere'

recipient_list = ['pawan@gmail.com', 'girdhar@gmail.com']

# Output from the healthcare code 1
df_final_req = pd.read_csv("output.csv")

unmapped_diags = df_final_req[df_final_req["final_diagnosis"].isna() == True]["mapped_DIAGNOSIS"].value_counts().reset_index()
unmapped_diags['status'] = "diagnosis"
unmapped_diags.columns = ["entity", "count", "status"]
unmapped_diags = unmapped_diags[unmapped_diags["entity"] != " "]

unmapped_complain = df_final_req[df_final_req["final_complain"].isna() == True]["mapped_COMPLAINT"].value_counts().reset_index()
unmapped_complain['status'] = "complain"
unmapped_complain.columns = ["entity", "count", "status"]
unmapped_complain = unmapped_complain[unmapped_complain["entity"] != " "]

final_unmapped_data = unmapped_diags.append(unmapped_complain)

final_unmapped_data.to_csv("unmapped_data.csv", index=False)

msg = EmailMessage()
msg['Subject'] = 'Unmapped Data'
msg['From'] = email_id
msg['To'] = recipient_list
msg.set_content('Here is the data for unmapped diagnosis/complains')

for each_file in os.listdir():
    if each_file == 'unmapped_data.csv':
        continue
    with open(each_file, 'rb') as f:
        file_data = f.read()
        file_name = f.name
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

with smtplib.SMTP_SSL('smtp.privatemail.com', 465) as smtp:
    smtp.login(email_id, email_pass)
    smtp.send_message(msg)

print("File has been deleted.")
os.remove("unmapped_data.csv")