import smtplib
import mimetypes
import os
from email.message import EmailMessage
from datetime import date

# CONFIG
SENDER_EMAIL = 'stockpredicteralert@outlook.com'
SENDER_PASSWORD = 'lsrzmzwyeeaocdli'
RECIPIENTS = ['archisdhar@gmail.com']  # Add any additional emails to this list
SUBJECT = f"Weekly Portfolio Update - {date.today()}"
BODY = "Here is your updated portfolio for this week."

# FILE TO SEND
file_path = os.path.join('portfolio', 'portfoliogrowth_with_shares.csv')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find file at: {file_path}")

# Build Email
msg = EmailMessage()
msg['From'] = SENDER_EMAIL
msg['To'] = ', '.join(RECIPIENTS)
msg['Subject'] = SUBJECT
msg.set_content(BODY)

# Attach file
ctype, encoding = mimetypes.guess_type(file_path)
maintype, subtype = ctype.split('/', 1)

with open(file_path, 'rb') as fp:
    msg.add_attachment(fp.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(file_path))

# Send Email
try:
    with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print("✅ Email sent successfully.")
except Exception as e:
    print(f"failed to send email: {e}")
