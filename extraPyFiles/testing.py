from distutils.text_file import TextFile
from operator import attrgetter, index
from venv import create
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from fileinput import filename
import pandas as pd 
import numpy as np
import os
from os import listdir
import yfinance as yf
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

path = os.getcwd()

def main():
    path = os.getcwd()
    path = path[0:path.find("/extraPyFiles")]
    #rankAll(path)
    #createGraphic(path)
    sendEmail(path)
    #print('hi')
    
    findDifference('{0}/logs/2022-08-15_portfolio.csv'.format(path), '{0}/results/portfolio.csv'.format(path))

def loop(path,results):
    if results:
        path += "/results"
    else:
        path += "/metrics"
    filenames = find_csv_filenames(path)
    return filenames

def sendEmail(path):
    sender_address = 'StocksPredictor123@outlook.com'
    sender_pass = 'Steelers2022!'
    #fileToSend = '{0}/results/portfolio.csv'.format(path)
    receiver_addresses = ['pdantu1234@gmail.com','archisdhar@gmail.com']
    attachments = ['{0}/extraPyFiles/difference.csv'.format(path)]
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = 'list@stockspredictor'
    message['Subject'] = 'NOT A SPAM!'
    #The subject line
    #The body and the attachments for the mail
    mail_content = 'hey'
    
    for fileToSend in attachments:
        
        name = fileToSend
        ctype, encoding = mimetypes.guess_type(fileToSend)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"

        maintype, subtype = ctype.split("/", 1)

        if maintype == "text":
            fp = open(fileToSend)
            # Note: we should handle calculating the charset
            attachment = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "image":
            fp = open(fileToSend, "rb")
            attachment = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "audio":
            fp = open(fileToSend, "rb")
            attachment = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(fileToSend, "rb")
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(fp.read())
            fp.close()
            encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename=name)
        message.attach(attachment)

    server = smtplib.SMTP("smtp-mail.outlook.com",587)
    server.starttls()
    server.login(sender_address,sender_pass)
    server.sendmail(sender_address, receiver_addresses, message.as_string())
    server.quit()
    print('Mail Sent')

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def createGraphic(path):
    portfolio = pd.read_csv('{0}/results/portfolio.csv'.format(path))
    filenames = loop(path,False)
    print(filenames)
    df2 = portfolio.groupby(['ETF'])['weight'].sum().reset_index()
    print(df2.head())
    
    df2.to_csv('{0}/results/sector_weights.csv'.format(path))



def rankAll(path):
    filenames = loop(path,True)
    df_list = []
    for file in filenames:
        if 'action' in file or 'QQQ' in file or 'SPY' in file or 'portfolio' in file:
            continue
        else:
            df = pd.read_csv('{0}/results/{1}'.format(path,file))
            df_list.append(df)
    
    rankings = pd.concat(df_list)
    rankings.sort_values(by='Score',ascending=False,inplace=True)
    rankings = rankings[rankings['Score'] > 0]
    rankings.to_csv('show.csv')

    portfolio = pd.read_csv('{0}/results/portfolio.csv'.format(path))

    a = rankings["Ticker"].tolist()
    b = portfolio["Ticker"].tolist()

    s = set(b)
    temp3 = [x for x in a if x not in s]
    print(temp3)
    newdf = rankings[rankings['Ticker'].isin(temp3)]
    newdf.to_csv('diff.csv')

def findDifference(df1,df2):
    df1 = pd.read_csv(df1)
    df2 = pd.read_csv(df2)
    series1 = df1['Ticker']
    series2 = df2['Ticker']
    
    union = pd.Series(np.union1d(series1, series2))
  
    # intersection of the series
    intersect = pd.Series(np.intersect1d(series1, series2))
    
    # uncommon elements in both the series 
    notcommonseries = union[~union.isin(intersect)]
    
    print(notcommonseries)
    x = list(notcommonseries)
    
    
    
    print(x)
    buys = df2.loc[df2['Ticker'].isin(x)]
    print(buys.head())
    sells = df1.loc[df1['Ticker'].isin(x)]
    sells['Technical Action'] = 'Sell'
    final_df = pd.concat([buys,sells])
    final_df.to_csv('difference.csv',index=False)
    
if __name__ == "__main__":
    main()
    
    
