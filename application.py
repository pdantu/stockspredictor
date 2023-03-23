from flask import Flask,render_template,request,redirect,url_for,session,flash
from os import listdir
import os
import calculate
import pandas as pd


application = Flask(__name__)

@application.route('/',methods = ['POST','GET'])
def index():
    return '<form action = "/calculateform", method = "post"><button type="submit">Calculate</button></form>'

@application.route('/calculateform',methods = ['POST', 'GET'])
def calculateform():
    filenames = listdir(os.getcwd() + '/metrics')
    metrics = [ filename for filename in filenames if filename.endswith( '.csv' ) ]
    return render_template('calculateform.html',metrics=metrics)

@application.route('/calculate',methods=['POST', 'GET'])
def calculatee():
    calculatec = calculate.CalculateStocks()

    weightsdict = {'Forward EPS': int(request.form.getlist('weights')[0]), 'Forward P/E': int(request.form.getlist('weights')[1]), 'PEG Ratio': int(request.form.getlist('weights')[2]), 'Market Cap': int(request.form.getlist('weights')[3]), 'Price To Book': int(request.form.getlist('weights')[4]), 'Return on Equity': int(request.form.getlist('weights')[5]), 'Free Cash Flow': int(request.form.getlist('weights')[6]), 'Revenue Growth': int(request.form.getlist('weights')[7]), 'Dividend Yield': int(request.form.getlist('weights')[8]), 'Deb To Equity': int(request.form.getlist('weights')[9])}
    # return str(request.form.getlist('weights'))
    metricswanted = request.form.getlist('metr')


    calculatec.weightdict = weightsdict
    calculatec.calcResults(calculatec.path,metricswanted, 'custom')
    df = pd.read_csv('{0}/portfolio/portfolio{1}.csv'.format(calculatec.path, 'custom'))
    # writePortfolioToLogs(path,df)
    # sendEmail(path)
    z = df.to_html()
    print (z)
    return z

    return "Hi"


if __name__ == '__main__':
    application.run(debug=True)