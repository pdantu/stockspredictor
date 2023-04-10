from flask import Flask,render_template,request,redirect,url_for,session,flash
from os import listdir
import os
import calculate
import csv
import pandas as pd


application = Flask(__name__)
application.secret_key = 'the random string'

@application.route('/',methods = ['POST','GET'])
def index():
    return render_template('index.html')

@application.route('/calculateform',methods = ['POST', 'GET'])
def calculateform():
    filenames = listdir(os.getcwd() + '/metrics')
    metrics = [ filename for filename in filenames if filename.endswith('.csv')]

    for i in range(len(metrics)):
        metrics[i] = metrics[i].replace('-metrics.csv', '')

    return render_template('calculateform.html',metrics=metrics)

@application.route('/calculate',methods=['POST', 'GET'])
def calculatee():
    print("CALCULATING...")

    calculatec = calculate.CalculateStocks()

    # weightsdict = {'Forward EPS': int(request.form.getlist('weights')[0]), 'Forward P/E': int(request.form.getlist('weights')[1]), 'PEG Ratio': int(request.form.getlist('weights')[2]), 'Market Cap': int(request.form.getlist('weights')[3]), 'Price To Book': int(request.form.getlist('weights')[4]), 'Return on Equity': int(request.form.getlist('weights')[5]), 'Free Cash Flow': int(request.form.getlist('weights')[6]), 'Revenue Growth': int(request.form.getlist('weights')[7]), 'Dividend Yield': int(request.form.getlist('weights')[8]), 'Deb To Equity': int(request.form.getlist('weights')[9])}
    # return str(request.form.getlist('weights'))
    # metricswanted = request.form.getlist('metr')

    tempMetrics = []

    for i in session['metricswanted']:
        tempMetrics.append(i + '-metrics.csv')

    calculatec.weightdict = session['weightsdict']
    calculatec.calcResults(calculatec.path, tempMetrics, 'custom')
    # df = pd.read_csv('{0}/portfolio/portfolio{1}.csv'.format(calculatec.path, 'custom'))
    results = []
    with open('{0}/portfolio/portfolio{1}.csv'.format(calculatec.path, 'custom')) as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                    row[i] = round(row[i], 2)
                except:
                    print("Not a float")
            results.append(row)
            print(row)
    # writePortfolioToLogs(path,df)
    # sendEmail(path)
    session['port'] = results
    print("-------------------------------------------------------------------------------------------")
    return redirect(url_for('viewPortfolio'))

@application.route('/calcHelper',methods=['POST', 'GET'])
def calcHelper():
    session['weightsdict'] = {'Forward EPS': int(request.form.getlist('weights')[0]), 'Forward P/E': int(request.form.getlist('weights')[1]), 'PEG Ratio': int(request.form.getlist('weights')[2]), 'Market Cap': int(request.form.getlist('weights')[3]), 'Price To Book': int(request.form.getlist('weights')[4]), 'Return on Equity': int(request.form.getlist('weights')[5]), 'Free Cash Flow': int(request.form.getlist('weights')[6]), 'Revenue Growth': int(request.form.getlist('weights')[7]), 'Dividend Yield': int(request.form.getlist('weights')[8]), 'Deb To Equity': int(request.form.getlist('weights')[9])}
    session['metricswanted'] = request.form.getlist('metr')
    return render_template('loading.html')

@application.route('/viewPortfolio',methods=['POST', 'GET'])
def viewPortfolio():
    return render_template('viewPortfolio.html', port = session['port'], rows = len(session['port']))

if __name__ == '__main__':
    application.run(debug=True)