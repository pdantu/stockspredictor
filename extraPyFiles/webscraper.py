import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import operator
import os
path = os.getcwd()

def main():
    lis = getETFList()
    bearmarket = getMarketDownturn()
    sortted = getLeastCorrelated(lis)
    getWeightings(sortted)

def getETFList():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds")
    html = r.text
    soup = BeautifulSoup(html, "html.parser")


    firstH3 = soup.find('h3') # Start here
    uls = []
    for nextSibling in firstH3.findNextSiblings():
        if nextSibling.name == 'h2':
            break
        if nextSibling.name == 'ul':
            uls.append(nextSibling)

    etfs = []
    for ul in uls:
        for li in ul.findAll('li'):
            
            a = li.text
            a = li.text.split('(')
            a = li.text.split(')')
            a = a[0].split()
            etfs.append(a[-1])
    return etfs
            
def getCorrelation(etf, etf2 = 'SPY'):
    etfdata = yf.Ticker(etf)
    etfprice = etfdata.history('max')
    if etfprice.shape[0] == 0:
        return 100
    etf2 = yf.Ticker(etf2)
    etf2price = etf2.history('max')
    corr = etfprice['Close'].corr(etf2price['Close'])
    return corr

def getLeastCorrelated(etf_list):
    corrdict = {}
    for x in etf_list:
        val = getCorrelation(x)
        if val < 0:
            corrdict[x] = val
    
    sortedcorrdict = dict(sorted(corrdict.items(),key=operator.itemgetter(1),reverse=False))
    return sortedcorrdict

def getMarketDownturn():
    spy = yf.Ticker('SPY')
    spyprice = spy.history('max')
    spyprice['50DayEMA'] = spyprice['Close'].ewm(span = 50).mean()
    spyprice['200DayEMA'] = spyprice['Close'].ewm(span = 200).mean()
    if spyprice['50DayEMA'].iloc[len(spyprice) - 1] < spyprice['200DayEMA'].iloc[len(spyprice) - 1]:
        return 1
    return 0

def getWeightings(corrdict):
    sumofcorr = sum(corrdict.values()) * -1
    weightdf = pd.DataFrame(columns = ['Symbol', 'Correlation', 'Weight'])
    for key in corrdict:
        weightdf.loc[len(weightdf.index)] = [key, corrdict[key], round(corrdict[key] * -1 / sumofcorr, 2)]
    weightdf['Dollar amt'] = weightdf['Weight'] * 1000

    weightdf.to_csv(path + '/results/hedging.csv')
if __name__ == "__main__":
    main()
"""
soup = BeautifulSoup(html, "html.parser")
data1 = soup.find('ul')
for li in data1.find_all("li"):
    soup2 = BeautifulSoup(li.text, "html.parser")
    data2 = soup2.find('li')
    print(type(li.text))
    for li2 in data2.find_all("a"):
        print(li2.text, end=" ")"""