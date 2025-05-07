import os
import pandas as pd
import numpy as np
import math
from datetime import datetime, date, timedelta
import requests

class CalculateStocks:
    def __init__(self) -> None:
        self.path = os.getcwd()
        self.weightdict = {}
        self.api_key = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'

    def main(self):
        print(self.hasUpcomingEarnings('VRSK'))
        # f_list = self.loop(self.path, False)
        # types = ['growth']
        # types = ['value', 'income']
        # for t in types:
        #     self.calcResults(self.path, f_list, t)
        #     df = pd.read_csv(f'{self.path}/portfolio/portfolio{t}.csv')
        #     self.addCompName(df, t)

    def loop(self, path, results):
        path += "/metrics" if not results else "results"
        return [f for f in os.listdir(path) if f.endswith(".csv")]
    
    def hasUpcomingEarnings(self, ticker):
        today = datetime.today().date()
        two_weeks = today + timedelta(days=14)
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar?symbol={ticker}&limit=5&apikey={self.api_key}"

        try:
            print(f"🔍 Checking earnings for {ticker}")
            resp = requests.get(url)
            data = resp.json()

            if not data:
                print(f"ℹ️ No earnings data returned for {ticker}")
                return False
            
            for entry in data:
                report_date = datetime.strptime(entry['date'], "%Y-%m-%d").date()
                print(f"📅 Earnings date for {ticker}: {report_date}")
                if today <= report_date <= two_weeks:
                    print(f"⚠️ Skipping {ticker} — earnings on {report_date}")
                    return True
        except Exception as e:
            print(f"❌ Error checking earnings for {ticker}: {e}")
        return False


    def calcResults(self, path, f_list, type_):
        d_list = []
        for name in f_list:
            if 'SPY' in name or 'QQQ' in name:
                continue
            df = pd.read_csv(f'{path}/metrics/{name}')
            print(f'Processing: {name}')
            d_list = self.process(d_list, df, name, type_)
        print(d_list)
        if d_list:
            portfolio = pd.concat(d_list)
            scoresum = portfolio['Score'].sum()
            portfolio['weight'] = (portfolio['Score'] / scoresum) * 100
            portfolio['Dollar Amount'] = portfolio['weight'] / 100 * 5000
            portfolio.sort_values(by='weight', ascending=False, inplace=True)
            portfolio_path = os.path.join(self.path, 'portfolio')
            os.makedirs(portfolio_path, exist_ok=True)
            portfolio.to_csv(os.path.join(portfolio_path, f'portfolio{type_}.csv'), index=False)

    def process(self, d_list, df, sector_file, type_):
        sector = sector_file.split('-')[0]
        etfFraction = self.getETFaction(sector)
        stocksdf = pd.DataFrame(columns=['ETF', 'Ticker', 'Technical Action', 'Score'])

        for symbol in df['Ticker']:
            if symbol == 'Other':
                continue
            prices = self.getPrices(symbol)
            if prices.empty:
                continue

            sharpe = self.calcSharpe(prices)
            measure = self.getSignalMeasure(prices)
            self.setWeightDict(type_)

            sc = self.getScore(sector, symbol, sharpe, self.weightdict)
            action = 'Buy' if measure == 1 else 'Sell'
            stocksdf.loc[len(stocksdf)] = [sector, symbol, action, sc]

        stocksdf.sort_values(by='Score', ascending=False, inplace=True)
        buys = stocksdf[stocksdf['Technical Action'] == 'Buy']
        stocksdf.to_csv(f'{self.path}/results/{sector}{type_}-action.csv', index=False)
        buys.to_csv(f'{self.path}/results/{sector}{type_}-buys.csv', index=False)

        strongbuys = buys[buys['Score'] > 0]
        x = buys[buys['Score'] > 70]
        if x.shape[0] > 3:
            etfFraction = x.shape[0]
        d_list.append(strongbuys.head(etfFraction))
        return d_list

    def getPrices(self, ticker):
        FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&apikey={FMP_API_KEY}"

        def get_last_weekday(date):
            while date.weekday() > 4:  # Skip Sat (5) and Sun (6)
                date -= timedelta(days=1)
            return date

        try:
            resp = requests.get(url)
            data = resp.json().get("historical", [])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').reset_index(drop=True)
            df = df[['date', 'close']].rename(columns={'close': 'Close'})

            # 🛑 Check if last available date is recent enough
            last_date = df['date'].max().date()
            expected_last = get_last_weekday(datetime.today().date())

            if last_date < expected_last:
                print(f"⚠️ Skipping {ticker} — last price is from {last_date}, expected at least {expected_last}")
                return pd.DataFrame()

            # ✅ Compute technical indicators
            df['20DayEMA'] = df['Close'].ewm(span=20).mean()
            df['100DayEMA'] = df['Close'].ewm(span=100).mean()
            df['50DaySMA'] = df['Close'].rolling(window=50).mean()
            df['200DaySMA'] = df['Close'].rolling(window=200).mean()
            df['12DayEMA'] = df['Close'].ewm(span=12).mean()
            df['26DayEMA'] = df['Close'].ewm(span=26).mean()
            df['MACD'] = df['12DayEMA'] - df['26DayEMA']
            df['MACD Signal'] = df['MACD'].ewm(span=9).mean()

            return df

        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()


    def getETFaction(self, etf):
        prices = self.getPrices(etf)
        if prices.empty:
            return 1
        return 3 if prices['20DayEMA'].iloc[-1] > prices['100DayEMA'].iloc[-1] else 1

    def calcSharpe(self, prices):
        subprices = prices.tail(253)
        z = subprices['Close'].diff()
        returns = [z.iloc[i] / subprices['Close'].iloc[i - 1] for i in range(1, len(z))]
        if not returns:
            return 0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return mean / math.sqrt(variance) * math.sqrt(253) if variance else 0

    def getSignalMeasure(self, prices):
        measure = 1 if prices['20DayEMA'].iloc[-1] > prices['100DayEMA'].iloc[-1] else -1
        return measure

    def rsi(self, df, periods=14):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=periods - 1, min_periods=periods).mean()
        avg_loss = loss.ewm(com=periods - 1, min_periods=periods).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def setWeightDict(self, type_):
        if type_ == 'growth':
            self.weightdict = {
                'Forward EPS': 3, 'Forward P/E': 3, 'PEG Ratio': 3, 'Market Cap': 1,
                'Price To Book': 1, 'Return on Equity': 3, 'Free Cash Flow': 1,
                'Revenue Growth': 3, 'Dividend Yield': 1, 'Debt to Equity': 1, 'Earnings Growth': 3
            }
        elif type_ == 'value':
            self.weightdict = {
                'Trailing P/E': 3,
                'Forward P/E': 3,
                'PEG Ratio': 2,
                'Price To Book': 3,
                'Return on Equity': 2,
                'Market Cap': 1,
                'Debt to Equity': 2,
                'E/V to EBITDA': 3,
                'Free Cash Flow': 2,
                'Dividend Yield': 1,
                'Revenue Growth': 1,
                'Beta': 1
            }
        elif type_ == 'income':
            self.weightdict = {
                'Dividend Yield': 3,
                'Free Cash Flow': 3,
                'Forward EPS': 3,
                'Trailing EPS': 3,
                'Debt to Equity': 2,
                'Return on Equity': 2,
                'Forward P/E': 2,
                'PEG Ratio': 1,
                'Market Cap': 1,
                'Price To Book': 1,
                'Revenue Growth': 1  # income stocks can still grow
            }


    def getScore(self, etf, stock, sharpe, columns):
        df = pd.read_csv(f'{self.path}/metrics/{etf}-metrics.csv')
        df = df.rename(columns={
            'Dividend_Yield': 'Dividend Yield',
            'Forward_PE': 'Forward P/E',
            'Trailing_PE': 'Trailing P/E',
            'Market_Cap': 'Market Cap',
            'Trailing_EPS': 'Trailing EPS',
            'Forward_EPS': 'Forward EPS',
            'PEG_Ratio': 'PEG Ratio',
            'Price_To_Book': 'Price To Book',
            'EV_to_EBITDA': 'E/V to EBITDA',
            'Free_Cash_Flow': 'Free Cash Flow',
            'Debt_to_Equity': 'Debt to Equity',
            'Earnings_Growth': 'Earnings Growth',
            'Ebitda_Margins': 'Ebitda Margins',
            'Quick_Ratio': 'Quick Ratio',
            'Target_Mean_Price': 'Target Mean Price',
            'Return_on_Equity': 'Return on Equity',
            'Revenue_Growth': 'Revenue Growth',
            'Current_Ratio': 'Current Ratio',
            'Current_Price': 'Current Price'
        })
        factordict = {key: (1 if val > 0 else -1) for key, val in columns.items()}
        row = df[df['Ticker'] == stock]
        if row.empty:
            return 0
        row = row.fillna(0)
        score = 0
        for x in columns:
            if x not in df.columns:
                continue
            mean, std = df[x].mean(), df[x].std()
            val = row[x].iloc[0]
            if std == 0:
                continue
            z = (val - mean) / std if val >= mean else (mean - val) / std
            score += z * factordict.get(x, 0) * columns[x]
        return score + sharpe * 2

    def addCompName(self, data, type_):
        all_metrics = pd.concat([
            pd.read_csv(f'{self.path}/metrics/{file}')
            for file in os.listdir(f'{self.path}/metrics')
            if file.endswith('-metrics.csv')
        ])
        
        name_map = dict(zip(all_metrics['Ticker'], all_metrics.get('companyName', ['']*len(all_metrics))))
        data['Name'] = data['Ticker'].map(name_map).fillna('')
        data.to_csv(f'{self.path}/portfolio/portfolio{type_}.csv', index=False)


if __name__ == "__main__":
    cs = CalculateStocks()
    cs.main()
