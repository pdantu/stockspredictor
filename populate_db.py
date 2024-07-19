from app import app, db, Action, Metrics, SECTORS
import pandas as pd
import os
from datetime import datetime

def populate_actions():
    for sector in SECTORS:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], f'{sector}-action.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            for _, row in data.iterrows():
                action = Action(
                    etf=sector,
                    ticker=row['Ticker'],
                    technical_action=row['Technical Action'],
                    score=row['Score'],
                    date=datetime.now()  # or parse date from CSV if available
                )
                db.session.add(action)
    db.session.commit()

def populate_metrics():
    for sector in SECTORS:
        file_path = os.path.join(app.config['CSV_FOLDER'], f'{sector}-metrics.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            for _, row in data.iterrows():
                metrics = Metrics(
                    beta=row['Beta'],
                    dividend_yield=row['Dividend Yield'],
                    forward_pe=row['Forward P/E'],
                    trailing_pe=row['Trailing P/E'],
                    market_cap=row['Market Cap'],
                    trailing_eps=row['Trailing EPS'],
                    forward_eps=row['Forward EPS'],
                    peg_ratio=row['PEG Ratio'],
                    price_to_book=row['Price To Book'],
                    ev_to_ebitda=row['E/V to EBITDA'],
                    free_cash_flow=row['Free Cash Flow'],
                    debt_to_equity=row['Debt to Equity'],
                    earnings_growth=row['Earnings Growth'],
                    ebitda_margins=row['Ebitda Margins'],
                    quick_ratio=row['Quick Ratio'],
                    target_mean_price=row['Target Mean Price'],
                    return_on_equity=row['Return on Equity'],
                    revenue_growth=row['Revenue Growth'],
                    current_ratio=row['Current Ratio'],
                    current_price=row['Current Price'],
                    date=datetime.now()  # or parse date from CSV if available
                )
                db.session.add(metrics)
    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        populate_actions()
        populate_metrics()
