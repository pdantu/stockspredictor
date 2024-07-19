from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import io
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    etf = db.Column(db.String(10), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    technical_action = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)

class Metrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    beta = db.Column(db.Float, nullable=False)
    dividend_yield = db.Column(db.Float, nullable=False)
    forward_pe = db.Column(db.Float, nullable=False)
    trailing_pe = db.Column(db.Float, nullable=False)
    market_cap = db.Column(db.Float, nullable=False)
    trailing_eps = db.Column(db.Float, nullable=False)
    forward_eps = db.Column(db.Float, nullable=False)
    peg_ratio = db.Column(db.Float, nullable=False)
    price_to_book = db.Column(db.Float, nullable=False)
    ev_to_ebitda = db.Column(db.Float, nullable=False)
    free_cash_flow = db.Column(db.Float, nullable=False)
    debt_to_equity = db.Column(db.Float, nullable=False)
    earnings_growth = db.Column(db.Float, nullable=False)
    ebitda_margins = db.Column(db.Float, nullable=False)
    quick_ratio = db.Column(db.Float, nullable=False)
    target_mean_price = db.Column(db.Float, nullable=False)
    return_on_equity = db.Column(db.Float, nullable=False)
    revenue_growth = db.Column(db.Float, nullable=False)
    current_ratio = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)

# Directories
RESULTS_FOLDER = 'results'
MERGED_CSV_PATH = 'merged.csv'
CSV_FOLDER = 'metrics'
PORTFOLIO_FOLDER = 'portfolio'
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['PORTFOLIO_FOLDER'] = PORTFOLIO_FOLDER


# List of available sectors (11 SPDR ETF sectors)
SECTORS = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

def merge_csv_files():
    all_data = pd.DataFrame()
    for sector in SECTORS:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], f'{sector}-action.csv')
        if os.path.exists(file_path):
            sector_data = pd.read_csv(file_path)
            # Add a 'Sector' column to identify which sector each row belongs to
            sector_data['Sector'] = sector
            all_data = pd.concat([all_data, sector_data], ignore_index=True)
    all_data.to_csv(MERGED_CSV_PATH, index=False)

# Merge CSV files on application startup
merge_csv_files()

# Load merged CSV into DataFrame
merged_data = pd.read_csv(MERGED_CSV_PATH)

@app.route('/')
def home():
    return render_template('indexx.html')

@app.route('/metrics', methods=['GET', 'POST'])
def metrics():
    selected_sector = None
    data = None
    if request.method == 'POST':
        selected_sector = request.form['sector']
        file_path = os.path.join(app.config['CSV_FOLDER'], f'{selected_sector}-metrics.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            data = None

    return render_template('metrics.html', sectors=SECTORS, selected_sector=selected_sector, data=data)

@app.route('/action', methods=['GET', 'POST'])
def action():
    selected_sector = None
    data = None
    if request.method == 'POST':
        selected_sector = request.form['sector']
        file_path = os.path.join(app.config['RESULTS_FOLDER'], f'{selected_sector}-action.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            data = None

    return render_template('action.html', sectors=SECTORS, selected_sector=selected_sector, data=data)

@app.route('/portfolios', methods=['GET', 'POST'])
def portfolios():
    portfolio_type = None
    data = None
    if request.method == 'POST':
        portfolio_type = request.form['portfolio']
        file_path = os.path.join(app.config['PORTFOLIO_FOLDER'], f'portfolio{portfolio_type.lower()}.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            data = None

    return render_template('portfolios.html', portfolio_type=portfolio_type, data=data)

@app.route('/search_stock', methods=['GET', 'POST'])
def search_stock():
    df = pd.read_csv('merged.csv')
    search_result = None
    if request.method == 'POST':
        stock_tickers = request.form['stock_ticker'].upper().split(',')
        stock_tickers = [ticker.strip() for ticker in stock_tickers]
        search_result = df[df['Ticker'].isin(stock_tickers)]
        if search_result.empty:
            search_result = None

    return render_template('search_stock.html', search_result=search_result)

@app.route('/download_search_results', methods=['POST'])
def download_search_results():
    df = pd.read_csv(MERGED_CSV_PATH)
    stock_tickers = request.form['stock_ticker'].upper().split(',')
    stock_tickers = [ticker.strip() for ticker in stock_tickers]
    search_result = df[df['Ticker'].isin(stock_tickers)]
    
    if search_result.empty:
        return "No data found for the given stock tickers."

    # Create a CSV file in memory
    output = io.StringIO()
    search_result.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, mimetype='text/csv', attachment_filename='search_results.csv', as_attachment=True)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
