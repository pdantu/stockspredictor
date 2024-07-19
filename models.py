from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    etf = db.Column(db.String(50), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    technical_action = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

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
    date = db.Column(db.DateTime, default=datetime.utcnow)
