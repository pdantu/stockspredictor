-- schema.sql

CREATE TABLE Action (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ETF TEXT,
    Ticker TEXT,
    TechnicalAction TEXT,
    Score REAL,
    date TEXT
);

CREATE TABLE Metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Beta REAL,
    DividendYield REAL,
    ForwardPE REAL,
    TrailingPE REAL,
    MarketCap REAL,
    TrailingEPS REAL,
    ForwardEPS REAL,
    PEGRatio REAL,
    PriceToBook REAL,
    EVtoEBITDA REAL,
    FreeCashFlow REAL,
    DebtToEquity REAL,
    EarningsGrowth REAL,
    EbitdaMargins REAL,
    QuickRatio REAL,
    TargetMeanPrice REAL,
    ReturnOnEquity REAL,
    RevenueGrowth REAL,
    CurrentRatio REAL,
    CurrentPrice REAL,
    date TEXT
);
