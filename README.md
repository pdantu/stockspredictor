# stockspredicter


# 08/12/2022

Tasks -- 
   1) Technical Indicators on ETF's (50vs200 Exponentional Moving Averages, MACD vs MACD Signal, Relative Strength Index (RSI))
   2) Technical Indicators on stocks in sector (50vs200 Exponentional Moving Averages, MACD vs MACD Signal, Relative Strength Index)
        - do all the scoring, and rank the stocks for each sector 
        - look at neural nets and see how to properly incorporate weighted metric inputs 
        - Top 5 stocks for etfs that are in (buy) mode , 2 stocks for etfs that are in (sell) mode 
        - outputs a csv that indicates which stocks to buy and sell for the day 

   3) For each stock -- 
        1) PEG Ratio lower than mean --used
        2) Current Price
        3) Sharpe Ratio (make sure its high) --used
        4) Beta	
        5) Forward P/E	--used
        6) Trailing P/E	
        7) Market Cap	--used
        8) Trailing EPS	
        9) Forward EPS	--used
        10) PEG Ratio	--used
        11) Price To Book	--used
        12) E/V to EBITDA	--used
        13) Free Cash Flow	--used
        14) Deb To Equity	--used
        15) Earnings Growth	
        16) Ebitda margins	
        17) Quick Ratio	
        18) Target Mean Price	
        19) Return on Equity  --used 	
        20) Revenue Growth	--used
        21) Current Ratio	
        22) Dividend Yield	

    4) scrape for list of etfs (free)
       - for hedging (correlation against the spy price)
    

    5) Emailing/text message with list of stocks to buy or sell 

    6) Automation -- set up a cron job to run the script to generate market data into csv every day 

    7) sentiment analysis on the 'news' section 

    8) scrape yahoo finance 

    9) LSTM models with RNN 




Things to Look at: 
  -- different modules to scrape market data 
  -- understand options/futures trading more 

