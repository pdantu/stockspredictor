import pandas as pd
import os

440.8360598244557
438.02468159144104
429.94590661404027
427.4927018875449
359.01931029564395
350.79640538377294
338.1920211772185
328.2637757339408
300.49832572615844
295.0527818180247
294.39163252990573
284.6241747540494
273.8340136688757
231.23683193217408
207.79137706275424

# Load previous and current portfolio
previous = pd.read_csv('logs/2025-05-18_portfoliogrowth.csv')  # 🔁 update this file name as needed
current = pd.read_csv('portfolio/portfoliogrowth.csv')

# Extract tickers
prev_tickers = set(previous['Ticker'])
curr_tickers = set(current['Ticker'])

# Find differences
added = curr_tickers - prev_tickers
removed = prev_tickers - curr_tickers

# Format for CSV
added_df = pd.DataFrame({'Ticker': list(added), 'Action': 'Added'})
removed_df = pd.DataFrame({'Ticker': list(removed), 'Action': 'Removed'})
diff_df = pd.concat([added_df, removed_df])

# Save
diff_df.to_csv('logs/ticker_changes.csv', index=False)
print("✅ Saved ticker changes to logs/ticker_changes.csv")
