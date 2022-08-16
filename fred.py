from fredapi import Fred
fred = Fred(api_key='13f1e0b5cbdcb8307bbf7bbca9852e4a')

categories = ['UNrate', 'GDP', 'SP500','CPALTT01USM657N' ]
data = fred.get_series_latest_release('CPALTT01USM657N')
print(data.tail())