# PROGRAM NAME: main.py
# PROGRAM PURPOSE: Runs the GUI client using the Predict.py code
# PROGRAMMER: Dillon Pietsch
# DATE WRITTEN: 7-17-2017

from final_code import run_predictor as rp

# Enter in valid ticker for stock you want to be predicted
ticker = 'TSLA'
rp(ticker, '2010-07-24')
