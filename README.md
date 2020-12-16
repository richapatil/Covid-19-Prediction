# Covid-19-Prediction
A Boosted Random Forest Classification model was implemented to predict the outcome of the CoV-2 positive patients. Based on multiple features.

The original dataset was taken from: https://github.com/Atharva-Peshkar/Covid-19-Patient-Health-Analytics/blob/master/final.csv 
But it is modified according to our use.

final(1).csv -> dataset
RF_model.py  -> Main code
app.py       -> flask code
templates    -> contains HTML code file
index.html   -> contains GUI interface of the model for predicting
model.pkl    -> file created for loading the code in app.py (it is generated while running RF_model.py)

Steps to reproduce the results:
1) Load RF_model.py 
2) Then run the app.py

Note : Run above two files on cmd using command : python file_name.py
