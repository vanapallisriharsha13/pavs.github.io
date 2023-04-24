import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
# # For time stamps
from datetime import datetime
import streamlit as st
import yfinance as yf
yf.pdr_override()
from datetime import datetime
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
st.title('Stock Trend Prediction')
user_input=st.text_input("Enter Stock Ticker",'GOOG')
df=yf.download(user_input, start, end)
st.subheader("Data from 2013 to 2023")
st.write(df.head(10))
st.subheader("Basic description of our stock")
st.write(df.describe())
st.write(df.info())
 
st.subheader("Historical Closing Price")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylabel('Adj Close')
plt.title(f"Closing Price of {user_input}")
st.pyplot(fig)

#moving averages of stock with 50 days
st.subheader("moving averages of stock with 50 days")
ma50=df.Close.rolling(50).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma50)
plt.plot(df.Close)
st.pyplot(fig)
 
#moving averages of stock with 100 days
st.subheader("moving averages of stock with 50 and 100 days")
ma50=df.Close.rolling(50).mean()
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma50,'r')
plt.plot(ma100,'b')
plt.plot(df.Close,'g')
st.pyplot(fig)
 
#the total volume of stock being traded each day
st.subheader("The total volume of stock being traded each day")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Volume)
plt.ylabel('Volume')
plt.title(f"Sales Volume for {user_input}")
st.pyplot(fig)

st.subheader("Compare the daily returns of two companies")
user_input2=st.text_input("Enter Another Stock ticker to compare",'MSFT')
tech_list=[user_input,user_input2]
# Comparing 2 companies
closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Close']
# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()
sns.jointplot(x=user_input, y=user_input2, data=tech_rets, kind='scatter')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape
from keras.models import Sequential
from keras.layers import Dense, LSTM
# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# Convert the data to a numpy array
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
st.subheader("The rmse : ")
st.write(rmse)
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
f1=plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(f1)
#plt.show()
# #st.write(model.summary())
st.subheader("Summary of our Model")
model.summary(print_fn=lambda x: st.text(x))
st.subheader("The Actual Close Prices and Prediction Prices")
st.write(valid)
