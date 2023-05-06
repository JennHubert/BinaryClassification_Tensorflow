import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
# Imports Above

df=pd.read_csv('classification/ad.data', header=None) # Import data
df.columns = df.columns.map(lambda x: f'column_{x+1}') # Column names

df = df[(df.column_1 != "   ?") & (df.column_2 != '   ?') & (df.column_3 !='     ?') & (df.column_4 !='?') ] # Subsetting to get rid of missing values

df['column_1']=df['column_1'].astype(float) # Changing to float dtype
df['column_2']=df['column_2'].astype(float) # Changing to float dtype
df['column_3']=df['column_3'].astype(float) # Changing to float dtype
df['column_4']=df['column_4'].astype(float) # Changing to float dtype

list=[] # making empty list
# Iterate over columns
for column in df:
     
    # The below code goes column by column to see which ones only have 1 value
    #columnSeriesObj = df[column]
    column_max=df[column].max() # Finding column max
    column_min=df[column].min() # Finding column min
    if column_max == column_min: # If the min and max are the same
        #df=df.drop(column)
        list.append(column) # Add the column name to the list
        #print('Column Name : ', column)
        #print(df[column].value_counts())
        
df.drop(columns=list, inplace=True) # Drop the columns with only one 

df["column_1559"]= np.where(df["column_1559"]=='ad.', 1, 0) # Encoding the class feature



'''
c = df.corr(method='spearman').abs().unstack().transpose()

c=c.drop_duplicates()
so = c.sort_values(ascending=False, kind="quicksort")
print(so)
so.to_csv('/Users/jenniferhubert/DSSA Spring 2023/Deep Learning/coding-assignment-week-8-JennHubert/images/SO.csv')

cdf=pd.read_csv('/Users/jenniferhubert/DSSA Spring 2023/Deep Learning/coding-assignment-week-8-JennHubert/images/SO.csv', header=None)
cdf.columns = cdf.columns.map(lambda x: f'column_{x+1}')
print(cdf)
relevant=cdf[(cdf.column_1 == "column_1559") | (cdf.column_2 == "column_1559")]
print(relevant)
relevant.to_csv('/Users/jenniferhubert/DSSA Spring 2023/Deep Learning/coding-assignment-week-8-JennHubert/images/SO23.csv')
'''
y=df['column_1559'] # Class feature
X=df.iloc[:,0:1558] # Other features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting data

scaler = MinMaxScaler() # Initializing scaler
X_train_scaled = scaler.fit_transform(X_train) # Scaling X_train
X_test_scaled = scaler.fit_transform(X_test) # Scaling X_test



model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=10, batch_size=1)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print('Test accuracy:', test_acc)
