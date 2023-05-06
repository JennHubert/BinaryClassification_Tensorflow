import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
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
    column_max=df[column].max() # Finding column max
    column_min=df[column].min() # Finding column min
    if column_max == column_min: # If the min and max are the same
        list.append(column) # Add the column name to the list

        

df.drop(columns=list, inplace=True) # Drop the columns with only one 

df["column_1559"]= np.where(df["column_1559"]=='ad.', 1, 0) # Encoding the class feature


# The code below was used to find the spearman correlation.
# I have commented it out so that it is still there for you to see, 
# But not run when this file is run.
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

# Separating features
y=df['column_1559'] # Class feature
X=df.iloc[:,0:1558] # Other features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting data

scaler = MinMaxScaler() # Initializing scaler
X_train_scaled = scaler.fit_transform(X_train) # Scaling X_train
X_test_scaled = scaler.transform(X_test) # Scaling X_test



model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
	tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
]) # Building the model


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')]) # Adding more settings & compiling the model

history=model.fit(X_train_scaled, y_train, epochs=10) # Train the model


from matplotlib import rcParams # Import for the paramaters
rcParams['figure.figsize'] = (18, 8) # Parameter
rcParams['axes.spines.top'] = False # Parameter
rcParams['axes.spines.right'] = False #Parameter

plt.plot(history.history['loss'], label='Loss') # Line for loss
plt.plot(history.history['accuracy'], label='Accuracy') # Line for accurarcy
plt.plot(history.history['precision'], label='Precision') # Line for precision
plt.plot(history.history['recall'], label='Recall') # Line for Recall
plt.title('Evaluation metrics', size=20) # Plot title
plt.ylabel('Score', size=14) # Y label
plt.xlabel('Epoch', size=14) # X label
plt.legend() # Legend
plt.show() # Show plot

predictions = model.predict(X_test_scaled) # Get predictions with test data 
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)] # Making the decision boundary (0.5)
loss, accuracy, precision, recall = model.evaluate(X_test_scaled, y_test) # Finding the scores for individual test
print('Loss: ', loss) # Display loss
print('accuracy: ', accuracy) # Display accurary
print('precision: ', precision) # Display Precision
print('recall: ', recall) # Display recall

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import for the picture of the cm
print("") # Empty line - easier to read
print("CONFUSION MATRIX:") # Easier to read with label
print(confusion_matrix(y_test, prediction_classes)) # CM Scores


cm = confusion_matrix(y_test, prediction_classes) # CM Scores again
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=[1,0]) # Making the display
disp.plot() # Plotting it
plt.show() # Showing it

from sklearn.metrics import classification_report # Importing the report
print(classification_report(y_test, prediction_classes)) # Printing the report this way because it is good for the report
