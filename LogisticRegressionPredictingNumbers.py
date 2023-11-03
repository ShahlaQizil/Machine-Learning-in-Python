#Predicting Numbers

from sklearn.datasets import load_digits
#Load and return the digits dataset (classification).
#Each datapoint is a 8x8 image of a digit.
from sklearn.model_selection import train_test_split
#Split arrays or matrices into random train and test subsets.
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.pyplot is a state-based interface to matplotlib. 
#It provides an implicit, MATLAB-like, way of plotting. 
#It also opens figures on your screen, and acts as the figure GUI manager.
import seaborn as sns
#Seaborn is a Python data visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive and informative statistical graphics.
#from sklearn import metrrics
#%matplotlib inline
#The line magic command %matplotlib inline enables the drawing of matplotlib figures in the IPython environment. 
#Once this command is executed in any cell, the matplotlib plots will appear directly below the cell in which the plot function was called for the rest of the session.

digits = load_digits()

#Determining the total number of images and labels
print("Image Data Shape" , digits.data.shape)
print("Label Data Shape" , digits.target.shape)

#Displaying some of the images and labels
#plt.figure(figsize=(20,4))
#Create a new figure, or activate an existing figure.
#for index , (image,label) in enumerate(zip(digits.data[0:5] , digits.target[0:5])):
    #The enumerate() function takes a collection (e.g. a tuple) and returns it as an enumerate object.
    #The enumerate() function adds a counter as the key of the enumerate object.
    #plt.subplot(1,5,index+1)
    #plt.imshow(np.reshape(image(8,8)) , cmap=plt.cm.gray)
    #plt.title("Training: %i\n" % label , fontsize = 20)

#Divide dataset into Training and Test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(digits.data , digits.target , test_size=0.23 , random_state= 2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Import the Logistic Regression model
from sklearn.linear_model import LogisticRegression

#Making an instance of the model and training it
LogisticReg = LogisticRegression()
LogisticReg.fit(x_train , y_train)

#Predicting the output of the first element of the test set
#print(LogisticReg.predict(x_test(0).reshape(1,-1)))

#Predicting for the entire dataset
predictions = LogisticReg.predict(x_test)

#Determining the accuracy of the model
score=LogisticReg.score(x_test, y_test)

#Represnting the Confusion Matrix in a heat map
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test , predictions)
plt.figure(figsixe=(9,9))
sns.heatmap(cm , annot=True , fmt=".3f" , linewidths=.5 , square = True , cmap='Blues_r');
plt.ylabel('Actual Label');
plt.xlabel('Predicted Lbalel');
all_sample_title = "accuracy Score: {0}" .format(score)
plt.title(all_sample_title , size = 15);

