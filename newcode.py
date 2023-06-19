
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
from scipy.stats import pearsonr
import numpy as np 
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression  #all packages import in above lines 


main = tkinter.Tk()
main.title("Predicting Personality") #designing main screen
main.geometry("1300x1200")

global filename
mrc = []   #list of MRC and LIWC words
liwc = []
emotion = []
global openness
global agreeable
global conscientious
global extroversion
global neuroticism
global open_count
global agree_count
global ext_count
global neu_count
global cons_count
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, naive_acc, lra_acc # all global variables names define in above lines

def traintest(train):     #method to generate test and train data from dataset
    X = train.values[:, 0:4] 
    Y = train.values[:, 5] 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv("dataset.txt")
    X, Y, X_train, X_test, y_train, y_test = traintest(train)

with open("LIWC.dic", "r") as file:  #reading LIWC dictinary
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        liwc.append(line.lower())
    

        
with open("MRC.txt", "r") as file: #reading MRC dictionary
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        mrc.append(line.lower())
    

with open("emotions.txt", "r") as file: #reading emotion word
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        emotion.append(line.lower())
      

def opennessFunction(words): #calculate number of openness words from tweets
    count = 0.0
    for i in range(len(liwc)):
        if words.find(liwc[i]) != -1:
            count = count + 1
    if count > 0:
        count = count/float(len(liwc))
    return count

def agreeableFunction(words): #calculate number of agreeable words from tweets
    count = 0.0
    for i in range(len(mrc)):
        if words.find(mrc[i]) != -1:
            count = count + 1
    if count > 0:
        count = count/float(len(mrc))            
    return count

def neuroticismFunction(words): #calculate number of emotion words from tweets
    count = 0.0
    for i in range(len(emotion)):
        if words.find(emotion[i]) != -1:
            count = count + 1
    if count > 0:
        count = count/float(len(emotion))            
    return count

def pearson(feature,retweet,followers,mention,hashtag,following): #perason calculation
    pearson_value = 0;
    x = [feature,retweet,followers]
    y = [mention,hashtag,following]
    pearson_value, _ = pearsonr(x, y)
    return pearson_value

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def extractFeatures(): #extract features from tweets
    global openness
    global agreeable
    global conscientious
    global extroversion
    global neuroticism

    openness = 0.0
    agreeable = 0.0
    conscientious = 0.0
    extroversion = 0.0
    neuroticism = 0.0
    
    text.delete('1.0', END)
    
    for root, dirs, files in os.walk(filename):
      for fdata in files:
        with open(root+"/"+fdata, "r") as file:
            data = json.load(file)
            textdata = data['text'].strip('\n')
            textdata = textdata.replace("\n"," ")
            textdata = re.sub('\W+',' ', textdata)
            retweet = data['retweet_count']
            followers = data['user']['followers_count']
            density = data['user']['listed_count']
            following = data['user']['friends_count']
            replies = data['user']['favourites_count']
            hashtag = data['user']['statuses_count']
            username = data['user']['screen_name']
            words = textdata.split(" ")
            text.insert(END,"Username : "+username+"\n");
            text.insert(END,"Tweet Text : "+textdata);
            text.insert(END,"Retweet Count : "+str(retweet)+"\n")
            text.insert(END,"Following : "+str(following)+"\n")
            text.insert(END,"Followers : "+str(followers)+"\n")
            text.insert(END,"Density : "+str(density)+"\n")
            text.insert(END,"Hashtag : "+str(hashtag)+"\n")
            text.insert(END,"Tweet Words Length : "+str(len(words))+"\n\n")
            

def pearsonFunction(): #calculating pearson for each feature value
    text.delete('1.0', END)
    global open_count
    global agree_count
    global ext_count
    global neu_count
    global cons_count
    global openness
    global agreeable
    global conscientious
    global extroversion
    global neuroticism
    
    open_count = 0.0
    agree_count = 0.0
    ext_count = 0.0
    neu_count = 0.0
    cons_count = 0.0

    headers = "Openness,Agreeable,Neuroticism,Extroversion,Conscientious,class\n"
    text.insert(END,"Username\t\tOpenness\tAgreeable\tNeuroticism\tExtroversion\tConscientious\n")
    for root, dirs, files in os.walk(filename):
      for fdata in files:
        with open(root+"/"+fdata, "r") as file:
            data = json.load(file)
            textdata = data['text'].strip('\n')
            textdata = textdata.replace("\n"," ")
            textdata = re.sub('\W+',' ', textdata)
            retweet = data['retweet_count']
            followers = data['user']['followers_count']
            density = data['user']['listed_count']
            following = data['user']['friends_count']
            replies = data['user']['favourites_count']
            hashtag = data['user']['statuses_count']
            username = data['user']['screen_name']
            words = textdata.split(" ")
            
            openness = opennessFunction(textdata.lower())#use open swear words in tweets
            agreeable = agreeableFunction(textdata.lower()) #use agreeable words in tweets
            neuroticism = neuroticismFunction(textdata.lower()) #sentiment
            extroversion = following/hashtag    #friendly
            conscientious = followers/hashtag  #hardwork and reliable

            openness = pearson(openness,retweet,hashtag,followers,hashtag,following)
            agreeable = pearson(agreeable,retweet,following,followers,hashtag,following)
            neuroticism = pearson(neuroticism,retweet,density,followers,hashtag,following)
            extroversion = pearson(extroversion,retweet,replies,followers,hashtag,following)
            conscientious = pearson(conscientious,retweet,retweet,followers,hashtag,following)
            classlabel = 0
            max = 0
            if openness > max:
                max = openness
                classlabel = 1

            if agreeable > max:
                max = agreeable
                classlabel = 2

            if neuroticism > max:
                max = neuroticism
                classlabel = 3
        
            if extroversion > max:
                max = extroversion
                classlabel = 4

            if conscientious > max:
                max = conscientious
                classlabel = 5

            values = str(openness)+","+str(agreeable)+","+str(neuroticism)+","+str(extroversion)+","+str(conscientious)+","+str(classlabel)+"\n"
            headers+=values
            if openness > 0.1:
                open_count = open_count + 1
            if agreeable > 0.1:
                agree_count = agree_count + 1
            if neuroticism > 0.1:
                neu_count = neu_count + 1
            if extroversion > 0.1:
                ext_count = ext_count + 1
            if conscientious > 0.1:
                cons_count = cons_count + 1    
            #print('Pearsons correlation: %.3f' % corr) 
                    
            text.insert(END,username+"\t\t"+str(round(openness,4))+"\t    "+str(round(agreeable,4))+"\t     "+str(round(neuroticism,4))+"\t         "+str(round(extroversion,4))+"\t        "+str(round(conscientious,4))+"\n")

    f = open("dataset.txt", "w")
    f.write(headers)
    f.close()
    generateModel()

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n\n\n\n")  
    return accuracy        

def runSVM():
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2) 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix') 
                
def runRandomForest():
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=1.0,random_state=None) 
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix')

def runNaiveBayes():
    global naive_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    naive_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Algorithm Accuracy, Classification Report & Confusion Matrix')

def runLRA():
    global lra_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    lra_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Algorithm Accuracy, Classification Report & Confusion Matrix')

def graph():
    height = [svm_acc,random_acc,naive_acc,lra_acc]
    bars = ('SVM Accuracy', 'Random Forest Accuracy','Naive Bayes Accuracy','Logistic Regression Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def personalityGraph():
    height = [open_count, agree_count,neu_count,ext_count,cons_count]
    bars = ('Openness', 'Agreeable','Neuroticism','Extroversion','Conscientious')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Predicting Personality from Twitter')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Twitter Profile Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=360,y=100)

extractButton = Button(main, text="Extract LIWC, MRC & Twitter Profile Features", command=extractFeatures)
extractButton.place(x=50,y=150)
extractButton.config(font=font1) 

pearsonButton = Button(main, text="Pearson Correlation Analysis", command=pearsonFunction)
pearsonButton.place(x=470,y=150)
pearsonButton.config(font=font1) 

runsvm = Button(main, text="Run SVM Algorithm", command=runSVM)
runsvm.place(x=740,y=150)
runsvm.config(font=font1) 

runrandomforest = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomforest.place(x=950,y=150)
runrandomforest.config(font=font1) 

runnb = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
runnb.place(x=50,y=200)
runnb.config(font=font1) 

lra = Button(main, text="Run Logistic Regression Algorithm", command=runLRA)
lra.place(x=330,y=200)
lra.config(font=font1) 

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=650,y=200)
graph.config(font=font1) 

pgraph = Button(main, text="Personality Prediction Graph", command=personalityGraph)
pgraph.place(x=820,y=200)
pgraph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
