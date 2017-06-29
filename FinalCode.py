# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/root/Documents/Job')
data=dataset.iloc[:,6].values
df=pd.DataFrame()
i=0
while i<len(data):
        if(data[i]==1):
            a=[dataset.iloc[i,:].values]
            df1=pd.DataFrame(a)
            df=df.append(df1)
            i=i+1
        else:
            i=i+1

df=df.reset_index()
df=df.drop('index',1)
 
  # Preprocessing

  
X=dataset.iloc[:,[4,7,9]].values
X=pd.DataFrame(X)
y=dataset.iloc[:,[6]].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label=LabelEncoder()
X[2]=label.fit_transform(X[2])

one=OneHotEncoder(categorical_features=[2])
X=one.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



# Plotting Of Graph 
       
plt.hist(df[0])
plt.hist(df[1])
plt.hist(df[2])
plt.hist(df[3])
plt.hist(df[4])
plt.hist(df[5])

# Reason For Leaving The Company

Reason1=pd.DataFrame()
i=0
WithorNotWorkAccident=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4):
            a=[dataset.iloc[i,[7,9]].values]
            df1=pd.DataFrame(a)
            Reason1=Reason1.append(df1)
            i=i+1
            WithorNotWorkAccident=WithorNotWorkAccident+1
        else:
            i=i+1

Reason1=Reason1.reset_index()
Reason1=Reason1.drop('index',1)


Reason2=pd.DataFrame()
i=0
WithWorkAccident=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],5].values==1):
            a=[dataset.iloc[i,[7,9]].values]
            df1=pd.DataFrame(a)
            Reason2=Reason2.append(df1)
            i=i+1
            WithWorkAccident=WithWorkAccident+1
        else:
            i=i+1

Reason2=Reason2.reset_index()
Reason2=Reason2.drop('index',1)

Reason2WithoutAcc=pd.DataFrame()
i=0
WithoutWorkAccident=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],5].values==0):
            a=[dataset.iloc[i,[7,9]].values]
            df1=pd.DataFrame(a)
            Reason2WithoutAcc=Reason2WithoutAcc.append(df1)
            i=i+1
            WithoutWorkAccident=WithoutWorkAccident+1
        else:
            i=i+1

Reason2WithoutAcc=Reason2WithoutAcc.reset_index()
Reason2WithoutAcc=Reason2WithoutAcc.drop('index',1)

Reason3=pd.DataFrame()
i=0
WithHigherProjects=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],2].values>3):
            a=[dataset.iloc[i,[7,8,9]].values]
            df1=pd.DataFrame(a)
            Reason3=Reason3.append(df1)
            i=i+1
            WithHigherProjects=WithHigherProjects+1
        else:
            i=i+1

Reason3=Reason3.reset_index()
Reason3=Reason3.drop('index',1)

Reason4=pd.DataFrame()
i=0
WithPromotion=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],2].values>3 and dataset.iloc[[i],7].values==1):
            a=[dataset.iloc[i,[9]].values]
            df1=pd.DataFrame(a)
            Reason4=Reason4.append(df1)
            i=i+1
            WithPromotion=WithPromotion+1
        else:
            i=i+1

Reason4=Reason4.reset_index()
Reason4=Reason4.drop('index',1)


Reason5=pd.DataFrame()
i=0
WithLowerSalary=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],2].values>3 and dataset.iloc[[i],9].values=='low'):
            a=[dataset.iloc[i,[7]].values]
            df1=pd.DataFrame(a)
            Reason5=Reason5.append(df1)
            i=i+1
            WithLowerSalary=WithLowerSalary+1
        else:
            i=i+1

Reason5=Reason5.reset_index()
Reason5=Reason5.drop('index',1)


Reason6=pd.DataFrame()
i=0
WithHigherTime=0
while i<len(data):
        if(dataset.iloc[[i],0].values> 0.5 and dataset.iloc[[i],1].values > 0.5 and dataset.iloc[[i],4].values> 4 and dataset.iloc[[i],2].values>3 and dataset.iloc[[i],3].values>200):
            a=[dataset.iloc[i,[9]].values]
            df1=pd.DataFrame(a)
            Reason6=Reason6.append(df1)
            i=i+1
            WithHigherTime=WithHigherTime+1
        else:
            i=i+1

Reason6=Reason6.reset_index()
Reason6=Reason6.drop('index',1)

b=[WithHigherProjects,WithLowerSalary,WithHigherTime,WithPromotion,WithWorkAccident,WithorNotWorkAccident,WithoutWorkAccident]
Visual=pd.DataFrame(b)

ax = Visual[0].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Factors Affecting Employess", fontsize=12)
ax.set_ylabel("Values", fontsize=12)
plt.show()

# Regression Models


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_linear = regressor.predict(X_test)


from sklearn.tree import DecisionTreeClassifier
Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Classifier.fit(X_train, y_train)

y_pred_decision = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_decision)


from sklearn.ensemble import RandomForestClassifier
Classifier1 = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
Classifier1.fit(X_train, y_train)
y_pred_RandomForest = Classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_RandomForest)

from sklearn.linear_model import LogisticRegression
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train,y_train)
y_pred_LR=classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred_LR)



