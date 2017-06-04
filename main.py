import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

data_set = pd.read_csv("train.csv",',')

#kill null:

#print(data_set.info())
data_set = data_set.drop(["Cabin"], axis=1) # удаляем, так как поле мало заполнено
data_set.Age[np.isnan(data_set["Age"])] = np.mean(data_set.Age[~np.isnan(data_set["Age"])]) #заполняем недостающие значения средними 
data_set.Embarked[data_set.Embarked.isnull()] = rn.choice(["Q","S","C"])
print(data_set.info())


#print(data_set.shape)#(891, 12)
# print(data_set["Parch"])
# print(data_set["Survived"])
# print(np.extract((np.extract(data_set["Parch"]>0  ,data_set["SibSp"])==0),data_set["Survived"]))


# Summary table:

# data_set.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
# print(data_set.pivot_table('PassengerId', 'Pclass', 'Survived', 'count'))
# data_set.pivot_table('PassengerId', 'Sex', 'Survived', 'count').plot(kind='bar', stacked=True)
# print(data_set.pivot_table('PassengerId', 'Sex', 'Survived', 'count'))
# data_set.pivot_table('PassengerId', 'Embarked', 'Survived', 'count').plot(kind='bar', stacked=True)
# print(data_set.pivot_table('PassengerId', 'Embarked', 'Survived', 'count'))



#formalization:

# print(data_set.Sex)
data_set['Sex'].replace('male',0,inplace= True)   
data_set['Sex'].replace('female',1,inplace= True)

for i in set(data_set["Embarked"]):
    g ="Em: "+str(i)
    data_set.insert(data_set.columns.size-1,g,data_set['Embarked']== i)
data_set = data_set.drop(["Embarked"], axis=1)



#corr.matrix:

correlation = data_set.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()





fields =  (np.extract(abs((correlation.Survived))>0.1, (correlation.columns)))
fields = fields[1:]

print("Important fields: ",fields)



#Formation of test sets:

train = data_set[fields]
target = data_set["Survived"]

X_train, X_test , y_train, y_test = train_test_split(train, target, test_size=0.94, random_state=0,)

#Training:
clf = GaussianNB()
clf.fit(X_train ,y_train)












 #Load test set:
x = pd.read_csv("test.csv",",")
x= x.drop(["Cabin"], axis=1)
x.Age[np.isnan(x["Age"])] = np.mean(x.Age[~np.isnan(x["Age"])])
x.Fare[np.isnan(x["Fare"])] = np.mean(x.Fare[~np.isnan(x["Fare"])])
x['Sex'].replace('male',0,inplace= True)   
x['Sex'].replace('female',1,inplace= True)
for i in set(x["Embarked"]):
    g ="Em: "+str(i)
    x.insert(x.columns.size-1,g,x['Embarked']== i)
x = x.drop(["Embarked"], axis=1)


x_test = x[fields]
y = pd.read_csv("gender_submission.csv",",")
y_test = y["Survived"]




#Test:
print(clf.score(x_test, y_test))
print(clf.predict(x_test))
