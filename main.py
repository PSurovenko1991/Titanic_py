import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
from sklearn.cross_validation import train_test_split



from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



data_set = pd.read_csv("train.csv",',')

#kill null:

#print(data_set.info())
# data_set = data_set.drop(["Cabin"], axis=1) # удаляем, так как поле мало заполнено
data_set.Age[np.isnan(data_set["Age"])] = np.mean(data_set.Age[~np.isnan(data_set["Age"])]) #заполняем недостающие значения средними 
data_set.Embarked[data_set.Embarked.isnull()] = rn.choice(["Q","S","C"])
# print(data_set.info())


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
# Feature that tells whether a passenger had a cabin on the Titanic
data_set['Has_Cabin'] = data_set["Cabin"].apply(lambda x: 0 if type(x) == float else 1) 
data_set = data_set.drop("Cabin",axis=1)
# data_set['Ticket'] = data_set["Ticket"].apply(lambda x: 0 if  x[0].isalpha() else 1)
data_set['Name_length'] = data_set['Name'].apply(len)
data_set['Ticket_length'] = data_set['Ticket'].apply(len)


print(data_set.info())
# print(data_set.Ticket)


correlation = data_set.corr(method='pearson')
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()




# Important fields:
fields =  (np.extract(abs((correlation.Survived))>0.25, (correlation.columns)))
fields = fields[1:]
print("Important fields: ",fields)



# Formation of test sets:

train = data_set[fields]
target = data_set["Survived"]

X_train, X_test , y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=0)





#???????????????????????????????????????????????????????????????????????????????????????????

itog_val={}

model_rfc = RandomForestClassifier(n_estimators = 70) #в параметре передаем кол-во деревьев
model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
model_lr = LogisticRegression(penalty='l1', tol=0.01) 
model_svc = svm.SVC()

scores = cross_validation.cross_val_score(model_rfc, X_train, y_train, cv = kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_knc, X_train, y_train, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr, X_train, y_train, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_validation.cross_val_score(model_svc, X_train, y_train, cv = kfold)
itog_val['SVC'] = scores.mean()


#?????????????????????????????????????????????????????????????????????????????????????????




#Training:
clf = GaussianNB()
clf.fit(X_train ,y_train)
print(clf.score(X_test, y_test)) #результат


 #Load test set:
x = pd.read_csv("test.csv",",")
x['Has_Cabin'] = x["Cabin"].apply(lambda x: 0 if type(x) == float else 1) 
x= x.drop(["Cabin"], axis=1)
x.Age[np.isnan(x["Age"])] = np.mean(x.Age[~np.isnan(x["Age"])])
x.Fare[np.isnan(x["Fare"])] = np.mean(x.Fare[~np.isnan(x["Fare"])])
x['Sex'].replace('male',0,inplace= True)   
x['Sex'].replace('female',1,inplace= True)
for i in set(x["Embarked"]):
    g ="Em: "+str(i)
    x.insert(x.columns.size-1,g,x['Embarked']== i)
x = x.drop(["Embarked"], axis=1)
x['Name_length'] = x['Name'].apply(len)

x_test = x[fields]





#Test:
print(clf.score(X_test, y_test))
# print(clf.predict(x_test))

#get_results
result = pd.DataFrame(x["PassengerId"])

result.insert(1,'Survived', clf.predict(x_test))
result.to_csv("t20.csv", index = False)
# print(result)
# print(type(clf.__str__()))
# my_file = open("some.txt", "w")
# my_file.write(clf.fit(X_train ,y_train).__str__())
# my_file.close()

