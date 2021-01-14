
#-----------------------importing the libraries-------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#-------------------------importing datasets----------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('gender_submission.csv')
train.info()
#-----------------------------------------------------------------------------
#--------------------------------preprocessing step---------------------------

#-------------------------handle with missing data----------------------------
''' train.isna().sum()
train.drop(['Cabin'], axis = 1, inplace = True)
train.isna().sum()
test.isna().sum()
test.drop(['Cabin'], axis = 1, inplace = True)
test.isna().sum() '''

#for checking missing no. of missing value present in training and testing dataset
print(train.isna().sum())
print('\n')
print(test.isna().sum())
#-------------------------------missing value in age--------------------------
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer = imputer.fit(train.iloc[:,[5]])
train.iloc[:,[5]] = imputer.transform(train.iloc[:,[5]])

imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer = imputer.fit(test.iloc[:,[4]])
test.iloc[:,[4]] = imputer.transform(test.iloc[:,[4]])

#-----------------------------mission value in Fare---------------------------
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer = imputer.fit(test.iloc[:,[8]])
test.iloc[:,[8]] = imputer.transform(test.iloc[:,[8]])

#----------------------------mission value in Ebarked-------------------------
print(train['Embarked'].value_counts()) # from here we knew that s is the most frequent word
imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
imputer = imputer.fit(train.iloc[:,[11]])
train.iloc[:,[11]] = imputer.transform(train.iloc[:,[11]])

#-------------------get dummy variable for sex and embarked-------------------
train = pd.get_dummies(data = train, columns = ["Sex","Embarked"] , prefix = ["Sex","Embarked"])
test = pd.get_dummies(data = test, columns = ["Sex","Embarked"] , prefix = ["Sex","Embarked"])

#-----------if data is present in cabin then cabin = 1 otherwise 0------------
train['Cabin'] = train['Cabin'].fillna(0)
train.loc[train.Cabin !=0, 'Cabin'] = 1 #it means, when cabin !=0 then cabin =1
        
test['Cabin'] = test['Cabin'].fillna(0)
test.loc[test.Cabin !=0, 'Cabin'] = 1

#create new feature ageencode 
train['Ageencode'] = train['Age']    
train['Ageencode'] = pd.qcut(train['Ageencode'], 4)
test['Ageencode'] = test['Age']    
test['Ageencode'] = pd.qcut(test['Ageencode'], 4)

train['FamilySize'] = train['SibSp'] + train['Parch'] +1
train['IsAlone'] = 1
train.loc[train.FamilySize >1 , 'IsAlone'] = 0
test['FamilySize'] = test['SibSp'] + test['Parch'] +1
test['IsAlone'] =1
test.loc[test.FamilySize >1 , 'IsAlone'] = 0

#-----------label encoding in ageencode--------------------------------------
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Ageencode = LabelEncoder()
train.iloc[:,[15]] = labelencoder_Ageencode.fit_transform(train.iloc[:,[15]])
labelencoder_Ageencode_y = LabelEncoder()
test.iloc[:,[14]] = labelencoder_Ageencode_y.fit_transform(test.iloc[:,[14]])

#-------------take first term from ticket and create new feature-------------- 
ticket = train['Ticket']
ticket1 =np.str()
for i in range(len(ticket)):
        ticket1 = ticket1 + ticket[i][0]
ticket1 = list(ticket1)
train['Ticket[0]'] = ticket1

ticket = test['Ticket']
ticket2 =np.str()
for i in range(len(ticket)):
        ticket2 = ticket2 + ticket[i][0]
ticket2 = list(ticket2)
test['Ticket[0]'] = ticket2

#------------------------encoding  of ticket----------------------------------
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
train.iloc[:,[18]] = labelencoder.fit_transform(train.iloc[:,[18]])
test.iloc[:,[17]] = labelencoder.fit_transform(test.iloc[:,[17]])

#-----------------------------seprate miss, mr etc----------------------------
train['Title'] = train['Name'].str.split(',' , expand = True)[1].str.split('.', expand = True)[0]
print(train['Title'].value_counts())
title1 = (train['Title'].value_counts() < 10)
train['Title'] = train['Title'].apply(lambda x: ' Misc' if title1.loc[x] == True else x)
print(train['Title'].value_counts())

test['Title'] = test['Name'].str.split(',' , expand = True)[1].str.split('.', expand = True)[0]
print(test['Title'].value_counts())
title1 = (test['Title'].value_counts() < 10)
test['Title'] = test['Title'].apply(lambda x: ' Misc' if title1.loc[x] == True else x)
print(test['Title'].value_counts())

#--------------------devide fare into 4 parts and encode it ------------------

train['Farecode'] = pd.qcut(train['Fare'], 4)
train.iloc[:,[20]] = labelencoder.fit_transform(train.iloc[:,[20]])

test['Farecode'] = pd.qcut(test['Fare'], 4)
test.iloc[:,[19]] = labelencoder.fit_transform(test.iloc[:,[19]])

#------------------------Describe training and testing set--------------------
print("train:-\n" ,train.describe(), '\n')
print('test:-\n', test.describe())

#-------------------get dummies of title , ageencode , farecode---------------
train = pd.get_dummies(data = train, columns = ["Title","Ageencode","Farecode"] , prefix = ["Title","Ageencode","Farecode"])
test = pd.get_dummies(data = test, columns = ["Title","Ageencode","Farecode"] , prefix = ["Title","Ageencode","Farecode"])

#--------------spliting dataset into training and testing set-----------------
x = train.iloc[:,[2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]].values
y = train.iloc[:, 1].values
x_test = test.iloc[:,[1,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]].values
y_test = gender.iloc[:,1].values

#------------split data into training and classvalidation state---------------
from sklearn.model_selection import train_test_split
x_train , x_cv , y_train , y_cv = train_test_split(x , y , test_size = .25 , random_state = 0)


#------------------------for finding correlation------------------------------
print(train.corr())

data1_x = ['Pclass', 'Age','SibSp',  'Parch', 'Fare','Cabin','Embarked_C','Embarked_Q','Embarked_S',
           'FamilySize','IsAlone', 'Ticket[0]']
Target = ['Survived']
for k in data1_x:
    if train[k].dtype != 'float64' :
        print('Survival Correlation by:', k)
        print(train[[k, Target[0]]].groupby(k, as_index=False).mean())
        print('-'*10, '\n')
        
#-----------------------------------------------------------------------------
#-----------------------plot data ageencode vs survived-----------------------
plt.figure(figsize=[16,12])
plt.subplot(241)
plt.boxplot(x = train['Fare'], showmeans = True, meanline = True)
plt.title('Fare boxplot')
plt.ylabel('Fare ($)')

plt.subplot(242)
plt.boxplot(train['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(243)
plt.boxplot(train['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(244)
plt.boxplot(x = train['Ticket[0]'], showmeans = True, meanline = True)
plt.title('Ticket boxplot')
plt.ylabel('Ticket starting term')

plt.subplot(245)
plt.hist(x = [train[train['Survived']==1]['Fare'] ,train[train['Survived']==0]['Fare']],stacked=True,
         color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(246)
plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']] ,stacked=True,
         color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(247)
plt.hist(x = [train[train['Survived']==1]['FamilySize'], train[train['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(248)
plt.hist(x = [train[train['Survived']==1]['Ticket[0]'], train[train['Survived']==0]['Ticket[0]']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Ticket[0] Histogram by Survival')
plt.xlabel('Ticket[0] ')
plt.ylabel('# of Passengers')
plt.legend()


import seaborn as sns
fig, axes = plt.subplots(2,3)
sns.barplot(x = 'IsAlone', y = 'Survived', data = train , ax = axes[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', data = train, ax = axes[0,1])
sns.barplot(x = 'Embarked_C',y = 'Survived', data = train, ax = axes[0,2])
sns.barplot(x = 'Embarked_Q',y = 'Survived', data = train, ax = axes[1,0])
sns.barplot(x = 'Embarked_S',y = 'Survived', data = train, ax = axes[1,1])
sns.pointplot(x = 'Ageencode_0', y = 'Survived', data = train, ax = axes[1,2])

fig, axes = plt.subplots(1,3)
sns.barplot(x = 'Pclass',y= 'Fare', hue = 'Survived', data = train, ax = axes[0])
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train, split = True, ax = axes[1])
sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = train, ax = axes[2])

fig, axes = plt.subplots(1,3)
sns.violinplot(x = 'Ticket[0]', y = 'Survived', data = train, split = True, ax = axes[0])
sns.barplot(x = 'Ticket[0]',y= 'Survived', data = train, ax = axes[1])
sns.barplot(x = 'Pclass', y = 'Ticket[0]', hue = 'Survived', data = train, ax = axes[2])


#------------------------explode 1st slice (pie plot)-------------------------
fig, axes = plt.subplots()
print(train['Survived'].value_counts())
plt.pie(train['Survived'].value_counts(),labels = ['Survived','Dead'],
        autopct = '%1.1f%%', shadow = True , startangle = 90)
plt.show()

"""#-------------------------------pair plot-------------------------------------
pp = sns.pairplot(train ,  hue = 'Survived' , palette = 'deep', height=1.2, diag_kind = 'kde',
             diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[]) """


#------------------------------3d Surface Image-------------------------------
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure(figsize =(14,6))
X = np.array([train['Pclass'],train['Age']])
Y = np.array([train['Fare']])
Z = np.array([train['Survived']])
#specify the 3d graphics to draw with projection = 3d
ax = fig.add_subplot(1,1,1,projection = '3d')
ax.plot_surface(X,Y,Z, rstride = 4 , cstride = 4 , linewidth = 0, color = 'r')

#------------------------------Heat Map---------------------------------------
fig , axes = plt.subplots(figsize =(24, 16))
fig = sns.heatmap(train.corr() ,square = True, annot=True, linewidths=0.1,vmax=1.0, linecolor='white',
                annot_kws={'fontsize':6})
plt.title('Pearson Correlation of Features', y=1.05, size=15)


#-----------------------------------------------------------------------------
#---------------------creating classifier model-------------------------------

#---------------fitting classifier to the training set------------------------

def ensemble_models(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    classifier1 = RandomForestClassifier(n_estimators = 3, criterion = 'entropy' , 
                                    random_state = 0)
    classifier1.fit(x_train, y_train)
    
    from sklearn.ensemble import AdaBoostClassifier
    classifier2 = AdaBoostClassifier()
    classifier2.fit(x_train, y_train)
    
    from sklearn.ensemble import BaggingClassifier
    classifier3 = BaggingClassifier()
    classifier3.fit(x_train, y_train)
    
    from sklearn.ensemble import ExtraTreesClassifier
    classifier4 = ExtraTreesClassifier()
    classifier4.fit(x_train, y_train)
    
    from sklearn.ensemble import GradientBoostingClassifier
    #----------------------------by parameter tuning-------------------------- 
    classifier5 = GradientBoostingClassifier()
    classifier5.fit(x_train, y_train)
    
    print('RandomForestClassifier training accuracy: ', classifier1.score(x_train, y_train))
    print('AdaBoostClassifier training accuracy: ', classifier2.score(x_train, y_train))
    print('BaggingClassifier training accuracy: ', classifier3.score(x_train, y_train))
    print('ExtraTreesClassifier training accuracy: ', classifier4.score(x_train, y_train))
    print('GradientBoostingClassifier training accuracy: ', classifier5.score(x_train, y_train))
    
    return classifier1, classifier2, classifier3 , classifier4 , classifier5

def gaussian_process_models(x_train, y_train):
    from sklearn.gaussian_process import GaussianProcessClassifier
    classifier1 = GaussianProcessClassifier()
    classifier1.fit(x_train, y_train)
    
    print('GaussianProcessClassifier training accuracy: ', classifier1.score(x_train, y_train))
    
    return classifier1

def linear_models(x_train , y_train):
    from sklearn.linear_model import LogisticRegression
    classifier1 = LogisticRegression(C=1.2, random_state = 0, max_iter = 1500)
    classifier1.fit(x_train, y_train)
    
    from sklearn.linear_model import PassiveAggressiveClassifier
    classifier2 = PassiveAggressiveClassifier()
    classifier2.fit(x_train, y_train)
    
    from sklearn.linear_model import RidgeClassifierCV
    classifier3 = RidgeClassifierCV()
    classifier3.fit(x_train , y_train)
    
    from sklearn.linear_model import SGDClassifier
    classifier4 = SGDClassifier()
    classifier4.fit(x_train , y_train)
    
    from sklearn.linear_model import Perceptron
    classifier5 = Perceptron()
    classifier5.fit(x_train , y_train)
    
    print('LogisticRegression training accuracy: ', classifier1.score(x_train, y_train))
    print('PassiveAggressiveClassifier training accuracy: ', classifier2.score(x_train, y_train))
    print('RidgeClassifierCV training accuracy: ', classifier3.score(x_train, y_train))
    print('SGDClassifier training accuracy: ', classifier4.score(x_train, y_train))
    print('Perceptron training accuracy: ', classifier5.score(x_train, y_train))
    
    return classifier1, classifier2, classifier3 , classifier4 , classifier5

def naive_bayes_models(x_train , y_train):
    from sklearn.naive_bayes import BernoulliNB
    classifier1 = BernoulliNB()
    classifier1.fit(x_train, y_train)
    
    from sklearn.naive_bayes import GaussianNB
    classifier2 = GaussianNB()
    classifier2.fit(x_train, y_train)
    
    print('naive_bayes BernoulliNB training accuracy: ', classifier1.score(x_train, y_train))
    print('naive_bayes GaussianNB training accuracy: ', classifier2.score(x_train, y_train))
    
    return classifier1, classifier2

def nearest_neighbors_models(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    classifier1 = KNeighborsClassifier(n_neighbors = 5 ,metric = 'minkowski' , 
                                  p = 2)
    classifier1.fit(x_train, y_train)
    
    print('KNeighborsClassifier training accuracy: ', classifier1.score(x_train, y_train))
    
    return classifier1

def svm_models(x_train , y_train):
    from sklearn.svm import SVC
    classifier1 = SVC(kernel = 'rbf' , random_state = 0)
    classifier1.fit(x_train, y_train)
    
    from sklearn.svm import NuSVC
    classifier2 = NuSVC(kernel = 'rbf' , random_state = 0)
    classifier2.fit(x_train, y_train)
    
    from sklearn.svm import LinearSVC
    classifier3 = LinearSVC(dual=False)
    classifier3.fit(x_train , y_train)
    
    print('SVC training accuracy: ', classifier1.score(x_train, y_train))
    print('NuSVC training accuracy: ', classifier2.score(x_train, y_train))
    print('LinearSVC training accuracy: ', classifier3.score(x_train, y_train))
    
    return classifier1 , classifier2 , classifier3

def trees_models(x_train , y_train):
    from sklearn.tree import DecisionTreeClassifier
    classifier1 = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
    classifier1.fit(x_train, y_train)
    
    from sklearn.tree import ExtraTreeClassifier
    classifier2 = ExtraTreeClassifier()
    classifier2.fit(x_train , y_train)
    
    print('DecisionTreeClassifier training accuracy: ', classifier1.score(x_train, y_train))
    print('ExtraTreesClassifier training accuracy: ', classifier2.score(x_train, y_train))
    
    return classifier1 , classifier2

def discriminant_analysis_models(x_train, y_train):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    classifier1 = LinearDiscriminantAnalysis()
    classifier1.fit(x_train ,y_train)
    print('LinearDiscriminantAnalysis training accuracy: ', classifier1.score(x_train, y_train))    
    return classifier1

def XGBClassifier_models(x_train ,y_train):
    import xgboost as xgb
    classifier1 =  xgb.XGBClassifier(colsample_bytree = 0.5 ,learning_rate = .25, 
                                     max_depth =4,min_child_weight = 7,gamma = 0.4)
    classifier1.fit(x_train , y_train)
    print('XGBClassifier training accuracy: ', classifier1.score(x_train, y_train))
    return classifier1

def models(x_train, y_train):
    model1 = ensemble_models(x_train, y_train)
    model2 = gaussian_process_models(x_train, y_train)
    model3 = linear_models(x_train , y_train)
    model4 = naive_bayes_models(x_train , y_train)
    model5 = nearest_neighbors_models(x_train, y_train)
    model6 = svm_models(x_train , y_train)
    model7 = trees_models(x_train , y_train)
    model8 = discriminant_analysis_models(x_train, y_train) 
    model9 = XGBClassifier_models(x_train ,y_train)
    
    return (model1 ,model2 , model3 , model4 , model5 , model6 , model7 ,model8, model9 )
    
model = models(x_train , y_train)

#-----------------------------------------------------------------------------
#---------------------confusion matrix----------------------------------------

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
#------------confusion matrix except model[1] and model[4]--------------------
def confusion_matrix_accuracy(models1):
    for i in range(len(models1)):
        cm = confusion_matrix(y_cv , models1[i].predict(x_cv))
        print(models1[i])
        print('\n')
        print(cm)
        print(classification_report(y_cv , models1[i].predict(x_cv)))
        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator = models1[i], X = x_train, y = y_train, cv = 10)
        print(accuracies.mean())
        print(accuracies.std())
        print()
        
n = [0,2,3,5,6]        
for j in n:
    confusion_matrix_accuracy(model[j])

#-----------------------confusion matrix for model[0]-------------------------
cm = confusion_matrix(y_cv , model[1].predict(x_cv))
print(model[1])
print('\n')
print(cm)
print(classification_report(y_cv , model[1].predict(x_cv)))
accuracies = cross_val_score(estimator = model[1], X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()

#-----------------------confusion matrix for model[4]-------------------------
cm = confusion_matrix(y_cv , model[4].predict(x_cv))
print(model[4])
print('\n')
print(cm)
print(classification_report(y_cv , model[4].predict(x_cv)))
accuracies = cross_val_score(estimator = model[4], X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()

#-----------------------confusion matrix for model[7]-------------------------
cm = confusion_matrix(y_cv , model[7].predict(x_cv))
print(model[7])
print('\n')
print(cm)
print(classification_report(y_cv , model[7].predict(x_cv)))
accuracies = cross_val_score(estimator = model[7], X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()

#-----------------------confusion matrix for model[8]-------------------------
cm = confusion_matrix(y_cv , model[8].predict(x_cv))
print(model[8])
print('\n')
print(cm)
print(classification_report(y_cv , model[8].predict(x_cv)))
accuracies = cross_val_score(estimator = model[8], X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()

#-------------------------------apply knn model-------------------------------
# Importing the Keras libraries and packages

# libraries for NN
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
     # Part 2 - Now let's make the ANN!
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 14, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

cm = confusion_matrix(y_cv , grid_search.predict(x_cv))
print(grid_search.score(x_train, y_train))
print('\n')
print(cm)
print(classification_report(y_cv , grid_search.predict(x_cv)))

'''
#-----------------hyper parameter tuning for xgboost model--------------------
import xgboost as xgb
classifier1 =  xgb.XGBClassifier(colsample_bytree = 0.5 ,learning_rate = .25, max_depth =4,
                                 min_child_weight = 7,gamma = 0.4)
classifier1.fit(x_train , y_train)
print('XGBClassifier training accuracy: ', classifier1.score(x_train, y_train))
    
from sklearn.model_selection import GridSearchCV
parameters = [{"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
               "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
               "min_child_weight" : [ 1, 3, 5, 7 ],
               "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
               "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }]
grid_search = GridSearchCV(estimator = classifier1,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
cm = confusion_matrix(y_cv , classifier1.predict(x_cv))
print(classifier1)
print('\n')
print(cm)
print(classification_report(y_cv , classifier1.predict(x_cv)))
accuracies = cross_val_score(estimator = classifier1, X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()
'''

#-----------------hyper parameter tuning for logistic regression-------------------
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(C=1.2, random_state = 0, max_iter = 1500)
classifier1.fit(x_train, y_train)

from sklearn.model_selection import GridSearchCV
parameters = {'C':[0.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5]}
grid_search = GridSearchCV(estimator = classifier1,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
print("best_accuracy:- ", best_accuracy)
best_parameters = grid_search.best_params_
print("best_parameters:- ", best_parameters)
cm = confusion_matrix(y_cv , classifier1.predict(x_cv))
print(classifier1)
print('\n')
print(cm)
print(classification_report(y_cv , classifier1.predict(x_cv)))
accuracies = cross_val_score(estimator = classifier1, X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
print()

#-----------------------------------------------------------------------------
#--------------------------predecting test value------------------------------

y_pred = model[2][0].predict(x_test)  
#------------------------------submission-------------------------------------
gender['Survived'] = model[2][0].predict(x_test)
gender[['PassengerId', 'Survived']].to_csv("submission.csv", index = False)         
