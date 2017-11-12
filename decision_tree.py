"""以信息熵为度量，构造信息熵下降最快的树
最大的优点：在使用者不了解过多的背景知识的情况下进行自学习。
三种生成算法：
1 ID3：信息增益最大的准则
2 C4.5：信息增益比最大的准则（不完全，使用启发式）
3 CART：回归树：平方误差最小的准则
        分类树：基尼系数最小的准则
随机森林：随机的形成许多决策树，使用每一颗随机树进行分类，去所有决策树中分类结果最多的为最终结果
"""
import numpy as np 
import pandas as pd
import operator as op

trainData = pd.read_csv('Titanic/train.csv')
testData = pd.read_csv('Titanic/test.csv')

#drop some data class
trainData = trainData.drop(['Cabin'], axis = 1)
testData = testData.drop(['Cabin'], axis = 1)
trainData = trainData.drop(['Ticket'], axis = 1)
testData = testData.drop(['Ticket'], axis = 1)

#fill in the age value
combine = [trainData, testData]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Donna'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mile', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
age_title_mapping = {1: 30, 2: 20, 3: 40, 4: 3, 5: 40, 6: 40}
trainData['Title'] = trainData['Title'].map(age_title_mapping)
testData['Title'] = testData['Title'].map(age_title_mapping)
#fill in the missing value
#fill in the Embarked value
trainData = trainData.fillna({'Embarked':'S'})
testData = testData.fillna({'Embarked': 'S'})
#sort the ages into logical categories
trainData['Age'] = trainData['Age'].fillna(0)
testData['Age'] = testData['Age'].fillna(0)
for x in range(len(trainData['Age'])):
    if trainData['Age'][x] == 0:
        trainData['Age'][x] = trainData['Title'][x]
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young adult', 'Adult', 'Senior']
trainData['AgeGroup'] = pd.cut(trainData['Age'], bins, labels = labels)
testData['AgeGroup'] = pd.cut(testData['Age'], bins, labels = labels)

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young adult': 5, 'Adult': 6, 'Senior': 7}

#convert the age group into a numerical value
trainData['AgeGroup'] = trainData['AgeGroup'].map(age_mapping)
testData['AgeGroup'] = testData['AgeGroup'].map(age_mapping)

#drop the Age value for now
trainData = trainData.drop(['Age'], axis = 1)
testData = testData.drop(['Age'], axis = 1)

#drop the Name value since it contains no useful information
trainData = trainData.drop(['Name'], axis = 1)
testData = testData.drop(['Name'], axis = 1)

#map each Sex value into a numerical value
sex_mapping = {'male' : 0, 'female' : 1}
trainData['Sex'] = trainData['Sex'].map(sex_mapping)
testData['Sex'] = testData['Sex'].map(sex_mapping)

#map each Embarked value into a numerical value
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
trainData['Embarked'] = trainData['Embarked'].map(embarked_mapping)
testData['Embarked'] = testData['Embarked'].map(embarked_mapping)

for x in range(len(testData['Fare'])):
    if pd.isnull(testData['Fare'][x]):
        pclass = testData['Pclass'][x]
        testData['Fare'][x] = round(trainData[trainData['Pclass'] == pclass]['Fare'].mean(), 4)

trainData['FareBand'] = pd.qcut(trainData['Fare'], 4, labels = [1, 2, 3, 4])
testData['FareBand'] = pd.qcut(trainData['Fare'], 4, labels = [1, 2, 3, 4])
#drop Fare values
trainData = trainData.drop(['Fare'], axis = 1)
testData = testData.drop(['Fare'], axis = 1)

predictors = trainData.drop(['Survived', 'PassengerId'], axis = 1)
target = trainData['Survived']
"""熵（entropy）： Ent(D) = -∑plogp ，Ent(D)值越小，这一集合的纯度越高
   信息增益：Gain（D， a）= Ent（D）- ∑Di/D * Ent（D)
   增益率：Gain_ratio(D|a) = Gain(D, a)/IV(a)
   IV(a) = """
Dataset = pd.concat([predictors, target], axis = 1)

#Ent
def Ent(Pro):
    Ents = no.dot(Pro, np.log(Pro)) * -1
    return Ents

#the probablity of subdataset
def probablity(dataset):
    length = len(dataset.iloc[:, -1].unique())
    pro = np.zeros(length)
    for x in range(length):
        pro[x] = len(dataset.iloc[:, -1][x]) / length
    return pro

#choose the feature to classify the dataset
def decide_class(dataset):
    col_len = len(dataset.columns)
    Gains = {}
    GainsList = zeros(col_len)
    Gain_ratios = {}
    for x in range(col_len):
        testFeature = dataset.iloc[:, x]
        testlabels = testFeature.unique()
        length = len(testFeature)
        lab_length = len(testlabels)
        ent = zeros(lab_length)
        pro = zeros(lab_length)
        for y in range(lab_length):
            testdata = dataset[testFeature == testlabels[y]]
            Ents = Ent(prpbablity(testdata))
            lab_inner_length = len(testData.iloc[:, 1])
            ent[y] = Ents
            pro[y] = lab_inner_length/lab_length
        Gain = Ent(probablity(dataset)) - np.dot(ent, pro)
        Gain_ratio = Gain / (-1 * (np.dot(pro, np.log(pro))))
        Gains[Gain] = x
        Gain_ratios[Gain_ratio] = x
    sorted_Gains = sorted(Gains.items, key = op.itemgetter(1), reverse=True)
    num_columns = sorted_Gains[:(col_len/2)]




            




    

    
    











