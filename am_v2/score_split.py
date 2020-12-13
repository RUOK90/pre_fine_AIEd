import csv
import random

def getOldTestUsers():
    path = '/shared/AAAI20/score_data/14d_10q/test_user_list.csv'
    return getUsers(path)

def getOldTrainUsers():
    path = '/shared/AAAI20/score_data/14d_10q/train_user_list.csv'
    return getUsers(path)

def getUsers(path):
    users = []
    with open(path, 'r') as file:
        spamReader = csv.reader(file, delimiter=',')
        for row in spamReader:
            users.append(row[0])
    return users

def saveUsers(path, users):
    with open(path, 'w') as file:
        spamWriter = csv.writer(file, delimiter=',')
        for user in users:
            spamWriter.writerow([user])

def split():
    train_rate = 0.7
    validation_rate = 0.1
    test_rate = 0.2
    oldTestUsers = getOldTestUsers()
    oldTrainUsers = getOldTrainUsers()

    allUsers = oldTestUsers + oldTrainUsers
    train_end_index = int(len(allUsers) * train_rate)
    validation_end_index = int(len(allUsers) * (train_rate + validation_rate))
    random.shuffle(allUsers)

    trainUsers = allUsers[:train_end_index]
    validationUsers = allUsers[train_end_index:validation_end_index]
    testUsers = allUsers[validation_end_index:]

    trainPath = './train_user_list.csv'
    validationPath = './validation_user_list.csv'
    testPath = './test_user_list.csv'

    saveUsers(trainPath, trainUsers)
    saveUsers(validationPath, validationUsers)
    saveUsers(testPath, testUsers)

split()
