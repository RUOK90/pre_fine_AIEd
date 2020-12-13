import csv

ROOT_PATH = '/shared/AAAI20/response'
OLD_TRAIN_PATH = f'{ROOT_PATH}/train_user_list.csv'
OLD_VALID_PATH = f'{ROOT_PATH}/val_user_list.csv'
OLD_TEST_PATH = f'{ROOT_PATH}/test_user_list.csv'
REVIEW_USER_PATH = f'{ROOT_PATH}/review_quiz_students_list.csv'
TRAIN_WITHOUT_REVIEW_PATH = f'./train_users_without_review'

def gatherAllUserIds():
    trainUserIds = readUserIds(OLD_TRAIN_PATH)
    testUserIds = readUserIds(OLD_TEST_PATH)
    validationUserIds = readUserIds(OLD_VALID_PATH)
    return trainUserIds + testUserIds + validationUserIds

def gatherReviewUserIds(path, skipHeader=False):
    ids = []
    with open(path) as file:
        reader = csv.reader(file)
        if skipHeader is True:
            _ = next(reader, None)
        for row in reader:
            ids.append(row[1])
    return ids

def getFlagByUserIdOfReview():
    flagByUserId = {}
    reviewUserIds = gatherReviewUserIds(REVIEW_USER_PATH, True)
    for userId in reviewUserIds:
        flagByUserId[userId] = True
    return flagByUserId

def readUserIds(path):
    ids = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            ids.append(row[0].split('.')[0])
    return ids

def getUserIdsWithoutReview():
    userIdsWithoutReview = []
    allUserIds = gatherAllUserIds()
    flagByUserIdOfReview = getFlagByUserIdOfReview()
    for userId in allUserIds:
        flag = flagByUserIdOfReview.get(userId, False)
        if flag is False:
            userIdsWithoutReview.append(userId)
    return userIdsWithoutReview

def saveUsersIdsWithoutReview(path):
    userIdsWithoutReview = getUserIdsWithoutReview()
    with open(path, 'w') as csvFile:
        spamWriter = csv.writer(csvFile, delimiter=',')
        for userId in userIdsWithoutReview:
            spamWriter.writerow([f'{userId}.csv'])

def analyzeScoreData():
    rootPath = "/shared/AAAI20/score_data/14d_10q"
    trainName = "train_user_list.csv"
    validationName = "validation_user_list.csv"
    testName = "test_user_list.csv"
    trainUserCount, trainScoreCount, trainResponseCount = countScore(rootPath, trainName)
    validationUserCount, validationScoreCount, validationResponseCount = countScore(rootPath, validationName)
    testUserCount, testScoreCount, testResponseCount = countScore(rootPath, testName)
    print('userCount scoreCount, responseCount')
    print('train:', trainUserCount, trainScoreCount, trainResponseCount)
    print('validation:', validationUserCount, validationScoreCount, validationResponseCount)
    print('test:', testUserCount, testScoreCount, testResponseCount)
    print('total:', trainUserCount + validationUserCount + testUserCount,
          trainScoreCount + validationScoreCount + testScoreCount,
          trainResponseCount + validationResponseCount + testResponseCount)

def countScore(rootPath, name):
    path = f'{rootPath}/{name}'
    fileNames = []
    userIdByUserId = {}
    with open(path, newline='') as csvFile:
        spamReader = csv.reader(csvFile, delimiter=',')
        for row in spamReader:
            fileName = row[0]
            userId = fileName.split('_')[0]
            fileNames.append(fileName)
            userIdByUserId[userId] = userId
    responseCount = 0
    for fileName in fileNames:
        responsePath = f'{rootPath}/response/{fileName}'
        lineCount = len(open(responsePath).readlines())
        responseCount += min(lineCount, 100)

    return len(userIdByUserId.keys()), len(fileNames), responseCount

def countReview(rootPath, name):
    path = f'{rootPath}/{name}'
    fileNames = []
    userIdByUserId = {}
    with open(path, newline='') as csvFile:
        spamReader = csv.reader(csvFile, delimiter=',')
        for row in spamReader:
            fileName = row[0]
            userId = fileName.split('_')[2]
            fileNames.append(fileName)
            userIdByUserId[userId] = userId
    responseCount = 0
    for fileName in fileNames:
        responsePath = f'{rootPath}/response/{fileName}'
        lineCount = len(open(responsePath).readlines())
        responseCount += min(lineCount, 100)

    return len(userIdByUserId.keys()), len(fileNames), responseCount


def analyzeReviewData():
    rootPath = "/shared/AAAI20/review_data"
    trainName = "train_data_names.csv"
    validationName = "validation_data_names.csv"
    testName = "test_data_names.csv"
    trainUserCount, trainScoreCount, trainResponseCount = countReview(rootPath, trainName)
    validationUserCount, validationScoreCount, validationResponseCount = countReview(rootPath, validationName)
    testUserCount, testScoreCount, testResponseCount = countReview(rootPath, testName)
    print('userCount reviewCount, responseCount')
    print('train:', trainUserCount, trainScoreCount, trainResponseCount)
    print('validation:', validationUserCount, validationScoreCount, validationResponseCount)
    print('test:', testUserCount, testScoreCount, testResponseCount)
    print('total:', trainUserCount + validationUserCount + testUserCount,
          trainScoreCount + validationScoreCount + testScoreCount,
          trainResponseCount + validationResponseCount + testResponseCount)


# userIds = getUserIdsWithoutReview()
# saveUsersIdsWithoutReview(TRAIN_WITHOUT_REVIEW_PATH)
analyzeReviewData()
