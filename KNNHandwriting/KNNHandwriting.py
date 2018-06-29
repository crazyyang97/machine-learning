# -*- coding: utf-8 -*-
'''
Copyright (C), 2018, fuujiro, DLUT (电创1601冯子扬 201688035)
Date:  2018-06-29 11:56

@author: fuujiro fuujiro@foxmail.com
@version: 1.0
@Environment: Python 2.7.15

'''

from numpy import *
from os import listdir
import KNN
from numpy.core import multiarray



def img2vector(filename):
    '图像文件转换成矩阵'
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):             #将32行合并成一行
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect               #一个样本最终成为一个1*1024的向量


def handwritingClassTest():
    '手写识别测试函数，调用了KNN模块的KNN分类器函数'
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "in #%d, the classifier came back with: %d, the real answer is: %d" % (i, classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
handwritingClassTest()
 
    
    


