#!/usr/bin/env python
# coding: utf-8

# In[31]:


##功能：对电子政务
##

# -*- coding:utf8 -*- 
import re    
import os


import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from numpy import *;

import re
import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec

from sklearn.datasets import load_files 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans

TaggededDocument = gensim.models.doc2vec.TaggedDocument
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)




def get_sentences(doc):
    line_break = re.compile('[\u3000 \r\n]')
    delimiter = re.compile('[\u3000 \b 。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            #print(sent)
            sentences.append(sent)
            
    return sentences

def get_stopword_list():
    stop_word_path = 'E:/abstract/stopwords.txt'
    stopword_list = [sw.replace('\n','') for sw in open(stop_word_path,encoding='utf-8',errors='ignore').readlines()]
    return stopword_list


def get_datasest(list_name):
    
    files = os.listdir(list_name)  
    for path in files:
        if(os.path.isfile(list_name + '/' + path)):   
            print(path+'---start---')
            logger.info(path+'---start---')
            file = open(list_name + '/' + path, 'r',encoding='utf8').read()
            #print("content:",file.split('\n')[2:])
            docs = get_sentences(file)
            x_train = []
            for i, text in enumerate(docs):
                ##如果是已经分好词的，不用再进行分词，直接按空格切分即可
                word_list = ' '.join(jieba.cut(text.split('\n')[0])).split(' ')
                l = len(word_list)
                word_list[l - 1] = word_list[l - 1].strip()
                document = TaggededDocument(word_list, tags=[i])
                x_train.append(document)
                #print(x_train)
    return x_train

def train(x_train, size=100, epoch_num=1): ##size 是你最终训练出的句子向量的维度，自己尝试着修改一下
 
    model_dm = Doc2Vec(x_train, min_count=1, window=5, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('dov2vec') ##模型保存的位置
    print("train end")
    return model_dm
 
def ceshi():
    model_dm = Doc2Vec.load("dov2vec")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    str1 = '“互联网+政务服务”让服务零距离 成为您的生活好帮手！'
    #print(type(str1))
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = ' '.join(jieba.cut(str1))
    #print(type(test_text))
    inferred_vector_dm = model_dm.infer_vector(test_text) ##得到文本的向量
    #print(inferred_vector_dm)
 
    return inferred_vector_dm

def text_to_vector(text):
   
    model_dm = Doc2Vec.load("dov2vec")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
   
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = ' '.join(jieba.cut(text)).split(' ')
 
    inferred_vector_dm = model_dm.infer_vector(test_text) ##得到文本的向量
    #print(inferred_vector_dm)
 
    return inferred_vector_dm


#训练doc2vec
#x_train = get_datasest('E:\datadel\passage_all')
#train(x_train)



# In[32]:


import numpy as np
import matplotlib.pyplot as plt

'''标志位统计递归运行次数'''
flag = 0

'''欧式距离'''
def ecludDist(x, y):
    return np.sqrt(sum(np.square(np.array(x) - np.array(y))))

'''曼哈顿距离'''
def manhattanDist(x, y):
    return np.sum(np.abs(x - y))

'''夹角余弦'''
def cos(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

'''计算簇的均值点'''
def clusterMean(dataset):
   # print("dataset------",sum(np.array(dataset),axis=0))
    if len(dataset)==0:
        sum_result = np.zeros(100)
    else:
        sum_result = sum(np.array(dataset),axis=0)/len(dataset)
    #print(sum_result)
    return sum_result

'''生成随机均值点'''
def randCenter(dataset, k):
    temp = []
    while len(temp) < k:
        index = np.random.randint(0, len(dataset)-1)
        if  index not in temp:
            temp.append(index)
    return np.array([dataset[i] for i in temp])

'''以数据集的前k个点为均值点'''
def orderCenter(dataset, k):
    return np.array([dataset[i] for i in range(k)])

'''聚类'''

def kMeans(dataset, dist, center, k,test_docs):
    global flag
    all_kinds=[]
    kinds_index=[]
    yindex = 0
    for _ in range(k):
        temp = []
        tmp_index = []
        all_kinds.append(temp)
        kinds_index.append(tmp_index)
    #计算每个点到各均值点的距离  
    for i in dataset:
        
        temp = []
        for j in center:
            temp.append(dist(i, j))
        
        all_kinds[temp.index(min(temp))].append(i)  #放入距离小的那个聚类中心所在的簇
        kinds_index[temp.index(min(temp))].append(yindex)
        yindex+=1
   
  
     #打印中间结果    
   #  for i in range(k):
   #      #print(np.array(all_kinds[i]))
   #     print('第'+str(i)+'组:', all_kinds[i], kinds_index[i], end='\n')

    flag += 1
    print('************************迭代'+str(flag)+'次***************************')
    logger.info('************************迭代{}次***************************'.format(str(flag)))
    #更新均值点
    center_ = np.array([clusterMean(i) for i in all_kinds])
    #print("center_",center_)
    #print("center",center)
    if np.all(center_ == center) or flag==10:
        global prediction
        prediction = ""
        print('结束')
       # for i in range(k):
        #    print('第'+str(i)+'组均值点：', center_[i], end='\n')
        #    plt.scatter([j[0] for j in all_kinds[i]], [j[1] for j in all_kinds[i]], marker='*')
       # plt.grid()
       # plt.show()
        result_index = []
  
        for i in range(k):
            tmp = []
            if len(all_kinds[i]) > 0:
                for j in  all_kinds[i]:
                    #print("j----",j)
                    tmp.append(dist(center[i], j))
                #print(len(tmp))
                #print(len(kinds_index[i]))
                #print((kinds_index[i][tmp.index(min(tmp))]))
                result_index.append(kinds_index[i][tmp.index(min(tmp))]) 
        #print(result_index)
        flag=0
        result_index.sort()
        prediction=""
        for i in result_index:
           # print(test_docs[i])
            prediction+= test_docs[i]
        logger.info("predict:", prediction)
        logger.info("---------------------------------")
        print("---------------------------------")
    else:
        #递归调用kMeans函数
        center = center_
        kMeans(dataset, dist, center, k,test_docs)
        

import copy
import math
import random
import time

global MAX
MAX=10000.0

global Epsilon
Epsilon=0.01

# In[34]:
def initialize_U(data, cluster_number):
	"""
	这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
	"""
	global MAX
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U
 
def distance(point, center):
	"""
	该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return math.sqrt(dummy)
 
def end_conditon(U, U_old):
    """
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
    global Epsilon
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):
		    if abs(U[i][j] - U_old[i][j]) > Epsilon :
			    return False
    return True
 
def normalise_U(U):
	"""
	在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
	"""
	for i in range(0, len(U)):
		maximum = max(U[i])
		for j in range(0, len(U[0])):
			if U[i][j] != maximum:
				U[i][j] = 0
			else:
				U[i][j] = 1
	return U
 
 
def fuzzy(data, cluster_number, m,weight):
	"""
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
	"""
    
	# 初始化隶属度矩阵U
	U = initialize_U(data, cluster_number)
	# print_matrix(U)
	# 循环更新U
	while (True):
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
		# 计算聚类中心
		C = []
		for j in range(0, cluster_number):
			current_cluster_center = []
			for i in range(0, len(data[0])):
				dummy_sum_num = 0.0
				dummy_sum_dum = 0.0
				for k in range(0, len(data)):
    				# 分子
					dummy_sum_num += (U[k][j] ** m) * data[k][i]
					# 分母
					dummy_sum_dum += (U[k][j] ** m)
				# 第i列的聚类中心
				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j簇的所有聚类中心
			C.append(current_cluster_center)
 
		# 创建一个距离向量, 用于计算U矩阵。
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)
 
		# 更新U
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0, cluster_number):
    				# 分母
					dummy += (distance_matrix[i][j ] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / dummy
 
		if end_conditon(U, U_old):
			print ("已完成聚类")
			break
 
	U = normalise_U(U)
	return U,C
def checker_iris(final_location,k,data,center,test_docs,weight):
    """
    和真实的聚类结果进行校验比对
    """
    all_kinds=[]
    weight_index=[]    #句子权重下标
    for _ in range(k):
        temp = []
        tmp_index = []
        all_kinds.append(temp)
        weight_index.append(tmp_index)

    for i in range(0,len(data)):
        for j in range(0, len(final_location[0])):
            if final_location[i][j] == 1:  #i+(50*k)表示 j表示第j类
                all_kinds[j].append(i)
                weight_index[j].append(i)
   # print(all_kinds)
    result_index = []

    prediction=""
    for i in range(k):
        tmp=[]
        if len(all_kinds[i]) > 0:
            wi=0
            for j in all_kinds[i]:
     #           print(j)
                #tmp.append(distance(center[i], data[j]))
                 tmp.append(ecludDist(center[i], j)*weight[weight_index[i][wi]])
                 wi+=1
            #print(tmp)
            if len(tmp)!=0:
                result_index.append(all_kinds[i][tmp.index(min(tmp))])
      #  print(result_index)
    result_index.sort()
    for i in result_index:
        #print(test_docs[i])
        prediction += test_docs[i]
    
    #print(prediction)

    #print("---------------------------------")
    return prediction


from rouge import Rouge
import jieba
import math
import os 
import re
#输入的两个字符串长度不应为0；
def Rouge_L(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    Rouge_L_R=maxNum/lstr2
    Rouge_L_P=maxNum/lstr1
    beta =Rouge_L_P/(Rouge_L_R+(math.e)**(-12))
    if Rouge_L_R==0 and Rouge_L_P==0:
        Rouge_L_F=0
    else:
        Rouge_L_F= ((1+beta**2)*Rouge_L_R*Rouge_L_P)/(Rouge_L_R+(beta**2*Rouge_L_P))
    return Rouge_L_P,Rouge_L_R,Rouge_L_F



def Rouge_1(model, reference):#terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    terms_reference= jieba.cut(reference)#默认精准模式
    terms_model= jieba.cut(model)
    
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    #print(grams_reference)
    #print(grams_model)
    temp = 0
    reference_ngram_all = len(grams_reference)
    model_ngram_all = len(grams_model)
    for x in grams_reference:
        if x in grams_model: temp=temp+1
    rouge_1_R=temp/reference_ngram_all
    rouge_1_P=temp/model_ngram_all
    if rouge_1_R==0 and rouge_1_P==0:
        rouge_1_F1=0
    else:
        rouge_1_F1=2*rouge_1_R*rouge_1_P/(rouge_1_R+rouge_1_P)#2 *  召回率 *  准确率/ (召回率+准确率)
    return rouge_1_P,rouge_1_R,rouge_1_F1
 
def Rouge_2(model, reference):#terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    terms_reference = jieba.cut(reference)
    terms_model = jieba.cut(model)
    grams_reference = list(terms_reference)
    grams_model = list(terms_model)
    gram_2_model=[]
    gram_2_reference=[]
    temp = 0
    
    reference_ngram_all = len(grams_reference)-1
    model_ngram_all = len(grams_model)-1
    if reference_ngram_all==0 or model_ngram_all==0:
        return 0.0, 0.0 ,0.0
    else:
        for x in range(len(grams_model)-1):
             gram_2_model.append(grams_model[x] + grams_model[x+1])
        #print(gram_2_model)
        for x in range(len(grams_reference)-1):
             gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
        #print(gram_2_reference)
        for x in gram_2_model:
            if x in gram_2_reference:temp=temp+1
        rouge_2_R=temp/reference_ngram_all
        rouge_2_P=temp/model_ngram_all
        if rouge_2_R>1:
            rouge_2_R=1.0
        #print("rouge_2:",rouge_2_R,rouge_2_P)
        if rouge_2_R==0 and rouge_2_P==0:
             rouge_2_F1 = 0
        else:
            rouge_2_F1=2*rouge_2_R*rouge_2_P/(rouge_2_R+rouge_2_P)
    return rouge_2_P,rouge_2_R,rouge_2_F1
 

def my_Rouge(model, reference):
    global ALL_Rouge_1_R,ALL_Rouge_2_R,ALL_Rouge_L_R
    global ALL_Rouge_1_P, ALL_Rouge_2_P, ALL_Rouge_L_P
    global ALL_Rouge_1_F, ALL_Rouge_2_F, ALL_Rouge_L_F
    print("rouge_1="+str(Rouge_1(model, reference)))
    print("rouge_2="+str(Rouge_2(model, reference)))
    print("rouge_L="+str(Rouge_L(model, reference)))
    logger.info("rouge_1={}".format(Rouge_1(model, reference)))
    logger.info("rouge_2={}".format(Rouge_2(model, reference)))
    logger.info("rouge_L={}".format(Rouge_L(model, reference)))
    ALL_Rouge_1_R += Rouge_1(model, reference)[1]
    ALL_Rouge_2_R += Rouge_2(model, reference)[1]
    ALL_Rouge_L_R += Rouge_L(model, reference)[1]
    ALL_Rouge_1_P += Rouge_1(model, reference)[0]
    ALL_Rouge_2_P += Rouge_2(model, reference)[0]
    ALL_Rouge_L_P += Rouge_L(model, reference)[0]
    ALL_Rouge_1_F += Rouge_1(model, reference)[2]
    ALL_Rouge_2_F += Rouge_2(model, reference)[2]
    ALL_Rouge_L_F += Rouge_L(model, reference)[2]
    



def processing(list_name):
    
    global count
    '''对每个语料'''
    # 所有文件夹，第一个字段是次目录的级别  
   
    # 所有文件  
   
    # 返回一个列表，其中包含在目录条目的名称(google翻译)  
    files = os.listdir(list_name) 
    files.sort(key= lambda x:int(x[:-4])) 
    for path in files:
        if(os.path.isfile(list_name + '/' + path)):   
            print(path+'---start---')
            logger.info(path+'---start---')
            file = open(list_name + '/' + path, 'r',encoding='utf8').read()
            title=file.split('\n')[0]
            contents=file.split('\n')[2:]
            print("title:",str(file.split('\n')[0]))
            logger.info("title={}".format((file.split('\n')[0])))
            filecontent =""
            for content in contents:
                filecontent+=content
            logger.info("filecontent={}".format(filecontent))
            weight=[]
            title_weight=[]
            para_weight=[]
            len_weight=[]

            #text = open("E:\\passage\  “互联网+政务服务”让服务零距离.txt",'r',encoding='utf-8').read( )  #读取文档转为str类型
            test_docs=[]

            if len(contents)>0:
                title_v=text_to_vector(title)
                #test_docs = get_sentences(filecontent)       #返回每一句话组成的list
                #print("test_docs:",len(test_docs))
                for s in contents:
                    gettext = get_sentences(s)
                    for t in gettext:
                        test_docs.append(t)
                        if len(gettext)==1:                 #一句话自成一段段落权重提高
                            para_weight.append(3)
                            break
                        else:
                            para_weight.append(1)

                if len(test_docs) >3:
                    count+=1
                    batches = []
                    for text in test_docs:
                        #print(text)
                        layer = text_to_vector(text)
                        title_weight.append(ecludDist(layer,title_v))   #与标题的相关性权重
                        batches.append(layer)
                        
                    x_texts=np.array(batches)
                    #print(x_texts)
                    k=3
                    #k=(int)(0.15 * len(test_docs))
                    #if k<=0:
                    #    k=1
                    for i in range(len(title_weight)):
                        weight.append(title_weight[i]/para_weight[i])

                    res,center=fuzzy(x_texts,k,2,weight)
                   # print(res)
                    prediction=checker_iris(res,k,x_texts,center,test_docs,weight)
                    '''
                    f:F1值  p：查准率  R：召回率
                    '''
                    logger.info("predict={}".format(prediction))
                    my_Rouge(prediction, title)
                    with open('fuzzy_res_all.txt', 'a+', encoding='utf-8') as f:
                        f.write('acticle:'+filecontent+'\n')
                        f.write('ref:'+title+'\n')
                        f.write('dec:'+prediction+"\n\n")
                    #print(rouge_score["rouge-2"])
                    #print(rouge_score[0]["rouge-l"])

    print(count)
    AVL_Rouge_1_P = ALL_Rouge_1_P/count
    AVL_Rouge_2_P = ALL_Rouge_2_P/count
    AVL_Rouge_L_P = ALL_Rouge_L_P/count
    AVL_Rouge_1_R = ALL_Rouge_1_R / count
    AVL_Rouge_2_R = ALL_Rouge_2_R / count
    AVL_Rouge_L_R = ALL_Rouge_L_R / count
    AVL_Rouge_1_F = ALL_Rouge_1_F / count
    AVL_Rouge_2_F = ALL_Rouge_2_F / count
    AVL_Rouge_L_F = ALL_Rouge_L_F / count
    print("rouge_1:({},{},{})".format(AVL_Rouge_1_P, AVL_Rouge_1_R, AVL_Rouge_1_F))
    print("rouge_2:({},{},{})".format(AVL_Rouge_2_P, AVL_Rouge_2_R, AVL_Rouge_2_F))
    print("rouge_L:({},{},{})".format(AVL_Rouge_L_P, AVL_Rouge_L_R, AVL_Rouge_L_F))


ALL_Rouge_1_P=0
ALL_Rouge_2_P=0
ALL_Rouge_L_P=0
ALL_Rouge_1_R=0
ALL_Rouge_2_R=0
ALL_Rouge_L_R=0
ALL_Rouge_1_F=0
ALL_Rouge_2_F=0
ALL_Rouge_L_F=0
count = 0
processing('E:\datadel\passage_test')


# In[ ]:





# In[ ]:




