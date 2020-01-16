#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:02:54 2019

@author: robabbott
"""
import csv
import numpy as np
import os
import re
import spacy

def make_header():
    header = "rec_id,"

    data_header = ""

    for i in range(0,20):
        for j in range(0,300):
            data_header += "S"+str(i)+"_"+str(j)+","

    for i in range(0,10):
        for j in range(0,300):
            data_header += "Q"+str(i)+"_"+str(j)+","

    for i in range(0,1):
        for j in range(0,300):
            data_header += "A"+str(i)+"_"+str(j)

            if j < 299:
                data_header += ","


    header += data_header

    return header

print(make_header())

nlp = spacy.load('en_core_web_lg')


fn = os.path.join('tasks_1-20_v1-2', 'en', 
                  'qa1_single-supporting-fact_test.txt')

proc_idx = re.compile(r'(^\d+) (.+)')

last_idx = 0
context = 1
iterator = 1
docs = {}

cur_sentence = 1

with open(fn, 'r') as fp:
    st_reader = csv.reader(fp, delimiter='\t')

    # print('Start context: ', context)

    np.set_printoptions(precision=9, threshold=500000, linewidth=999999999999)

    arry_idx = 0
    arry = np.zeros((31,300), dtype='f')
    
    for row in st_reader:
        if len(row) == 1: # this is just a statement

            s_type = 0 # type 0 = statement

            g = proc_idx.findall(row[0])
            idx = int(g[0][0])
            stmt_str = g[0][1]
            if idx < last_idx:
                context = context + 1
                cur_sentence = 1;
                # print('Start context: ', context)
                docs = {}     

            doc = nlp(stmt_str)
            


            for token in doc:
                arry[arry_idx] = token.vector
                # arry = np.array([token.head.norm, 
                #                 token.norm, 
                #                 token.lemma, 
                #                 token.tag, 
                #                 token.pos])
                arry_idx += 1

    
            # docs[idx] = {'type':'statement', 'body' : doc, 'body_str':arry}

            # print("Context: ", context, " Statement: ", cur_sentence, " \tbody: ", doc)
            # arry_string = np.array2string(np.reshape(arry, (3000)), formatter={'float_kind':lambda x: "%.7f" % x},
            #     separator=",")


            # print(str(context)+"_"+str(cur_sentence)+",",arry_string[1:len(arry_string)-1],",",str(s_type))                           
#            print(stmt_str)
            
            arry_idx = iterator*10
            iterator += 1

        else: # this is a question row
            g = proc_idx.findall(row[0])
            idx = int(g[0][0])
            question_str = g[0][1]
            answer_str = row[1]

            s_type = 1 # s_type 1 = question

            if idx < last_idx:
                context = context + 1
                cur_sentence = 1;
                # print('Start context: ', context)  
                docs = {}
            
            doc = nlp(question_str)
            ans = nlp(answer_str)

            # arry = np.zeros((10, 300), dtype='f')
            # arry_idx = 0

            for token in doc:
                arry[arry_idx] = token.vector                
                # arry = np.array([token.head.norm, 
                #                 token.norm, 
                #                 token.lemma, 
                #                 token.tag, 
                #                 token.pos])
                arry_idx += 1

            arry_idx=30

            for token in ans:
                arry[arry_idx] = token.vector                
            
            # docs[idx] = {'type':'question', 'body' : doc, 'body_str':arry, 'answer':ans, 'answer_str':arry2}

            # print("Context: ", context, " Question : ", cur_sentence, " \tbody: ", doc)
            arry_string = np.array2string(np.reshape(arry, (9300)), formatter={'float_kind':lambda x: "%.7f" % x}, 
                separator=",")

            print(str(context)+"_"+str(cur_sentence)+",",arry_string[1:len(arry_string)-1])                           

            arry_idx = 0
            iterator = 1
            arry = np.zeros((31,300), dtype='f')

#            print('question: ', question_str, '\t answer: ', answer_str)

        last_idx = idx
        cur_sentence+=1;  

# TODO:  try constructing verb-prep-obj ngrams


