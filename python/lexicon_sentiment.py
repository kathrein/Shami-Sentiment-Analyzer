#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# read csv file and write it in splited data set according to the first label colum pos or neg or no


"""
Created on Thu Feb 28 11:48:26 2019

@author: xabuka


this code read a csv  lexicon file and store every polarity in list 
read a csv data set with human annotation
create new file contain the roduced sentiment and the old one with the sentence
"""
import csv
#/Users/xabuka/PycharmProjects/measuring_acceptability/corpora/magid/train
lex_file = '../../MoArLex/MoArLexSecondIteration.csv'


pos_list, neg_list = [],[]
with open(lex_file) as csv_lex:
    csv_reader = csv.reader(csv_lex, delimiter=',')
    for row in csv_reader:
        #print(row[1])
        if row[1] =='Positive':
            pos_list.append(row[0])
        elif row[1] =='Negative':
            neg_list.append(row[0])
print(len(neg_list))
print(len(pos_list))
        
       
#
#
data_dir = '../data/'
file_sa = 'Paldataset.csv'
lexi_sa = 'Paldataset_lex.txt'

#dir_type = 'train/'
lexi_sa_file = open(lexi_sa,'a+',encoding = 'utf-8')

with open(data_dir+file_sa) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 1
    for row in csv_reader:
        neg_counter = 0
        pos_counter = 0
        for word in row[1].split(' '):
            if word in neg_list:
                neg_counter += 1
            elif word in pos_list:
                pos_counter += 1
        if pos_counter > neg_counter:
            sent = 1
        elif pos_counter < neg_counter:
            sent = 0
        else:
            sent = 2
        lexi_sa_file.write(str(row[0])+','+str(sent)+','+row[1]+'\n')
