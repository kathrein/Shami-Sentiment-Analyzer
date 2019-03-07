#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# read csv file and write it in splited data set according to the first label colum pos or neg or no


"""
Created on Thu Feb 28 11:48:26 2019

@author: xabuka


this code read a csv file and store every sentence in a seperate file 
according to the name of the labelled folder
"""
import csv
#/Users/xabuka/PycharmProjects/measuring_acceptability/corpora/magid/train
data_dir = '../data/labr3/'
dir_type = 'train/'
with open(data_dir+'3_labr_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 1
    for row in csv_reader:
        if row[0] == '4' or row[0] == '5':
            write_file = open(data_dir+dir_type+'POS/'+str(line_count)+'.txt','w+')
            write_file.write(row[-1])
            line_count += 1
        elif row[0] == '1' or row[0] == '2':
            write_file = open(data_dir+dir_type+'NEG/'+str(line_count)+'.txt','w+')
            write_file.write(row[-1])
            line_count += 1
        elif row[0]== '3':
           write_file = open(data_dir+dir_type+'NO/'+str(line_count)+'.txt','w+')
           write_file.write(row[-1])
           line_count += 1
        
        
        
print(f'Processed {line_count} lines.')
    
#with open('data.txt','a') as a_file:
#    a_file.writelines('new Lin 1a')
#    a_file.write('new Lin 2a')
#    a_file.close()