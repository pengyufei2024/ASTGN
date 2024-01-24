import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import os

#Read traffic matrix dataset
def readTM(fileName, size):
    TM = np.zeros((size, size), dtype=np.float64)
    with open(fileName) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                row = int(line.split(' ')[0])
                column = int(line.split(' ')[1])
                value = float(line.split(' ')[2])
                TM[row - 1][column - 1] = value
    return TM

G=nx.Graph()
#Set nodes for network topology
nodes=[]
G.add_nodes_from(nodes)

#Set edges and weights for network topology
edges=[]
G.add_weighted_edges_from(edges)

#A folder for storing network matrix datasets
path=""
path_list=os.listdir(path)
path_list.sort()
txt_len=len(path_list)
H5traffic = np.zeros((1, 12), dtype=np.float)

#Calculate traffic data
for filename in path_list:
    name=path+filename
    TM=readTM(name,12)
    nodeTM = np.zeros((1,12), dtype=np.float)
    for i in range(1, 13):
        for j in range(1, 13):
            if i != j:
                path_all = nx.shortest_path(G, source=i, target=j)
                for a in range(1, len(path_all)):
                    nodeTM[0][path_all[a] - 1] = nodeTM[0][path_all[a] - 1] + TM[i - 1][j - 1]
    H5traffic = np.append(H5traffic, nodeTM, axis=0)
H5traffic=H5traffic[1:]
H5traffic=H5traffic/1024

# Create index for. h5 file
time_h5=pd.date_range(start='2004-03-01 00:00:00',periods= 5000,freq='5T')

#Create columns for. h5 file
id=list(range(1,13))
id_str=[str(x) for x in id]
id_h5=pd.Index(id_str,dtype=object)

#Create values for. h5 file
frame=pd.DataFrame(H5traffic,columns=id_h5,index=time_h5)
frame.to_hdf('Abilene.h5',key='df',mode='w')
res=pd.read_hdf('Abilene.h5')
print(res)





































# path=[[1,1,3,4,2,2,3,3,3,4,5,2],
#       [1,1,2,3,1,1,2,2,2,3,4,1],
#       [3,2,1,3,3,1,2,4,1,4,4,2],
#       [4,3,3,1,2,2,1,2,4,1,1,4],
#       [2,1,3,2,1,2,1,1,3,2,3,2],
#       [2,1,1,2,2,1,1,3,2,3,3,2],
#       [3,2,2,1,1,1,1,2,3,2,2,3],
#       [3,2,4,2,1,3,2,1,4,1,2,3],
#       [3,2,1,4,3,2,3,4,1,5,5,1],
#       [4,3,4,1,2,3,2,1,5,1,1,4],
#       [5,4,4,1,3,3,2,2,5,1,1,5],
#       [2,1,2,4,2,2,3,3,1,4,5,1]]
# Path=np.array(path)
# Weight=np.zeros((12,12),dtype=np.float)
# print(Path)
# for i in range(12):
#     for j in range(12):
#         if i!=j:
#           Weight[i][j]=Path[i][j]
#         else:
#             Weight[i][j]=0
# print(Weight)
# STG=pd.DataFrame(Weight)
# STG.to_csv('W_12.csv',index=False)
# std=np.std(Weight, dtype = np.float)
# print(std)
# Weight=np.zeros((12,12),dtype=np.float)
# for i in range(12):
#     for j in range(12):
#         if i!=j:
#             if math.exp(-((Path[i][j]**2)/(std**2)))>=0.1:
#                 Weight[i][j]=math.exp(-((Path[i][j]**2)/(std**2)))
#             else:
#                 Weight[i][j]=0
#         else:
#             Weight[i][j]=0
# Weight= np.around(Weight, 6)
# print(Weight)
# STG=pd.DataFrame(Weight)
# STG.to_csv('W_12.csv',index=False,)
# for i in range(12):
#     for j in range(12):
#         f = open("Adj.txt", "a")
#         f.write(f"{i+1} {j+1} {Weight[i][j]}\n")
#         f.close()




# testY=np.resize(testY,[726,144])
# testPred=np.resize(testPred,[726,144])
# np.savetxt("True_Traffic.csv",testY,delimiter=',')
# np.savetxt("Predict_Traffic.csv",testPred,delimiter=',')
# testY=np.array(testY)
# testPred=np.array(testPred)
# True_Traffic=pd.DataFrame(testY)
# True_Traffic.to_csv('True_Traffic.csv',index=False)
# Predict_Traffic=pd.DataFrame(testPred)
# Predict_Traffic.to_csv('Predict_Traffic.csv',index=False)

# f = open('True_Traffic.csv','w',newline='')
# writer = csv.writer(f)
# for i in testY:
#     writer.writerow(i)
# f.close()
#
# f = open('Predict_Traffic.csv','w',newline='')
# writer = csv.writer(f)
# for i in testPred:
#     writer.writerow(i)
# f.close()