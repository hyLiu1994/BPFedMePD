import h5py
import os
import glob

dataset = 'Mnist'

datasize = 'small'

algorithm = 'pFedBayes'

alg = dataset + '_' + datasize + '_' + algorithm

if(algorithm == 'pFedMe' or algorithm == 'pFedMe_p'):
    alg = alg + '_0.001_10_15_10u_50b_20_5_0.001_avg.h5'
else:
    alg = alg + '_0.001_1.0_15_10u_100b_20_0.h5'

file_path = os.path.join('results' , alg)

with h5py.File(file_path , 'r') as f: 
    for key in f.keys(): 
        print(key) 
        data = f[key][:] 
        print(max(data)) 