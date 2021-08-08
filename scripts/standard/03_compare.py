import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lbfcs.analyze as analyze

plt.style.use(r'C:\Users\flori\Documents\mpi\repos\lbFCS\styles\paper.mplstyle')


#################### Define directories of solved series
dir_names = []
dir_names.extend([r'C:\Data\p17.lbFCS2\21-04-30_B23_N1-4\04_s2_Pm2-8nt-c5000_p40uW_s50_2FOVs_1'])
dir_names.extend([r'C:\Data\p17.lbFCS2\20-12-17_N1-2x5xCTC_cseries\id180_Pm2-1d25nM_p40uW_exp400_1'])

data_init, files = analyze.load_all_pickedprops(dir_names,'Pos0')
print(files)

#%%
data = data_init.copy()
print(files)
print()

####################
'''
Analyze
'''
####################
exp = 0.4
success = 99
ids = [1]


query_str = 'id in @ids '
query_str += 'and abs(frame-M/2)*(2/M) < 0.5 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= @success '
query_str += 'and N < 3 '
query_str += 'and koff/@exp < 0.17 '


data = data.query(query_str)
print(files.query('id in @ids'))
print('Remaining groups: %i'%len(data))


####################
'''
Plotting
'''
####################
################### Plot results

f = plt.figure(0,figsize = [5,9])
f.clear()
f.subplots_adjust(left=0.2,bottom=0.08,hspace=0.5)

#################### success
field = 'success'
bins = np.linspace(80,100,100)
ax = f.add_subplot(511)

ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')
ax.set_xlabel(field)

#################### N
field = 'N'
bins = np.linspace(0,3,45)
ax = f.add_subplot(512)
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')
ax.set_xlabel(field)


#################### koff
field = 'koff'
bins = np.linspace(0,0.6,60)
ax = f.add_subplot(513)
ax.hist(data.query(query_str)[field]/exp,
        bins=bins,histtype='step',ec='k')
ax.set_xlabel(field)

#################### kon
field = 'konc'
bins = np.linspace(0,0.1,60)
ax = f.add_subplot(514)
ax.hist(data.query(query_str)[field],#*(1e-6/(exp*data.conc*1e-12)),
        bins=bins,histtype='step',ec='k')
ax.set_xlabel(field)

#################### eps
field = 'occ'
bins = np.linspace(0,0.8,60)
ax = f.add_subplot(515)
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')
ax.set_xlabel(field)