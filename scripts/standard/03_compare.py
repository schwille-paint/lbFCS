import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lbfcs.analyze as analyze

plt.style.use('~/lbFCS/styles/paper.mplstyle')


#################### Define directories of solved series
dir_names = []
### Cy3B in B&C&B(new sample)&B(new sample)
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/01_s1_Cy3B-B-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/02_s1_Cy3B-C-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/08_s2_Cy3B-B-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/09_s2_Cy3B-B-c10000_p40uW_s50_1'])

### Cy3B in B&C&B (next day, new sample)
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/01_s1_Cy3B-B-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/04_s1_Cy3B-C-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/06_s1_Cy3B-B-c10000_p40uW_s50_1'])

### Atto550 in B&C&B(new sample)
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/03_s1_Atto550-B-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/04_s1_Atto550-C-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks_green/05_s2_Atto550-B-c10000_p40uW_s50_repeat_1'])

### Atto550 in B&C&B (next day, new sample)
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/02_s2_Atto550-B-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/03_s2_Atto550-C-c10000_p40uW_s50_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_imager-checks_green/05_s2_Atto550-B-c10000_p40uW_s50_1'])

### Atto565 in B&C (very strange dye -> Do not use!)
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks/06_s2_Atto565-B-c10000_p40uW_s50_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-20_imager-checks/07_s2_Atto565-C-c10000_p40uW_s50_1'])



data_init, files = analyze.load_all_pickedprops(dir_names)
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
success = 96
# ids = [10,12] # At550 in B equilibrating
ids = [11] # At550 in C equilibrating

# ids = [4,6] # Cy3B in B equilibrating
# ids = [5] # Cy3B in C equilibrating

query_str = 'id in @ids '
query_str += 'and abs(frame-M/2)*(2/M) < 0.5 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= @success '
query_str += 'and N < 3 '
query_str += 'and abs(eps_normstat-1) < 0.2 '

# query_str += 'and koff < 0.17*@exp '
# query_str += 'and sqrt((x-350)**2+(y-350)**2) < 300 '
# query_str += 'and konc*(1e-6/(@exp*conc*1e-12)) > 0.5'
# query_str += 'and nn_d > 5 '



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
f.subplots_adjust(bottom=0.1,wspace=0.3)
#################### success
field = 'success'
bins = np.linspace(80,100,100)
ax = f.add_subplot(511)

ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')

#################### N
field = 'N'
bins = np.linspace(0,3,45)
ax = f.add_subplot(512)
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')
ax.axvline(1)

#################### koff
field = 'koff'
bins = np.linspace(0,0.6,60)
ax = f.add_subplot(513)
ax.hist(data.query(query_str)[field]/exp,
        bins=bins,histtype='step',ec='k')
ax.axvline(0.11,ls='--')
ax.axvline(0.26,ls='--')
ax.axvline(0.085)
ax.axvline(0.2)

#################### kon
field = 'konc'
bins = np.linspace(0,30,60)
ax = f.add_subplot(514)
ax.hist(data.query(query_str)[field]*(1e-6/(exp*data.conc*1e-12)),
        bins=bins,histtype='step',ec='k')
ax.axvline(4.8*1,ls='--')
ax.axvline(16*1,ls='--')
ax.axvline(4.8*0.8)
ax.axvline(16*0.8)

#################### eps
field = 'eps'
bins = np.linspace(50,600,60)
ax = f.add_subplot(515)
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')