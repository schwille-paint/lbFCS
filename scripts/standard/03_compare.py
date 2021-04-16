import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lbfcs.analyze as analyze

plt.style.use('~/lbFCS/styles/paper.mplstyle')


#################### Define directories of solved series
dir_names = []

##################### NUPs a20
### @2nM
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_a20/02_s2w6_c1500_561-p40uW-s23_FOV2_1'])
### @250pM
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_a20/05_s2w4_c1500_561-p60uW-s23_FOV3_1'])

##################### NUPs 2x5xCTC
# ## @2nM
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_2x5xCTC/02_s2w1_c1500_561-p60uW-s39_FOV1_1'])
## @250pM
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_2x5xCTC/05_s1w7_c1500_561-p60uW-s39_FOV2_1'])
## @50pM
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_2x5xCTC/08_s1w6_c1500_561-p60uW-s11_FOV3_1'])

##################### NUPs 1x5xCTC
### @1nM
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_1x5xCTC/02_s1w3_c1500_561-p60uW-s13_FOV1_1'])

### @250pM & @24.5C
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_1x5xCTC/10_s1w2_c1500_561-p60uW-s13_FOV2_1'])

data_init, files = analyze.load_props(dir_names)
print(files)

#%%
data = data_init.copy()

####################
'''
Analyze
'''
####################
exp = 0.4
success = 92

query_str = 'id >= 0'
query_str += 'and abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= @success '

query_str += 'and occ > 0.25'
# query_str += 'and koff > 0.1*@exp '
# query_str += 'and konc*(1e-6/(@exp*conc*1e-12)) > 2'
# query_str += 'and nn_d > 1 '
query_str += 'and N < 25 '


data = data.query(query_str)
print(len(data))

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
bins = np.linspace(0,100,100)
ax = f.add_subplot(411)

ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')

#################### N
field = 'N'
bins = np.linspace(0,10,45)
ax = f.add_subplot(412)
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',ec='k')

#################### koff
field = 'koff'
bins = np.linspace(0,0.4,60)
ax = f.add_subplot(413)
ax.hist(data.query(query_str)[field]/exp,
        bins=bins,histtype='step',ec='k')


#################### kon
field = 'konc'
bins = np.linspace(0,30,60)
ax = f.add_subplot(414)
ax.hist(data.query(query_str)[field]*(1e-6/(exp*data.conc*1e-12)),
        bins=bins,histtype='step',ec='k')
