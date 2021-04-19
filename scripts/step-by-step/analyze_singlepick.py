'''
Script to analze single picks.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lbfcs.pick_fcs as fcs
import lbfcs.pick_other as other
import lbfcs.pick_combine as pickprops
import lbfcs.analyze as analyze

plt.style.use('~/lbFCS/styles/paper.mplstyle')

############################## Load props & picked
dir_names = ['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/21-04-13_N4_JS_test']

props_init,files_props = analyze.load_all_pickedprops(dir_names,must_contain = ['2500pM'],filetype = 'props')
picked_init,files_picked = analyze.load_all_pickedprops(dir_names,must_contain = ['2500pM'],filetype = 'picked')

#%%
print(files_props)
##############################
'''
Filter props
'''
##############################

### Copy initial DataFrames
props = props_init.copy()
picked = picked_init.copy()

### Define filter
success = 98 # Success criterium

query_str = 'id >= 0'
query_str += 'and abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= @success '
query_str += 'and N < 7 '

### Query props an get remaining groups
props = props.query(query_str)
groups = props.group.values

#%%
##############################
'''
Analyze one group
'''
##############################

### Select one specific group 
g = 105
g = groups[g]
df = picked.query('group == @g')

### Necessary parameters for analysis
M = df.M.iloc[0]
ignore = 1
weights = [1,1,1,1]
photons_field = 'photons_ck'
exp = 0.4

### Get all properties of this group
s = pickprops.combine(df,M,ignore,weights)
print(s.iloc[:10])

### Get trace&ac
trace, ac = fcs.trace_ac(df,M,field = photons_field)

### Get normalization histogram
eps ,x, y, y2, y_diff = other.extract_eps(df[photons_field].values)

##############################
'''
Visualize
'''
##############################
p_uplim = 3200
level_uplim = 4

############################ Normalization photon histogram
f = plt.figure(0,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.23,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.step(x,y,where='mid',
        c='tomato',label='original')
ax.step(x,-y2,where='mid',
        c='darkblue',label='neg. doubled')
ax.step(x,y_diff,where='mid',
        c='darkmagenta',label='difference')
ax.axvline(eps,ls='--',c='k',label=r'$\epsilon_{norm}$')
ax.axhline(0,ls='-',c='k')
patch = plt.Rectangle([0.6*eps,0],
                      0.8*eps,
                      ax.get_ylim()[-1],
                      ec='k',fc='grey',alpha=0.3)
ax.add_patch(patch)

ax.legend()
ax.set_xlabel('Photon Counts')
ax.set_xlim(0,p_uplim)
ax.set_ylabel('Occurences')
ax.set_ylim(-ax.get_ylim()[-1],ax.get_ylim()[-1])


############################ Trace
f = plt.figure(1,figsize=[6,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.15,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.plot(trace,c='tomato')
for i in range(1,level_uplim+1):
    ax.axhline(i*eps,ls='--',c='k')
ax.set_xlabel('Frames')
ax.set_ylabel('Photon Counts')
ax.set_ylim(-100,p_uplim/2)


############################ Trace
f = plt.figure(3,figsize=[2,3])
f.subplots_adjust(bottom=0.03,top=0.98,left=0.02,right=1)
f.clear()
ax = f.add_subplot(111)
patch = plt.Rectangle([0,0],
                      0.25,
                      s.occ,
                      ec='none',fc='lightgrey')
ax.add_patch(patch)
patch = plt.Rectangle([0,0],
                      0.25,
                      1,
                      ec='k',fc='none')
ax.add_patch(patch)
ax.set_axis_off()
ax.text(0.07,0.5,r'$\rho$',fontsize=18,rotation=90)
ax.text(0.3,0.7,'N = %.2f'%s.N,fontsize=11)
ax.text(0.3,0.5,r'$k_{off}$' +' = %.2f (1/s)'%(s.koff/exp),fontsize=11)
ax.text(0.3,0.3,r'$k_{on}c$' +' = %.2f (1/s)'%(s.konc/exp),fontsize=11)
