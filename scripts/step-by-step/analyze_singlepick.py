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

plt.style.use(r'C:\Users\flori\Documents\mpi\repos\lbFCS\styles\paper.mplstyle')
# plt.style.use('~/lbFCS/styles/paper.mplstyle')

############################## Load props & picked
dir_names = [r'C:\Data\p17.lbFCS2\21-07-13_sumN1']

props_init,files_props = analyze.load_all_pickedprops(dir_names,must_contain = ['k2'],filetype = 'props')
picked_init,files_picked = analyze.load_all_pickedprops(dir_names,must_contain = ['k2'],filetype = 'picked')

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
success = 99 # Success criterium

query_str = 'id >= 0'
query_str += 'and abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= @success '
query_str += 'and N < 2.3 '
query_str += 'and N >  1.8 '
query_str += 'and occ > 0.3 '

### Query props an get remaining groups
props = props.query(query_str)
groups = props.group.values
print(len(groups))


##############################
'''
Analyze one group
'''
##############################

### Select one specific group 
g = 14
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
p_uplim = 1200
level_uplim = 4
window = [2000,3000]

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
ax.set_xlabel('I (t)')
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
ax.set_xlabel('Time')
ax.set_xticks([])
ax.set_xlim(window)
ax.set_ylabel('I (t)')
ax.set_ylim(-100,p_uplim/2)
ax.set_yticks([])

############################ Show occupancy barcode
f = plt.figure(11,figsize=[6,0.9])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.15,right=0.95)
f.clear()
ax = f.add_subplot(111)
x = np.arange(0,len(trace))
y_up = trace.copy()
y_low = np.zeros(len(trace)) 
y_up[y_up>0] = 1
ax.fill_between(x,
                y_low,y_up,
                color='tomato',
                alpha=0.5,
                )
ax.step(x,y_up,'-',c='k',lw=0.5)
ax.set_xlim(window)
ax.set_xticks([])
ax.set_ylim(0.1,0.9)
ax.set_yticks([])

############################ Info
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
