import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
M1 = [1000, 2000, 3000, 4000, 4500, 5000 ,5500,6000, 6500,7000]
x1 = [5.419 , 8.231, 23.85, 950.8, 377.0, 143.7, 54.61 , 21.53, 8.999,4.059]

M2 = [1000, 2000, 3000, 4000, 4500,5000 ,5500 ,6000, 6500,7000]
x2 = [0   ,13.11,37.63,1476 ,622.9 ,253.0, 102.1, 42.42, 18.29,8.361]
"""
M1 = [1000,2000,3000,4000,4500,5000, 5500, 6000, 6500, 7000]
M2 = M1

x1 = (1.)*np.array([2531.0 ,109.9   ,11.51    ,1.796    ,0.8133   ,0.410400, 0.230700, 0.1423000,0.0957100,0.0673900],'d')
dx1= (1.)*np.array([16.84  ,0.7159  ,0.07034  ,0.009896 ,0.004596 ,0.002336, 0.001361, 0.0008805,0.0006138,0.0004375],'d')

x1p = (1.)*np.array([2470.0 ,104.8   ,10.23    ,1.353    ,0.525100 , 0.213700, 0.092130, 0.0422900, 0.0209100, 0.0110800],'d')
dx1p= (1.)*np.array([15.41  ,0.6374  ,0.06126  ,0.008078 ,0.002860 , 0.001111, 0.000491, 0.0002299, 0.0001148, 0.0000631],'d')

x1pp =(0.001)*np.array([5.419 , 8.231, 23.85, 950.8, 377.0, 143.7, 54.61, 21.53, 8.999, 4.059],'d')
x2pp= (0.001)*np.array([8.666 , 13.11, 37.63, 1476 , 622.9, 253.0, 102.1, 42.42, 18.29, 8.361],'d')


x2 = (1.)*np.array([2940.0 ,136.1   ,15.33    , 2.513   ,1.148    ,0.568600, 0.308900, 0.1842000,0.1187000,0.0824700],'d')
dx2= (1.)*np.array([19.55  ,0.8948  ,0.09525  ,0.01419  ,0.00632  ,0.003318, 0.001769, 0.0010960,0.0007404,0.0005294],'d')

x2p = (1.)*np.array([2863.0 ,130.7   ,13.84   , 2.018   ,0.8197    ,0.3448   , 0.1517000, 0.0699300 , 0.0342000 , 0.01783],'d')
dx2p= (1.)*np.array([17.85  ,0.7987  ,0.08288 ,0.01200  ,0.004711  ,0.001852 , 0.0007851, 0.0003784 , 0.0001860 , 0.00009799],'d')

R=[] 
dR=[]
Rp=[] 
dRp=[]
Rpp =[]
np.random.seed(1234)
for i in range(len(M2)):
    R.append( x2[i]/ x1[i])
    Nums = np.random.normal(x2[i],dx2[i],10000)
    Dens = np.random.normal(x1[i],dx1[i],10000)
    dR.append( np.std(Nums/Dens) )
    
    Rp.append( x2p[i]/ x1p[i])
    Nums = np.random.normal(x2p[i],dx2p[i],10000)
    Dens = np.random.normal(x1p[i],dx1p[i],10000)
    dRp.append( np.std(Nums/Dens) )

    Rpp.append( x2pp[i]/ x1pp[i])

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(M1,x1,'b^-',label=r'LHC $\sqrt{s}$=13 TeV')
ax1.plot(M2,x2,'r^-',label=r'LHC $\sqrt{s}$=14 TeV')

ax1.plot(M1,x1p,'bx-',label=r"LHC $\sqrt{s}$=13 TeV, $m_{\tau\nu}>0.3M_{W'}$")
ax1.plot(M2,x2p,'rx-',label=r"LHC $\sqrt{s}$=14 TeV, $m_{\tau\nu}>0.3M_{W'}$")

ax1.plot(M1,x1pp,'bo-',label=r"LHC $\sqrt{s}$=13 TeV, $m_{\tau\nu}>3.75TeV$")
ax1.plot(M2,x2pp,'ro-',label=r"LHC $\sqrt{s}$=14 TeV, $m_{\tau\nu}>3.75TeV$")

ax1.fill_between(M1, x1-dx1, x1+dx1,
                alpha=0.5, edgecolor='b', facecolor='b')
ax1.fill_between(M2, x2-dx2, x2+dx2,
                alpha=0.5, edgecolor='r', facecolor='r')
                
ax1.fill_between(M1, x1p-dx1p, x1p+dx1p,
                alpha=0.5, edgecolor='b', facecolor='b')
ax1.fill_between(M2, x2p-dx2p, x2p+dx2p,
                alpha=0.5, edgecolor='r', facecolor='r')
                
ax1.set_yscale('log')

#handles, labels = ax1.get_legend_handles_labels()
#lgd = ax1.legend(handles, labels, loc='best', bbox_to_anchor=(1.04,1.0))
#text = ax1.text(-0.2,1.05, "Aribitrary text", transform=ax1.transAxes)

ax1.legend(loc='best')#, bbox_to_anchor=(1.04, 1.0))
#ax1.set_ylabel(r"$\sigma \times 10^{9}$[nb]")
ax1.set_ylabel(r"$\sigma( pp\to W')$ [fb]")
print(R)
print(dR)
ax2.errorbar(M1, R, yerr =dR, marker="^",linewidth=0, c='r')#, 'k-')
ax2.errorbar(M1, Rp, yerr =dRp, linestyle='-',linewidth=0, marker="x", c='r')#, 'k-')
ax2.scatter(M1 , Rpp, linestyle='-',linewidth=0, marker='o', c=('k',))
ax2.plot( M1 , np.ones(len(M1)),'k-',linewidth=1)
ax2.set_ylabel(r"Ratio $R_{14/13}$")
ax2.yaxis.grid(True)
ax2.set_xlabel("$M_{W'}$ [GeV]")
ax2.set_ylim(0,2.5)
ax1.set_ylim(0.001,10000)

lines1 = [matplotlib.lines.Line2D([], [], linewidth=1, linestyle='-', color='b'), matplotlib.lines.Line2D([], [], linewidth=1, linestyle='-', color='r') ]
lines2 = [matplotlib.lines.Line2D([], [], linewidth=0, marker='^', color='k'), 
          matplotlib.lines.Line2D([], [], linewidth=0, marker='x', color='k'),
          matplotlib.lines.Line2D([], [], linewidth=0, marker='o', color='k')]


legend1 = ax1.legend( lines1+lines2, ["$\sqrt{s}$=13 TeV", "$\sqrt{s}$=14 TeV","$m_{inv}>10 GeV$", "$m_{inv}>0.3M_{W'}$", "$m_{inv}>3.75 TeV$"], 
loc='upper right',framealpha=0.3,prop={"size":8})
ax1.set_title("LHC W' production cross section")

#legend2 = ax1.legend( lines2 ,[], loc=4)
#legend1 = plt.legend(
#  [], 
#  ["algo1", "algo2", "algo3"], loc=1)
#pyplot.legend([l[0] for l in plot_lines], parameters, loc=4)
#plt.gca().add_artist(legend1)

dataset = pd.DataFrame(
  {
  "Mass" : M1,
  "Xsec13Total": x1,
  "Xsec13percent":x1p,
  "Xsec13fixed":x1pp,
  "Xsec14Total": x2,
  "Xsec14percent":x2p,
  "Xsec14fixed":x2pp,
  "Ratio_total": x2/x1,
  "Ratio_perc": x2p/x1p,
  "Ratio_fixed": x2pp/x1pp,
  }
  )
print(dataset)
#fig.savefig('samplefigure', bbox_extra_artists=(lgd,text), bbox_inches='tight')

plt.show()
