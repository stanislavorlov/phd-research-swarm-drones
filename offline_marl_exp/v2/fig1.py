import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
BLUE,ORANGE,AQUA,INK,INK2,MUTED,GRID='#2a78d6','#eb6834','#1baf7a','#0b0b0b','#52514e','#9a9892','#dcdbd6'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5})
fig,ax=plt.subplots(figsize=(6.5,2.5)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off')
def node(x,y,t,c=INK2,fc='white'):
    ax.scatter([x],[y],s=620,color=fc,edgecolor=c,linewidth=1.3,zorder=4); ax.text(x,y,t,ha='center',va='center',fontsize=8.5,zorder=5)
def arr(a,b,c,ls='-',rad=0.0,lw=1.4):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=c,lw=lw,linestyle=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=14,shrinkB=14,zorder=3))
xs=[1.0,4.0,7.0]
node(xs[0],2.6,'$s_t$'); node(xs[1],2.6,'$s_{t+1}$'); node(xs[2],2.6,'$s_{t+2}$')
node(xs[1],0.9,"$s'_B$",c=ORANGE); node(xs[2],0.9,"$s''_B$",c=ORANGE)
arr((xs[0],2.6),(xs[1],2.6),BLUE); arr((xs[1],2.6),(xs[2],2.6),BLUE); arr((xs[2],2.6),(9.3,2.6),BLUE)
arr((xs[0],2.6),(xs[1],0.9),ORANGE,ls=(0,(4,3))); arr((xs[1],2.6),(xs[2],0.9),ORANGE,ls=(0,(4,3)))
arr((xs[1],0.9),(xs[0],2.6),MUTED,ls=(0,(1,2)),rad=-0.35,lw=1.0); arr((xs[2],0.9),(xs[1],2.6),MUTED,ls=(0,(1,2)),rad=-0.35,lw=1.0)
ax.text(2.5,2.86,'branch A: committed',color=BLUE,ha='center',fontsize=8)
ax.text(5.6,2.05,'branch B: evaluated, then rolled back',color=ORANGE,ha='left',fontsize=8)
ax.text(1.55,0.72,'restore\n(uav, ch, t, prev)',color=MUTED,ha='center',fontsize=7.5,style='italic')
ax.text(5.5,3.55,r"$Q_A = r(s_t,a_A) + J(s_{t+1})$",color=BLUE,ha='center',fontsize=9)
ax.text(5.5,0.12,r"$Q_B = r(s_t,a_B) + J(s'_B)$",color=ORANGE,ha='center',fontsize=9)
ax.text(9.35,2.86,'...',color=BLUE,fontsize=11)
ax.text(8.75,1.2,'J: planner return\nfrom the next state',color=INK2,ha='center',fontsize=7.8,style='italic')
fig.tight_layout(pad=0.2); fig.savefig('figs/fig_pairs.png',dpi=400,facecolor='white'); print('ok')
