"""One-time fold0 audit of the frozen all-development production NN."""
from common import *
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
OUT=output_directory()
def main():
 journal_style();d=pd.read_parquet(DATA/'SI3_fold0_metrics.parquet')
 def score(t,m,k):
  stat='seed_mean' if m=='NN' else 'baseline';pop='primary' if m=='NN' else m+'_defined'
  v=d[d.trait.eq(t)&d.model.eq(m)&d.metric.eq(k)&d.statistic.eq(stat)&d.population.eq(pop)]
  assert len(v)==1,(t,m,k)
  return float(v.estimate.iloc[0])
 methods=['NN','pixel_mean','pixel_seasonal_mean'];rv=[score(t,m,'r2') for t in ORDER for m in methods];ev=[score(t,m,'nrmse_P95_P05') for t in ORDER for m in methods]
 limits=(min(-.08,np.nanmin(rv)-.035),max(.75,np.nanmax(rv)+.035));err=(min(.16,np.nanmin(ev)-.015),max(.45,np.nanmax(ev)+.015))
 fig=plt.figure(figsize=(WIDTH,96/25.4));ar=fig.add_axes((.090,.115,.410,.750));ae=fig.add_axes((.555,.115,.410,.750),sharey=ar)
 for start,end,c in [(0,2,'#009E73'),(2,4,'#D55E00'),(4,5,'#0072B2'),(5,11,'#9B5A9A')]:
  for ax,xlim,reverse in [(ar,limits,False),(ae,err,True)]:
   rgba=np.array(to_rgba(c));alpha=np.linspace(.015,.270,256);alpha=alpha[::-1] if reverse else alpha;band=np.broadcast_to(rgba,(2,256,4)).copy();band[:,:,3]=alpha
   ax.imshow(band,extent=(*xlim,end-.5,start-.5),aspect='auto',interpolation='bicubic',zorder=0)
   if end<11:ax.axhline(end-.5,color='white',lw=.4)
 for i,t in enumerate(ORDER):
  for ax,k in [(ar,'r2'),(ae,'nrmse_P95_P05')]:
   ax.scatter(score(t,'pixel_mean',k),i,s=12,facecolor='white',edgecolor='#363B3E',linewidth=.5,zorder=4)
   ax.scatter(score(t,'pixel_seasonal_mean',k),i,s=26,marker=(8,2,0),color='#828B90',linewidth=.5,zorder=5)
   ax.scatter(score(t,'NN',k),i,s=28,marker='^',color=TC[t],zorder=6)
 for ax,xlim,label in [(ar,limits,'R²'),(ae,err,'P95–P5 nRMSE')]:
  ax.set(ylim=(10.5,-.5),xlim=xlim,xlabel=label);ax.grid(axis='x',color='#C8CDD0',lw=.3);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',length=2,pad=2)
 ar.set_xticks([v for v in np.arange(-.75,.751,.25) if v>=limits[0]]);ar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v:.2f}'.replace('-','\u2212')));ar.axvline(0,color='#6F7477',ls='--',lw=.6);ar.set_yticks(range(11),ORDER);ae.set_xticks(np.arange(.20,err[1]+.001,.05));ae.tick_params(labelleft=False)
 for tick in ar.get_yticklabels():tick.set(color='black',fontweight='normal',fontsize=7)
 handles=[Line2D([],[],marker='^',ms=5,color='#009E73',lw=0,label='Fold-0 NN'),Line2D([],[],marker='o',ms=3.5,mfc='white',mec='#363B3E',lw=0,label='Pixel mean'),Line2D([],[],marker=(8,2,0),ms=5,color='#828B90',lw=0,label='Pixel seasonal mean')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.965),ncol=3,frameon=False,fontsize=7,columnspacing=1.6,handletextpad=.5)
 export_journal(fig,'SI3_fold0_test',OUT)
if __name__=='__main__':main()
