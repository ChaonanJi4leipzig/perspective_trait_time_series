"""Reuse the final Fig3 data/drawing, with journal-sized typography and caption notes."""
from common import *
OUT=output_directory()
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
def main():
 journal_style();d=pd.read_parquet(DATA/'Fig3a_metrics.parquet');b=pd.read_parquet(DATA/'Fig3b_budget.parquet');p=pd.read_parquet(DATA/'Fig3a_pixel_baselines.parquet').set_index(['trait','model'])
 fig=plt.figure(figsize=(WIDTH,122/25.4));ar=fig.add_axes((.078,.095,.205,.735));ae=fig.add_axes((.310,.095,.205,.735),sharey=ar);ab=fig.add_axes((.625,.095,.355,.735))
 score=lambda t,s,m,k:float(d.loc[d.trait.eq(t)&d.scope.eq(s)&d.model.eq(m)&d.metric.eq(k),'mean'].iloc[0])
 vals=[score(t,s,'NN','r2') for t in ORDER for s in ['temporal','spatial']]+[score(t,'temporal',m,'r2') for t in ORDER for m in ['pixel_mean','pixel_seasonal_mean']]
 limits=(min(-.08,min(vals)-.035),max(.75,max(vals)+.035));err=(.16,.45)
 for start,end,c in [(0,2,'#009E73'),(2,4,'#D55E00'),(4,5,'#0072B2'),(5,11,'#9B5A9A')]:
  for ax,xlim,reverse in [(ar,limits,False),(ae,err,True)]:
   rgba=np.array(to_rgba(c));alpha=np.linspace(.015,.270,256);alpha=alpha[::-1] if reverse else alpha;band=np.broadcast_to(rgba,(2,256,4)).copy();band[:,:,3]=alpha
   ax.imshow(band,extent=(*xlim,end-.5,start-.5),aspect='auto',interpolation='bicubic',zorder=0)
   if end<11:ax.axhline(end-.5,color='white',lw=.4)
 for i,t in enumerate(ORDER):
  for s,dy,mark in [('temporal',-.18,'o'),('spatial',.18,'s')]:
   ar.scatter(score(t,s,'NN','r2'),i+dy,s=10,marker=mark,color=TC[t],zorder=5);ae.scatter(score(t,s,'NN','nrmse_P95_P05'),i+dy,s=10,marker=mark,color=TC[t],zorder=5)
  for ax,k in [(ar,'r2'),(ae,'nrmse_P95_P05')]:
   value=lambda m:score(t,'temporal',m,k) if k=='r2' else float(p.loc[(t,m),'nrmse_P95_P05'])
   ax.scatter(value('pixel_mean'),i-.18,s=8,facecolor='white',edgecolor='#363B3E',linewidth=.5,zorder=4)
   ax.scatter(value('pixel_seasonal_mean'),i-.18,s=20,marker=(8,2,0),color='#828B90',linewidth=.5,zorder=4)
 for ax,xlim,label in [(ar,limits,'R²'),(ae,err,'P95–P5 nRMSE')]:
  ax.set(ylim=(10.5,-.5),xlim=xlim,xlabel=label);ax.grid(axis='x',color='#C8CDD0',lw=.3);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',length=2,pad=2)
 ar.set_xticks([0,.25,.50,.75]);ar.axvline(0,color='#6F7477',ls='--',lw=.6);ar.set_yticks(range(11),ORDER);ae.set_xticks(np.arange(.20,.451,.05));ae.tick_params(labelleft=False)
 for tick in ar.get_yticklabels():tick.set(color='black',fontweight='normal',fontsize=7)
 handles=[Line2D([],[],marker='o',ms=3,color='#009E73',lw=0,label='Temporal'),Line2D([],[],marker='s',ms=3,color='#009E73',lw=0,label='Spatial'),Line2D([],[],marker='o',ms=3,mfc='white',mec='#363B3E',lw=0,label='Pixel mean'),Line2D([],[],marker=(8,2,0),ms=4,color='#828B90',lw=0,label='Pixel seasonal mean')]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.944),ncol=2,frameon=False,fontsize=7,columnspacing=1,handletextpad=.4)
 pales=['#D5DEE8','#D8D8D8','#E5D9CB'];dark=['#496882','#676767','#997049']
 for j,t in enumerate(ORDER):
  left=0
  for i,l in enumerate(['site','date','scene']):
   r=b[b.trait.eq(t)&b.level.eq(l)].iloc[0];ab.barh(j,r.share,left=left,height=.62,color=pales[i],edgecolor='white',linewidth=.3);ab.barh(j,r.recovered,left=left,height=.62,color=dark[i]);ab.plot([left+r.recovered_low,left+r.recovered_high],[j,j],color='black',lw=.5);ab.plot([left+r.recovered]*2,[j-.08,j+.08],color='black',lw=.5)
   if l=='date':ab.plot(left+r.pixel_seasonal_recovered_date,j-.44,marker='^',ms=3.5,mfc='white',mec='black',mew=.5,ls='',zorder=6)
   left+=r.share
 ab.set(ylim=(10.5,-.5),xlim=(0,1),xlabel='Reference variance (%)');ab.set_yticks(range(11),ORDER,fontsize=7,fontweight='normal',color='black');ab.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100']);ab.tick_params(axis='y',length=0,pad=2);ab.tick_params(axis='x',length=2,pad=2);ab.spines[['left','top','right']].set_visible(False)
 pairs=[(Patch(facecolor=c),Patch(facecolor=pale)) for c,pale in zip(dark,pales)]
 fig.legend(handles=pairs+[Line2D([],[],marker='^',ms=3.5,mfc='white',mec='black',lw=0)],labels=['Between sites','Between dates','Within scenes','Pixel seasonal mean'],handler_map={tuple:HandlerTuple(ndivide=None,pad=.15)},loc='upper left',bbox_to_anchor=(.605,.944),ncol=2,frameon=False,fontsize=7,columnspacing=.6,handletextpad=.4,handlelength=1.8)
 for x,letter,title in [(.038,'a','Held-out evaluation'),(.555,'b','Variance recovered (temporal)')]:
  fig.text(x,.976,letter,fontsize=9,fontweight='bold',va='top');fig.text(x+.033,.976,title,fontsize=8,va='top')
 export_journal(fig,'Fig3',OUT)
if __name__=='__main__':main()
