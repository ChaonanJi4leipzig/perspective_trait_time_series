"""Reuse the final Fig3 data/drawing, with journal-sized typography and caption notes."""
from common import *
OUT=output_directory()
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
def tint(color,white=.72):
 """Opaque light tint: mix the trait colour with white (one factor for all traits)."""
 return tuple((1-white)*np.array(to_rgba(color)[:3])+white)
def main():
 journal_style();totals=pd.read_parquet(DATA/'Fig3b_totals.parquet').set_index('trait');d=pd.read_parquet(DATA/'Fig3a_metrics.parquet');b=pd.read_parquet(DATA/'Fig3b_budget.parquet');p=pd.read_parquet(DATA/'Fig3a_pixel_baselines.parquet').set_index(['trait','model'])
 fig=plt.figure(figsize=(WIDTH,122/25.4));ar=fig.add_axes((.078,.095,.205,.735));ae=fig.add_axes((.310,.095,.205,.735),sharey=ar)
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
 # All four x-scales are percentages of the same reference variance.
 # Extra headroom allows the right-hand value labels without narrowing panel a.
 levels=['site','date','scene','total'];titles=['Between\nsites','Between\ndates','Within\nscenes','Total\nrecovered']
 ranges=[float(np.ceil(max(b[b.level.eq(l)].share.max(),b[b.level.eq(l)].recovered_high.max())*100/5)*5+30) for l in levels[:3]]+[100.]
 ranges[2]+=10
 left=.550;right=.985;gap=.010;unit=(right-left-3*gap)/sum(ranges);budget_axes=[];labels=[]
 for k,(level,title,span) in enumerate(zip(levels,titles,ranges)):
  ax=fig.add_axes((left,.095,span*unit,.735),sharey=ar);budget_axes.append(ax)
  ax.set(xlim=(0,span),ylim=(10.5,-.5));ax.tick_params(axis='y',left=False,labelleft=False);ax.tick_params(axis='x',length=2,pad=2)
  ax.spines[['left','top','right']].set_visible(False);ax.set_title(title,fontsize=7,pad=5)
  ax.set_xticks(([0,20,40] if k<2 else [0,50,100]));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v:g}'))
  for j,t in enumerate(ORDER):
   if level=='total':
    share=100.;rec=100*totals.loc[t,'total_recovered'];lo=100*totals.loc[t,'total_low'];hi=100*totals.loc[t,'total_high']
   else:
    row=b[b.trait.eq(t)&b.level.eq(level)].iloc[0];share=100*row.share;rec=100*row.recovered;lo=100*row.recovered_low;hi=100*row.recovered_high
   ax.barh(j,share,height=.62,color=tint(TC[t]),zorder=1);ax.barh(j,rec,height=.62,color=TC[t],zorder=2)
   ax.plot([lo,hi],[j,j],color='black',lw=.5,zorder=3)
   if level=='total':ax.plot(100*totals.loc[t,'pixel_seasonal_total'],j,marker='|',ms=5,mew=.8,color='black',ls='',zorder=4)
   text=ax.annotate(f'{rec:.0f}/{share:.0f}%',(share,j+.24),xytext=(1.1,0),textcoords='offset points',ha='left',va='center',fontsize=7,color='#666666',annotation_clip=False)
   # Total background reaches 100%; keep its value label just past the recovered bar.
   if level=='total':text.xy=(rec,j+.24)
   labels.append((k,text))
  left+=span*unit+gap
 # Panel-b x title is a real x-label, so matplotlib places it at the same height as panel-a labels
 # (same tick-label font and axes bottom); its x is moved to the centre of the four budget panels.
 first=budget_axes[0].get_position();centre=(first.x0+budget_axes[-1].get_position().x1)/2
 budget_axes[0].set_xlabel('Share of total variance (%)');budget_axes[0].xaxis.label.set_x((centre-first.x0)/first.width)
 bh=[Patch(facecolor=TC['LAI'],label='Recovered'),Patch(facecolor=tint(TC['LAI']),label='Not recovered'),Line2D([],[],color='black',lw=.5,label='95% CI'),Line2D([],[],marker='|',ms=5,mew=.8,color='black',lw=0,label='Pixel seasonal mean')]
 fig.legend(handles=bh,loc='upper left',bbox_to_anchor=(.535,.944),ncol=4,frameon=False,fontsize=7,columnspacing=.5,handletextpad=.3,handlelength=.8)
 # sharey guarantees exact vertical alignment with both evaluation axes.
 assert all(ax.get_ylim()==ar.get_ylim() for ax in budget_axes)
 def check_alignment(figure):
  renderer=figure.canvas.get_renderer();ys=ar.transData.transform(np.column_stack([np.zeros(11),np.arange(11)]))[:,1]
  for ax in budget_axes:assert np.allclose(ax.transData.transform(np.column_stack([np.zeros(11),np.arange(11)]))[:,1],ys,atol=1e-9)
  for k,text in labels:
   if k<3:assert text.get_window_extent(renderer).x1<budget_axes[k+1].bbox.x0,(k,text.get_text())
  top=lambda ax:ax.xaxis.label.get_window_extent(renderer).y1
  assert abs(top(budget_axes[0])-top(ae))<=.1/25.4*figure.dpi and abs(top(ar)-top(ae))<=.1/25.4*figure.dpi,(top(ar),top(ae),top(budget_axes[0]))
 fig._paper_layout_check=check_alignment
 fig._fixed_width_axes=[ar,ae]
 for x,letter,title in [(.038,'a','Held-out evaluation'),(.535,'b','Variance recovered (temporal)')]:
  fig.text(x,.976,letter,fontsize=9,fontweight='bold',va='top');fig.text(x+.033,.976,title,fontsize=8,va='top')
 export_journal(fig,'Fig3',OUT)
if __name__=='__main__':main()
