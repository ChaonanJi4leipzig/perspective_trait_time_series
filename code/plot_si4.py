"""Two plates adapted from v6 draw_panel_b(), retaining its QQ collision rule."""
from common import *
OUT=output_directory()
MAP_TRAITS = ["LMA", "Chl", "EWT", "N"]
UNITS = {"LMA": r"g m$^{-2}$", "Chl": r"µg cm$^{-2}$", "EWT": r"mg cm$^{-2}$", "N": r"mg cm$^{-2}$"}
MAP_TICKS = {
    "LMA": [60, 70, 80],
    "Chl": [30, 34, 38],
    "EWT": [10, 12],
    "N": [0.1, 0.15],
}
def legacy_style() -> None:
    si3_initial_style()
def place_qq_equations(fig: plt.Figure, qq_panels: list) -> None:
    """Same rule for every inset: clear top-left, else clear lower-right."""
    renderer = fig.canvas.get_renderer()
    for ax, equation in qq_panels:
        initial_low, initial_high = ax.get_xlim()
        span = initial_high - initial_low
        points = np.asarray(ax.collections[0].get_offsets())
        radius = np.sqrt(ax.collections[0].get_sizes().max()) * fig.dpi / 144 + 2
        placed = False
        # Extra lower margin is a fallback if both text positions contain points.
        for extra in (0, 0.10, 0.20, 0.30):
            limits = (initial_low - extra * span, initial_high)
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            display_points = ax.transData.transform(points)
            for position, ha, va in [((0.03, 0.80), "left", "top"),
                                     ((0.97, 0.03), "right", "bottom")]:
                equation.set(position=position, ha=ha, va=va)
                equation.update_bbox_position_size(renderer)
                box = equation.get_bbox_patch().get_window_extent(renderer)
                inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                padded = box.padded(radius)
                occupied = any(padded.contains(x, y) for x, y in display_points)
                if inside and not occupied:
                    placed = True
                    break
            if placed:
                break
        assert placed, "No clear equation position inside QQ inset"
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle

def plate(fig,maps,name,date,left):
 qq=[];scale=1.17*1.05;height=.140*1.07*scale;factor=.47/.48;width=.098*scale*factor;cb_width=.012*1.07*scale*factor
 x=lambda oldx:left+(oldx-.56)*factor
 top,bottom_edge=.944,.080;row_step=(top-bottom_edge-height)/3
 for row,trait in enumerate(MAP_TRAITS):
  bottom=top-height-row*row_step;ref,pred,mask=maps[trait];observed,fitted=ref[mask],pred[mask];low,high=map(float,np.quantile(observed,[.05,.95]));ticklo,tickhi=np.quantile(observed,[.1,.9]);ticks=MAP_TICKS[trait]
  if not all(ticklo<=v<=tickhi for v in ticks):
   span=float(tickhi-ticklo);precision=max(0,int(-np.floor(np.log10(span/(len(ticks)+1)))));ticks=np.clip(np.round(np.linspace(ticklo,tickhi,len(ticks)),precision),low,high).tolist()
  labels=[f'{v:g}' for v in ticks];q=np.linspace(.05,.95,19);qr,qp=np.quantile(observed,q),np.quantile(fitted,q);a,b=np.polyfit(qr,qp,1);rlo,rhi=np.quantile(observed,[.02,.98]);qlow=min(float(rlo),float(qr.min()),float(qp.min()));qhigh=max(float(rhi),float(qr.max()),float(qp.max()));margin=.06*(qhigh-qlow);qlow-=margin;qhigh+=margin
  color=TC[trait];cmap=LinearSegmentedColormap.from_list(trait,['white',color]);norm=Normalize(low,high,clip=True)
  axq=fig.add_axes((x(.580),bottom,width,height));axr=fig.add_axes((x(.708),bottom,width,height));axp=fig.add_axes((x(.836),bottom,width,height));axc=fig.add_axes((x(.973)-cb_width/2,bottom,cb_width,height))
  axq.plot([qlow,qhigh],[qlow,qhigh],ls='--',lw=.7,color='#888888');axq.plot([qlow,qhigh],a*np.array([qlow,qhigh])+b,lw=1,color='#333333');axq.scatter(qr,qp,s=12,color=color,zorder=3);axq.set(xlim=(qlow,qhigh),ylim=(qlow,qhigh));axq.grid(color='#D2D2D2',ls='--',lw=.4)
  axq.text(.03,.97,trait,transform=axq.transAxes,va='top',ha='left',fontsize=21,fontweight='bold',color=color,bbox=dict(facecolor='white',edgecolor='none',pad=1))
  eq=axq.text(.03,.80,f'y={a:.2f}x\n{b:+.2f}',transform=axq.transAxes,va='top',fontsize=16,linespacing=.95,bbox=dict(facecolor='white',edgecolor='none',pad=1));qq.append((axq,eq))
  if trait=='EWT':axq.set_ylabel('Reconstruction',fontsize=21,labelpad=7)
  qqticks=np.linspace(qlow,qhigh,5)[1:-1] if min(np.diff(ticks))/(qhigh-qlow)<.16 else ticks; qqlabels=[f'{v:.0f}' if trait!='N' else f'{v:.2f}' for v in qqticks]; axq.set_xticks(qqticks,qqlabels);axq.set_yticks(qqticks,qqlabels);axq.set_xlim(qlow,qhigh);axq.set_ylim(qlow,qhigh);axq.tick_params(length=2,pad=1,labelsize=19)
  if trait=='N':
   axq.tick_params(axis='y',pad=2)
   for label in axq.get_yticklabels():label.set(rotation=0,ha='right',va='center',rotation_mode='default')
  for ax,array in [(axr,ref),(axp,pred)]:
   im=ax.imshow(np.ma.masked_where(~mask,array),cmap=cmap,norm=norm,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   y0,x0=(480,460) if name=='Hesse' else (475,475);ax.add_patch(Rectangle((x0-.5,y0-.5),50,50,fill=False,edgecolor='black',lw=.75,zorder=10))
  box=axr.get_position();axq.set_position((x(.580)+width/2-box.width/2,box.y0,box.width,box.height));axc.set_position((x(.973)-cb_width/2,box.y0,cb_width,box.height));cb=fig.colorbar(im,cax=axc)
  for ax in [axq,axr,axp,axc]:
   for spine in ax.spines.values():spine.set_linewidth(.9);spine.set_edgecolor('#4A4A4A')
  cb.outline.set_linewidth(.9);cb.outline.set_edgecolor('#4A4A4A');assert all(low<=v<=high for v in qqticks);cb.set_ticks(qqticks,labels=qqlabels);cb.ax.set_ylim(low,high);cb.ax.tick_params(labelsize=18,length=2,pad=2)
  if trait=='N':
   cb.ax.tick_params(axis='y',pad=2)
   for label in cb.ax.get_yticklabels():label.set(rotation=0,ha='left',va='center',rotation_mode='default')
  cb.set_label(f'{trait} ({UNITS[trait]})',fontsize=20,labelpad=2)
 fig.text(x(.580),.976,f'{"FR-Hes Hesse" if name=="Hesse" else "IT-Cp2 Castelporziano"} · {date}',fontsize=22,fontweight='bold',va='center')
 for pos,label in [(.580,'Reference'),(.708,'Reference'),(.836,'Reconstructed')]:fig.text(x(pos)+width/2,.0419,label,fontsize=21,ha='center',va='top')
 return qq

def main():
 legacy_style();fig=plt.figure(figsize=(22,12.5));qq=[];records=[]
 for name,path,left in [('Hesse',DATA/'SI3_maps_Hesse.npz',.01),('Castelporziano',DATA/'SI3_maps_Castelporziano.npz',.515)]:
  with np.load(path) as z:
   mask=z['mask'];maps={t:(z['reference'][i],z['prediction'][i],mask) for i,t in enumerate(MAP_TRAITS)};event=str(z['event']);date=event.split(':')[1]
  qq.extend(plate(fig,maps,name,date,left))
 normalize(fig);journal_style();fig.canvas.draw();place_qq_equations(fig,qq);fig.canvas.draw()
 for ax,text in qq:
  assert ax.get_xlim()==ax.get_ylim();points=ax.transData.transform(ax.collections[0].get_offsets());radius=np.sqrt(ax.collections[0].get_sizes().max())*fig.dpi/144+2;assert all(ax.bbox.padded(-radius).contains(x,y) for x,y in points)
 # Layout-only polish: retain the existing numerical CSV unchanged.
 ticks=[t for ax in fig.axes for t in ax.get_yticklabels() if t.get_visible() and t.get_text()];assert all(t.get_rotation()==0 for t in ticks)
 export_journal(fig,'SI4_maps_qq',OUT)
if __name__=='__main__':main()
