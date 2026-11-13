"""Vector reconstruction of the established neuron-diagram design, not a flowchart."""
from common import *
OUT=output_directory()
from matplotlib.patches import Circle,Ellipse,FancyBboxPatch,PathPatch
from matplotlib.path import Path as MPath

def main():
 journal_style();assert ORDER==['LAI','LMA','Chl','Car','EWT','N','C','NSC','Lignin','Fibre','Cellulose']
 fig=plt.figure(figsize=(WIDTH,108/25.4));ax=fig.add_axes([.01,.01,.98,.98]);ax.set(xlim=(0,1),ylim=(1,0));ax.set_axis_off()
 # Preserve the established input/trunk/head/output neurons and auxiliary branch.
 grey='#66737C';blue='#2E79B1';green='#2E8263';purple='#8960B2'
 ax.add_patch(FancyBboxPatch((.205,.225),.385,.555,boxstyle='round,pad=.014,rounding_size=.022',facecolor='#EDF1F3',edgecolor='#CCD4DA',lw=.7,zorder=0))
 ys=[.265,.370,.475,.635,.740];trunks=[.225,.400,.575];input_x=.072;head_x=.735;head_y=[.205,.305,.405,.535,.635];output_x=.885;output_y=np.linspace(.145,.745,11)
 def network(xs,xx,yy,color,alpha=.3):
  for x,y in xs:
   for y2 in yy:ax.plot([x,xx],[y,y2],color=color,alpha=alpha,lw=.4,zorder=1)
 network([(input_x,y) for y in ys],trunks[0],ys,'#A8B6C1')
 for xa,xb in zip(trunks[:-1],trunks[1:]):network([(xa,y) for y in ys],xb,ys,'#9DAEBB')
 network([(trunks[-1],y) for y in ys],head_x,head_y,'#7CAE98',.36)
 network([(head_x,y) for y in head_y],output_x,output_y,'#7CAE98',.34)
 auxiliary=[]
 # Route every dashed auxiliary connection through a corridor left of the head,
 # then below the auxiliary heading; neither label is crossed.
 for route,y in enumerate([ys[0],ys[-1]]):
  for end_y in [.875,.955]:
   path=MPath([(trunks[-1],y),(.612+.015*route,y),(.612+.015*route,.845+.012*route),(.81,.845+.012*route),(output_x,end_y)],[MPath.MOVETO,MPath.LINETO,MPath.LINETO,MPath.LINETO,MPath.LINETO])
   patch=PathPatch(path,fill=False,edgecolor=purple,alpha=.35,lw=.5,linestyle=(0,(3,3)),zorder=1);ax.add_patch(patch);auxiliary.append(patch)
 def nodes(x,yy,edge,fill,r=.012):
  for y in yy:ax.add_patch(Ellipse((x,y),2*r,2*r*183/108,facecolor=fill,edgecolor=edge,lw=.8,zorder=3))
 nodes(input_x,ys,blue,'#DCEAF7');
 for x in trunks:nodes(x,ys,grey,'white')
 nodes(head_x,head_y,green,'#E0F0E8');nodes(output_x,output_y,green,green,.009)
 for x in [input_x,*trunks]:
  for y in [.535,.560,.585]:ax.add_patch(Ellipse((x,y),.004,.004*183/108,facecolor=grey,edgecolor='none',zorder=3))
 for y in [.455,.480]:ax.add_patch(Ellipse((head_x,y),.004,.004*183/108,facecolor=green,edgecolor='none',zorder=3))
 for y in [.875,.955]:ax.add_patch(Ellipse((output_x,y),.024,.024*183/108,facecolor='#F0E8FA',edgecolor=purple,lw=.8,linestyle=(0,(3,2)),zorder=3))
 for y in [.910,.925]:ax.add_patch(Ellipse((output_x,y),.004,.004*183/108,facecolor=purple,edgecolor='none',zorder=3))
 ax.text(.073,.145,'Input: 804\nSentinel-2 features',ha='center',va='center',fontsize=7.5,color='#35434A')
 ax.text(.400,.170,'Hidden layers (3 × 128)',ha='center',va='center',fontsize=8,color='#35434A')
 ax.text(.716,.072,'Shared trait head (64)',ha='center',va='center',fontsize=7.5,color=green)
 ax.text(.926,.072,'11 traits',ha='center',va='center',fontsize=7.5,color=green)
 for y,t in zip(output_y,ORDER):ax.text(.910,y,t,va='center',fontsize=7,color='#35434A')
 aux_label=ax.text(.785,.795,'Auxiliary: 16 hyperspectral PCs\n(training only)',ha='center',va='center',fontsize=7.5,color=purple)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for line in auxiliary:
  for text in ax.texts:
   box=text.get_window_extent(renderer).padded(1)
   assert not line.get_path().transformed(line.get_transform()).intersects_bbox(box,filled=False),text.get_text()
 assert not ax.images,'SI2 must be entirely vector'
 export_journal(fig,'SI2_model',OUT)
if __name__=='__main__':main()
