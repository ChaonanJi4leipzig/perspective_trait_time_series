"""Shared relative paths, colours, Arial style and final export settings."""
import os
from pathlib import Path
import argparse,json,hashlib
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLOSC_NTHREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import paths
ROOT=paths.REPO
DATA=ROOT/'data';FIGURES=ROOT/'figures'
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplconfig')
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
WIDTH=183/25.4

def output_directory():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=FIGURES);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);return a.output_dir.resolve()

ORDER=['LAI','LMA','Chl','Car','EWT','N','C','NSC','Lignin','Fibre','Cellulose']
TC={t:'#9B5A9A' for t in ORDER};TC.update(LAI='#009E73',LMA='#009E73',Chl='#D55E00',Car='#D55E00',EWT='#0072B2')
def style(size=14):
 import matplotlib as mpl
 mpl.use('Agg');mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':size,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8,'figure.facecolor':'white','axes.facecolor':'white','axes.unicode_minus':False})
def journal_style():
 style(7)
 mpl.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial'],'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.sf':'Arial','mathtext.tt':'Arial','mathtext.cal':'Arial','mathtext.default':'regular','mathtext.fallback':None,'axes.unicode_minus':False,'savefig.pad_inches':0})
 assert Path(font_manager.findfont('Arial',fallback_to_default=False)).is_file()

def normalize(fig):
 """Resize the reused geometry and its point units together; floor all text at7pt."""
 ratio=WIDTH/fig.get_figwidth();height=fig.get_figheight()*ratio
 for text in fig.findobj(Text):
  text.set_fontfamily('Arial');text.set_math_fontfamily('custom');text.set_fontsize(max(7,text.get_fontsize()*ratio))
 for line in fig.findobj(Line2D):
  line.set_linewidth(max(.25,line.get_linewidth()*ratio));line.set_markersize(line.get_markersize()*ratio);line.set_markeredgewidth(max(.2,line.get_markeredgewidth()*ratio))
 for c in fig.findobj(PathCollection):c.set_sizes(c.get_sizes()*ratio**2)
 for ax in fig.axes:
  for which in ['major','minor']:
   for axis in [ax.xaxis,ax.yaxis]:
    axis.labelpad*=ratio
    for tick in axis.get_major_ticks() if which=='major' else axis.get_minor_ticks():tick.set_pad(tick.get_pad()*ratio)
 fig.set_size_inches(WIDTH,height);journal_style();fig.canvas.draw()
UNIT_LABELS={
 'g m$^{-2}$':r'$\mathrm{g\ m^{-2}}$',
 '$\\mu$g cm$^{-2}$':r'$\mathrm{µg\ cm^{-2}}$',
 'mg cm$^{-2}$':r'$\mathrm{mg\ cm^{-2}}$',
 'LMA (g m$^{-2}$)':r'LMA ($\mathrm{g\ m^{-2}}$)',
 'Chl ($\\mu$g cm$^{-2}$)':r'Chl ($\mathrm{µg\ cm^{-2}}$)',
 'Chl (µg cm$^{-2}$)':r'Chl ($\mathrm{µg\ cm^{-2}}$)',
 'EWT (mg cm$^{-2}$)':r'EWT ($\mathrm{mg\ cm^{-2}}$)',
 'N (mg cm$^{-2}$)':r'N ($\mathrm{mg\ cm^{-2}}$)',
}
def fix_units(fig):
 labels=[]
 for text in fig.findobj(Text):
  if text.get_text() in UNIT_LABELS:text.set_text(UNIT_LABELS[text.get_text()])
  if '\\mathrm{' in text.get_text() and '^{-2}' in text.get_text():
   text.set_fontfamily('Arial');text.set_math_fontfamily('custom');labels.append(text)
 return labels

def export_journal(fig,name,outdir):
 journal_style();fix_units(fig)
 for text in fig.findobj(Text):text.set_fontfamily('Arial');text.set_math_fontfamily('custom')
 save_tight(fig,name,outdir)

def read_plot_table(path,parse_dates=()):
 frame=pd.read_parquet(path)
 for col in parse_dates:frame[col]=pd.to_datetime(frame[col])
 return frame

# Reused geometry starts at its original point scale before journal normalization.
def fig4_initial_style():
 mpl.rcParams.update({
     "font.family": "sans-serif",
     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
     "font.size": 17,
     "pdf.fonttype": 42,
     "ps.fonttype": 42,
     "svg.fonttype": "none",
 })

def si1_initial_style():
 mpl.rcParams.update({'font.family':'Arial', 'font.size':7, 'pdf.fonttype':42,
         'ps.fonttype':42, 'axes.linewidth':.8, 'figure.facecolor':'white',
         'axes.facecolor':'white', 'axes.unicode_minus':False})

def si3_initial_style():
 plt.rcParams.update({
         "font.family": "sans-serif",
         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
         "font.size": 21,
         "axes.labelsize": 21,
         "axes.titlesize": 22,
         "xtick.labelsize": 20,
         "ytick.labelsize": 20,
         "axes.linewidth": 0.9,
         "figure.facecolor": "white",
         "axes.facecolor": "white",
     })

def si4_initial_style():
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
                      'axes.titlesize':8,'axes.linewidth':.5,'xtick.labelsize':7,
                      'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
                      'savefig.pad_inches':0,'axes.unicode_minus':True,'text.color':'black',
                      'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular','mathtext.sf':'Arial','mathtext.tt':'Arial','mathtext.cal':'Arial','mathtext.fallback':None,
                      'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})

YEAR_COLORS = {
    2022: "#0072B2",
    2023: "#D55E00",
    2024: "#009E73",
    2025: "#9B5A9A",
    2026: "#E69F00",
}

IGBP_COLORS = {'forest':'#009E73', 'grassland':'#E69F00', 'cropland':'#0072B2',
          'wetland':'#56B4E9', 'shrubland_savanna':'#D55E00', 'other':'#CC79A7'}

SIGNAL_COLORS = {'Chl':'#D55E00','CCC':'#009E73','LAI':'#0072B2'}

def content_bbox(fig):
 """Bounds of visible content; empty off-axis canvases do not add whitespace."""
 from matplotlib.transforms import Bbox
 from matplotlib.spines import Spine
 from matplotlib.axis import Axis
 renderer=fig.canvas.get_renderer();boxes=[]
 for ax in fig.axes:
  if not ax.get_visible():continue
  if ax.axison:
   box=ax.get_tightbbox(renderer)
   if box is not None:boxes.append(box)
  else:
   for artist in ax.get_children():
    if artist is ax.patch or isinstance(artist,(Spine,Axis)) or not artist.get_visible():continue
    if isinstance(artist,Text) and not artist.get_text():continue
    try:box=artist.get_tightbbox(renderer)
    except (AttributeError,TypeError):continue
    if box is not None and np.isfinite(box.extents).all() and box.width+box.height>0:boxes.append(box)
 for artist in list(fig.texts)+list(fig.legends):
  if artist.get_visible():boxes.append(artist.get_window_extent(renderer))
 return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())

def save_tight(fig,name,outdir,pad_mm=1):
 """Crop visible content with 1-mm padding, retaining 183-mm exported width."""
 from matplotlib.transforms import Bbox
 pad=pad_mm/25.4;target=WIDTH-2*pad
 fixed=[(ax,ax.get_position().x0*fig.get_figwidth(),ax.get_position().width*fig.get_figwidth()) for ax in getattr(fig,'_fixed_width_axes',[])]
 for iteration in range(40):
  fig.canvas.draw();box=content_bbox(fig);difference=target-box.width
  if abs(difference)<2e-7:break
  fig.set_size_inches(fig.get_figwidth()+difference,fig.get_figheight(),forward=True)
  for ax,x,width in fixed:
   pos=ax.get_position();ax.set_position((x/fig.get_figwidth(),pos.y0,width/fig.get_figwidth(),pos.height))
 fig.canvas.draw();box=content_bbox(fig)
 if hasattr(fig,'_paper_layout_check'):fig._paper_layout_check(fig)
 assert abs(box.width-target)<1e-5,(name,box.width,target)
 bounds=Bbox.from_bounds(box.x0-pad,box.y0-pad,WIDTH,box.height+2*pad)
 texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
 assert min(t.get_fontsize() for t in texts)>=6.5,(name,min(t.get_fontsize() for t in texts))
 for suffix in ['pdf','png']:fig.savefig(outdir/(name+'.'+suffix),dpi=600,bbox_inches=bounds,pad_inches=0,facecolor='white')
 plt.close(fig)
