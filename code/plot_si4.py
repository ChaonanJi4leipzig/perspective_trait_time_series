"""Final SI4 consistency plot: no flux/model reprocessing."""
from common import *
import matplotlib.dates as mdates
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch
O=DATA
OUT=output_directory()
# Journal contract: quantitative grid; the evidence and numerical inputs stay
# fixed. All sizes below are physical points at the final 183 mm PDF width.
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
si4_initial_style()
sites=['Hesse','Castelporziano'];quantities=['Chl','CCC','LAI']
colors = SIGNAL_COLORS
# Whole-unit mathtext keeps each exponent anchored to its base in Arial.
labels=['Tower GPP\n'+r'($\mathrm{g\ C\ m^{-2}\ d^{-1}}$)',
        'Reconstructed Chl\n'+r'($\mathrm{µg\ cm^{-2}}$)',
        'Reconstructed CCC\n'+r'$\mathrm{g\ m^{-2}}$ (ground)',
        'Reconstructed LAI / ground PAI\n'+r'($\mathrm{m^{2}\ m^{-2}}$)']
stats=read_plot_table(O/'spearman_GPP.parquet');stats=stats[stats.region.eq('Fig4_window')]
fig,axes=plt.subplots(4,2,figsize=(183/25.4,190/25.4),sharex=True,sharey='row',gridspec_kw={'hspace':.17,'wspace':.16})
fig.subplots_adjust(left=.135,right=.995,bottom=.066,top=.924)
start,end=pd.Timestamp('2022-06-01'),pd.Timestamp('2026-09-30')
for j,site in enumerate(sites):
    model=read_plot_table(O/f'model_series_{site}.parquet',parse_dates=['date']).set_index('date')
    daily=read_plot_table(O/f'daily_GPP_{site}.parquet',parse_dates=['date']).set_index('date')
    field=read_plot_table(O/f'field_PAI_{site}.parquet',parse_dates=['date'])
    ax=axes[0,j]
    ax.plot(daily.index,daily.GPP,color='.70',lw=.35,zorder=2)
    ax.plot(model.index,model.GPP,ls='none',marker='o',ms=2.2,mew=0,color='#303030',zorder=3)
    ax.legend(handles=[Line2D([],[],color='.70',lw=.8,label='Daily, QC-passed'),
                       Line2D([],[],color='#303030',marker='o',ms=3.5,ls='none',label='±7 d mean (≥8/15 days)')],
              loc='upper left',frameon=False,fontsize=6.5,handlelength=1.1,borderaxespad=.3,
              labelspacing=.25,handletextpad=.4)
    for i,q in enumerate(quantities,1):
        ax=axes[i,j];c=colors[q]
        ax.fill_between(model.index,model[q+'_lo'],model[q+'_hi'],color=c,alpha=.20,lw=0,zorder=2)
        ax.plot(model.index,model[q],color=c,lw=.8,marker='o',ms=1.7,mew=0,zorder=3)
        s=stats[(stats.site==site)&(stats.quantity==q)].set_index('season')
        a,b=s.loc['all_dates'],s.loc['Apr-Oct']
        ax.text(.018,.955,f'ρ = {a.spearman_rho:.2f} (n = {int(a.n)}); Apr–Oct {b.spearman_rho:.2f} ({int(b.n)})',
                transform=ax.transAxes,va='top',fontsize=6.5,color='black',
                bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':1.0},zorder=6)
        ax.margins(y=.23)
    ax=axes[3,j]
    ax.errorbar(field.date,field.PAI_mean,yerr=field.PAI_sd,fmt='s',color='#292929',
                mfc='white',mec='#292929',mew=.6,ms=2.8,elinewidth=.55,capsize=1.5,zorder=5)
    method='SunScan' if site=='Hesse' else 'DHP'
    ax.legend(handles=[Line2D([],[],ls='none',marker='s',mfc='white',mec='#292929',ms=4,
                             label=f'{method} overstory PAI (mean ± SD)')],
              loc='lower right',frameon=False,fontsize=6.5,handlelength=1,borderaxespad=.35)
    title='FR-Hes Hesse' if site=='Hesse' else 'IT-Cp2 Castelporziano'
    axes[0,j].set_title(title,pad=6,fontweight='normal')
    for i,ax in enumerate(axes[:,j]):
        for year in range(2021,2027):
            ax.axvspan(pd.Timestamp(year,11,1),pd.Timestamp(year+1,3,1),color='#BBC9D6',alpha=.15,lw=0,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',length=2.5,width=.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlim(start,end)
        if j==0:ax.set_ylabel(labels[i],labelpad=5)
        ax.text(.018,.05,chr(ord('a')+i*2+j),transform=ax.transAxes,fontsize=8,fontweight='bold',color='black')
    axes[-1,j].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1,j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1,j].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))
    axes[-1,j].set_xlabel('Date',labelpad=4)
all_daily=pd.concat([read_plot_table(O/f'daily_GPP_{site}.parquet').GPP for site in sites])
axes[0,0].set_ylim(min(-.8,float(all_daily.min())-.4),max(17,float(all_daily.max())+1))
fig.legend(handles=[Line2D([],[],color=colors[q],lw=1.5,label=q) for q in quantities]+
           [Patch(facecolor='.55',alpha=.20,edgecolor='none',label='Ensemble range')],
           loc='upper center',bbox_to_anchor=(.55,.993),frameon=False,ncol=4,fontsize=6.5,
           handlelength=1.8,columnspacing=1.9)
# Tight export with zero padding, but enforce the final *exported* width;
# changing the canvas width never rescales the physical font sizes.
unit_labels=fix_units(fig)
target_width=183/25.4
for iteration in range(20):
    fig.canvas.draw()
    tight=fig.get_tightbbox(fig.canvas.get_renderer())
    difference=target_width-tight.width
    if abs(difference)<1e-7:break
    fig.set_size_inches(fig.get_figwidth()+difference,fig.get_figheight(),forward=True)
fig.canvas.draw();renderer=fig.canvas.get_renderer();tight=fig.get_tightbbox(renderer)
assert abs(tight.width-target_width)<1e-5
bbox=Bbox.from_bounds(tight.x0,tight.y0,target_width,tight.height)
texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
assert min(t.get_fontsize() for t in texts)>=6.5
assert all(t.get_fontfamily()==['Arial'] for t in texts)
for t in texts:
    font=font_manager.get_font(font_manager.findfont(t.get_fontproperties(),fallback_to_default=False))
    assert all(ord(c) in font.get_charmap() for c in t.get_text() if not c.isspace())
assert not any(any(word in t.get_text().lower() for word in ['w7','seed','gate']) for t in texts)
# Verify each correlation label fits its panel at its final physical size.
for ax in axes.flat:
    for t in ax.texts:
        if t.get_text().startswith('ρ'):
            assert t.get_window_extent(renderer).x1 <= ax.get_window_extent(renderer).x1
png=OUT/'SI4_flux_consistency.png';pdf=OUT/'SI4_flux_consistency.pdf'
fig.savefig(png,dpi=600,bbox_inches=bbox,pad_inches=0)
fig.savefig(pdf,bbox_inches=bbox,pad_inches=0)
plt.close(fig)
