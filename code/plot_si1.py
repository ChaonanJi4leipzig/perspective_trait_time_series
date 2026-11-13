"""Final SI1 timeline: only relative packaged figure data."""
from common import *
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.font_manager as fm
OUT=output_directory()
COLORS = IGBP_COLORS
LABELS = {'forest':'Forest', 'grassland':'Grassland', 'cropland':'Cropland',
          'wetland':'Wetland', 'shrubland_savanna':'Shrubland/savanna', 'other':'Other'}
LOCAL_COLOR, LOCAL_WIDTH, LOCAL_HEIGHT = '#C4C4C4', .35, .25
PC_COLOR, PC_WIDTH, PC_HEIGHT = '#DEDEDE', .55, .14

def main():
    font_path = fm.findfont(fm.FontProperties(family='Arial'), fallback_to_default=False)
    si1_initial_style()
    s = pd.read_parquet(DATA / 'SI1_S2_observations.parquet')
    e = pd.read_parquet(DATA / 'SI1_development_events.parquet')
    pc = pd.read_parquet(DATA / 'SI1_pc_fill_dates.parquet')
    cl = pd.read_parquet(DATA / 'SI1_site_classes_filled.parquet').set_index('site_id')
    s.date_utc = pd.to_datetime(s.date_utc, utc=True, format='mixed')
    pc.date_utc = pd.to_datetime(pc.date_utc, utc=True)
    e.date = pd.to_datetime(e.date, utc=True)
    n_events = e.groupby('site_id').size()
    sites = sorted(set(e.site_id) | set(s.site_id), key=lambda sid:
        (list(COLORS).index(cl.loc[sid, 'display_class']), -int(n_events.get(sid, 0)), sid))
    assert len(sites)==41 and set(sites)==set(cl.index)
    fig, ax = plt.subplots(figsize=(183/25.4, 218/25.4))
    fig.subplots_adjust(left=43/183, right=165/183, bottom=12/218, top=198/218)
    row_counts = []
    for j, sid in enumerate(sites):
        sg = s[s.site_id.eq(sid) & s.valid_observation]
        pg = pc[pc.site_id.eq(sid)]
        eg = e[e.site_id.eq(sid)]
        # Same timestamps, row-relative lengths, colours and point linewidths as before.
        ax.vlines(pg.date_utc, j-PC_HEIGHT, j+PC_HEIGHT, color=PC_COLOR, lw=PC_WIDTH)
        ax.vlines(sg.date_utc, j-LOCAL_HEIGHT, j+LOCAL_HEIGHT, color=LOCAL_COLOR, lw=LOCAL_WIDTH)
        ax.scatter(eg.date, np.full(len(eg), j), s=9,
            color=COLORS[cl.loc[sid, 'display_class']], linewidth=0, zorder=3)
        ax.text(1.01, j, f'{len(eg):2d} / {len(sg)+len(pg):3d}',
            transform=ax.get_yaxis_transform(), fontsize=7, va='center')
        row_counts.append(dict(site_id=int(sid), plot_label=cl.loc[sid,'plot_label'],
            display_class=cl.loc[sid,'display_class'], n_enmap=len(eg),
            n_local_valid=len(sg), n_pc_fill=len(pg), n_s2_combined=len(sg)+len(pg)))
    ax.set_yticks(range(len(sites)), [cl.loc[sid,'plot_label'] for sid in sites])
    ax.set_ylim(len(sites)-.4, -.7)
    ax.set_xlim(pd.Timestamp('2022-01-01',tz='UTC'), pd.Timestamp('2026-10-01',tz='UTC'))
    ax.set_xticks([pd.Timestamp(f'{y}-01-01',tz='UTC') for y in range(2022,2027)],
                  [str(y) for y in range(2022,2027)])
    ax.set_xlabel('Date', fontsize=7)
    ax.tick_params(axis='y',length=0, pad=4)
    ax.tick_params(axis='x', labelsize=7, length=3)
    ax.spines[['left','top','right']].set_visible(False)
    ax.text(1.01,1.012,'EnMAP / S2',transform=ax.transAxes,fontsize=7)
    class_legend = fig.legend(handles=[Line2D([],[],marker='o',ls='',ms=3.4,
        color=color,label=LABELS[key]) for key,color in COLORS.items()],
        loc='upper center',bbox_to_anchor=(.5,.995), ncol=6, frameon=False,
        columnspacing=1, handlelength=1.1, handletextpad=.4, fontsize=7)
    tick_legend = fig.legend(handles=[
        Line2D([],[],marker='|',ls='',ms=14,markeredgewidth=LOCAL_WIDTH,
               color=LOCAL_COLOR,label='S2, local valid'),
        Line2D([],[],marker='|',ls='',ms=7.84,markeredgewidth=PC_WIDTH,
               color=PC_COLOR,label='S2, PC cloud ≤ 50%')],
        loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False,
        columnspacing=1.6,handlelength=1.1,handletextpad=.4,fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Validate actual text extents at the final physical size before exporting.
    all_text = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for text in all_text:
        bb=text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert bb.x0>=0 and bb.x1<=1 and bb.y0>=0 and bb.y1<=1, (text.get_text(),bb.bounds)
        assert text.get_fontfamily()==['Arial'] and 6.5<=text.get_fontsize()<=8
    label_boxes=[t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    for a,b in zip(label_boxes,label_boxes[1:]): assert not a.overlaps(b)
    legend_boxes=[lg.get_window_extent(renderer).transformed(fig.transFigure.inverted()) for lg in [class_legend,tick_legend]]
    assert not legend_boxes[0].overlaps(legend_boxes[1])
    assert legend_boxes[1].y0>ax.get_position().y1
    for ext in ['png','pdf']:
        # Fixed page width preserves 183 mm and 7-pt type; there is no added save padding.
        fig.savefig(OUT/f'SI1_data_volume.{ext}',dpi=600,pad_inches=0,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
