"""Production cuboids and continuous-time forest trajectories for two sites."""
from common import *
OUT=output_directory()
import matplotlib.dates as mdates
from matplotlib.transforms import Bbox
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D


fig4_initial_style()

# All cuboids use this exact orthographic basis.  The proportions are a
# horizontal mirror of the supplied reference: near-square front and long depth.
FRONT_WIDTH = 1.20
FRONT_HEIGHT = 1.06
TIME_VECTOR = np.array([-1.10, 1.14])


def front_face_bounds(origin: tuple[float, float]) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` for the unprojected front face."""
    x, y = origin
    return x, y, x + FRONT_WIDTH, y + FRONT_HEIGHT


def _blend_with_white(color: str, fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1.0 - fraction) * rgb + fraction)


def trait_colormap(color: str, lightening: float = 0.0) -> LinearSegmentedColormap:
    """Return a named Matplotlib cmap or a white-to-hex sequential fallback."""
    if color in mpl.colormaps:
        rgba = mpl.colormaps[color](np.linspace(0, 1, 256))
        rgba[:, :3] = (1.0 - lightening) * rgba[:, :3] + lightening
        return LinearSegmentedColormap.from_list(f"cube_{color}_{lightening:.2f}", rgba, N=256)
    return LinearSegmentedColormap.from_list(
        f"cube_{color.lstrip('#')}_{lightening:.2f}",
        ["#F9FAFA", _blend_with_white(color, lightening)], N=256,
    )


def trait_normalization(*arrays: np.ndarray) -> Normalize:
    """Return a shared P5--P95 per-trait normalization with integer endpoints."""
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Each cuboid needs at least one finite value.")
    lower = float(np.floor(np.nanpercentile(values, 5)))
    upper = float(np.ceil(np.nanpercentile(values, 95)))
    if np.isclose(lower, upper):
        padding = max(0.02 * abs(lower), 1.0)
        lower, upper = lower - padding, upper + padding
    return Normalize(lower, upper, clip=True)


def _heatmap_in_polygon(
    ax: plt.Axes,
    values: np.ndarray,
    origin: np.ndarray,
    column_vector: np.ndarray,
    row_vector: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
    zorder: float,
) -> None:
    """Render ``values`` as a heatmap clipped to an affine polygon face."""
    ox, oy = origin
    ux, uy = column_vector
    vx, vy = row_vector
    face = Polygon(
        [(ox, oy), (ox + ux, oy + uy), (ox + ux + vx, oy + uy + vy), (ox + vx, oy + vy)],
        closed=True, facecolor="none", edgecolor="none", transform=ax.transData,
    )
    ax.add_patch(face)
    transform = Affine2D.from_values(ux, uy, vx, vy, ox, oy) + ax.transData
    image = ax.imshow(
        np.asarray(values, dtype=float), extent=(0, 1, 0, 1), origin="lower",
        cmap=cmap, norm=norm, interpolation="nearest", transform=transform,
        zorder=zorder, rasterized=True,
    )
    image.set_clip_path(face)


def _line(ax: plt.Axes, start: np.ndarray, end: np.ndarray, zorder: float, *, subtle: bool = False) -> None:
    ax.plot(
        [start[0], end[0]], [start[1], end[1]],
        color="#75818A" if subtle else "#4A5055",
        linewidth=0.25 if subtle else 0.85,
        alpha=0.13 if subtle else 1.0,
        solid_capstyle="butt" if subtle else "round",
        zorder=zorder,
    )


def _coordinate_triad(ax: plt.Axes, origin: np.ndarray, zorder: float, fontsize: float = 17) -> None:
    """Draw the one lon--lat--time guide from the front lower-left corner."""
    time_direction = TIME_VECTOR / np.linalg.norm(TIME_VECTOR)
    vectors = {
        "lon": np.array([0.27, 0.0]),
        "lat": np.array([0.0, 0.25]),
        "time": 0.27 * time_direction,
    }
    for vector in vectors.values():
        ax.annotate(
            "", xy=origin + vector, xytext=origin,
            arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": "#4A5055", "shrinkA": 0, "shrinkB": 0},
            zorder=zorder,
        )
    ax.text(*(origin + vectors["lon"] + [0.035, -0.028]), "lon", ha="left", va="top", fontsize=fontsize, zorder=zorder)
    ax.text(*(origin + vectors["lat"] + [0.025, 0.030]), "lat", ha="left", va="bottom", fontsize=fontsize, zorder=zorder)
    ax.text(*(origin + vectors["time"] + [-0.016, 0.035]), "time", ha="right", va="bottom", fontsize=fontsize, zorder=zorder)


def draw_trait_cuboid(
    ax: plt.Axes,
    front: np.ndarray,
    top: np.ndarray,
    left: np.ndarray,
    color: str,
    label: str,
    origin: tuple[float, float],
    show_axes: bool = False,
    zorder: float = 1,
    norm: Normalize | None = None,
    fontsize: float = 17,
) -> Normalize:
    """Draw one trait cuboid at the front-face lower-left ``origin``.

    Parameters are face arrays with shapes ``front=(lat, lon)``,
    ``top=(time, lon)``, and ``left=(time, lat)``.  The normalization is
    independent per trait but shared across all three faces.
    """
    front, top, left = (np.asarray(face, dtype=float) for face in (front, top, left))
    if front.ndim != 2 or top.ndim != 2 or left.ndim != 2:
        raise ValueError("front, top and left must each be two-dimensional.")
    if top.shape[1] != front.shape[1] or left.shape[1] != front.shape[0]:
        raise ValueError("Expected top=(time, lon) and left=(time, lat).")

    norm = trait_normalization(front, top, left) if norm is None else norm
    front_cmap = trait_colormap(color, 0.00)
    top_cmap = trait_colormap(color, 0.28)
    left_cmap = trait_colormap(color, 0.36)
    for cmap in (front_cmap, top_cmap, left_cmap):
        cmap.set_bad("#F0F1F1")

    flb = np.asarray(origin, dtype=float)
    frb = flb + [FRONT_WIDTH, 0.0]
    flt = flb + [0.0, FRONT_HEIGHT]
    frt = flt + [FRONT_WIDTH, 0.0]
    blb, blt, brt = flb + TIME_VECTOR, flt + TIME_VECTOR, frt + TIME_VECTOR

    _heatmap_in_polygon(ax, top, flt, np.array([FRONT_WIDTH, 0.0]), TIME_VECTOR, top_cmap, norm, zorder)
    _heatmap_in_polygon(ax, left, flb, np.array([0.0, FRONT_HEIGHT]), TIME_VECTOR, left_cmap, norm, zorder + 0.01)
    _heatmap_in_polygon(ax, front, flb, np.array([FRONT_WIDTH, 0.0]), np.array([0.0, FRONT_HEIGHT]), front_cmap, norm, zorder + 0.02)

    for fraction in (0.20, 0.40, 0.60, 0.80):
        _line(ax, flt + fraction * np.array([FRONT_WIDTH, 0.0]), flt + fraction * np.array([FRONT_WIDTH, 0.0]) + TIME_VECTOR, zorder + 0.05, subtle=True)
        _line(ax, flb + fraction * np.array([0.0, FRONT_HEIGHT]), flb + fraction * np.array([0.0, FRONT_HEIGHT]) + TIME_VECTOR, zorder + 0.05, subtle=True)

    for start, end in (
        (flb, frb), (frb, frt), (frt, flt), (flt, flb),
        (flt, blt), (blt, brt), (brt, frt), (flb, blb), (blb, blt),
    ):
        _line(ax, start, end, zorder + 0.10)

    if show_axes:
        _coordinate_triad(ax, flb, zorder + 0.20, fontsize=fontsize)
    ax.text(
        flb[0] + 0.06, flt[1] - 0.06, label,
        ha="left", va="top", fontsize=fontsize, color="#151719", zorder=zorder + 0.20,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    return norm

TRAITS = (
    ("LMA", "LMA (g/m²)", "LMA", "#009E73", r"LMA (g m$^{-2}$)"),
    ("CHL", "Chl content (μg/cm²)", "Chl", "#D55E00", r"Chl ($\mu$g cm$^{-2}$)"),
    ("EWT", "EWT (mg/cm²)", "EWT", "#0072B2", r"EWT (mg cm$^{-2}$)"),
    ("N", "N content (mg/cm²)", "N", "#9B5A9A", r"N (mg cm$^{-2}$)"),
    ("LIGNIN", "Lignin (mg/cm²)", "Lignin", "#9A6A2F", r"Lignin (mg cm$^{-2}$)"),
)
DISPLAY_STEP = {"LMA": 1.0, "CHL": 1.0, "EWT": 0.5, "N": 0.01, "LIGNIN": 0.1}
SELECTED = tuple(trait for trait in TRAITS if trait[0] != "LIGNIN")
TRAIT_STYLE = {short: (label, color) for short, _source, label, color, _unit in SELECTED}
FONT_BODY = 20
FONT_TRAIT = 23
FONT_PANEL = 28
YEARS = (2022, 2023, 2024, 2025, 2026)
UNITS = {
    "LMA": r"g m$^{-2}$",
    "CHL": r"$\mu$g cm$^{-2}$",
    "EWT": r"mg cm$^{-2}$",
    "N": r"mg cm$^{-2}$",
}
def draw_trajectory(ax: plt.Axes, frame: pd.DataFrame, trait: str,
                    bottom: bool) -> None:
    subset = frame.loc[frame.trait.eq(trait)]
    for year in YEARS:
        color = YEAR_COLORS[year]
        year_data = subset.loc[subset.year.eq(year)]
        predicted = year_data.loc[year_data.kind.eq("Production NN")]
        predicted = predicted.sort_values("seasonal_date")
        ax.plot(predicted.seasonal_date, predicted["mean"], "-o", color=color,
                linewidth=1.1, markersize=4.2 * np.sqrt(2),
                markeredgecolor="#3A3A3A",
                markeredgewidth=0.25, zorder=3)
        reference = year_data.loc[year_data.kind.eq("EnMAP reference") & year_data["mean"].notna()]
        reference = reference.sort_values("seasonal_date")
        ax.scatter(reference.seasonal_date, reference["mean"], s=56,
                   facecolor="white", edgecolor=color, linewidth=1.25, zorder=5)

    finite = subset["mean"].dropna().to_numpy(float)
    margin = max(float(np.ptp(finite)) * 0.08, 0.02 * abs(float(np.median(finite))))
    lower, upper = float(finite.min() - margin), float(finite.max() + margin)
    if trait == "N":
        lower, upper = np.floor(lower * 100) / 100, np.ceil(upper * 100) / 100
        tick_step = 0.02
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    else:
        lower, upper = np.floor(lower), np.ceil(upper)
        tick_step = {"LMA": 10, "CHL": 2, "EWT": 1}[trait]
    if (upper-lower)/tick_step > 10:
        tick_step *= np.ceil((upper-lower)/(10*tick_step))
    ax.set_ylim(lower, upper)
    ticks = np.arange(np.ceil(lower / tick_step) * tick_step,
                      upper + tick_step / 2, tick_step)
    # The rows are separated by exactly 1 mm, so omit ticks on the shared edge.
    if bottom:
        ticks = ticks[ticks < upper - tick_step * 0.01]
    else:
        ticks = ticks[ticks > lower + tick_step * 0.01]
    ax.set_yticks(ticks)
    ax.set_xlim(pd.Timestamp("2001-02-15"), pd.Timestamp("2001-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#B8B8B8", linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_BODY, length=2, width=0.55, pad=2)
    ax.set_ylabel(UNITS[trait], fontsize=FONT_BODY,
                  labelpad=5 if trait == "N" else 3)
    ax.tick_params(axis="x", bottom=bottom, labelbottom=bottom)
    if bottom:
        ax.set_xlabel("Month", fontsize=FONT_BODY)
    name, color = TRAIT_STYLE[trait]
    ax.text(0.03, 0.97, name, transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT_TRAIT, fontweight="bold", color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=6)

def place_n_ylabel(fig: plt.Figure, ewt_ax: plt.Axes, n_ax: plt.Axes) -> None:
    """Keep axes fixed; move the N unit label vertically if EWT occupies it."""
    renderer = fig.canvas.get_renderer()
    label = n_ax.yaxis.label
    ticks = [tick.get_window_extent(renderer).padded(6)
             for tick in n_ax.get_yticklabels() if tick.get_visible()]
    for y in (0.50, 0.80, 0.20, 0.65, 0.35):
        label.set_y(y)
        box = label.get_window_extent(renderer)
        if box.y0 < n_ax.bbox.y0 + 8 or box.y1 > n_ax.bbox.y1 - 8:
            continue
        if any(box.overlaps(tick) for tick in ticks):
            continue
        occupied = False
        for line in ewt_ax.lines:
            radius = line.get_markersize() * fig.dpi / 144 + 6
            area = Bbox.intersection(box.padded(radius), ewt_ax.bbox)
            if area is not None and line.get_path().transformed(
                    ewt_ax.transData).intersects_bbox(area, filled=False):
                occupied = True
                break
        if not occupied:
            for collection in ewt_ax.collections:
                radius = np.sqrt(collection.get_sizes().max()) * fig.dpi / 144 + 6
                area = Bbox.intersection(box.padded(radius), ewt_ax.bbox)
                points = ewt_ax.transData.transform(collection.get_offsets())
                if area is not None and any(area.contains(x, yy) for x, yy in points):
                    occupied = True
                    break
        if not occupied:
            return
    raise ValueError("No clear N y-label position within the existing layout")

def cube_limits(cube: np.ndarray, front_index: int, short: str) -> tuple[float, float]:
    """Per-site colour limits: P5--P95 of the three visible faces, rounded outwards."""
    faces = np.concatenate((cube[front_index].ravel(), cube[:, 0, :].ravel(),
                            cube[:, :, 0].ravel()))
    finite = faces[np.isfinite(faces)]
    step = DISPLAY_STEP[short]
    lower, upper = np.percentile(finite, (5, 95))
    if short != "N":
        step = 1.0
    return float(np.floor(lower / step) * step), float(np.ceil(upper / step) * step)

def draw_cuboid(ax: plt.Axes, cube: np.ndarray, trait: tuple[str, ...],
                front_index: int, origin: tuple[float, float],
                zorder: int, limits: tuple[float, float] | None = None) -> mpl.colors.Normalize:
    short, _source, label, color, _unit = trait
    norm = mpl.colors.Normalize(*(limits or cube_limits(cube, front_index, short)), clip=True)
    draw_trait_cuboid(ax, front=cube[front_index], top=cube[:, 0, :],
                      left=cube[:, :, 0], color=color, label=label,
                      origin=origin, norm=norm, show_axes=False,
                      zorder=zorder, fontsize=FONT_BODY)
    return norm

def draw_colorbar(fig: plt.Figure, ax: plt.Axes, trait: tuple[str, ...],
                  norm: mpl.colors.Normalize, ticks: tuple[float, ...] | None = None) -> None:
    short, _source, _label, color, unit = trait
    ax.set_axis_off()
    color_axis = ax.inset_axes((0.12, 0.30, 0.76, 0.16))
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=trait_colormap(color))
    bar = fig.colorbar(mapper, cax=color_axis, orientation="horizontal")
    middle = (norm.vmin + norm.vmax) / 2
    if short != "N":
        middle = round(middle)
    bar.set_ticks(ticks if ticks is not None else [norm.vmin, middle, norm.vmax])
    precision = 2 if short == "N" else 0
    bar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{precision}f"))
    bar.ax.tick_params(labelsize=FONT_BODY, length=1.5, width=0.5, pad=1.5,
                       colors="#303438")
    bar.outline.set_linewidth(0.5)
    bar.ax.set_title(unit, fontsize=FONT_BODY, color="#151719", pad=2)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def seasonal_dates(dates):
 d=pd.DatetimeIndex(pd.to_datetime(dates,utc=True,format='mixed'))
 return pd.to_datetime(dict(year=2000,month=d.month,day=d.day))

def draw_continuous(ax,frame,ensemble,baseline,trait,bottom,ylim=None,ticks=None):
 label,color=TRAIT_STYLE[trait]
 part=ensemble[ensemble.trait.eq(trait)].copy().sort_values('date')
 dates=pd.to_datetime(part.date,utc=True).dt.tz_localize(None)
 ax.fill_between(dates,part.seed_min,part.seed_max,color=color,alpha=.20,lw=0,zorder=2)
 ax.plot(dates,part['mean'],color=color,lw=1.5,marker='o',ms=4,mew=0,zorder=3)
 reference=frame[frame.trait.eq(trait)&frame.kind.eq('EnMAP reference')]
 rd=pd.to_datetime(reference.date,utc=True).dt.tz_localize(None)
 ax.scatter(rd,reference['mean'],facecolor='white',edgecolor=color,s=25,lw=1,zorder=5)
 b=baseline[baseline.trait.eq(trait)].copy();bd=pd.to_datetime(b.date)
 for year in YEARS:
  dd=[];vv=[]
  for stamp,value in zip(bd,b['mean']):
   try:date=stamp.replace(year=year)
   except ValueError:continue
   dd.append(date);vv.append(value)
  dd=pd.DatetimeIndex(dd);vv=np.array(vv,float);supported=np.zeros(len(dd),bool)
  valid=dates[part['mean'].notna()].to_numpy()
  for a,z in zip(valid[:-1],valid[1:]):
   if z-a<=np.timedelta64(16,'D'):supported|=(dd>=a)&(dd<=z)
  vv[~supported]=np.nan
  ax.plot(dd,vv,color='#777777',ls='--',lw=1.1,zorder=2)
 for year in range(2021,2027):ax.axvspan(pd.Timestamp(year,11,1),pd.Timestamp(year+1,3,1),color='#D9D9D9',alpha=.35,lw=0,zorder=0)
 ax.set_xlim(pd.Timestamp('2022-06-01'),pd.Timestamp('2026-09-30'))
 ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 ax.set_xlabel('Date' if bottom else '',fontsize=20)
 ax.tick_params(axis='x',labelbottom=bottom,length=2,pad=2)
 vals=np.concatenate([part[['seed_min','seed_max']].to_numpy().ravel(),reference['mean'].to_numpy()]);vals=vals[np.isfinite(vals)]
 margin=max(np.ptp(vals)*.1,abs(np.median(vals))*.02);ax.set_ylim(vals.min()-margin,vals.max()+margin)
 ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4));ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f' if trait=='N' else '%g'))
 if ylim is not None:ax.set_ylim(*ylim)
 if ticks is not None:
  low,high=ax.get_ylim();assert low<=min(ticks) and max(ticks)<=high,(trait,ticks,low,high);ax.set_yticks(ticks)
 low,high=ax.get_ylim();assert low<=vals.min() and vals.max()<=high,(trait,low,high,vals.min(),vals.max())
 ax.set_ylabel(UNITS[trait],fontsize=20,labelpad=4);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#D3D3D3',ls='--',lw=.4)
 title=ax.annotate(label,(.025,1),xytext=(0,3),textcoords='offset points',color=color,fontweight='bold',fontsize=20,va='bottom',ha='left',annotation_clip=False,xycoords='axes fraction')
 return title

# Fig4: per-site scales; only Hesse EWT and N use the requested explicit y ticks.
MAIN_TICKS={('Hesse','EWT'):(12,13,14,15),('Hesse','N'):(.15,.20,.25)}
# Fig4_v2: one range per trait for a-d. Endpoints are multiples of V2_UNIT; y ticks are multiples of V2_STEP.
V2_UNIT={'LMA':5,'CHL':2,'EWT':.5,'N':.02}
V2_STEP={'LMA':15,'CHL':5,'EWT':1,'N':.05}
SITES=('Hesse','Castelporziano')
# Trajectory columns: the left column's left whitespace is GAP_RATIO x its right whitespace,
# and that right whitespace is RIGHT_GAP_MM (display mm at final size).
GAP_RATIO=1.5;RIGHT_GAP_MM=2.5

def load_site(name):
 with np.load(DATA/f'Fig4_cube_{name}.npz') as z:
  names=z['trait_names'].astype(str).tolist();cubes={t:z['data'][names.index(t)] for t in UNITS};front=int(z['t0']);assert str(z['display_dates'][front])[:10]=='2024-07-27'
 frame=pd.read_parquet(DATA/f'Fig4_trajectories_{name}.parquet')
 ensemble=pd.read_csv(DATA/f'Fig4_ensemble_range_{name}.csv',float_precision='round_trip')
 baseline=pd.read_parquet(DATA/f'Fig4_multi_year_mean_{name}.parquet')
 return cubes,front,frame,ensemble,baseline

def shared_ranges(sites):
 """Smallest V2_UNIT-multiple range holding both sites' colour limits and every plotted trajectory value."""
 ranges={}
 for trait in UNITS:
  values=[]
  for cubes,front,frame,ensemble,_ in sites.values():
   part=ensemble[ensemble.trait.eq(trait)];reference=frame[frame.trait.eq(trait)&frame.kind.eq('EnMAP reference')]
   values+=[*cube_limits(cubes[trait],front,trait),*part[['seed_min','seed_max']].to_numpy().ravel(),*reference['mean'].to_numpy()]
  values=np.asarray(values,float);values=values[np.isfinite(values)];unit,step=V2_UNIT[trait],V2_STEP[trait]
  lower,upper=round(np.floor(values.min()/unit)*unit,6),round(np.ceil(values.max()/unit)*unit,6)
  ticks=tuple(round(float(v),6) for v in np.arange(np.ceil(lower/step-1e-9)*step,upper+step*1e-3,step))
  assert len(ticks)%2==1,(trait,ticks)  # colour bars label first, middle and last tick
  ranges[trait]=dict(limits=(lower,upper),ticks=ticks,bar_ticks=(ticks[0],ticks[len(ticks)//2],ticks[-1]))
 return ranges

def visible_boxes(axes,renderer):
 """Drawn content of each axes (same rule as common.content_bbox, which is hash-pinned by the fold-0 run)."""
 from matplotlib.spines import Spine
 from matplotlib.axis import Axis
 boxes=[]
 for ax in axes:
  if ax.axison:boxes.append(ax.get_tightbbox(renderer));continue
  for artist in ax.get_children():
   if artist is ax.patch or isinstance(artist,(Spine,Axis)) or not artist.get_visible():continue
   if isinstance(artist,Text) and not artist.get_text():continue
   try:box=artist.get_tightbbox(renderer)
   except (AttributeError,TypeError):continue
   if box is not None and np.isfinite(box.extents).all() and box.width+box.height>0:boxes.append(box)
 return boxes

def column_gaps(fig,rows):
 """Per site row: whitespace left and right of the left trajectory column (display pixels)."""
 renderer=fig.canvas.get_renderer();gaps=[]
 for left_block,left_column,right_column in rows:
  boundary=Bbox.union(visible_boxes(left_block,renderer)).x1
  column=Bbox.union([ax.get_tightbbox(renderer) for ax in left_column])
  gaps.append((column.x0-boundary,min(ax.get_tightbbox(renderer).x0 for ax in right_column)-column.x1))
 return gaps

def build(output,sites,ranges=None):
 # export_journal leaves journal rcParams behind; restore the Fig4 start state for every build.
 mpl.rcdefaults();fig4_initial_style();style(20);fig=plt.figure(figsize=(17,15.8),dpi=100)
 trajectory_titles=[];hesse_bottom=[];second_headings=[];panel_headings=[];column_rows=[]
 rows=fig.add_gridspec(2,1,left=.015,right=.985,bottom=.04,top=.91,hspace=.13)
 for k,name in enumerate(SITES):
  cubes,front,frame,ensemble,baseline=sites[name]
  blocks=rows[k].subgridspec(1,2,width_ratios=[1,2],wspace=.10);panel_a=blocks[0].subgridspec(2,1,height_ratios=[2.6,1],hspace=0)
  cube_ax=fig.add_subplot(panel_a[0]);cube_ax.set(xlim=(0,3.90),ylim=(.88,4.39));cube_ax.set_aspect('equal');cube_ax.set_axis_off()
  colors=panel_a[1].subgridspec(2,2,wspace=.08,hspace=.30);traj=blocks[1].subgridspec(2,2,wspace=.38,hspace=.16)
  origins=((1.13,2.17),(2.61,2.17),(1.13,1.01),(2.61,1.01));norms=[None]*4
  for index in (3,2,1,0):norms[index]=draw_cuboid(cube_ax,cubes[SELECTED[index][0]],SELECTED[index],front,origins[index],10*(4-index),limits=ranges[SELECTED[index][0]]['limits'] if ranges else None)
  _coordinate_triad(cube_ax,np.array(origins[0]),zorder=100,fontsize=FONT_BODY);lon=next(t for t in cube_ax.texts if t.get_text()=='lon');lon.set_x(lon.get_position()[0]+.25)
  left_block=[cube_ax];columns=([],[])
  for i,trait in enumerate(SELECTED):
   short=trait[0];r,c=divmod(i,2);bar_ax=fig.add_subplot(colors[r,c]);left_block.append(bar_ax)
   draw_colorbar(fig,bar_ax,trait,norms[i],ticks=ranges[short]['bar_ticks'] if ranges else None)
   ax=fig.add_subplot(traj[r,c]);columns[c].append(ax)
   ylim,ticks=(ranges[short]['limits'],ranges[short]['ticks']) if ranges else (None,MAIN_TICKS.get((name,short)))
   title=draw_continuous(ax,frame,ensemble,baseline,short,r==1,ylim=ylim,ticks=ticks);trajectory_titles.append((ax,title))
   if k==0:
    ax.set_xlabel('')
    if r==1:hesse_bottom.append(ax)
  column_rows.append((left_block,*columns))
  top=rows[k].get_position(fig).y1
  site_heading=fig.text(.02,top+.03,f'{chr(97+2*k)}  '+('FR-Hes Hesse' if name=='Hesse' else 'IT-Cp2 Castelporziano'),fontsize=24,fontweight='bold',va='bottom')
  panel_heading=fig.text(.35,top+.03,chr(98+2*k),fontsize=24,fontweight='bold',va='bottom');panel_headings.append(panel_heading)
  if k==1:second_headings=[site_heading,panel_heading]
 normalize(fig)
 for t in fig.findobj(Text):t.set_fontsize(min(8,t.get_fontsize()))
 # Only the needed clearance separates Hesse ticks from the next site headings.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();millimetre=fig.dpi/25.4
 tick_bottom=min(t.get_window_extent(renderer).y0 for ax in hesse_bottom for t in ax.get_xticklabels() if t.get_visible())
 for text in second_headings:
  shift=max(0,text.get_window_extent(renderer).y1-(tick_bottom-millimetre))
  x,y=text.get_position();text.set_position((x,y-shift/fig.bbox.height))
 # Move the left trajectory column (and its b/d letter) right by `shift`; narrow both trajectory columns
 # by `trim` on their facing sides so they keep equal widths. In the tighter row this gives a left gap of
 # GAP_RATIO*RIGHT_GAP_MM and a right gap of RIGHT_GAP_MM. Unit labels are finalised first.
 fix_units(fig);fig.canvas.draw();millimetre=fig.dpi/25.4
 left,right=(min(g) for g in zip(*column_gaps(fig,column_rows)))
 shift=GAP_RATIO*RIGHT_GAP_MM*millimetre-left;trim=max(0.,(RIGHT_GAP_MM*millimetre-right+shift)/2)
 shift,trim=shift/fig.bbox.width,trim/fig.bbox.width
 for _,left_column,right_column in column_rows:
  for ax in left_column:pos=ax.get_position();ax.set_position((pos.x0+shift,pos.y0,pos.width-trim,pos.height))
  for ax in right_column:pos=ax.get_position();ax.set_position((pos.x0+trim,pos.y0,pos.width-trim,pos.height))
 for text in panel_headings:x,y=text.get_position();text.set_position((x+shift,y))
 def check_layout(figure):
  renderer=figure.canvas.get_renderer();tick_bottom=min(t.get_window_extent(renderer).y0 for ax in hesse_bottom for t in ax.get_xticklabels() if t.get_visible())
  for text in second_headings:assert tick_bottom-text.get_window_extent(renderer).y1>=figure.dpi/25.4-1e-6
  for ax,text in trajectory_titles:assert text.get_window_extent(renderer).y0>=ax.bbox.y1+figure.dpi/25.4-1e-6
  for left,right in column_gaps(figure,column_rows):assert left>right>=1.5*figure.dpi/25.4,(output,left,right)
 fig._paper_layout_check=check_layout
 from matplotlib.patches import Patch
 for k in [0]:
  top=rows[k].get_position(fig).y1
  handles=[Line2D([],[],color='#333333',lw=.7,label='Trait reconstruction'),Patch(facecolor='#777777',alpha=.2,label='Ensemble range'),Line2D([],[],ls='--',color='#777777',lw=.6,label='Multi-year mean'),Line2D([],[],ls='',marker='o',mfc='white',mec='#444444',ms=3,label='EnMAP reference (in-sample)'),Patch(facecolor='#D9D9D9',alpha=.35,label='Nov–Feb')]
  fig.legend(handles=handles,ncol=3,loc='lower right',bbox_to_anchor=(.992,top+.012),frameon=False,fontsize=7,handlelength=1.2,columnspacing=.8,handletextpad=.5)
 export_journal(fig,output,OUT)

def main():
 sites={name:load_site(name) for name in SITES}
 build('Fig4',sites)
 ranges=shared_ranges(sites)
 for trait,spec in ranges.items():print('Fig4_v2',trait,spec,flush=True)
 build('Fig4_v2',sites,ranges)
if __name__=='__main__':main()
