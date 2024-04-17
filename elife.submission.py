import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn
from nilearn.connectome import ConnectivityMeasure
import umap
import plotly.express as pex
import plotly.graph_objects as go
import pingouin # only version 0.2.1 
import nibabel
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from brainspace.gradient.utils import dominant_set
from surfplot import Plot
from enigmatoolbox.plotting import plot_subcortical
import bct
from scipy.stats import pearsonr
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm
import warnings

warnings.filterwarnings("ignore")

# todo check for no local addressing. maybe add pathjoin.
DIR = '/Users/qasem/PycharmProjects/large-manifold-nbs/data/'
REGIONS = pd.read_csv(DIR+'regions_sorted.csv')[['region', '7net', '17net',]]
REGIONS_STR_ORDER = pd.read_csv(DIR+'striatum_order.csv').iloc[:,0].values
NETS7_ORDER = ['Striatum', 'Limbic', 'Default', 'DorsAttn', 
              'Cont', 'SalVentAttn', 'SomMot', 'Vis',]

vertices = nibabel.load(DIR + 'Schaefer_atlas_1000.nii').get_fdata()[0]
l_hemi, r_hemi = load_conte69()

SUBJECTS = pd.read_csv(DIR + 'subjects.csv')
# matching str and int ids. erroneous subjects removed

EPOCHS = ('baseline', 'early', 'late')  # in future add 'rest'
EPOCH_REF = 'baseline'

RANGE_G1 = (-5, 7)
RANGE_G2 = (-5, 4)
RANGE_G3 = (-4.5, 4)

COLOR_RANGE_ECC = (1, 6);       COLOR_MAP_ECC = 'viridis'
COLOR_RANGE_G   = (-3.7, 3.7);  COLOR_MAP_G = 'bwr'
COLOR_RANGE_T   = (-3, 3);      COLOR_MAP_T = 'bwr'
COLOR_MAP_SHIFT = {'b2e': '#7fb0b4', 'e2l': '#bf7fa1', 'b2l': '#f5c26b'}

OLD_ORDER_17 = ['ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 
                'DorsAttnA', 'DorsAttnB', 'LimbicA', 'LimbicB', 'SalVentAttnA', 'SalVentAttnB',
                'SomMotA', 'SomMotB', 'StriatumLeft', 'StriatumRight', 'TempPar', 'VisCent', 'VisPeri']


# saved new version in ts1, cmat/cnt1
def ts(subject: int, epoch):
    """
    returns a pd.DataFrame of time series of a subject in a specific epoch
    epoch: str in 'rest', 'baseline', 'early', 'late' (in learning phase)
    """
    # todo dump time-series. must add 'rest'. should update with new subcortical data
    # in data/ts/ we don't have cerebellum regions included. only 1012 regions.
    return pd.read_csv(DIR+'ts1/ts_'+str(subject)+'_'+epoch+'.csv')


def compute_conn_mat(timeseries, fill_diag=False, kind='covariance'):
    """ not centred connectivity matrix
    timeseries: pd.DataFrame of time series of a subject in a specific epoch
    fill_diag: bool, if True, diagonal elements of the matrix will be filled with 0
    kind: str, 'covariance' or 'correlation'
    """
    conn_measure = ConnectivityMeasure(kind=kind)
    cmat = conn_measure.fit_transform([timeseries.to_numpy()])[0]
    if fill_diag:   np.fill_diagonal(cmat, 0)
    return pd.DataFrame(cmat, index=timeseries.columns, columns=timeseries.columns)


def load_conn_mat_grand_mean():
    """ loads the grand mean connectivity matrix """
    # print('computing grand mean conn mat')
    # grand_mean = mean_riemann(np.stack([
    #     compute_conn_mat(ts(s, e)) for e in EPOCHS for s in SUBJECTS.int_id
    # ]), maxiter=5)
    # pd.DataFrame(grand_mean).to_csv(DIR+'cmat/cnt1/grand_mean.csv', index=False)
    return pd.read_csv(DIR+'cmat/cnt1/grand_mean.csv').to_numpy() # riemann mean of all subjects over all epochs


def load_conn_mat_ref_mean(epoch_ref=EPOCH_REF):
    """ loads the reference mean connectivity matrix """
    # print('computing reference conn mat')
    # ref_mean = mean_riemann(np.stack([
    #     compute_conn_mat(ts(s, epoch_ref)) for s in SUBJECTS.int_id
    # ]), maxiter=5)
    # pd.DataFrame(ref_mean).to_csv(DIR+f'cmat/cnt1/{epoch_ref}_mean.csv', index=False)
    return pd.read_csv(DIR+f'cmat/cnt1/{epoch_ref}_mean.csv').to_numpy() # the ref epoch for all subjects


def dump_conn_mat_mean():
    """ dumps mean connectivity matrices for each subject """
    print('computing mean conn mat for each subject')
    for subject in SUBJECTS.int_id:
        c = mean_riemann(np.stack([compute_conn_mat(ts(subject, e)) for e in EPOCHS]), maxiter=5)
        pd.DataFrame(c).to_csv(DIR+'cmat/cnt1/mean_'+str(subject)+'.csv', index=False)

    
def load_conn_mat_mean(subject):
    return pd.read_csv(DIR+'cmat/cnt1/mean_'+str(subject)+'.csv').to_numpy()


def _to_tangent(s, mean):
    # Covariance centering
    p = sqrtm(mean)
    p_inv = invsqrtm(mean)
    return p @ logm(p_inv @ s @ p_inv) @ p

def _gl_transport(t, sub_mean, grand_mean):
    g = sqrtm(grand_mean) @ invsqrtm(sub_mean)
    return g @ t @ g.T

def _from_tangent(t, grand_mean):
    p = sqrtm(grand_mean)
    p_inv = invsqrtm(grand_mean)
    return p @ expm(p_inv @ t @ p_inv) @ p

def center_cmat(c, sub_mean, grand_mean):
    """Center covariance matrix using tangent transporting procedure
    https://github.com/danjgale/adaptation-manifolds/blob/main/adaptman/connectivity.py """
    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)


def dump_conn_mat_centred(grand_mean):
    """ dumps centred connectivity matrices for each subject """
    print('computing centered conn mat for each subject')
    for s in SUBJECTS.int_id:
        for e in EPOCHS:
            c = center_cmat(compute_conn_mat(ts(s, e)).to_numpy(), 
                            load_conn_mat_mean(s), grand_mean)
            pd.DataFrame(c).to_csv(DIR+'cmat/cnt1/cnt_'+str(s)+'_'+e+'.csv', index=False)


def load_conn_mat_centred(subject, epoch) -> pd.DataFrame:
    """ loads a centred connectivity matrix of a subject in a specific epoch """
    return pd.DataFrame(pd.read_csv(DIR+'cmat/cnt1/cnt_'+str(subject)+'_'+epoch+'.csv').values,
                        index=pd.MultiIndex.from_frame(REGIONS),
                        columns=pd.MultiIndex.from_frame(REGIONS))
    
    
def conn_mat(subject, epoch, centred=True) -> pd.DataFrame:
    """ returns a connectivity matrix of a subject in a specific epoch """
    if centred: return load_conn_mat_centred(subject, epoch)
    else:       return compute_conn_mat(ts(subject, epoch))


def generate_umap():
    """ generates umap embeddings of connectivity matrices
    filename: str, if not None, will save the plot with this name
    """
    print('generating umap embeddings of connectivity matrices')
    l = []
    for cnt in (True, False):
        cmats = np.stack([conn_mat(s, e, centred=cnt).values.flatten()
                             for s in SUBJECTS.int_id for e in EPOCHS])
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(cmats)
        l.append(pd.DataFrame(embedding, columns=['dim1', 'dim2'],
                          index=pd.MultiIndex.from_product([SUBJECTS.int_id, EPOCHS])))
    df_emb = pd.concat(l, axis=0, keys=['centred', 'original'], names=['centred', 'subject', 'epoch'])
    df_emb.index = df_emb.index.set_levels(df_emb.index.levels[1].astype(str), level=1)
    
    print('plotting umap embeddings of connectivity matrices')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i,ax in enumerate(axes.flatten()):
        cnt = 'centred' if i//2 else 'original'
        hue = 'epoch' if i%2 else 'subject'
        seaborn.scatterplot(data=df_emb.loc[cnt], x='dim1', y='dim2',
                            hue=hue, legend=False, ax=ax)
        ax.set_title(cnt+' conn mat -- color='+hue)
    plt.savefig('fig2_umap.png', dpi=300); plt.savefig('fig2_umap.svg')


def dump_gradients(approach='pca'):
    gmaps_ref = GradientMaps(random_state=42, approach=approach, kernel='cosine')
    gmaps_ref.fit(load_conn_mat_ref_mean())
    gmaps = GradientMaps(random_state=42, approach=approach, alignment='procrustes', kernel='cosine')

    gmaps.fit([conn_mat(s, e, centred=True).to_numpy()
            for e in EPOCHS for s in SUBJECTS.int_id], reference=gmaps_ref.gradients_)

    S = SUBJECTS.int_id.tolist()
    dfg = pd.concat([pd.DataFrame({'subject': s, 'epoch': e,
                                'g1': gmaps.aligned_[EPOCHS.index(e) * len(S) + S.index(s)][:, 0],
                                'g2': gmaps.aligned_[EPOCHS.index(e) * len(S) + S.index(s)][:, 1],
                                'g3': gmaps.aligned_[EPOCHS.index(e) * len(S) + S.index(s)][:, 2]},
                            index=pd.MultiIndex.from_frame(REGIONS))
                for e in EPOCHS for s in S], axis=0)
    dfg['ecc'] = np.sqrt(dfg['g1']**2 + dfg['g2']**2 + dfg['g3']**2)
    dfg.to_csv(DIR+'gradients1.csv')


def load_color_map_yeo(nets='7'):
    """ color map dict for yeo networks. should overlap and be ordered by current networks """
    assert nets in ('7', '17')
    cmap = {}
    with open(DIR + '_cmap'+nets+'.txt') as f:
        for line in f.readlines():
            k, r, g, b = line.split()
            cmap[k] = (int(r), int(g), int(b))
    cmap = {k: np.array(v) / 255 for k, v in cmap.items()}
    return cmap
    # should overlap current networks
    # take only cmap values in used_nets and ordered by it
    # cmap = {k: cmap[k] for k in regions.get_level_values('network').unique()}


def plot_cortex(data, p, **kwargs):
    filename = kwargs.pop('filename')
    cortex_ordered = REGIONS.loc[REGIONS['7net'] != 'Striatum', 'region']
    data = data.loc[cortex_ordered]    # order for plotting

    plot = Plot(surf_lh=l_hemi, surf_rh=r_hemi, label_text=[kwargs.pop('label_text', filename)],
                layout=kwargs.pop('layout', 'grid'), size=kwargs.pop('size', (900, 700)))
    
    # assuming data is stored in first column
    if p is None:
        if isinstance(data, pd.Series):     data1 = data.values
        elif isinstance(data, pd.DataFrame):   data1 = data.iloc[:, 0].values
        else: raise ValueError('data must be a Series or DataFrame')
    else:
        data1 = data.apply(lambda row: row[0] if row[p] < 0.05 else None, axis=1).values

    data1 = map_to_labels(data1, vertices, mask=(vertices != 0))
    plot.add_layer(data1, cbar=True, cmap=kwargs.pop('color_map', None),
                    color_range=kwargs.pop('color_range', None),)

    if kwargs.pop('outline', False):    # asumming outline only when p is given
        if p is None:   raise ValueError('p must be given to outline')
        data2 = map_to_labels(data[p] < 0.05, vertices, mask=(vertices != 0))
        plot.add_layer(data2, cbar=False, cmap='binary', as_outline=True)

    if 'highlight' in data.columns:
        data2 = map_to_labels(data['highlight'], vertices, mask=(vertices != 0))
        plot.add_layer(data2, cbar=False, cmap='Wistia')  # seed region yellowed

    fig = plot.build(); fig.savefig(filename+'.png', dpi=300)



def plot_striatum(data, p, **kwargs):
    data = data.loc[REGIONS_STR_ORDER]    # order for plotting
    
    if p is None:
        if isinstance(data, pd.Series):     data1 = data.values
        elif isinstance(data, pd.DataFrame):   data1 = data.iloc[:, 0].values
        else: raise ValueError('data must be a Series or DataFrame')
    else:
        data1 = data.apply(lambda row: row[0] if row[p] < 0.05 else None, axis=1).values
        
    plot_subcortical(data1,
                ventricles=False, size=kwargs.pop('size', (800, 400)), color_bar=True, 
                cmap=kwargs.pop('color_map', None),  color_range=kwargs.pop('color_range', None),
                screenshot=True, filename=kwargs.pop('filename')+'.png',
                nan_color=(0.5, 0.5, 0.5, 0)    # None to be gray
    )


def plot_brain(data, p=None, **kwargs):
    """ data can be an iterable of proper size
    if data is pd.DataFrame, takes the first column to plot
    if a pd.Series, index should be the region names. will match inside sub-functions
    if p (str) is given, mask everything with p < 0.05
    if 'highlight' in data columns, highlights the regions with highlight==True. only in plot_cortex
    """
    fname = kwargs.pop('filename', None)
    plot_cortex(data, p, filename=fname+'_cortex', **kwargs)
    plot_striatum(data, p, filename=fname+'_striatum', **kwargs)
    

def plot_radar_shifts(df, num_nets, filename, pairs=('b2e', 'e2l'), order=None):
    assert num_nets in ('7', '17'); num_nets = num_nets+'net'
    if order is None:   order = df.index.get_level_values(num_nets).unique()
    
    df = df.groupby(['shift', num_nets])['t'].mean()
    fig = go.Figure()
    for shift in pairs:
        fig.add_trace(go.Scatterpolar(
            r=df.loc[shift].loc[order], theta=order, fill='toself', 
            name=shift, marker_color=COLOR_MAP_SHIFT[shift]))
    fig.add_trace(go.Barpolar(
        r=[0.05]*len(order), theta=order, name='zero', marker_color='black'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=(-3, 3))), showlegend=True)
    fig.show()
    filename = f'{filename}-radar-{num_nets}'
    fig.write_image(filename+'.svg'); fig.write_image(filename+'.png')
    


def plot3d_networks(df, num_nets):
    assert num_nets in ('7', '17'); cmap = load_color_map_yeo(num_nets); num_nets = num_nets+'net'
    nets_used = df.index.get_level_values(num_nets).unique()
    cmap = {k: [int(v*255) for v in cmap[k]] for k in cmap} # convert to 0-255
    cmap = {k: f'rgb({v[0]}, {v[1]}, {v[2]})' for k, v in cmap.items()}  # plotly format
    
    # only this is different from plot3d_ecc
    ax = pex.scatter_3d(x='g1',y='g2',z='g3', color=num_nets,
                        data_frame=df.reset_index(), opacity=.5,
                        color_discrete_map=cmap,
                        category_orders={num_nets: nets_used})                  
    
    ax.update_traces(marker_size=3)
    ax.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='centroid',
                            marker=dict(size=7, color='white', symbol='circle',
                                        line=dict(color='black', width=5)), opacity=1))
    ax.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    ax.write_image('fig3C.png'); ax.write_image('fig3C.svg')


def plot3d_ecc(df, num_nets):
    assert num_nets in ('7', '17'); cmap = load_color_map_yeo(num_nets); num_nets = num_nets+'net'
    nets_used = df.index.get_level_values(num_nets).unique()
    cmap = {k: f'rgb({v[0]}, {v[1]}, {v[2]})' for k, v in cmap.items()}  # plotly format
    
    # only this is different from plot3d_networks
    ax = pex.scatter_3d(x='g1',y='g2',z='g3', color='ecc',
                        data_frame=df.reset_index(), opacity=.5,
                        color_continuous_scale='viridis',
                        range_color=COLOR_RANGE_ECC)
    
    ax.update_traces(marker_size=3)
    ax.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='centroid',
                            marker=dict(size=7, color='white', symbol='circle',
                                        line=dict(color='black', width=5)), opacity=1))
    ax.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    ax.write_image('fig3E.png'); ax.write_image('fig3E.svg')


def jointplot_gradients(df, num_nets):
    assert num_nets in ('7', '17'); cmap = load_color_map_yeo(num_nets); num_nets = num_nets+'net'
    nets_used = df.index.get_level_values(num_nets).unique()

    seaborn.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    seaborn.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    seaborn.jointplot(y='g2', x='g1', hue=num_nets, hue_order=nets_used,
                        data=df.reset_index(), palette=cmap, alpha=.5, legend=False,
                        height=8, space=0, xlim=RANGE_G1, ylim=RANGE_G2)
    plt.savefig('fig3D_1.png', dpi=300, format='png'); plt.savefig('fig3D_1.svg', format='svg')
    seaborn.jointplot(y='g3', x='g1', hue=num_nets, hue_order=nets_used,
                        data=df.reset_index(), palette=cmap, alpha=.5, legend=False,
                        height=8, space=0, xlim=RANGE_G1, ylim=RANGE_G3)
    plt.savefig('fig3D_2.png', dpi=300, format='png'); plt.savefig('fig3D_2.svg', format='svg')


def compute_ttests(df):
    """ paired t-tests between epochs. correct for multiple comparisons fdr_bh.
    baseline, early, late. not rest 
    """
    print('computing t-tests')
    dft = df.reset_index('epoch').groupby(['region', '7net', '17net']).apply(
        pingouin.pairwise_tests, subject='subject', padjust='fdr_bh',
        dv='ecc',       # dependent variable: ecc
        within='epoch'  # within instead of between, makes it paired
        )
    dft.loc[:, 'shift'] = dft.A.str[0] + '2' + dft.B.str[0]
    dft = dft.set_index('shift', append=True).droplevel(3)
    dft = dft.reorder_levels([3,0,1,2])
    dft = dft.loc[:, ['T', 'p-unc', 'p-corr']]
    dft.columns = ['t', 'p', 'p_corr']
    dft.loc[:, 't'] = -1 * dft.loc[:, 't']  # change t-statistics, look at appendix
    return dft


def plot3d_t_shifts(pair, df, df_ttests, num_nets):
    """ plot lines for significant shifts between epoch pairs
    pair:       tuple of two epochs, e.g. ('baseline', 'early')
    df:         average loading of each region in each epoch. not indexed.
    df_ttests:  t-tests between epochs to find significant shifts
    num_nets:   '7' or '17' to color regions by network affiliation
    """
    assert num_nets in ('7', '17'); cmap = load_color_map_yeo(num_nets); num_nets = num_nets+'net'

    nets_used = df[num_nets].unique() # df.index.get_level_values(num_nets).unique()
    regions = df['region'].unique() # df.index.get_level_values('region').unique()

    df = df.set_index(['region', 'epoch'])

    cmap = {k: [int(v*255) for v in cmap[k]] for k in cmap} # convert to 0-255
    cmap = {k: f'rgb({v[0]}, {v[1]}, {v[2]})' for k, v in cmap.items()}  # plotly format

    pair_ = pair[0][0] + '2' + pair[1][0]

    fig = go.Figure()

    for r in regions:   # for all regions

        data = df.loc[r]
        
        net = data[num_nets][0]
        hex_color = cmap[net]
        
        # plot tail of the line
        fig.add_trace(go.Scatter3d(x=[data.loc[pair[0]].g1],
                                y=[data.loc[pair[0]].g2],
                                z=[data.loc[pair[0]].g3],
                                mode='markers', marker=dict(size=.7, color=hex_color,opacity=0.7),name=pair[0]))

        region_is_significant = df_ttests.loc[(pair_, r),'p_corr'][0] < 0.05
        if region_is_significant:
            # plot head of the line
            fig.add_trace(go.Scatter3d(x=[data.loc[pair[1]].g1],
                                    y=[data.loc[pair[1]].g2],
                                    z=[data.loc[pair[1]].g3],
                                    mode='markers', marker=dict(size=3, color=hex_color,opacity=0.7),name=pair[1]))

            # add a line between head to tail for example, baseline to early
            fig.add_trace(
                go.Scatter3d(x=[data.loc[pair[0], 'g1'], data.loc[pair[1], 'g1']],
                            y=[data.loc[pair[0], 'g2'], data.loc[pair[1], 'g2']],
                            z=[data.loc[pair[0], 'g3'], data.loc[pair[1], 'g3']],
                            mode='lines', line=dict(color=hex_color, width=2)
                            ))
    # centroid
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='centroid',
                                marker=dict(size=7, color='white', symbol='circle',
                                            line=dict(color='black', width=5)), opacity=1))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                      scene=dict(xaxis=dict(range=[-4,6]), yaxis=dict(range=[-4,4]), zaxis=dict(range=[-3,4])))
    fig.show(); fig.write_image('fig4_' + pair_ + '.svg', width=800, height=800)


def dump_seed_corr(seed):
    """ seed connectivity shift between epochs to csv file
    """
    list1 = []
    for e in EPOCHS:
        list2 = []
        for subj in SUBJECTS.int_id:
            ts_ = ts(subj, e)
            seed_ts = ts_[seed]
            # we neglect pvalue of whether r is significant
            r = [pearsonr(seed_ts, ts_[idx])[0] for idx in ts_]   # pearsonr()[0] is rvalue
            r = pd.DataFrame({'r': r}, index=ts_.columns)
            r.rename_axis('region', inplace=True)
            r = pd.concat([r], keys=[subj], names=['subject'])
            list2.append(r)
        r = pd.concat(list2, 0)
        r = pd.concat([r], keys=[e], names=['epoch'])   # adds epoch as lower level index
        list1.append(r)
    dfr = pd.concat(list1, 0)
    dfr.to_csv(DIR+f'corr_{seed}.csv')


def ttest_epochs(df):
    df.index = df.index.droplevel([0,1])
    df = df.unstack(0)
    df.columns = df.columns.droplevel(0)
    b2e = pingouin.ttest(df['early'], df['baseline'], paired=True, alternative='greater').loc['T-test']
    e2l = pingouin.ttest(df['late'], df['early'], paired=True, alternative='greater').loc['T-test']
    return pd.Series([b2e['T'], b2e['p-val'], e2l['T'], e2l['p-val']], index=['b2e_t', 'b2e_p', 'e2l_t', 'e2l_p'])


def stripplot_ecc(df, regions):
    seaborn.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    seaborn.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, axes = plt.subplots(3, 2, figsize=(6,11), sharex=True, sharey=True)

    for i, region in enumerate(regions):
        seaborn.stripplot(data=df[df.region=='7Networks_' + region], x='epoch', y='ecc',
                        ax=axes[i//2, i%2], color='black', alpha=.5)
        seaborn.pointplot(data=df[df.region=='7Networks_' + region], x='epoch', y='ecc', markers='X',
                        color='black', ci=None, join=True, scale=.3, linestyles='dashed', ax=axes[i//2, i%2])

        axes[i//2, i%2].set_ylim(0, 8)
        axes[i//2, i%2].set_title(region)

    plt.tight_layout()
    plt.savefig('fig5_and_suppfig5_stripplots.svg', format='svg')
    plt.show()


def dump_seeds_conn(seeds):
    seeds = ['7Networks_'+s for s in seeds]    # prepare names
    for seed in seeds:  dump_seed_corr(seed)
    df_seeds_corr = pd.concat([pd.read_csv(DIR+f'corr_{s}.csv', index_col=[0, 1, 2])
                            for s in seeds], keys=seeds, names=['seed'], axis=0).round(4) # always round corr values
    df_seeds_shift = df_seeds_corr.reorder_levels([0,3,1,2]).groupby(level=[0,1]).apply(ttest_epochs)
    df_seeds_shift.to_csv(DIR+'df_seeds_shift.csv')


def plot_seed_conn_shift(seeds, df_, **kwargs):
    seeds = ['7Networks_'+s for s in seeds]    # prepare names
    for seed_region in seeds:
        df = df_.loc[seed_region].copy()
        df['highlight'] = df.index.get_level_values('region') == seed_region
        plot_brain(df, filename=f'fig5_b2e_{seed_region}',              # b2e
                    color_map=kwargs.pop('color_map', COLOR_MAP_T),
                    color_range=kwargs.pop('color_range', COLOR_RANGE_T),)
        plot_brain(df.iloc[:, 2:], filename=f'fig5_e2l_{seed_region}',  # e2l
                    color_map=kwargs.pop('color_map', COLOR_MAP_T),
                    color_range=kwargs.pop('color_range', COLOR_RANGE_T),)


def _prepare_for_radar_plot(df):
    df = df.reorder_levels([1,0]).join(REGIONS.set_index('region')).set_index(['7net', '17net'], append=True).drop(columns=['b2e_p', 'e2l_p']).droplevel(0)
    df.columns = ['b2e', 'e2l']; df.columns.name = 'shift'
    df = df.stack().to_frame('t')
    return df


def compute_graph_measures(cmat, num_nets, thresh=.9, regions=REGIONS):
    """
    from https://github.com/danjgale/adaptation-manifolds
    Calculate region-level connectivity properties pertaining to integration
    and segregation

    Parameters
    ----------
    cmat : np.ndarray
        Dense connectivity matrix
    thresh : float, optional
        Row-wise threshold. By default .9, which is identical to the threshold
        in cograd.gradients
        
    Returns
    -------
    pd.DataFrame
        Functional connectivity measures for each region
    """
    assert num_nets in ('7', '17'); num_nets = num_nets+'net'
    print(f'computing measures for {num_nets} networks affilation')
    
    # affilitation vectors
    regions = regions.copy()
    network_aff = regions[num_nets].to_numpy()
    regions['hemi'] = regions['region'].apply(lambda rname: rname[10] if rname[0]=='7' else rname[0])
    network_hemi_aff = (regions['hemi'] + regions[num_nets].astype(str)).to_numpy()

    # threshold and binarize
    x = dominant_set(cmat, k=1 - thresh, is_thresh=False, as_sparse=False)
    x_bin = (x != 0).astype(float)

    return pd.DataFrame({
        'participation': bct.participation_coef(x, network_aff, 'out'),
        'participation_h': bct.participation_coef(x, network_hemi_aff, 'out'),
        'module_degree': bct.module_degree_zscore(x_bin, network_aff, 2),
        'module_degree_h': bct.module_degree_zscore(x_bin, network_hemi_aff, 2),
        'strength': np.sum(x, axis=1)},
        index=pd.MultiIndex.from_frame(regions),
        )


def scatterplot_graph_ecc(df, num_nets):
    assert num_nets in ('7', '17'); cmap = load_color_map_yeo(num_nets); num_nets = num_nets+'net'
    nets_used = df.index.get_level_values(num_nets).unique()
    cmap = {net: cmap[net] for net in nets_used}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(['participation', 'module_degree', 'strength']):
        seaborn.scatterplot(x='ecc', y=col, data=df, ax=axes[i], alpha=.6,
                            # style=df.index.get_level_values(num_nets).isin(['Striatum']),
                            hue=num_nets, hue_order=nets_used, palette=seaborn.color_palette(cmap.values()))
        corr, p = pearsonr(df['ecc'], df[col])
        x, y = df['ecc'], df[col]
        m, b = np.polyfit(x, y, 1)
        axes[i].plot(x, m*x + b, color='black', linewidth=1)
        axes[i].set_title(f'corr: {corr:.2f} p: {p:.2f}')
    axes[0].get_legend().remove(); axes[1].get_legend().remove()
    axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    fname = 'supp_fig2_left'; plt.savefig(fname+'.png', dpi=300); plt.savefig(fname+'.svg', format='svg')


def scatterplot(df, epoch):
    plt.figure(figsize=(10, 10))
    seaborn.regplot(data=df, x=epoch, y='score', color='black')
    r, p = pearsonr(df[epoch], df['score'])
    plt.title(f'{epoch} {epoch} r={r:.4f} p={p:.4f}')
    plt.xlim(-1.4, 1.4)
    plt.ylim(-2, 2)
    plt.savefig(f'DAN-A-{epoch}.svg')


def generate_spin_permutation_nulls(df, spins, true_ordering, n_perms=2000):
    def f(ordering):
        df1 = df.copy()
        df1['idx_perm'] = 36 * (ordering if isinstance(ordering, list) else ordering.tolist())
        return df1.merge(REGIONS, left_on='idx_perm', right_on='idx').groupby(['17net_y', 'subject']).mean().groupby('17net_y').corr()['score']

    df_nulls = pd.concat([f(ordering) for ordering in spins.T], axis=0, keys=range(n_perms))
    return df_nulls


def plot_permutation_results(df, df_nulls, true_values, spins, true_ordering):
    nums_net = '17net'

    fig, axes = plt.subplots(2, REGIONS[nums_net].nunique(), figsize=(REGIONS[nums_net].nunique() * 3, 6), sharex=True, sharey=True)

    for i, epoch in enumerate(['b2e', 'e2l']):
        for j, net in enumerate(REGIONS[nums_net].unique()):
            null_values = df_nulls.loc[:, net, epoch].values
            axes[i, j].hist(null_values, bins=20)
            axes[i, j].axvline(true_values.loc[net, epoch], color='red')
            pval = 1 - percentileofscore(null_values, true_values.loc[net, epoch]) / 100
            axes[i, j].set_title(f'{pval:.3f}')
            axes[i, j].set_xlabel(net)

    axes[0, 0].set_ylabel('b2e')
    axes[1, 0].set_ylabel('e2l')

    plt.tight_layout()
    plt.savefig('DAN-B-vasa.svg', format='svg')


def plot_permutation_results_boxplot(df, df_nulls, true_values, true_ordering, nums_net='17net'):
    fig, axes = plt.subplots(2, REGIONS[nums_net].nunique(), figsize=(REGIONS[nums_net].nunique() * 1.5, 6), sharex=False, sharey=True)

    for i, epoch in enumerate(['b2e', 'e2l']):
        for j, net in enumerate(sorted(REGIONS[nums_net].unique())):
            null_values = df_nulls.loc[:, net, epoch].values
            # instead of histogram, plot a box plot of the null distribution. do not show outliers.
            # show the true value in red dot (not line) on top of the box plot. not with different x.
            axes[i, j].boxplot(null_values, showfliers=False)
            axes[i, j].plot(1, true_values.loc[net, epoch], 'ro')
            pval = 1 - percentileofscore(null_values, true_values.loc[net, epoch]) / 100
            axes[i, j].set_title(f'{pval:.3f}')
            axes[i, j].set_xlabel(net)

    axes[0, 0].set_ylabel('b2e')
    axes[1, 0].set_ylabel('e2l')

    plt.tight_layout()
    plt.savefig('DAN-B-vasa.svg', format='svg')


if __name__ == '__main__':
    # dump_gradients()
    print('loading gradients and eccentricity')
    df_gradients = pd.read_csv(DIR+'gradients1.csv').set_index(['epoch']+REGIONS.columns.tolist())

    # fig 1. ignore

    # fig 2
    # generate_umap()

    dfg_mean = df_gradients.groupby(level=df_gradients.index.names).mean().drop('subject', axis=1)

    # fig 3A
    for g in ['g1', 'g2', 'g3']:
        plot_brain(dfg_mean.loc[EPOCH_REF, g], 
                filename=f'fig3A_{EPOCH_REF}_{g}',
                color_map=COLOR_MAP_G, color_range=COLOR_RANGE_G,
                layout='row', size=(1600,300))


    # fig 3B. Taken from Keanna's script

    # fig 3C
    plot3d_networks(dfg_mean.loc[EPOCH_REF], '7')

    # fig 3E
    plot3d_ecc(dfg_mean.loc[EPOCH_REF], '7')

    # fig 3D
    jointplot_gradients(dfg_mean, '7')  # can .loc[EPOCH_REF] if one epoch needed

    # fig 3G
    plot_brain(dfg_mean.loc[EPOCH_REF, 'ecc'],
            filename=f'fig3G_{EPOCH_REF}_ecc',
            color_map=COLOR_MAP_ECC, color_range=COLOR_RANGE_ECC,
            layout='row', size=(1600,300))

    # compute_ttests(df_gradients).to_csv(DIR+'ttests1.csv')
    print('loading ttets results')
    df_ttests = pd.read_csv(DIR+'ttests1.csv').set_index(['shift', 'region', '7net', '17net'])

    # fig 4 A,B
    for shift in ('b2e', 'e2l'):
        plot_brain(df_ttests.loc[shift], p='p_corr', filename=f'fig4_{shift}', 
                outline=True, color_map=COLOR_MAP_T, color_range=COLOR_RANGE_T)

    # fig 4 radar
    plot_radar_shifts(df_ttests, '7', 'fig4_ttest', order=None)     # give '17' for yeo's 17 networks

    # fig 4C,D
    plot3d_t_shifts(('baseline', 'early'), dfg_mean.reset_index(), df_ttests, '7')
    plot3d_t_shifts(('early', 'late'), dfg_mean.reset_index(), df_ttests, '7')
    
    
    # fig 5
    seeds = [
        'LH_Default_PFC_19',        'RH_Default_PFCdPFCm_8',        # PFC
        'LH_DorsAttn_FEF_5',        'RH_DorsAttn_FEF_6',            # FEF
        'LH_Default_pCunPCC_32',    'RH_Default_pCunPCC_18',        # PCC
    ]
    left_hemi_seeds, right_hemi_seeds = seeds[::2], seeds[1::2] 

    # dump_seeds_conn(seeds)
    print('loading seed connectivity shift between epochs')
    df_seeds_shift = pd.read_csv(DIR+'df_seeds_shift.csv', index_col=[0,1])

    # fig 5 brain plots. highlighted, no mask
    plot_seed_conn_shift(left_hemi_seeds, df_seeds_shift)

    # fig 5 radar plots
    df_seeds_shift_pivot = _prepare_for_radar_plot(df_seeds_shift)
    for s in left_hemi_seeds:   plot_radar_shifts(df_seeds_shift_pivot.loc['7Networks_'+s], '17', 'fig5_'+s, order=OLD_ORDER_17)
    
    # fig 5 strip plots
    stripplot_ecc(df_gradients.reset_index(), seeds)
    
    # fig 6 from Tianyao

    # fig 7
    spins = nnstats.gen_spinsamples(coords, hemi, seed=1234, n_rotate=2000, method='vasa')
    true_ordering = REGIONS.set_index('region').loc[df['region'].unique()]['idx'].values
    true_values = f(true_ordering).unstack()
    df_nulls = generate_spin_permutation_nulls(df, spins, true_ordering)
    plot_permutation_results(df, df_nulls, true_values, spins, true_ordering)
    plot_permutation_results_boxplot(df, df_nulls, true_values, true_ordering)

    
    # supplementary figures
    
    # supp fig 1. ignore

    # supp fig 2
    graph_measures = compute_graph_measures(load_conn_mat_ref_mean(), '7',
                                            thresh=.9, regions=REGIONS.copy()) # affiliation=yeo's 7 networks
    # supp fig 2. rigth panel
    for measure, color_range in zip(['participation', 'module_degree', 'strength'], [(0, 1), (-2.9, 2.9), (.5, 1.6)]):
        plot_brain(graph_measures[measure], filename=f'supp_fig2_{measure}', label_text=measure,
                color_map='viridis', color_range=color_range, layout='row', size=(1600,300))
    # supp fig 2. left panel
    scatterplot_graph_ecc(graph_measures.join(dfg_mean.loc[EPOCH_REF, 'ecc']), '7')

    # supp fig 3
    for shift in ('b2e', 'e2l'):
        plot_brain(df_ttests.loc[shift], filename=f'supp_fig3_{shift}_not_masked', 
                color_map=COLOR_MAP_T, color_range=COLOR_RANGE_T)
    
    # supp fig 4
    plot_brain(df_ttests.loc['b2l'], p='p_corr', filename=f'supp_fig4',
            outline=True, color_map=COLOR_MAP_T, color_range=COLOR_RANGE_T)
    plot_radar_shifts(df_ttests, '7', 'supp_fig4_radar', pairs=['b2l'], order=None)
    
    # supp fig 5
    plot_seed_conn_shift(right_hemi_seeds, df_seeds_shift)
    for s in right_hemi_seeds:   plot_radar_shifts(df_seeds_shift_pivot.loc['7Networks_'+s], '17', 'supp_fig5_'+s, order=OLD_ORDER_17)
    
    # supp fig 6
