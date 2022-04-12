import os
import io
import PyIF
import tqdm
import json
import fiona
import datetime
import requests
import urllib.request

import numpy as np
import xarray as xr
import pandas as pd
import rsfuncs as rs
import rasterio as rio
import geopandas as gp
import seaborn as sns
import multiprocessing as mp

# import cartopy.crs as ccrs
# import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
from sklearn import metrics
from scipy import stats, spatial
from affine import Affine
from itertools import compress
from datetime import timedelta
from rasterio import features, mask
from climata.usgs import DailyValueIO
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta

# from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests, pacf
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.api import VAR
# from statsmodels.stats.stattools import durbin_watson
# from statsmodels.tsa.vector_ar.vecm import coint_johansen

import warnings
warnings.filterwarnings('ignore')



# # Relevant Equations (and refs)
# 
# #### Cross Correlation
# 
# of two signals is defined: 
#  $[f*g](t) = \sum_{i=1}^{n} f(t) g(t-\tau)$
# 
# the characteristic time $\tau_{lag}$ can be computed: 
# 
#  $\tau_{lag} = argmax|[f*g](t)|$
# 
# #### Entropy (Shannon, 1948): 
# 
# Given a discrete random variable $X$, with possible outcomes $ x_{1},...,x_{n} $ which occur with probability $  \mathrm {P} (x_{1}),...,\mathrm {P} (x_{n}) $ the entropy (units of nats) of $X$ is defined as: <br>
# 
# $ H(X) =  - \sum_{i=1}^{n} P(x) \ln P(x) $
# 
# #### Joint Entropy:
# of two discrete random variables $X$ and $Y$ is defined as the entropy of the joint distribution of $X$ and $Y$:
# 
# $ H(X,Y) =  - \sum_{i=1}^{n} P(x,y) \ln P(x,y) $
# 
# 
# #### Conditional Entropy: 
# 
# The amount of information needed to describe the outcome of a random variable $Y$ given that the value of another random variable $X$ is known. Here, information is measured in shannons, nats, or hartleys. The entropy of $Y$ conditioned on $X$ is:
# 
# $ H (Y|X) = -\sum p(x,y)\ln {\frac {p(x,y)}{p(x)}} $
# 
# 
# #### Relative Entropy, aka K-L Divergence,
# 
# The Relative Entropy (aka K-L divergence, $ D_{\text{KL}}(P\parallel Q)$ ), which measures how one probability distribution $P(x)$ is different from a second $Q(x)$ is defined as:
# 
# $ D_{\text{KL}}(P\parallel Q)=\sum _{x\in {\mathcal {X}}}P(x)\ln \left({\frac {P(x)}{Q(x)}}\right)$
# 
# #### Jensen Shannon Distance:
# 
# The Jensen Shannon Distance (JSD) also measures how one probability distribution $P(x)$ is different from a second $Q(x)$, but has desirable properties of always being finite and symmetric: 
# 
# $ JSD(X) = \sqrt{\frac{D(p \parallel m) + D(q \parallel m)}{2}}\$
# 
# where $D(x \parallel y)$ is the K-L Divergence, defined above.
# 
# 
# #### Mutual information
# 
# measures how much information can be obtained about one random variable by observing another. The mutual information of $X$ relative to $Y$ (which represents conceptually the average amount of information about $X$ that can be gained by observing $Y$ is given by:
# 
# $ I(X; Y)=H(X)− H(X|Y)= -\sum p(x,y)\ln \frac{p(x,y)}{p(x) p(y)} $
# 
# #### Transfer entropy (Schreiber, 2000)
# 
# is the amount of directed (time-asymmetric) transfer of information between two random processes. Transfer entropy from a process X to another process Y is the amount of uncertainty reduced in future values of Y by knowing the past values of X given past values of Y.
# 
# $ T_{X→Y} = \sum p(y_{t+1}, y_{t}, x_{t}) ln( \frac{p(y_{t+1} | y_{t} , x_{t})} {p(y_{t+1} | y_{t})}) $
# 
# Can be thought of as the deviation from independence
# (in bits) of the state transition (from the previous state
# to the next state) of an information destination X from
# the (previous) state of an information source Y
# 
# Transfer entropy can be thought of as Conditional mutual Information (Lizier, 2008): 
# 
# $ T_{X→Y} = I(X ; Y{t+1}|Y) = H(Y_{t+1}|Y) − H(Y_{t+1}|Y,X) $ 
# 
# #### References
# 
# Shannon, C. E. (1948). A mathematical theory of communication. The Bell system technical journal, 27(3), 379-423.
# 
# Schreiber, T. (2000). Measuring information transfer. Physical review letters, 85(2), 461.
# 
# Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2008). Local information transfer as a spatiotemporal filter for complex systems. Physical Review E, 77(2), 026110.


# Functions

def get_snodas_swe(shppath,startdate,enddate, data_dir ="/Users/aakash/Desktop/SatDat/SNODAS/SNODAS_CA_processed/" ):
    '''
    Given a path to a shapefile, compute the monthly SWE
    Input: (str) - path to shapefile
    Output: (pd.DataFrame) - daily SWE 
    '''
    # Datetime the start/end
    start = datetime.datetime.strptime(startdate, "%Y-%m-%d")
    end = datetime.datetime.strptime(enddate, "%Y-%m-%d")
    dt_idx = pd.date_range(start,end, freq='D')
    
    # Find SWE files
    files = [os.path.join(data_dir,x) for x in os.listdir(data_dir) if x.endswith(".tif")]
    files.sort()

    # Read shapefile
    with fiona.open(shppath, "r") as shapefile:
        area_geom = [feature["geometry"] for feature in shapefile]

    # Read the files, mask nans, clip to area, extract dates
    imdict = {}

    for i in tqdm(files[:]):
        date = datetime.datetime.strptime(i[-16:-8],'%Y%m%d')# Get the date 
        datestr = date.strftime('%Y%m%d') # Format date
        if date >= start and date <= end:
            src = rio.open(i) # Read file
            src2 = rio.mask.mask(src, area_geom, crop=True) # Clip to shp 
            arr = src2[0].astype(float) # read as array
            arr = arr.reshape(arr.shape[1], arr.shape[2]) # Reshape bc rasterio has a different dim ordering 
            arr[arr < 0 ] = np.nan # Mask nodata vals 
            imdict[datestr] = arr/1000 # divide by scale factor to get SWE in m 
    
    all_dates = {}

    for i in dt_idx:
        date = i.strftime("%Y%m%d") 

        if date in imdict.keys():
            im = imdict[date]
        else:
            im = np.zeros_like(list(imdict.values())[0])
            im[im==0] = np.nan
        all_dates[date] = im
    
    return all_dates
            
def get_res_data(stid, startdate, enddate, freq = 'D', var = 'inflow'):
    '''
    Fetch CDEV reservoir data via api 
    Input Params: 
        stid (str) - 3 letter station id (ex: ISB)
        startdate - 'YYYY-mm-dd'
        startdate - 'YYYY-mm-dd'
        freq - "D" = Day, "M" = Month
    
    Output: inflow - (pd.DataFrame)
    '''
    varlookup = {
                    'storage':'65',
                    'inflow':'76'}
    
    # build the api query from params
    query = '''https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums={}&dur_code={}&Start={}&End={}'''.format(stid,varlookup[var],freq,startdate,enddate)
    print(query)
    # Read, extract relevant cols as float and datetime, return df
    dat = pd.read_csv(query)
    vals = pd.to_numeric(dat['VALUE'], errors = 'coerce')
    dt = pd.to_datetime(dat['DATE TIME'])
    indf = pd.DataFrame(zip(dt,vals* 0.0283168), columns = ['date',"q_cms"]) # cfs --> CMS 
    
    return indf

def normalize(x):
    return(x-np.nanmin(x))/(np.nanmax(x)- np.nanmin(x))


def dict2im(outdict, empty_im, rclist=None):
    outim = np.zeros_like(empty_im)
    outdf = pd.DataFrame.from_dict(outdict)

    # Populate the per-pixel entropy 
    for rc, dfcolidx in zip(rclist,outdf.columns):

        rowidx, colidx = rc
        val = outdf[dfcolidx].values[0]
        outim[rowidx,colidx] = val
        
    return outim

def calc_nbins(N):
    
    '''
    A. Hacine-Gharbi, P. Ravier, "Low bias histogram-based estimation of mutual information for feature selection", Pattern Recognit. Lett (2012).
    '''
    ee = np.cbrt(8 + 324*N + 12*np.sqrt(36*N + 729*N**2))
    bins = np.round(ee/6 + 2/(3*ee) + 1/3)

    return int(bins)

def split_before_after(imstack, beforeidx,afteridx):
    beforeim = imstack[:,:,:beforeidx]
    afterim = imstack[:,:,-afteridx:]
    return beforeim, afterim

def mask_unpack(imlist, meanim):
    im1, im2, im3 = [np.ma.masked_array(x, mask=np.isnan(meanim)) for x in imlist]
    return im1, im2, im3

def df_shifted(df, target=None, lag=0):
    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag)
    return  pd.DataFrame(data=new)

def calc_info(imstack, inflow):
    
    # Build the out image
    te_im = np.zeros_like(np.mean(imstack, axis = 2))
    mi_im = np.zeros_like(np.mean(imstack, axis = 2))
    js_im = np.zeros_like(np.mean(imstack, axis = 2))

    rows, cols, time = imstack.shape
    px_ts = []
    rclist = []

    # extract pixelwise timeseries
    for row in range(rows):
        for col in range(cols):
            ts_arr = imstack[row,col,:]
            if not np.isnan(ts_arr).all():
                px_ts.append(pd.Series(ts_arr))
                rclist.append([row,col])

    if len(px_ts) == 0:
        return te_im, js_im, mi_im

    pxdf = pd.concat(px_ts, axis = 1)
    pxdf.columns = pxdf.columns.map(str)

    # Populate the per-pixel lags 
    for rc, dfcolidx in zip(rclist,pxdf.columns):

        a=np.ma.masked_invalid(pxdf[dfcolidx])
        b=np.ma.masked_invalid(inflow)

        msk = (~a.mask & ~b.mask)

        tempdf = pd.DataFrame([a[msk],b[msk]]).T
        tempdf.columns = ['var','q_cms']
        
        # get n bins
        nbins = calc_nbins(len(tempdf))

        # compute info theory stuffs
        try: 
            mi = metrics.mutual_info_score(tempdf['var'].value_counts(normalize=True,bins = nbins),tempdf['q_cms'].value_counts(normalize=True,bins = nbins))
        except:
            mi = np.nan
        try:
            js_dist = spatial.distance.jensenshannon(tempdf['var'].value_counts(normalize=True,bins = nbins),tempdf['q_cms'].value_counts(normalize=True,bins = nbins))
        except:
            js_dist = np.nan
        try:
            TE = PyIF.te_compute.te_compute(np.array(tempdf['var'].values),np.array(tempdf['q_cms'].values))
        except:
            TE = np.nan

        # fill ims
        rowidx, colidx = rc
        te_im[rowidx,colidx] = TE
        js_im[rowidx,colidx] = js_dist
        mi_im[rowidx,colidx] = mi

    return te_im, js_im, mi_im

def calc_mi(imstack, inflow):
    
    # Build the out image
    mi_im = np.zeros_like(np.mean(imstack, axis = 2))

    rows, cols, time = imstack.shape
    px_ts = []
    rclist = []

    # extract pixelwise timeseries
    for row in range(rows):
        for col in range(cols):
            ts_arr = imstack[row,col,:]
            if not np.isnan(ts_arr).all():
                px_ts.append(pd.Series(ts_arr))
                rclist.append([row,col])

    if len(px_ts) == 0:
        return te_im, js_im, mi_im

    pxdf = pd.concat(px_ts, axis = 1)
    pxdf.columns = pxdf.columns.map(str)

    # Populate the per-pixel lags 
    for rc, dfcolidx in zip(rclist,pxdf.columns):

        a=np.ma.masked_invalid(pxdf[dfcolidx])
        b=np.ma.masked_invalid(inflow)

        msk = (~a.mask & ~b.mask)

        tempdf = pd.DataFrame([a[msk],b[msk]]).T
        tempdf.columns = ['var','q_cms']
        
        # get n bins
        nbins = calc_nbins(len(tempdf))

        # compute info theory stuffs
        try: 
            mi = metrics.mutual_info_score(tempdf['var'].value_counts(normalize=True,bins = nbins),tempdf['q_cms'].value_counts(normalize=True,bins = nbins))
        except:
            mi = np.nan
       
        # fill ims
        rowidx, colidx = rc
        mi_im[rowidx,colidx] = mi

    return mi_im

def cross_correlation_using_fft(x, y):
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    return np.fft.fftshift(cc)

def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def calc_xcorr_fft(imstack, qarr):
    rows, cols, time = imstack.shape
    px_ts = []
    rclist = []

    # extract pixelwise timeseries
    for row in range(rows):
        for col in range(cols):
            ts_arr = imstack[row,col,:]

            if not np.isnan(ts_arr).all():
                px_ts.append(pd.Series(ts_arr))
                rclist.append([row,col])
            else:
                px_ts.append(pd.Series(np.zeros_like(ts_arr)))
                rclist.append([row,col])

    pxdf = pd.concat(px_ts, axis = 1)
    pxdf.columns = pxdf.columns.map(str)

    # Build the out image
    lagim = np.zeros_like(np.mean(imstack, axis = 2))
    corrim = np.zeros_like(np.mean(imstack, axis = 2))

    # Populate the per-pixel lags 
    for rc, dfcolidx in zip(rclist,pxdf.columns):

        a=np.ma.masked_invalid(pxdf[dfcolidx])
        b=np.ma.masked_invalid(qarr)
        
        msk = (~a.mask & ~b.mask)

        # compute shift n mag, if <0, shift by 45d (midpoint of 90d time window)
        try:
            shiftval = compute_shift(a[msk],b[msk])
            if shiftval <0:
                shiftval = 45 - abs(shiftval)
            elif shiftval==0:
                shiftval = np.nan
            corrmat = np.ma.corrcoef(a[msk],b[msk])
            corr = np.nanmean(corrmat[np.where(~np.eye(corrmat.data.shape[0], dtype=bool))].data)
        except:
            shiftval, corr = np.nan, np.nan
        
        # fill ims
        rowidx, colidx = rc
        lagim[rowidx,colidx] = shiftval
        corrim[rowidx,colidx] = corr

    return lagim, corrim

def calc_mi_ev_means(events,imstack, shed_ts):
    '''
    wrapper for calc_info
    '''
    mi_ims = []

    dt_idx = pd.date_range('2003-09-30','2020-10-01', freq='D')
    
    for t1 in tqdm(events):
        timespan = t1 + relativedelta(months = 3) #  timespan after event
        window = (dt_idx[dt_idx > t1]& dt_idx[dt_idx <= timespan])

        # Copy the df for indices to filter the array
        ts = shed_ts.copy()

        ts['dt'] = ts.index
        ts.reset_index(inplace = True)
        start = ts[ts.dt == window[0]].index
        end = ts[ts.dt == window[-1]].index

        s, e = int(start.values), int(end.values)

        # Call infotheory function on imstack and Q
        try:
            mi_im = calc_mi(imstack[:,:,s:e+1],shed_ts.loc[window]['q_cms'].interpolate(how = 'linear'))
            mi_ims.append(np.where(mi_im!=0,mi_im, np.nan))
        except:
            continue

    # Compute event means
    mi_mean = np.nanmean(np.dstack(mi_ims), axis = 2)
 
    return mi_mean

def calc_it_ev_means(events,imstack, shed_ts):
    '''
    wrapper for calc_info
    '''
    mi_ims = []
    js_ims = []
    te_ims = []

    dt_idx = pd.date_range('2003-09-30','2020-10-01', freq='D')
    
    for t1 in tqdm(events):
        timespan = t1 + relativedelta(months = 3) #  timespan after event
        window = (dt_idx[dt_idx > t1]& dt_idx[dt_idx <= timespan])

        # Copy the df for indices to filter the array
        ts = shed_ts.copy()

        ts['dt'] = ts.index
        ts.reset_index(inplace = True)
        start = ts[ts.dt == window[0]].index
        end = ts[ts.dt == window[-1]].index

        s, e = int(start.values), int(end.values)

        # Call infotheory function on imstack and Q
        try:
            te_im, js_im, mi_im = calc_info(imstack[:,:,s:e+1],shed_ts.loc[window]['q_cms'].interpolate(how = 'linear'))
            js_ims.append(np.where(js_im!=0,js_im, np.nan))
            mi_ims.append(np.where(mi_im!=0,mi_im, np.nan))
            te_ims.append(np.where(te_im!=0,te_im, np.nan))
        except:
            continue

    # Compute event means
    mi_mean = np.nanmean(np.dstack(mi_ims), axis = 2)
    js_mean = np.nanmean(np.dstack(js_ims), axis = 2)
    te_mean = np.nanmean(np.dstack(te_ims), axis = 2)
    
    return mi_mean, js_mean, te_mean

def calc_xc_ev_means(events,imstack, shed_ts):
    '''
    wrapper for calc_xcorr_fft
    '''
    cor_ims = []
    lag_ims = []

    dt_idx = pd.date_range('2003-09-30','2020-10-01', freq='D')
    
    for t1 in tqdm(events):
        timespan = t1 + relativedelta(months = 3) #  timespan after event
        window = (dt_idx[dt_idx > t1]& dt_idx[dt_idx <= timespan])

        # Copy the df for indices to filter the array
        ts = shed_ts.copy()

        ts['dt'] = ts.index
        ts.reset_index(inplace = True)
        start = ts[ts.dt == window[0]].index
        end = ts[ts.dt == window[-1]].index

        s, e = int(start.values), int(end.values)

        # Call xc function on imstack and Q
        try:
            lag_im, cor_im = calc_xcorr_fft(imstack[:,:,s:e+1],shed_ts.loc[window]['q_cms'].interpolate(how = 'linear'))
            lag_ims.append(np.where(cor_im!=0,lag_im, np.nan))
            cor_ims.append(np.where(cor_im!=0,cor_im, np.nan))
        except:
            continue

    # Compute event means
    cor_mean = np.nanmean(np.dstack(cor_ims), axis = 2)
    lag_mean = np.where(cor_mean != np.nan, np.nanmean(np.dstack(lag_ims), axis = 2), np.nan)
    
    return cor_mean, lag_mean


def write_extreme_rasters(seasonal_dict, stn_id,outdir, var = None):

    catch_shp = "../shape/{}.shp".format(stn_id)

    if var is None:
        print("supply variable")
        return

    for k,v in seasonal_dict.items():
        st_se = stn_id + "_" + var + "_" + k + ".tiff"
        outfn = os.path.join(outdir, st_se)
        print("writing " + outfn)
        rs.write_raster(v, gp.read_file(catch_shp), outfn)

def check_if_outfns_ext(stn_id):

    seasonal_res_dir = "../rasters/ext_TEST"

    outfns = []
    for seas in ['W','Sp','Su','F']:        
        st_ses = [stn_id + "_" + var + "_" + seas + ".tiff" for var in ['pmi','pjsd','pte','pcor','plag','dmi','dlag','dcor','dte','djsd']]
        outfns.append([os.path.join(seasonal_res_dir, x) for x in st_ses])

    outfns_all =  [item for sublist in outfns for item in sublist]

    if all([os.path.isfile(f) for f in outfns_all]):
        print(" {} OUT FILES EXIST, SKIPPING ============== ".format(stn_id))
        return False
    else:
        return True



def main(stn_id):

    print("=======" * 15)
    print("PROCESSING: {}".format(stn_id))

    # Set start / end date
    dt_idx = pd.date_range('2003-09-30','2020-10-01', freq='D')

    ###### DEFINE RESULTS DIR
    ext_res_dir = "../rasters/ext_TEST"
    print("****" * 15)
    print("OUTDIR = {}".format(ext_res_dir))
    print("****" * 15)
    if not os.path.exists(ext_res_dir):
        os.mkdir(ext_res_dir)

    # Read catchment shapefile 
    catch_shp = "../shape/{}.shp".format(stn_id)

    # Set filepaths for hydro data 
    resfn = os.path.join('../data/res_inflow/{}_res.csv'.format(stn_id))
    swestack_fn = os.path.join('../data/swe/{}_swe.npy'.format(stn_id))
    meltstack_fn = os.path.join('../data/smlt/{}_smlt.npy'.format(stn_id))
    pstack_fn = os.path.join('../data/plqd/{}_plqd.npy'.format(stn_id))
    
    # Get the daily reservoir inflow
    if not os.path.exists(resfn):
        try:
            inflow = rs.col_to_dt(get_res_data(stn_id, startdate, enddate))
            inflow.to_csv(resfn)
        except:
            print("no data for {}".format(stn_id))
            pass
    else:
        inflow = rs.col_to_dt(pd.read_csv(resfn))
        if len(inflow) == 0:
            print("no data for {}".format(stn_id))
            pass

    print('convolving arrays')

    # Get the SWE
    swestack = np.load(swestack_fn)
    swestack = swestack / 1000. # apply scaling factor 
    swevals = [swestack[:,:,t] for t in range(0, swestack.shape[2])]

    # Load dSWE
    dswe_unfilt = np.load(meltstack_fn)
    dswe = dswe_unfilt[:,:,-swestack.shape[2]:] # chop off the many trailing nans? 
    dswe = dswe / 100000. # apply scaling factor 
    dswevals = [dswe[:,:,t] for t in range(0, dswe.shape[2])]

    # compute hte rolling 3 and 5 day sums via convolution 
    conv_win_3d = np.ones(3)
    dswe_rolling_3d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_3d, mode='full'), axis=2, arr=dswe)
    dswe_3d = dswe_rolling_3d[:,:,2:]
    dswevals_3d = [dswe_3d[:,:,t] for t in range(0, dswe_3d.shape[2])]

    print('done 3d dswe ==============================')
    conv_win_5d = np.ones(5)
    dswe_rolling_5d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_5d, mode='full'), axis=2, arr=dswe)
    dswe_5d = dswe_rolling_5d[:,:,4:]
    dswevals_5d = [dswe_5d[:,:,t] for t in range(0, dswe_5d.shape[2])]

    print('done 5d dswe ==============================')
    # Load Precip
    pstack_unfilt = np.load(pstack_fn)
    pstack = pstack_unfilt[:,:,-swestack.shape[2]:] # chop off the many trailing nans? 
    pstack = pstack / 10. # apply scaling factor 
    pvals = [pstack[:,:,t] for t in range(0, pstack.shape[2])]

    # compute hte rolling 3 and 5 day sums via convolution 
    p_rolling_3d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_3d, mode='full'), axis=2, arr=pstack)
    p_3d = p_rolling_3d[:,:,:-2]
    pvals_3d = [p_3d[:,:,t] for t in range(0, p_3d.shape[2])]

    print('done 3d prcp ==============================')

    p_rolling_5d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_5d, mode='full'), axis=2, arr=pstack)
    p_5d = p_rolling_5d[:,:,:-4]
    pvals_5d = [p_5d[:,:,t] for t in range(0, p_5d.shape[2])]


    # Make a watershed mean df
    shed_ts  = inflow.loc[dt_idx[0]:dt_idx[-1]]
    swevals = np.array([np.nansum(x) for x in [swestack[:,:,t] for t in range(0, swestack.shape[2])]])
    dswevals = np.array([np.nansum(x) for x in [dswe[:,:,t] for t in range(0, dswe.shape[2])]])
    pvals = np.array([np.nansum(x) for x in [pstack[:,:,t] for t in range(0, pstack.shape[2])]])

    shed_ts['prcp_1d'] = pvals
    shed_ts['dswe_1d'] = dswevals
    
    # calc watershed avg rolling sums
    shed_ts['prcp_3d'] = shed_ts['prcp_1d'].rolling(3).sum()
    shed_ts['dswe_3d'] = shed_ts['dswe_1d'].rolling(3).sum()

    shed_ts['prcp_5d'] = shed_ts['prcp_1d'].rolling(5).sum()
    shed_ts['dswe_5d'] = shed_ts['dswe_1d'].rolling(5).sum()
        
    ######## Preprocess and EDA #########
    dt_idx = pd.date_range('2003-09-30','2020-10-01', freq='D')

    # Get annual mean ims 
    pmean, swemean, dswemean = [np.nanmean(x, axis = 2)*365 for x in [pstack, swestack,dswe]]
    
    # Compute annual means
    annual_pmean = pmean 
    annual_swemean = swemean
    annual_dswemean = dswemean 
        
    # Get annual mean ims 
    pmean, swemean, dswemean = [np.nanmean(x, axis = 2)*365 for x in [pstack, swestack,dswe]]
    pmean[pmean == 0] = np.nan


    ########### Main Routine: Info Theory #############

    ############## Extremes of 1,3,5d #############

    # for the extremes (1d, 3d, 5d), find the dates of the events that exceed the 99th %ile, and rank based on value for N events. 
    num_events = 10

    p_ev_1d = shed_ts[shed_ts['prcp_1d']>= np.nanpercentile(shed_ts['prcp_1d'], 99)].sort_values(by = 'prcp_1d', ascending = True).index[:num_events]
    d_ev_1d = shed_ts[shed_ts['dswe_1d']>= np.nanpercentile(shed_ts['dswe_1d'], 99)].sort_values(by = 'dswe_1d', ascending = True).index[:num_events]

    p_ev_3d = shed_ts[shed_ts['prcp_3d']>= np.nanpercentile(shed_ts['prcp_3d'], 99)].sort_values(by = 'prcp_3d', ascending = True).index[:num_events]
    d_ev_3d = shed_ts[shed_ts['dswe_3d']>= np.nanpercentile(shed_ts['dswe_3d'], 99)].sort_values(by = 'dswe_3d', ascending = True).index[:num_events]

    p_ev_5d = shed_ts[shed_ts['prcp_5d']>= np.nanpercentile(shed_ts['prcp_5d'], 99)].sort_values(by = 'prcp_5d', ascending = True).index[:num_events]
    d_ev_5d = shed_ts[shed_ts['dswe_5d']>= np.nanpercentile(shed_ts['dswe_5d'], 99)].sort_values(by = 'dswe_5d', ascending = True).index[:num_events]

    ############# MAIN ROUTINE #############
    imstack_lookup = {
    'prcp_1d': pstack,
    'prcp_3d': p_3d,
    'prcp_5d': p_5d,
    'dswe_1d': dswe,
    'dswe_3d': dswe_3d,
    'dswe_5d': dswe_5d}

    # Zip together the dates we defined by thresholding with the keys to the relevant arrays 
    iterdict = dict(zip(list(imstack_lookup.keys()),[p_ev_1d,p_ev_3d,p_ev_5d,d_ev_1d,d_ev_3d,d_ev_5d]))

    for k,v in iterdict.items():
        dates = v
        stack = imstack_lookup[k]

        # Wrappers for main routines 
        # cor_out, lag_out = calc_xc_ev_means(dates, stack)
        # mi_out = calc_mi_ev_means(dates, stack)

        # Wrappers for main routines 
        mi_out, js_out, te_out = calc_it_ev_means(dates, stack, shed_ts)
        cor_out, lag_out = calc_xc_ev_means(dates, stack, shed_ts)

        print(np.nanmean(lag_out))

        rs.write_raster(te_out, gp.read_file(catch_shp),os.path.join(ext_res_dir,"{}_{}_te_ex.tif".format(stn_id,k)))
        rs.write_raster(mi_out, gp.read_file(catch_shp),os.path.join(ext_res_dir,"{}_{}_mi_ex.tif".format(stn_id,k)))
        rs.write_raster(js_out, gp.read_file(catch_shp),os.path.join(ext_res_dir,"{}_{}_js_ex.tif".format(stn_id,k)))                       
        rs.write_raster(cor_out, gp.read_file(catch_shp),os.path.join(ext_res_dir,"{}_{}_cor_ex.tif".format(stn_id,k)))
        rs.write_raster(lag_out, gp.read_file(catch_shp),os.path.join(ext_res_dir,"{}_{}_lag_ex.tif".format(stn_id,k)))

    return print((" FIN *******")* 5)

if __name__ == "__main__":
    # Read catchments, reservoirs

    gdf = gp.read_file("../shape/sierra_catchments.shp")

    stids_all = list(gdf['stid'].values)
    nodata_stids = ["MCR", "CFW", "NHG"]

    stids = [x for x in stids_all if x not in nodata_stids]

    # Check if otufns already exist 
    checked = [check_if_outfns_ext(x) for x in stids]
    # Filter all and select those that do not exist 
    stids_filt = list(compress(stids, checked))[::-1]

    # Run a single watershed
    # main(gdf, ext_res_dir)

    main("SHA")

    # Parallelize
    # pool = mp.Pool(mp.cpu_count())
    
    # print("Processing {} watersheds".format(str(len(stids_filt))))

    # for i in tqdm(pool.imap_unordered(main, stids_filt), total=len(stids_filt)):
    #     pass
