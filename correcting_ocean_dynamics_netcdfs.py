# %%
import math
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l

# %%
def get_nodename_list(infilename="C:/Users/fanslow/Work/SLR/location_list.lst"):
    # Read in the locations list
    df = pd.read_csv(infilename,sep='\t', header=None,names=['nodename','grid_id','lat','lon'])
    return df

def load_data(infilename=
    'C:/Users/fanslow/Work/SLR/ar6-regional_novlm-confidence/' +
    'medium_confidence/ssp585/oceandynamics_ssp585_medium_confidence_values.nc'
    ):
    # Connor found the RSLR components, let's take a look
    ds = xr.open_dataset(infilename)
    return(ds)
    
def get_quantiles_years(ds):
    return ds['quantiles'].values, ds['years'].values

def get_grid_nodenames(nodenamedf,minlat,maxlat,minlon,maxlon):
    #Filter the data by the locations that we want.
    keepdf = nodenamedf.loc[
        (nodenamedf['lat'] >=minlat) & 
        (nodenamedf['lat'] <= maxlat) & 
        (nodenamedf['nodename'].str.contains('grid')) &
        (nodenamedf['lon'] >= minlon) & 
        (nodenamedf['lon'] <= maxlon)
        ,:]
    keepdf = keepdf.copy()
    return keepdf

def get_lat_lon_sl_from_nodename(ds,locdf,quantile,year):
    
    datadf = pd.DataFrame(
        {
            'lat': ds.sel(locations=ds.locations.isin(locdf['grid_id']),quantiles=quantile,years=year).lat.values,
            'lon': ds.sel(locations=ds.locations.isin(locdf['grid_id']),quantiles=quantile,years=year).lon.values,
            'rsl': ds.sel(locations=ds.locations.isin(locdf['grid_id']),quantiles=quantile,years=year).sea_level_change.values,
        }
    )
    if (all(datadf['lat'].values == locdf['lat'].values))&\
        (all(datadf['lon'].values == locdf['lon'].values)):
        #We have lat and lon alignment, can insert values into the 
        #existing dataframe
        locdf['rsl'] = datadf['rsl'].values
    else:
        print('data misalignment returning Null')
        return Null  
    
    return locdf

def reinsert_sl_from_nodename(ds,datadf,quantile,year):
    """
    Using a dataframe with columns that contain grid ids and rsl data
    Insert that data into the correct grid_id slot. Doing this one by
    one because I am not convinced that mapping the vector with isin
    will work correctly. It's only a few points, so perhaps not a 
    terrible inefficiency.
    
    """
    
    for index, arow in datadf.iterrows():
        if  (not math.isnan(arow['rsl'])):
            ds.sel(locations=arow['grid_id'],quantiles=quantile,years=year)['sea_level_change'] = arow['rsl_predicted']
    return ds
    
    

def interpolate_rsl(rsldf,verbose=False):
    
    #Ditch the nans
    rsldf = rsldf.loc[~(rsldf['rsl'].isna()),]
    
    #Fit an initial model
    gam = LinearGAM(s(0) + s(1)).fit(rsldf[['lon', 'lat']], rsldf['rsl'])
    
    #Do a gridsearch to tune it
    lam = np.logspace(-2, 8, 7)
    lams = [lam] * 2

    gam.gridsearch(rsldf[['lon', 'lat']], rsldf['rsl'], lam=lams, progress=verbose)
    if verbose:
        print(gam.summary())
    return gam

def make_results_plot(datadf,ssp,quantile,year,rsq,rmse,mae,me):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),)#sharex=True,sharey=True)
    vmin = datadf['rsl'].min()
    vmax= datadf['rsl'].max()
    ptsize=100
    scat0 = axes[0,0].scatter(datadf['lon'], datadf['lat'], c=datadf['rsl'], s=ptsize,vmin=vmin,vmax=vmax)
    axes[0,0].set_title(f'Raw Ocean Dynamics SSP {ssp} yr {str(year)} median %ile')
    fig.colorbar(scat0, ax=axes[0,0], label='Raw values (mm)')
    scat1 = axes[0,1].scatter(datadf['lon'], datadf['lat'], c=datadf['rsl_predicted'], s=ptsize,vmin=vmin,vmax=vmax)
    axes[0,1].set_title(f'Interpolated Ocean Dynamics SSP {ssp} yr {str(year)} median %ile')
    fig.colorbar(scat1, ax=axes[0,1], label='Interpolated values (mm)')
    scat2 = axes[1,0].scatter(datadf['lon'], datadf['lat'], c=(datadf['rsl_predicted'] - datadf['rsl']), s=ptsize,vmin=-25,vmax=25)
    axes[1,0].set_title('Raw - Interpolated')
    fig.colorbar(scat2, ax=axes[1,0], label='Raw - Interpolated (mm)')
    axes[1,1].scatter(datadf['rsl'], datadf['rsl_predicted'], c='black',s=ptsize/4)
    axes[1,1].set_title('Raw vs Interpolated scatter plot')
    axes[1,1].set_xlim(vmin,vmax)
    axes[1,1].set_ylim(vmin,vmax)
    font_props = {
        'family': 'monospace',  # e.g., 'serif', 'sans-serif', 'monospace'
        'color': 'black',
        'size': 15,
    }

    axes[1,1].annotate(f' RSQ: {rsq:.2f} mm\nRMSE: {rmse:.2f} mm\n MAE: {mae:.2f} mm\n  ME: {me:.2f} mm',xy=(0.05,0.70),xycoords='axes fraction',fontfamily='monospace',)
    plt.suptitle('Interpolation/Extrapolation with regularized splines GAM.')
    plt.savefig(f'C:/Users/fanslow/Work/SLR/figures/ssp_{ssp}_year_{str(year)}_interpolation_spline_comparison_q_{quantile:.3f}.jpg')
    return fig,axes
    

# %%
if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser('Script to correct the land/sea mask error in the ocean dynamics portion of AR6 sea level projections.')
    parser.add_argument(
        '-f', 
        '--file-name', 
        type=str, 
        default='C:/Users/fanslow/Work/SLR/ar6-regional_novlm-confidence/' +
            'medium_confidence/ssp585/oceandynamics_ssp585_medium_confidence_values.nc', 
        help='Full path filename to the ocean dynamics netCDF to correct.',
    )
    parser.add_argument(
        '-v', 
        '--verbose',
        action='store_true',
        help="Verbose implies printing model progress and statistics to screen and showing figures.",
    )
    parser.add_argument(
        '-s',
        '--stats-path',
        type=str,
        default='C:/Users/fanslow/Work/SLR/',
        help='Base file path for the output statistics.'
    )
    args = parser.parse_args(
#        [
#            '-f','C:/Users/fanslow/Work/SLR/ar6-regional_novlm-confidence/medium_confidence/ssp585/oceandynamics_ssp585_medium_confidence_values.nc',
#            '-s','C:/Users/fanslow/Work/SLR/'    
#        ]
    )
    verbose=args.verbose
    nodenamedf = get_nodename_list()
    ssp = args.file_name.split('_')[-4]
    #print(ssp)
    ds = load_data(args.file_name)
    quantilelist, yearlist = get_quantiles_years(ds)
    year_list = []
    q_list = []
    rsquare_list = []
    rmse_list = []
    mae_list = []
    me_list = []
    for year in yearlist: 
        for quantile in quantilelist:
            
            locdf = get_grid_nodenames(nodenamedf,minlat=42.0, maxlat=51.0,minlon=-74.0, maxlon=-65.0)
            datadf = get_lat_lon_sl_from_nodename(ds,locdf,quantile=quantile,year=year)
            gam = interpolate_rsl(datadf,verbose)
            datadf['rsl_predicted'] = gam.predict(datadf[['lon','lat']])
            ds = reinsert_sl_from_nodename(ds,datadf,quantile=quantile,year=year,)
            datadf_nonan = datadf.loc[~(datadf['rsl'].isna()),].copy()
            datadf_nonan['diff'] = datadf_nonan['rsl_predicted'] - datadf_nonan['rsl']
            MAE = datadf_nonan['diff'].abs().mean()
            SSR = (datadf_nonan['diff']**2).sum()
            SST = ((datadf_nonan['rsl'] - datadf_nonan['rsl'].mean())**2).sum()
            RMSE = np.sqrt((datadf_nonan['diff']**2).mean())
            ME = datadf_nonan['diff'].median()
            RSQ = 1 - SSR/SST
            print(f'Correction complete for quantile {quantile:.3f}! R**2: {RSQ:.3f} RMSE: {RMSE:.2f} MAE: {MAE:.2f} ME: {ME:.2f}')
            fig,axes = make_results_plot(datadf,ssp,quantile,year,RSQ,RMSE,MAE,ME)
            if verbose:
                fig.show()
            plt.close(fig)
            year_list.append(year)
            q_list.append(quantile)
            rsquare_list.append(RSQ)
            rmse_list.append(RMSE)
            mae_list.append(MAE)
            me_list.append(ME)
            
    stats_df = pd.DataFrame(
        {
            'year': year_list,
            'quantile': q_list,
            'rsquare': rsquare_list,
            'rmse': rmse_list,
            'mae': mae_list,
            'me': me_list
        }
    )
    
    #Write out the stats dataframe
    outfilename = args.stats_path + '/' + f'correction_statistics_ssp_{ssp}_{str(year)}_.csv'
    stats_df.to_csv(outfilename,index=False)
    
    #Write out the revised netcdf
    ds.to_netcdf((args.file_name + '_corrercted.nc'))




