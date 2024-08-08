
def calculate_soilmoisture_diag(month):
    import heat as ht
    import numpy as np
    import sys
    import os
    import glob
    import argparse
    import netCDF4 as nc
    import datetime

    import sloth.diagnostics
    import sloth.IO
    import sloth.analysis

    '''
    python VolumetricSoilMoisture_test_separate.py --satur=/p/scratch/cjjsc39/zhang36/ERA5eval/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline_data/tar_file/1997120100/parflow/*satur* 
    --saturVarName=saturation --porosityFile=/p/scratch/cjjsc39/zhang36/ERA5eval/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline_data/static_var/porosity/porosity.nc 
    --LLSMFile=/p/scratch/cjjsc39/zhang36/ERA5eval/TSMP_EUR-11/static/land-lake-sea-mask/EUR-11_TSMP_FZJ-IBG3_444x432_LAND-LAKE-SEA-MASK.nc 
    --outDir=/p/scratch/cjjsc39/zhang36/ERA5eval/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline_data/tar_file/1997120100/parflow 
    --griddesFile=/p/scratch/cjjsc39/zhang36/ERA5eval/TSMP_EUR-11/static/grids/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_griddes.txt

    ''' 

    # https://github.com/HPSCTerrSys/TSMP_WorkflowGettingStarted/blob/main/ctrl/postpro/calcParFlowDiagnosticVars.py
    # https://github.com/HPSCTerrSys/ParFlowDiagnostics/blob/master/Diagnostics.py

    ################################################################################
    # Handel commandline arguments
    ################################################################################

    #parser = argparse.ArgumentParser(description='Tell me what this script can do!.')
    #parser.add_argument('--satur', type=str, required=True, 
    #                    help='Full path to ParFlow satur files (pass pattern)') 
    #parser.add_argument('--saturVarName', type=str, required=True,
    #                    help='Name of satur variable in satur files') # should be always saturation
    #parser.add_argument('--porosityFile', type=str, required=True,
    #                    help='Full path to file holding ParFlow porosity values')
    #parser.add_argument('--LLSMFile', type=str, required=True,
    #                    help='Full path to file holding domain LLSM')
    #parser.add_argument('--outDir', type=str, required=True,
    #                    help='Full path to dir where to store data')
    #parser.add_argument('--griddesFile', type=str, required=True,
    #                    help='Full path to file holding griddefinition for used grid')

    #args = parser.parse_args()
    #saturFilesPattern = args.satur
    saturFilesPattern = f"/p/scratch/cslts/miaari1/detect/raw/{month}/*satur*"
    #porosityFile      = args.porosityFile
    porosityFile = "/p/scratch/cslts/miaari1/detect/static/porosity.nc"
    #LLSMFile             = args.LLSMFile
    LLSMFile = "/p/scratch/cslts/miaari1/detect/static/EUR-11_TSMP_FZJ-IBG3_444x432_LAND-LAKE-SEA-MASK.nc"
    #griddesFile          = args.griddesFile # should be the txt file
    griddesFile = "/p/scratch/cslts/miaari1/detect/static/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_griddes.txt"
    #outDir               = args.outDir
    outDir = "/p/scratch/cslts/miaari1/detect/soilmoisture/"
    #saturVarName      = args.saturVarName
    saturVarName = "saturation"

    # Get environment variables
    author_name      = os.getenv('AUTHOR_NAME')
    author_mail      = os.getenv('AUTHOR_MAIL')
    author_institute = os.getenv('AUTHOR_INSTITUTE')

    globAttrs = dict(author=os.getenv('AUTHOR_NAME', 'unset'),
                email=os.getenv('AUTHOR_MAIL', 'unset'),
                institute=os.getenv('AUTHOR_INSTITUTE', 'unset'),
                EXPID=os.getenv('EXPID', 'unset'),
                ENSMname=os.getenv('ENSMname', 'unset'))

    with nc.Dataset(porosityFile, 'r') as nc_file:
        porosity = nc_file.variables['porosity'][0,...]

    with nc.Dataset(LLSMFile, 'r') as nc_file:
        LLSM = nc_file.variables['LLSM'][0,...]

    # each script calculate for all the nc file in one folder in one month -- YOU CAN CUSTIMIZE TO YOUR DATASET
    saturFiles   = sorted(glob.glob(saturFilesPattern)) # Full path to ParFlow pressure files for each month (in my case)

    # load porosity data as heat
    split=None
    porosity = ht.array(porosity, split=split)

    # define the heat diagnostic
    diag = sloth.diagnostics.Diagnostics.Diagnostics(Mask=None, 
            Perm=None, Poro=porosity, Sstorage=None,
            Ssat=None, Sres=None, Nvg=None, Alpha=None,
            Mannings=None, Slopex=None, Slopey=None,
            Dx=None, Dy=None, Dz=None, Dzmult=None, 
            Nx=None, Ny=None, Nz=None,
            Terrainfollowing=True, Split=split)

    # calculate the volumetric soil moisture using diagnosis:

    for count, saturFile in enumerate(saturFiles): # the path for a list of saturation nc files
        print(f"month: {month}, day: {count}")
        with nc.Dataset(saturFile, 'r') as nc_file:
                time_max = nc_file['time'].shape[0]
                volsm = []
                times=[]
                outPrepandName       = saturFile[-34:-14]
            
                for t in range(time_max): # loop for all time steps in one nc file
                        satur        = nc_file.variables[saturVarName][t,...]

                        # save the time variable for outputing the nc file
                        nc_time      = nc_file.variables['time']
                        nc_times     = nc_time[t]
                        times.append(nc_times)
                        nc_calendar  = nc_time.calendar
                        
                        nc_timeUnits = nc_time.units

                        #print(f'-- file count: {count} -- t={t} (of {time_max}); satur.shape: {satur.shape}')
                        # convert press to ht-array as we deal with DIagnostcs.py
                        satur = ht.array(satur)

                        temp_volsm = diag.VolumetricMoisture(satur)
                        # collect the output at each time step
                        volsm.append(temp_volsm)

                # save the output to a nc file -- we output the nc file one by one otherwise crashed by the memory problem
                

                if count==0 or (t==0 and time_max==1): # in this case the stack cannot be used
                    volsm       = temp_volsm.numpy()

                    llsm_b           = np.broadcast_to(LLSM, volsm.shape)
                    volsm              = np.ma.masked_where((llsm_b < 2), volsm)
                    volsm              = volsm.reshape(1,15, 432, 444)
                    times       = nc_times
                    #volsm       = volsm.numpy()
                
                else:
                    volsm_ht    = ht.stack(volsm, axis=0)
                    #volsm_ht    = ht.array(volsm)
                    volsm       = volsm_ht.numpy()
                    #volsm       = volsm.numpy()
                
                    del volsm_ht

                    llsm_b           = np.broadcast_to(LLSM, volsm.shape)
                    volsm              = np.ma.masked_where((llsm_b < 2), volsm)

                # mask the calculated volsm into each
                

                times        = np.array(times)
                timeUnit     = nc_timeUnits
                timeCalendar = nc_calendar

                ###############################################################################
                #### volsm output as nc file
                ###############################################################################
                saveFile = f'{outDir}/{outPrepandName}_volsm.nc' # HERE YOU CAN RENAME THE OUTPUT AS YOU LIKE
                if not os.path.exists(f'{outDir}'):
                    os.makedirs(f'{outDir}')


                # add annotation inside the nc file
                description_str = [f'volumetric soil moisture',]
                description_str = ' '.join(description_str)

                # initalize the nc file based on Niklas's sloth library
                netCDFFileName = sloth.IO.createNetCDF(saveFile, domain=griddesFile,
                        nz=15, calcLatLon=True, timeUnit=timeUnit, timeCalendar=timeCalendar,
                        author=author_name, contact=author_mail,
                        institution=author_institute, 
                        history=f'Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
                        description=description_str)
                # nz should be TRUE so the function will create the z layer -- 15 layers

                # load to the nc file
                with nc.Dataset(netCDFFileName, 'a') as nc_file:
                    
                    ncVar = nc_file.createVariable('volsm', 'f8', ('time', 'lvl', 'rlat', 'rlon'),
                                                        fill_value=-9999, zlib=True)
                    ncVar.standard_name = 'volsm'
                    ncVar.long_name = 'volumetric soil moisture'
                    ncVar.units ='%'
                    ncVar.grid_mapping = 'rotated_pole'
                    
                    ncVar[...] = volsm[...]
                    ncTime = nc_file.variables['time']
                    ncTime[...] = times[...]