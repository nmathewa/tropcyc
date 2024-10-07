
from datetime import datetime , timedelta , timezone
import os
import pandas as pd
import pytz
import subprocess
import numpy as np

class nucast_downloader:

    def __init__(self):
        self.today_utc = datetime.now(timezone.utc)
        self.today = datetime.now()
        self.utc_adjust = self.today_utc - pd.Timedelta(hours=4) # it takes 4 hours to update the model data


    def get_init_time(self):
        """
        Get the initial time of the model run
        """
        utc_adjust = self.utc_adjust
        if 0 <= utc_adjust.hour < 6:
            init_time = '00'
        elif 6 <= utc_adjust.hour < 12:
            init_time = '06'
        elif 12 <= utc_adjust.hour < 18:
            init_time = '12'
        else:
            init_time = '18'
  
        return init_time 
    
    def create_hwrf_dft(self,name,lead,local_zone):
        """
        name of event (eg.14l)
        lead time in hours (multiple of 3) if 24 hours, lead = 24 (steps of 3 hours)= total 8 steps
        local_zone (eg. 'atlantic')
        """
        steps = int(lead/3)
        zone = pytz.timezone(local_zone)


        len_lead = int(lead/3)
        dft = pd.DataFrame(data=[name]*steps,columns=['name'])
        dt_prefix = self.today_utc.strftime('%Y%m%d')
        init_time = self.get_init_time()
        dt_final = dt_prefix + init_time
        dft['dt'] = dt_final
        dft['forecast_hour'] = np.arange(0,lead,3)
        dft['prefix'] = 'hfsb.storm.atm'
        dft['utc_time'] = pd.to_datetime(dft['dt'],format='%Y%m%d%H') + pd.to_timedelta(dft['forecast_hour'],unit='h')
        dft['local_time'] = dft['utc_time'].dt.tz_localize('UTC').dt.tz_convert(zone)

        stats_file = f'{name}.{dt_final}.hfsb.grib.stats.short'
        dft['filename'] = dft['name'] + '.' + dft['dt'] + '.' +dft['prefix'] + '.f' + dft['forecast_hour'].astype(str).str.zfill(3) + '.grb2'
        # add a new row for the stats file in the column filename (last row) add everything else as the same
        dft.loc[len(dft)] = [name,dt_final,lead,'hfsb.storm.atm',dft['utc_time'].iloc[-1],dft['local_time'].iloc[-1],stats_file]
        return dft


        
    def download_hwrf(self,outfol):
        """
        Download the HWRF data
        """
        get_hwrf_dft = self.create_hwrf_dft('14l',24,'Etc/GMT+5')

        name = get_hwrf_dft['name'][0]
        prefix_time1 = get_hwrf_dft['dt'][0][:-2]
        prefix_time2 = get_hwrf_dft['dt'][0][-2:]
        fol_name = name + prefix_time1 + prefix_time2
        fol_path = os.path.join(outfol,fol_name)
        os.makedirs(fol_path,exist_ok=True)

        

        for ii in range(len(get_hwrf_dft)):
            filename = get_hwrf_dft['filename'][ii]
            url_dir = f's3://noaa-nws-hafs-pds/hfsb/{prefix_time1}/{prefix_time2}/{filename}'
            print(url_dir)
            subprocess.run(['aws','s3','cp', '--no-sign-request' ,url_dir,fol_path],check=True)            
            #subprocess.run(['aws','s3','cp', '--no-sign-request' ,f's3://noaa-nws-hafs-pds/hfsb/20241006/12/{dft.loc[i,"filename"]}',outfol+fol_name],check=True)


        return get_hwrf_dft
        
        