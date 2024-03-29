# coding=utf-8
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import telebot.util
import xarray as xr
from __init__ import *
import scipy.io as sio
global_land_tif = join(this_root,'conf/land.tif')
global_start_year = 1982
# global_end_year = 2018
# global_end_year = 2015
global_season_dic = [
    'early',
    'peak',
    'late',
]

vars_info_dic = {
'LAI_3g': {
'path':join(data_root, 'LAI_3g/per_pix'),
'unit': 'm2/m2',
'start_year':1982,
},
# 'SPEI': {
# 'path':join(data_root, 'original_dataset/SPEI3_dic'),
# 'unit': 'SPEI',
# 'start_year':1982,
# },
'Temperature': {
'path':join(data_root, 'original_dataset/temperature_dic'),
'unit': 'Celsius',
'start_year':1982,
},
'Soil moisture': {
'path':join(data_root, 'original_dataset/CCI_SM_2020_dic'),
'unit': 'm3/m3',
'start_year':1982,
},
'Precipitation': {
'path':join(data_root, 'original_dataset/Precip_dic'),
'unit': 'mm',
'start_year':1982,
},
'CO2': {
'path':join(data_root, 'original_dataset/CO2_dic'),
'unit': 'ppm',
'start_year':1982,
},
'Aridity': {
'path':join(data_root, 'original_dataset/aridity_P_PET_dic'),
'unit': 'P/PET(mm/mm)',
'start_year':1982,
},
'VPD': {
'path':join(data_root, 'original_dataset/VPD_dic'),
'unit': 'VPD',
'start_year':1982,
},
'VOD': {
'path':join(data_root, 'VODCA/per_pix_05'),
'unit': 'VPD',
'start_year':1988,
},
'LAI4g_101': {
'path':join(data_root, 'LAI4g_101/per_pix'),
'unit': 'm2/m2',
'start_year':1982,
},
'MODIS_LAI_CMG': {
'path':join(data_root, 'BU_MCD_LAI_CMG/resample_bill_05_monthly_max_compose_per_pix'),
'unit': 'm2/m2',
'start_year':2000,
},
        }

# class HANTS:
#
#     def __init__(self):
#         '''
#         HANTS algorithm for time series smoothing
#         '''
#         pass
#
#     def hants_interpolate(self, values_list, dates_list, valid_range): #
#         '''
#         :param values_list: 2D array of values
#         :param dates_list:  2D array of dates, datetime objects
#         :param valid_range: min and max valid values, (min, max)
#         :return: Dictionary of smoothed values, key: year, value: smoothed value
#         '''
#         year_list = []
#         for date in dates_list:
#             year = date[0].year
#             if year not in year_list:
#                 year_list.append(year)
#         values_list = np.array(values_list)
#         std_list = []
#         for vals in values_list:
#             std = np.std(vals)
#             std_list.append(std)
#         std = np.mean(std_list)
#         std = 2. * std # larger than twice the standard deviation of the input data is rejected
#         interpolated_values_list = []
#         for i,values in enumerate(values_list):
#             dates = dates_list[i]
#             xnew, ynew = self.__interp_values_to_DOY(values, dates)
#             interpolated_values_list.append(ynew)
#
#         interpolated_values_list = np.array(interpolated_values_list)
#         results = HANTS().__hants(sample_count=365, inputs=interpolated_values_list, low=valid_range[0], high=valid_range[1],
#                                 fit_error_tolerance=std)
#         # plt.imshow(interpolated_values_list, aspect='auto',vmin=0,vmax=3)
#         # plt.colorbar()
#         #
#         # plt.figure()
#         # plt.imshow(results, aspect='auto',vmin=0,vmax=3)
#         # plt.colorbar()
#         # plt.show()
#         results_dict = dict(zip(year_list, results))
#         return results_dict
#
#     def __date_list_to_DOY(self,date_list):
#         '''
#         :param date_list: list of datetime objects
#         :return:
#         '''
#         start_year = date_list[0].year
#         start_date = datetime.datetime(start_year, 1, 1)
#         date_delta = date_list - start_date + datetime.timedelta(days=1)
#         DOY = [date.days for date in date_delta]
#         return DOY
#
#     def __interp_values_to_DOY(self, values, date_list):
#         DOY = self.__date_list_to_DOY(date_list)
#         inx = DOY
#         iny = values
#         x_new = list(range(1, 366))
#         func = interpolate.interp1d(inx, iny, fill_value="extrapolate")
#         y_new = func(x_new)
#         return x_new, y_new
#
#     def __makediag3d(self,M):
#         b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
#         b[:, ::M.shape[1] + 1] = M
#         return b.reshape((M.shape[0], M.shape[1], M.shape[1]))
#
#     def __get_starter_matrix(self,base_period_len, sample_count, frequencies_considered_count):
#         nr = min(2 * frequencies_considered_count + 1,
#                  sample_count)  # number of 2*+1 frequencies, or number of input images
#         mat = np.zeros(shape=(nr, sample_count))
#         mat[0, :] = 1
#         ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
#         cs = np.cos(ang)
#         sn = np.sin(ang)
#         # create some standard sinus and cosinus functions and put in matrix
#         i = np.arange(1, frequencies_considered_count + 1)
#         ts = np.arange(sample_count)
#         for column in range(sample_count):
#             index = np.mod(i * ts[column], base_period_len)
#             # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
#             mat[2 * i - 1, column] = cs.take(index)
#             mat[2 * i, column] = sn.take(index)
#         return mat
#
#     def __hants(self,sample_count, inputs,
#               frequencies_considered_count=3,
#               outliers_to_reject='Hi',
#               low=0., high=255,
#               fit_error_tolerance=5.,
#               delta=0.1):
#         """
#         Function to apply the Harmonic analysis of time series applied to arrays
#         sample_count    = nr. of images (total number of actual samples of the time series)
#         base_period_len    = length of the base period, measured in virtual samples
#                 (days, dekads, months, etc.)
#         frequencies_considered_count    = number of frequencies to be considered above the zero frequency
#         inputs     = array of input sample values (e.g. NDVI values)
#         ts    = array of size sample_count of time sample indicators
#                 (indicates virtual sample number relative to the base period);
#                 numbers in array ts maybe greater than base_period_len
#                 If no aux file is used (no time samples), we assume ts(i)= i,
#                 where i=1, ..., sample_count
#         outliers_to_reject  = 2-character string indicating rejection of high or low outliers
#                 select from 'Hi', 'Lo' or 'None'
#         low   = valid range minimum
#         high  = valid range maximum (values outside the valid range are rejeced
#                 right away)
#         fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
#                 fit are rejected)
#         dod   = degree of overdeterminedness (iteration stops if number of
#                 points reaches the minimum required for curve fitting, plus
#                 dod). This is a safety measure
#         delta = small positive number (e.g. 0.1) to suppress high amplitudes
#         """
#         # define some parameters
#         base_period_len = sample_count  #
#
#         # check which setting to set for outlier filtering
#         if outliers_to_reject == 'Hi':
#             sHiLo = -1
#         elif outliers_to_reject == 'Lo':
#             sHiLo = 1
#         else:
#             sHiLo = 0
#
#         nr = min(2 * frequencies_considered_count + 1,
#                  sample_count)  # number of 2*+1 frequencies, or number of input images
#
#         # create empty arrays to fill
#         outputs = np.zeros(shape=(inputs.shape[0], sample_count))
#
#         mat = self.__get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)
#
#         # repeat the mat array over the number of arrays in inputs
#         # and create arrays with ones with shape inputs where high and low values are set to 0
#         mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
#         p = np.ones_like(inputs)
#         p[(low >= inputs) | (inputs > high)] = 0
#         nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries
#
#         # prepare for while loop
#         ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false
#
#         dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
#         noutmax = sample_count - nr - dod
#         for _ in range(sample_count):
#             if ready.all():
#                 break
#             # print '--------*-*-*-*',it.value, '*-*-*-*--------'
#             # multiply outliers with timeseries
#             za = np.einsum('ijk,ik->ij', mat, p * inputs)
#
#             # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
#             diag = self.__makediag3d(p)
#             A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
#             # add delta to suppress high amplitudes but not for [0,0]
#             A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
#             A[:, 0, 0] = A[:, 0, 0] - delta
#
#             # solve linear matrix equation and define reconstructed timeseries
#             zr = np.linalg.solve(A, za)
#             outputs = np.einsum('ijk,kj->ki', mat.T, zr)
#
#             # calculate error and sort err by index
#             err = p * (sHiLo * (outputs - inputs))
#             rankVec = np.argsort(err, axis=1, )
#
#             # select maximum error and compute new ready status
#             maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
#             ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)
#
#             # if ready is still false
#             if not ready.all():
#                 j = rankVec.take(sample_count - 1, axis=-1)
#
#                 p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
#                     int)  # *check
#                 nout += 1
#
#         return outputs

class GLASS_LAI:

    def __init__(self):
        self.datadir = join(data_root,'GLASS_LAI')
        pass

    def run(self):
        # self.hdf_to_tif()
        # self.resample()
        # self.monthly_compose()
        self.perpix()
        pass


    def hdf_to_tif(self):
        fdir = join(self.datadir,'AVHRR')
        outdir = join(self.datadir,'tif_8d_005')
        T.mk_dir(outdir)
        for folder in tqdm(T.listdir(fdir)):
            for f in T.listdir(join(fdir,folder)):
                if not f.endswith('.hdf'):
                    continue
                fpath = join(fdir,folder,f)
                date = f.split('.')[-3]
                date = date.replace('A','')
                outf = join(outdir,f'{date}.tif')
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr = np.array(arr,dtype=float)
                arr[arr>2500] = np.nan
                ToRaster().array2raster(outf,originX, originY, pixelWidth, pixelHeight, arr)

    def resample(self):
        fdir = join(self.datadir,'tif_8d_005')
        outdir = join(self.datadir,'tif_8d_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath, outpath, 0.5)

    def doy_to_date(self,doy):
        date_init = datetime.datetime(2000,1,1)
        date_delta = datetime.timedelta(doy-1)
        date = date_init + date_delta
        month = date.month
        day = date.day
        return month,day

    def monthly_compose(self):
        fdir = join(self.datadir, 'tif_8d_05')
        outdir = join(self.datadir, 'tif_monthly_05')
        T.mk_dir(outdir)
        month_dic = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            year = f[:4]
            doy = f[4:7]
            year = int(year)
            doy = int(doy)
            month,day = self.doy_to_date(doy)
            month_dic[(year,month)] = []
        # print(month_dic)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            year = f[:4]
            doy = f[4:7]
            year = int(year)
            doy = int(doy)
            month,day = self.doy_to_date(doy)
            month_dic[(year,month)].append(fpath)
        for year,month in month_dic:
            flist = month_dic[(year,month)]
            outf = f'{year}{month:02d}.tif'
            outpath = join(outdir,outf)
            Pre_Process().compose_tif_list(flist,outpath)

        pass

    def perpix(self):
        fdir = join(self.datadir, 'tif_monthly_05')
        outdir = join(self.datadir, 'per_pix')
        flist = []
        for y in range(1982,2016):
            for m in range(1,13):
                f = f'{y}{m:02d}.tif'
                flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,flist)


    def seasonal_split(self):
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'per_pix_seasonal')
        T.mk_dir(outdir)
        dic = T.load_npy_dir(fdir)
        for season in global_season_dic:
            gs_range = global_season_dic[season]
            annual_dic = {}
            for pix in tqdm(dic,desc=season):
                vals = dic[pix]
                vals = np.array(vals)
                T.mask_999999_arr(vals)
                vals[vals == 0] = np.nan
                if np.isnan(np.nanmean(vals)):
                    continue
                annual_vals = T.monthly_vals_to_annual_val(vals,gs_range)
                annual_dic[pix] = annual_vals
            outf = join(outdir,season)
            T.save_npy(annual_dic,outf)



class GEE_AVHRR_LAI:
    def __init__(self):
        self.datadir = join(data_root,'GEE_AVHRR_LAI')
        self.product = 'NOAA/CDR/AVHRR/LAI_FAPAR/V5'
        T.mk_dir(self.datadir)
        pass

    def run(self):
        # self.download_AVHRR_LAI_from_GEE()
        # zip_folder = join(self.datadir,'download_AVHRR_LAI_from_GEE')
        # tif_dir = join(self.datadir,'tif')
        # self.unzip(zip_folder,tif_dir)
        # self.resample()
        # self.unify_raster()
        # self.per_pix()
        # self.per_pix_clean()
        self.seasonal_split()
        pass

    def get_download_url(self,datatype="LANDSAT/LC08/C01/T1_32DAY_NDVI",
                         start='2000-01-01',
                         end='2000-02-01',
                         ):
        dataset = ee.ImageCollection(datatype).filterDate(start, end)
        LAI = dataset.select('LAI')
        image = LAI.max()
        LAI = image.divide(1000).rename('LAI')
        path = LAI.getDownloadUrl({
            'scale': 40000,
            'crs': 'EPSG:4326',
        })
        return path
    def download_AVHRR_LAI_from_GEE(self):
        ee.Initialize()
        outdir = join(self.datadir, 'download_AVHRR_LAI_from_GEE')
        T.mk_dir(outdir)
        date_list = []
        for y in range(1982,2016):
            for m in range(1,13):
                date = f'{y}-{m:02d}-01'
                date_list.append(date)
        date_list.append('2016-01-01')
        # print(date_list)
        for i in tqdm(range(len(date_list))):
            if i + 1 >= len(date_list):
                continue
            start = date_list[i]
            end = date_list[i+1]
            url = self.get_download_url(datatype=self.product,start=start,end=end)
            f = f'{start}.zip'
            fpath = join(outdir,f)
            self.download_data(url,fpath)



    def download_data(self,url, file_name):

        path = file_name
        if not os.path.isfile(path):
            # success = 0
            attempt = 0
            while 1:
                try:
                    with open(path, "wb") as f:
                        response = requests.get(url, stream=True)
                        total_length = 25. * 1024. * 1024.

                        if total_length is None:  # no content length header
                            f.write(response.content)
                        else:
                            dl = 0
                            total_length = int(total_length)
                            for data in response.iter_content(chunk_size=1024):
                                dl += len(data)
                                f.write(data)
                                done = int(50 * dl / total_length)
                                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                                sys.stdout.flush()
                    success = 1
                except Exception as e:
                    attempt += 1
                    time.sleep(1)
                    success = 0
                if success == 1:
                    break
                if attempt > 10:
                    break
        else:
            pass

    def unzip(self,zipfolder,move_dst_folder):
        T.mk_dir(move_dst_folder)
        for zipf in T.listdir(zipfolder):
            zipf_split = zipf.split('-')
            tif_name = ''.join([zipf_split[0],zipf_split[1]]) + '.tif'
            zip_path = join(zipfolder,zipf)
            # move_dst_folder = this_root+'tif\\'
            outpath = join(move_dst_folder,tif_name)
            if not os.path.isfile(outpath):
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                zip_ref.extractall(temporary_root)
                zip_ref.close()

                file_list = T.listdir(temporary_root)
                for i in file_list:
                    if i.endswith('.tif'):
                        temp_f = join(temporary_root,i)
                        shutil.move(temp_f,outpath)
            else:
                print(move_dst_folder+tif_name+' is existed')


    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath, outpath, 0.5)

    def unify_raster(self):
        fdir = join(self.datadir, 'tif_05')
        outdir = join(self.datadir, 'tif_unify')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = array[1:]
            ToRaster().array2raster(outpath,-180, 90, pixelWidth, pixelHeight, array)
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif_unify')
        outdir = join(self.datadir,'per_pix')
        f_list = []
        for y in range(1982,2016):
            for m in range(1,13):
                f = f'{y}{m:02d}.tif'
                f_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,f_list)
        pass

    def per_pix_clean(self):
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'per_pix_clean')
        Pre_Process().clean_per_pix(fdir,outdir,mode='climatology')

    def interp_nan_climatology(self,vals):
        vals = np.array(vals)
        vals_reshape = np.reshape(vals,(-1,12))
        vals_reshape_T = vals_reshape.T
        month_mean = []
        for m in vals_reshape_T:
            mean = np.nanmean(m)
            month_mean.append(mean)
        nan_index = np.isnan(vals)
        val_new = []
        for i in range(len(nan_index)):
            isnan = nan_index[i]
            month = i % 12
            interp_val = month_mean[month]
            if isnan:
                val_new.append(interp_val)
            else:
                val_new.append(vals[i])
        val_new = np.array(val_new)

    def __get_pheno_df(self):
        fdir = join(data_root,'Phenology')
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            col_name = f.replace('.npy','')
            dic = T.load_npy(fpath)
            spatial_dic = {}
            for pix in dic:
                vals = dic[pix]
                if len(vals) == 0:
                    continue
                val = vals[0]
                spatial_dic[pix] = val
            spatial_dics[col_name] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        early_range_list = []
        peak_range_list = []
        late_range_list = []
        for i,row in df.iterrows():
            early_end_mon = row.early_end_mon
            early_start_mon = row.early_start_mon
            late_end_mon = row.late_end_mon
            late_start_mon = row.late_start_mon
            early_range = list(range(early_start_mon,early_end_mon+1))
            peak_range = list(range(early_end_mon,late_start_mon+1))
            late_range = list(range(late_start_mon,late_end_mon+1))
            early_range_list.append(early_range)
            peak_range_list.append(peak_range)
            late_range_list.append(late_range)
        early_range_list = np.array(early_range_list)
        peak_range_list = np.array(peak_range_list)
        late_range_list = np.array(late_range_list)
        df['early'] = early_range_list
        df['peak'] = peak_range_list
        df['late'] = late_range_list

        return df


    def seasonal_split(self):
        seasonal_df = self.__get_pheno_df()
        fdir = join(self.datadir,'per_pix_clean')
        outdir = join(self.datadir,'per_pix_seasonal')
        T.mk_dir(outdir)
        dic = T.load_npy_dir(fdir)
        # T.print_head_n(seasonal_df)
        season_dic = T.df_to_dic(seasonal_df,'pix')
        for season in global_season_dic:
            annual_dic = {}
            for pix in tqdm(dic,desc=season):
                if not pix in season_dic:
                    continue
                gs_range = season_dic[pix][season]
                vals = dic[pix]
                vals = np.array(vals)
                T.mask_999999_arr(vals)
                vals[vals == 0] = np.nan
                if np.isnan(np.nanmean(vals)):
                    continue
                annual_vals = T.monthly_vals_to_annual_val(vals,gs_range)
                annual_dic[pix] = annual_vals
            outf = join(outdir,season)
            T.save_npy(annual_dic,outf)


class GEE_MODIS_LAI:
    def __init__(self):
        self.datadir = join(data_root,'GEE_MODIS_LAI')
        self.product = 'MODIS/006/MOD15A2H'
        T.mk_dir(self.datadir)
        pass

    def run(self):
        # self.download_AVHRR_LAI_from_GEE()
        # zip_folder = join(self.datadir,'download_AVHRR_LAI_from_GEE')
        # tif_dir = join(self.datadir,'tif')
        # self.unzip(zip_folder,tif_dir)
        # self.resample()
        # self.unify_raster()
        # self.per_pix()
        self.per_pix_clean()
        # self.seasonal_split()
        pass

    def get_download_url(self,datatype="LANDSAT/LC08/C01/T1_32DAY_NDVI",
                         start='2000-01-01',
                         end='2000-02-01',
                         ):
        print(start,end)
        dataset = ee.ImageCollection(datatype).filterDate(start, end)
        LAI = dataset.select('Lai_500m')
        image = LAI.max()
        LAI = image.divide(10).rename('LAI')
        try:
            path = LAI.getDownloadUrl({
                'scale': 40000,
                'crs': 'EPSG:4326',
            })
        except:
            return None

        return path
    def download_AVHRR_LAI_from_GEE(self):
        ee.Initialize()
        outdir = join(self.datadir, 'download_AVHRR_LAI_from_GEE')
        T.mk_dir(outdir)
        date_list = []
        for y in range(2000,2021):
            for m in range(1,13):
                date = f'{y}-{m:02d}-01'
                date_list.append(date)
        date_list.append('2021-01-01')
        # print(date_list)
        for i in tqdm(range(len(date_list))):
            if i + 1 >= len(date_list):
                continue
            start = date_list[i]
            end = date_list[i+1]
            url = self.get_download_url(datatype=self.product,start=start,end=end)
            f = f'{start}.zip'
            fpath = join(outdir,f)
            self.download_data(url,fpath)



    def download_data(self,url, file_name):

        path = file_name
        if not os.path.isfile(path):
            # success = 0
            attempt = 0
            while 1:
                try:
                    with open(path, "wb") as f:
                        response = requests.get(url, stream=True)
                        total_length = 25. * 1024. * 1024.

                        if total_length is None:  # no content length header
                            f.write(response.content)
                        else:
                            dl = 0
                            total_length = int(total_length)
                            for data in response.iter_content(chunk_size=1024):
                                dl += len(data)
                                f.write(data)
                                done = int(50 * dl / total_length)
                                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                                sys.stdout.flush()
                    success = 1
                except Exception as e:
                    attempt += 1
                    time.sleep(1)
                    success = 0
                if success == 1:
                    break
                if attempt > 10:
                    break
        else:
            pass

    def unzip(self,zipfolder,move_dst_folder):
        T.mk_dir(move_dst_folder)
        for zipf in T.listdir(zipfolder):
            zipf_split = zipf.split('-')
            tif_name = ''.join([zipf_split[0],zipf_split[1]]) + '.tif'
            zip_path = join(zipfolder,zipf)
            # move_dst_folder = this_root+'tif\\'
            outpath = join(move_dst_folder,tif_name)
            if not os.path.isfile(outpath):
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                zip_ref.extractall(temporary_root)
                zip_ref.close()

                file_list = T.listdir(temporary_root)
                for i in file_list:
                    if i.endswith('.tif'):
                        temp_f = join(temporary_root,i)
                        shutil.move(temp_f,outpath)
            else:
                print(move_dst_folder+tif_name+' is existed')


    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath, outpath, 0.5)

    def unify_raster(self):
        fdir = join(self.datadir, 'tif_05')
        outdir = join(self.datadir, 'tif_unify')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = array[1:]
            ToRaster().array2raster(outpath,-180, 90, pixelWidth, pixelHeight, array)
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif_unify')
        outdir = join(self.datadir,'per_pix')
        f_list = []
        for y in range(2001,2021):
            for m in range(1,13):
                f = f'{y}{m:02d}.tif'
                f_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,f_list)
        pass

    def per_pix_clean(self):
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'per_pix_clean')
        Pre_Process().clean_per_pix(fdir,outdir,mode='climatology')

    def interp_nan_climatology(self,vals):
        vals = np.array(vals)
        vals_reshape = np.reshape(vals,(-1,12))
        vals_reshape_T = vals_reshape.T
        month_mean = []
        for m in vals_reshape_T:
            mean = np.nanmean(m)
            month_mean.append(mean)
        nan_index = np.isnan(vals)
        val_new = []
        for i in range(len(nan_index)):
            isnan = nan_index[i]
            month = i % 12
            interp_val = month_mean[month]
            if isnan:
                val_new.append(interp_val)
            else:
                val_new.append(vals[i])
        val_new = np.array(val_new)

    def __get_pheno_df(self):
        fdir = join(data_root,'Phenology')
        spatial_dics = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            col_name = f.replace('.npy','')
            dic = T.load_npy(fpath)
            spatial_dic = {}
            for pix in dic:
                vals = dic[pix]
                if len(vals) == 0:
                    continue
                val = vals[0]
                spatial_dic[pix] = val
            spatial_dics[col_name] = spatial_dic
        df = T.spatial_dics_to_df(spatial_dics)
        early_range_list = []
        peak_range_list = []
        late_range_list = []
        for i,row in df.iterrows():
            early_end_mon = row.early_end_mon
            early_start_mon = row.early_start_mon
            late_end_mon = row.late_end_mon
            late_start_mon = row.late_start_mon
            early_range = list(range(early_start_mon,early_end_mon+1))
            peak_range = list(range(early_end_mon,late_start_mon+1))
            late_range = list(range(late_start_mon,late_end_mon+1))
            early_range_list.append(early_range)
            peak_range_list.append(peak_range)
            late_range_list.append(late_range)
        early_range_list = np.array(early_range_list)
        peak_range_list = np.array(peak_range_list)
        late_range_list = np.array(late_range_list)
        df['early'] = early_range_list
        df['peak'] = peak_range_list
        df['late'] = late_range_list

        return df


    def seasonal_split(self):
        seasonal_df = self.__get_pheno_df()
        fdir = join(self.datadir,'per_pix_clean')
        outdir = join(self.datadir,'per_pix_seasonal')
        T.mk_dir(outdir)
        dic = T.load_npy_dir(fdir)
        # T.print_head_n(seasonal_df)
        season_dic = T.df_to_dic(seasonal_df,'pix')
        for season in global_season_dic:
            annual_dic = {}
            for pix in tqdm(dic,desc=season):
                if not pix in season_dic:
                    continue
                gs_range = season_dic[pix][season]
                vals = dic[pix]
                vals = np.array(vals)
                T.mask_999999_arr(vals)
                vals[vals == 0] = np.nan
                if np.isnan(np.nanmean(vals)):
                    continue
                annual_vals = T.monthly_vals_to_annual_val(vals,gs_range)
                annual_dic[pix] = annual_vals
            outf = join(outdir,season)
            T.save_npy(annual_dic,outf)


class GLC2000:

    def __init__(self):
        self.datadir = join(data_root,'GLC2000')
        pass

    def run(self):
        # self.resample()
        self.lc_spatial_dic()
        # self.gen_spatial_dic()
        # self.check_pix()
        pass


    def resample(self):
        tif = join(self.datadir,'glc2000_v1_1_resample_7200_3600.tif')
        outtif = join(self.datadir,'glc2000_v1_1_resample_720_360.tif')

        ToRaster().resample_reproj(tif,outtif,0.5)

    def lc_spatial_dic(self):
        tif = join(self.datadir,'glc2000_v1_1_resample_720_360.tif')
        outf = join(self.datadir,'lc_dic_reclass.npy')
        dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        lc_reclass_spatial_dic = {}
        evergreen = [1,4]
        deciduous = [2,3,5]
        for pix in tqdm(dic):
            val = dic[pix]
            if val in evergreen:
                cls = 'Evergreen'
            elif val in deciduous:
                cls = 'Deciduous'
            elif 11 <= val <= 12:
                cls = 'Shrubs'
            elif 13 <= val <= 14:
                cls = 'Grass'
            elif 16<= val <= 18:
                cls = 'Crop'
            else:
                continue
            lc_reclass_spatial_dic[pix] = cls
        T.save_npy(lc_reclass_spatial_dic,outf)


        pass


    def gen_spatial_dic(self):
        # outf = data_root + 'landcover/gen_spatial_dic'
        spatial_dic = {}
        for val in tqdm(valid_dic):
            pixs = valid_dic[val]
            for pix in pixs:
                spatial_dic[pix] = val
        np.save(outf,spatial_dic)


    def check_pix(self):
        # f = data_root + 'landcover/forests_pix.npy'
        f = data_root + 'landcover/gen_spatial_dic.npy'
        dic = T.load_npy(f)
        # spatial_dic = {}
        # for val in tqdm(dic):
        #     pixs = dic[val]
        #     for pix in pixs:
        #         # print(pix)
        #         spatial_dic[pix] = 1
        arr = DIC_and_TIF(Global_vars().tif_template_7200_3600).pix_dic_to_spatial_arr(dic)
        plt.imshow(arr)
        plt.show()
        pass


class CCI_SM:

    def __init__(self):
        self.datadir = join(data_root,'CCI')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.daily_to_monthly()
        # self.count_annual_tif_number()
        # self.resample()
        # self.per_pix()
        # self.pick_early_peak_late()
        self.interpolate_early_peak_late()
        # self.check_cci_sm()
        pass


    def __nc_to_tif(self,nc):
        ncin = Dataset(nc, 'r')
        lat = ncin['lat']
        lon = ncin['lon']
        longitude_grid_distance = abs((lon[-1] - lon[0]) / (len(lon) - 1))
        latitude_grid_distance = -abs((lat[-1] - lat[0]) / (len(lat) - 1))
        longitude_start = lon[0]
        latitude_start = lat[0]

        start = datetime.datetime(1970, 1, 1)
        time = ncin.variables['time']
        date = start + datetime.timedelta(days=int(time[0]))
        array = ncin['sm'][0][::]
        array = np.array(array)
        return date, array,longitude_start,latitude_start,longitude_grid_distance,latitude_grid_distance

        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED_1978-2020-v06.1')
        outdir = join(self.datadir,'tif_daily')
        T.mk_dir(outdir)
        for year_folder in T.listdir(fdir):
            outdir_i = join(outdir,year_folder)
            T.mk_dir(outdir_i)
            for f in tqdm(T.listdir(join(fdir,year_folder)),desc=year_folder):
                fpath = join(fdir,year_folder,f)
                date, array,longitude_start,latitude_start,longitude_grid_distance,latitude_grid_distance = self.__nc_to_tif(fpath)
                year = date.year
                month = date.month
                day = date.day
                outf = f'{year}{month:02d}{day:02d}.tif'
                outpath = join(outdir_i,outf)
                newRasterfn = outpath
                longitude_start = longitude_start
                latitude_start = latitude_start
                pixelWidth = longitude_grid_distance
                pixelHeight = latitude_grid_distance
                array = np.array(array)
                array[array < 0] = np.nan
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array)

        pass

    def daily_to_monthly(self):
        fdir = join(self.datadir,'tif_daily')
        outdir = join(self.datadir,'tif_monthly')
        T.mk_dir(outdir)
        for year_str in T.listdir(fdir):
            month_dic = {}
            for m in range(1,13):
                key = f'{year_str}{m:02d}'
                month_dic[key] = []
            for f in T.listdir(join(fdir,year_str)):
                yy_mm = f[:6]
                month_dic[yy_mm].append(join(fdir,year_str,f))
            for m in month_dic:
                print(m)
                flist = month_dic[m]
                if len(flist) == 0:
                    continue
                outf = join(outdir,f'{m}.tif')
                if isfile(outf):
                    continue
                Pre_Process().compose_tif_list(flist, outf)
                # exit()

    def count_annual_tif_number(self):
        fdir = join(self.datadir, 'tif_daily')
        for year_str in T.listdir(fdir):
            flag = 0
            for f in T.listdir(join(fdir, year_str)):
                flag += 1
            print(year_str,flag)
        exit()
        pass


    def resample(self):
        fdir = join(self.datadir,'tif_monthly')
        outdir = join(self.datadir,'tif_monthly_05deg')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            fpath = join(fdir,f)
            ToRaster().resample_reproj(fpath,outf,res=0.5)

        pass


    def per_pix(self):
        fdir = join(self.datadir,'tif_monthly_05deg')
        outdir = join(self.datadir,'per_pix/1982-2018')
        T.mk_dir(outdir,force=True)
        date_list = []
        for y in range(1982,2019):
            for m in range(1,13):
                date_list.append(f'{y}{m:02d}.tif')
        Pre_Process().data_transform_with_date_list(fdir,outdir,date_list)


    def load_phenology_dic(self):
        phenology_dir = join(data_root, 'Phenology')
        early_start_dic = T.load_npy(join(phenology_dir, 'early_start_mon.npy'))
        early_end_dic = T.load_npy(join(phenology_dir, 'early_end_mon.npy'))
        late_end_dic = T.load_npy(join(phenology_dir, 'late_end_mon.npy'))
        late_start_dic = T.load_npy(join(phenology_dir, 'late_start_mon.npy'))
        dic_all = {
            'early_start':early_start_dic,
            'early_end':early_end_dic,
            'late_start':late_start_dic,
            'late_end':late_end_dic,
        }

        df = T.spatial_dics_to_df(dic_all)
        early_range_list = []
        peak_range_list = []
        late_range_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            early_start = row.early_start
            early_end = row.early_end
            late_start = row.late_start
            late_end = row.late_end
            if len(early_start) == 0:
                early_range_list.append(np.nan)
                peak_range_list.append(np.nan)
                late_range_list.append(np.nan)
                continue
            early_range = list(range(early_start[0],early_end[0]+1))
            peak_range = list(range(early_end[0],late_start[0]+1))
            late_range = list(range(late_start[0],late_end[0]+1))
            early_range_list.append(early_range)
            peak_range_list.append(peak_range)
            late_range_list.append(late_range)
        df['early_range'] = early_range_list
        df['peak_range'] = peak_range_list
        df['late_range'] = late_range_list
        df = df.dropna()
        phenology_dic = T.df_to_dic(df,'pix')
        return phenology_dic


    def pick_early_peak_late(self):
        fdir = join(self.datadir,'per_pix/1982-2018')
        outdir = join(self.datadir,'pick_early_peak_late')
        T.mk_dir(outdir)
        outf = join(outdir,'dataframe.df')
        phenology_dic = self.load_phenology_dic()

        dic = T.load_npy_dir(fdir)
        result_dic = {}
        for pix in tqdm(dic):
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            if np.isnan(np.nanmean(vals)):
                continue
            if not pix in phenology_dic:
                continue
            early_range = phenology_dic[pix]['early_range']
            peak_range = phenology_dic[pix]['peak_range']
            late_range = phenology_dic[pix]['late_range']
            early_vals = T.monthly_vals_to_annual_val(vals,early_range)
            peak_vals = T.monthly_vals_to_annual_val(vals,peak_range)
            late_vals = T.monthly_vals_to_annual_val(vals,late_range)
            dic_i = {
                'early':early_vals,
                'peak':peak_vals,
                'late':late_vals,
            }
            result_dic[pix] = dic_i
        df = T.dic_to_df(result_dic,'pix')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        pass

    def interpolate_early_peak_late(self):
        fdir = join(self.datadir,'pick_early_peak_late')
        outdir = join(self.datadir,'pick_early_peak_late_npy')
        T.mk_dir(outdir)
        dff = join(fdir,'dataframe.df')
        df = T.load_df(dff)
        for season in global_season_dic:
            spatial_dic = {}
            outf = join(outdir,f'during_{season}_CCI_SM.npy')
            for i,row in tqdm(df.iterrows(),total=len(df)):
                pix = row['pix']
                vals = row[season]
                val_interp = T.interp_nan(vals,valid_percent=0.8)
                if len(val_interp) == 1:
                    continue
                spatial_dic[pix] = val_interp
            T.save_npy(spatial_dic,outf)

    def check_cci_sm(self):
        fdir = join(self.datadir, 'pick_early_peak_late')
        dff = join(fdir, 'dataframe.df')
        df = T.load_df(dff)
        # spatial_dic = T.df_to_spatial_dic(df,'early')
        spatial_dic = T.df_to_spatial_dic(df,'peak')
        spatial_dic_num = {}
        for pix in spatial_dic:
            vals = spatial_dic[pix]
            if np.isnan(np.nanmean(vals)):
                continue
            val_valid = T.remove_np_nan(vals)
            valid_num = len(val_valid)
            if not valid_num == 444//12:
                continue
            spatial_dic_num[pix] = 1

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_num)
        DIC_and_TIF().arr_to_tif(arr,join(this_root,'conf/sm_valid.tif'))
        # DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()


class LAI_4g:

    def __init__(self):
        self.datadir = '/Volumes/NVME2T/greening_project_redo/data/LAI4g/new_221109'
        pass

    def run(self):
        # self.mat_to_tif()
        # self.resample()
        # self.per_pix()
        self.hants()
        # self.check_hants()
        # self.climatology_mean()
        # self.per_pix_climatology_mean()
        # self.hants_climatology_mean()
        # self.transform_hants()
        self.check_original_data()
        pass

    def mat_to_tif(self):
        import h5py
        fdir = join(self.datadir,'mat')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath=join(fdir,f)
            outf = join(outdir,f.replace('.mat','.tif'))
            if isfile(outf):
                continue
            mat = h5py.File(fpath)
            # print(f,mat.keys())
            try:
                arr = mat['data_gimms_lai'][::]
            except:
                arr = mat['data_modis_lai'][::]
            arr = np.array(arr)
            arr = arr.T
            ToRaster().array2raster(outf, -180, 90, 360/4320., -(180./2160), arr)


    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            if isfile(outf):
                continue
            ToRaster().resample_reproj(fpath,outf,0.5)

    def per_pix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def get_date_list(self):
        fdir = join(self.datadir,'tif_05')
        date_list = []
        for f in T.listdir(fdir):
            date = f.split('_')[1]
            date = date.split('.')[0]
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[6:])
            if day == 2:
                day = 15
            date_obj = datetime.datetime(year,month,day)
            date_list.append(date_obj)
        return date_list

    def hants(self):
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'hants')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            outf = join(outdir,f'{f}')
            spatial_dict = T.load_npy(join(fdir,f))
            date_list = self.get_date_list()
            all_result = {}
            for pix in tqdm(spatial_dict):
                r,c = pix
                if r > 120:
                    continue
                vals = spatial_dict[pix]
                vals[vals<-9999] = np.nan
                if T.is_all_nan(vals):
                    continue
                try:
                    hants_results_dict = HANTS().hants_interpolate(vals,date_list,valid_range=(0,10))
                except:
                    continue
                all_result[pix] = hants_results_dict
            if len(all_result) == 0:
                continue
            T.save_npy(all_result,outf)
            # df = T.dic_to_df(all_result,'pix')
            # T.save_df(df,outf)
            # T.df_to_excel(df,outf.replace('.df',''))

    def per_pix_climatology_mean(self):
        fdir = join(self.datadir,'climatology_mean')
        outdir = join(self.datadir,'per_pix_climatology_mean')
        Pre_Process().data_transform(fdir,outdir)

    def hants_climatology_mean(self):
        fdir = join(self.datadir,'per_pix_climatology_mean')
        outdir = join(self.datadir,'hants_climatology_mean')
        T.mk_dir(outdir)
        date_list = self.get_date_list()
        date_list = date_list[:24]
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_hants = {}
        for pix in tqdm(spatial_dict):
            r,c = pix
            if r > 120:
                continue
            vals = spatial_dict[pix]
            vals[vals<-9999] = np.nan
            if T.is_all_nan(vals):
                continue
            try:
                hants_results_dict = HANTS().hants_interpolate(vals, date_list, valid_range=(0, 10),nan_value=0)
            except:
                continue
            # plt.plot(vals)
            vals = hants_results_dict[1982]
            spatial_dict_hants[pix] = vals
        T.save_npy(spatial_dict_hants,join(outdir,'hants_climatology_mean'))



    def check_hants(self):

        fdir = join(self.datadir,'hants')

        spatial_dict_1 = {}
        for f in T.listdir(fdir):
            print(f)
            spatial_dict = T.load_npy(join(fdir,f))
            for pix in spatial_dict:
                spatial_dict_1[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_1)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

    def climatology_mean(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'climatology_mean')
        T.mk_dir(outdir)
        date_list = self.get_date_list()
        fpath_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            fpath_list.append(fpath)
        fpath_dict = dict(zip(date_list,fpath_list))
        y_list = []
        m_list = []
        d_list = []
        for date in date_list:
            y,m,d = date.year,date.month,date.day
            y_list.append(y)
            m_list.append(m)
            d_list.append(d)
        y_list = list(set(y_list))
        m_list = list(set(m_list))
        d_list = list(set(d_list))
        y_list.sort()
        m_list.sort()
        d_list.sort()
        for m in m_list:
            for d in d_list:
                fpath_list = []
                outf = join(outdir,f'{1982}{m:02d}{d:02d}.tif')
                for y in y_list:
                    date = datetime.datetime(y,m,d)
                    fpath = fpath_dict[date]
                    fpath_list.append(fpath)
                Pre_Process().compose_tif_list(fpath_list,outf)


    def transform_hants(self):
        fdir = join(self.datadir,'hants')
        outdir = join(self.datadir,'hants_transform')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            for pix in tqdm(spatial_dict,desc=f):
                dict_i = spatial_dict[pix]
                year_list = []
                for y in dict_i:
                    year_list.append(y)
                year_list.sort()
                vals = []
                for y in year_list:
                    vals.append(dict_i[y])
                vals = np.array(vals)
                spatial_dict[pix] = vals
            T.save_npy(spatial_dict,join(outdir,f))

    def check_original_data(self):
        fdir = join(self.datadir,'per_pix')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_new = {}
        for pix in spatial_dict:
            r,c = pix
            if r > 120:
                continue
            vals = spatial_dict[pix]
            if T.is_all_nan(vals):
                continue
            spatial_dict_new[pix] = vals
        df = T.spatial_dics_to_df({'LAI4g':spatial_dict_new})
        vals_all = df['LAI4g'].tolist()
        vals_all = np.array(vals_all)
        vals_all_mean = np.nanmean(vals_all,axis=0)
        date_list = self.get_date_list()
        plt.plot(date_list,vals_all_mean)
        plt.show()
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        # plt.show()


class LAI_3g:

    def __init__(self):
        self.datadir = join(data_root,'LAI3g')
        pass

    def run(self):
        pass



class VODCA:

    def __init__(self):
        self.datadir = join(data_root,'VODCA')
        pass

    def run(self):
        # self.check_tif()
        # self.origin_data_025_per_pix()
        # self.clean_origin_data_025()
        # self.dict_to_tif()
        self.resample()
        # self.per_pix_05()
        # self.pick_early_peak_late()
        pass

    def check_tif(self):
        fdir = join(self.datadir,'tif025')
        for f in T.listdir(fdir):
            print(f)
            fpath = join(fdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            plt.imshow(array)
            plt.show()


    def origin_data_025_per_pix(self):
        fdir = join(self.datadir,'tif025')
        outdir = join(self.datadir,'per_pix_025')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def clean_origin_data_025(self):

        fdir = join(self.datadir,'per_pix_025')
        outdir = join(self.datadir,'per_pix_025_clean')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy_dir(fdir,'')
        spatial_dict_nan_number = {}
        gs_list = list(range(4,10))
        new_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            if T.is_all_nan(vals):
                continue
            vals_annual = T.monthly_vals_to_annual_val(vals,gs_list,method='mean')
            vals_nan_number = np.sum(np.isnan(vals_annual))
            ratio = 1 - vals_nan_number / len(vals_annual)
            if ratio != 1:
                continue
            new_spatial_dict[pix] = vals
        T.save_distributed_perpix_dic(new_spatial_dict,outdir)



    def dict_to_tif(self):
        fdir = join(self.datadir,'per_pix_025_clean')
        spatial_dict = T.load_npy_dir(fdir)
        outdir = join(self.datadir,'tif_per_pix_025_clean')
        T.mk_dir(outdir)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif_every_time_stamp(spatial_dict,outdir)

        pass



    def resample(self):
        fdir = join(self.datadir,'tif_per_pix_025_clean')
        outdir = join(self.datadir,'tif_resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_resample = T.resample_nan(array,0.5,pixelWidth)
            array_resample[array_resample == 0] = np.nan
            # array_resample = array_resample[::-1]
            DIC_and_TIF().arr_to_tif(array_resample,outf)


    def per_pix_05(self):
        fdir = join(self.datadir,'tif_resample')
        outdir = join(self.datadir,'per_pix_05')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def pick_early_peak_late(self):
        fdir = join(self.datadir,'per_pix_05')
        spatial_dict = T.load_npy_dir(fdir)
        outdir = join(self.datadir,'per_pix_05_pick_early_peak_late')
        T.mk_dir(outdir)


        pass


class LAI_4g_v101:
    def __init__(self):
        self.datadir = join(data_root, 'LAI4g_101')
        pass

    def run(self):
        # self.rename()
        # self.monthly_compose()
        # self.resample()
        self.per_pix()

    def rename(self):
        fdir = join(self.datadir,'GIMMS_LAI_4g_ver101')
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            new_name = f.split('_')[-1]
            new_name = new_name.replace('.tif','')
            new_name = new_name + '0.tif'
            os.rename(fpath,join(fdir,new_name))
        pass

    def monthly_compose(self):
        fdir = join(self.datadir,'GIMMS_LAI_4g_ver101')
        outdir = join(self.datadir,'GIMMS_LAI_4g_ver101_monthly_compose')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')


    def resample(self):
        fdir = join(self.datadir,'GIMMS_LAI_4g_ver101_monthly_compose')
        outdir = join(self.datadir,'GIMMS_LAI_4g_ver101_monthly_compose_resample')
        T.open_path_and_file(outdir)
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            target_res = 0.5
            original_res = pixelWidth
            array_resample = T.resample_nan(array,target_res,original_res)
            # array_resample[array_resample == 0] = np.nan
            DIC_and_TIF().arr_to_tif(array_resample,outf)
            # exit()

    def per_pix(self):
        fdir = join(self.datadir,'GIMMS_LAI_4g_ver101_monthly_compose_resample')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

def seasonal_split_ly_NDVI():
    fdir = join(data_root, 'NDVI_ly/per_pix')
    outdir = join(data_root, 'NDVI_ly/per_pix_seasonal')
    T.mk_dir(outdir)
    dic = T.load_npy_dir(fdir)
    for season in global_season_dic:
        gs_range = global_season_dic[season]
        annual_dic = {}
        for pix in tqdm(dic,desc=season):
            vals = dic[pix]
            vals = np.array(vals)
            T.mask_999999_arr(vals)
            vals[vals == 0] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            annual_vals = T.monthly_vals_to_annual_val(vals,gs_range)
            annual_dic[pix] = annual_vals
        outf = join(outdir,season)
        T.save_npy(annual_dic,outf)


class MODIS_LAI:
    def __init__(self):
        data_root = '/Volumes/NVME2T/greening_project_redo/data'
        self.datadir = join(data_root,'MODIS_LAI')
        pass

    def run(self):
        # self.per_pix()
        # self.hants()
        self.hants_format_transform()
        pass

    def per_pix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def hants(self):
        tif_dir = join(self.datadir,'tif_05')
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'per_pix_hants')
        T.mk_dir(outdir)
        date_obj_list = []
        for f in T.listdir(tif_dir):
            yyyy,mm,dd = Pre_Process().get_year_month_day(f,date_fmt='doy')
            date_obj = datetime.datetime(int(yyyy),int(mm),int(dd))
            date_obj_list.append(date_obj)
        for f in T.listdir(fdir):
            outf = join(outdir, f'{f}')
            spatial_dict = T.load_npy(join(fdir, f))
            date_list = date_obj_list
            all_result = {}
            for pix in tqdm(spatial_dict):
                r, c = pix
                if r > 120:
                    continue
                vals = spatial_dict[pix]
                vals[vals < -9999] = np.nan
                if T.is_all_nan(vals):
                    continue
                vals[np.isnan(vals)] = 0
                # print(vals)
                try:
                    hants_results_dict = HANTS().hants_interpolate(vals, date_list, valid_range=(0, 10))
                except:
                    continue
                all_result[pix] = hants_results_dict
            if len(all_result) == 0:
                continue
            T.save_npy(all_result, outf)
            # df = T.dic_to_df(all_result,'pix')
            # T.save_df(df,outf)
            # T.df_to_excel(df,outf.replace('.df',''))

        pass

    def hants_format_transform(self):
        fdir = join(self.datadir,'per_pix_hants')
        outdir = join(self.datadir,'per_pix_hants_format_transform')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            print(f)
            dict_i = T.load_npy(join(fdir,f))
            year_list = list(range(2000,2019))
            spatial_dict_i = {}
            for pix in dict_i:
                dict_j = dict_i[pix]
                arrs = []
                for year in year_list:
                    vals = dict_j[year]
                    vals = np.array(vals)
                    arrs.append(vals)
                arrs = np.array(arrs)
                spatial_dict_i[pix] = arrs
            outf = join(outdir,f)
            T.save_npy(spatial_dict_i,outf)

class MODIS_LAI_BU_CMG:

    def __init__(self):
        self.datadir = join(data_root,'BU_MCD_LAI_CMG')

    def run(self):
        # self.resample_bill()
        # self.rename()
        # self.monthly_compose()
        # self.per_pix()
        self.per_pix_biweekly()
        pass

    def __mat_to_tif(self,mat_f,outf):
        mat_f_r = scipy.io.loadmat(mat_f)
        print(mat_f_r.items())
        print(mat_f_r)
        exit()
        # print(mat_f_r.keys())
        matrix = mat_f_r['outmat']
        lai_arr = matrix[0][0][0]  ## band 1
        # lai_arr = matrix[0][0][1] ## band 2
        # lai_arr = matrix[0][0][2]
        # lai_arr = matrix[0][0][3]
        # lai_arr = matrix[0][0][4]
        lai_arr = np.array(lai_arr)
        lai_arr[lai_arr > 200] = np.nan
        lai_arr = lai_arr / 10.
        DIC_and_TIF(pixelsize=0.05).arr_to_tif(lai_arr,outf)

    def mat_to_tif(self):

        fdir = join(self.datadir,'BU_MCD_LAI_CMG005_NSustain_00_19')
        outdir = join(self.datadir,'tif')
        # outdir = join(self.datadir,'weight_tif')
        # outdir = join(self.datadir,'n_good_veg_tif')
        # outdir = join(self.datadir,'n_bad_veg_tif')
        # outdir = join(self.datadir,'n_non_veg_tif')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f.replace('.mat','.tif'))
            self.__mat_to_tif(join(fdir,f),outf)

        pass

    def resample_gdal(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            ToRaster().resample_reproj(join(fdir,f),outf,res=0.5)

    def resample_bill(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'resample_bill')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            outf = join(outdir, f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_resample = T.resample_nan(array, 0.5, pixelWidth)
            # array_resample[array_resample == 0] = np.nan
            # array_resample = array_resample[::-1]
            DIC_and_TIF().arr_to_tif(array_resample, outf)
    def rename(self):
        fdir = join(self.datadir,'resample_bill')
        for f in tqdm(T.listdir(fdir)):
            f_split = f.split('_')
            doy = f_split[-3]
            new_name = doy + '.tif'
            os.rename(join(fdir,f),join(fdir,new_name))


    def monthly_compose(self):
        # compose_method = 'mean'
        compose_method = 'max'
        fdir = join(self.datadir,'resample_bill')
        outdir = join(self.datadir,f'resample_bill_05_monthly_{compose_method}_compose')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        Pre_Process().monthly_compose(fdir,outdir,date_fmt='doy',method=compose_method)

    def per_pix(self):
        fdir = join(self.datadir,'resample_bill_05_monthly_max_compose')
        outdir = join(self.datadir,'resample_bill_05_monthly_max_compose_per_pix')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        year_list = [str(i) for i in range(2000,2020)]
        month_list = [f'{i:02d}' for i in range(1,13)]
        date_list = []
        for y in year_list:
            for m in month_list:
                date_list.append(f'{y}{m}.tif')
        Pre_Process().data_transform_with_date_list(fdir,outdir,date_list=date_list)
        pass

    def per_pix_biweekly(self):
        fdir = join(self.datadir,'resample_bill')
        outdir = join(self.datadir,'per_pix_biweekly')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        year_list = []
        for f in T.listdir(fdir):
            year = f[:4]
            if not year in year_list:
                year_list.append(year)
        for year in year_list:
            annual_flist = []
            for f in T.listdir(fdir):
                if f[:4] == year:
                    annual_flist.append(f)
            print(len(annual_flist))



        pass

def check_per_pix_data():
    dff = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Pick_Early_Peak_Late_value/Pick_variables/MODIS_LAI_CMG.df'
    df = T.load_df(dff)
    spatial_dic = T.df_to_spatial_dic(df,'peak')
    for pix in spatial_dic:
        if not pix == (26,567):
            continue
        vals = spatial_dic[pix]
        plt.plot(vals)
        plt.grid()
        plt.show()


    pass


class TRENDY:

    def __init__(self):

        pass


    def run(self):
        self.nc_to_tif()
        pass

    def nc_to_tif(self):
        f = '/Volumes/NVME2T/wen_proj/Trendy_nc/DLEM_S2_lai.nc'
        outdir = '/Volumes/NVME2T/wen_proj/Trendy_nc/tif'
        T.mk_dir(outdir)
        T.nc_to_tif(f,'lai',outdir)
        pass

class VODCA_GPP:

    def __init__(self):
        self.datadir = join(data_root,'VODCA_GPP')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        self.per_pix()
        self.daily()
        # self.check_daily()
        pass

    def nc_to_tif(self):
        f = join(self.datadir,'nc/VODCA2GPP_v1.nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        T.nc_to_tif(f,'GPP',outdir)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            ToRaster().resample_reproj(join(fdir,f),outf,res=0.5)

    def per_pix(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_05_per_pix')
        Pre_Process().data_transform(fdir,outdir)
        date_obj_list = []
        for f in T.listdir(fdir):
            date = f.split('.')[0]
            year = date[:4]
            month = date[4:6]
            day = date[6:]
            year = int(year)
            month = int(month)
            day = int(day)
            date_obj = datetime.datetime(year,month,day)
            date_obj_list.append(date_obj)
        np.save(join(self.datadir,'date_obj_list.npy'),date_obj_list)

    def kernel_daily(self,params):
        fdir,outdir,f,year_list_unique,date_obj_list = params
        fpath = join(fdir, f)
        outpath = join(outdir, f)
        if isfile(outpath):
            return
        dict_i = T.load_npy(fpath)
        spatial_dict = {}
        for pix in tqdm(dict_i):
            r,c = pix
            if r > 120:
                continue
            vals = dict_i[pix]
            if T.is_all_nan(vals):
                continue
            vals_list = []
            dates_list = []
            for year in year_list_unique:
                picked_year_index = [i for i in range(len(date_obj_list)) if date_obj_list[i].year == year]
                picked_date = date_obj_list[picked_year_index[0]:picked_year_index[-1] + 1]
                vals = np.array(vals)
                points = copy.copy(vals)
                vals[vals <= 0] = -1
                points[points <= 0] = np.nan
                picked_vals = vals[picked_year_index]
                picked_points = points[picked_year_index]
                if T.is_all_nan(picked_points):
                    continue
                vals_list.append(picked_vals)
                dates_list.append(picked_date)
            results = HANTS().hants_interpolate(vals_list, dates_list, (0, 10))
            spatial_dict[pix] = results
        T.save_npy(spatial_dict, outpath)
        pass

    def daily(self):
        date_f = join(self.datadir,'date_obj_list.npy')
        fdir = join(self.datadir,'tif_05_per_pix')
        outdir = join(self.datadir,'tif_05_per_pix_daily')
        T.mk_dir(outdir)
        T.open_path_and_file(outdir)

        date_obj_list = np.load(date_f,allow_pickle=True)
        year_list = [i.year for i in date_obj_list]
        year_list_unique = list(set(year_list))
        year_list_unique.sort()
        params_list = []
        for f in T.listdir(fdir):
            params = (fdir,outdir,f,year_list_unique,date_obj_list)
            params_list.append(params)
        MULTIPROCESS(self.kernel_daily,params_list).run(process=7)

    def check_daily(self):
        fdir = join(self.datadir,'tif_05_per_pix_daily')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_2 = {}
        for pix in tqdm(spatial_dict):
            dict_i = spatial_dict[pix]
            len_dict_i = len(dict_i)
            spatial_dict_2[pix] = len_dict_i
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_2)
        plt.imshow(arr)
        plt.show()


class VOD_AMSRU:

    def __init__(self):
        self.datadir = join(data_root,'AMSRU_VOD')
        pass

    def run(self):
        self.per_pix()
        pass

    def per_pix(self):
        fdir = join(self.datadir, 'tif/D')
        outdir = join(self.datadir, 'tif_per_pix')
        outdir_dateobj = join(self.datadir, 'dateobj')
        T.mk_dir(outdir)
        T.mk_dir(outdir_dateobj)
        for year in T.listdir(fdir):
            print(year)
            print('----------------')
            year_dir = join(fdir, year)
            outdir_year = join(outdir, year)
            Pre_Process().data_transform(year_dir, outdir_year)
            date_obj_list = []
            for f in T.listdir(year_dir):
                date = f.split('.')[0]
                year = date[:4]
                doy = date[4:7]
                year = int(year)
                doy = int(doy)
                date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
                date_obj_list.append(date_obj)
            np.save(join(outdir_dateobj, f'{year}.npy'), date_obj_list)



class FLUX_Matt:

    def __init__(self):
        self.data_dir = join(data_root,'FLUX_Matt')
        # T.open_path_and_file(self.data_dir)
        pass

    def run(self):
        # self.get_site_info()
        phenology_f = ''
        fdir = join(self.data_dir,'csv')
        fpath = join(fdir,'AMF_US-ARc_daily_wMODIS.csv')
        NDVI_dict = self.get_var_value_dict(fpath,'NDVI')
        gpp_dict = self.get_var_value_dict(fpath,'GPP_f')
        date_list = gpp_dict.keys()
        date_list = list(date_list)
        date_list.sort()
        ndvi_list = []
        gpp_list = []
        for date in date_list:
            NDVI = NDVI_dict[date]
            gpp = gpp_dict[date]
            ndvi_list.append(NDVI)
            gpp_list.append(gpp)
        plt.plot(date_list,ndvi_list,label='NDVI',color='r')
        plt.legend()
        plt.twinx()
        plt.plot(date_list,gpp_list,label='GPP',color='b')
        plt.legend()
        plt.show()
        exit()


    def get_site_info(self):
        f = join(self.data_dir,'Ameriflux_site_info.csv')
        df = pd.read_csv(f)
        T.print_head_n(df)

    def get_var_value_dict(self,csv_f,variable):
        df_i = pd.read_csv(csv_f)
        value_dict = {}
        for i,row in df_i.iterrows():
            year = row['Year']
            doy = row['DoY']
            year = int(year)
            doy = int(doy)
            value = row[variable]
            date_obj = datetime.datetime(year,1,1) + datetime.timedelta(doy - 1)
            value_dict[date_obj] = value
        return value_dict



class Terraclimate:
    def __init__(self):
        data_root = '/Volumes/NVME2T/greening_project_redo/data/'
        self.datadir = join(data_root,'Terraclimate')
        pass

    def run(self):
        # self.nc_to_tif_srad()
        # self.resample()
        # self.per_pix()
        self.extract_seasonal_period()
        pass

    def nc_to_tif_srad(self):
        outdir = self.datadir + '/srad/tif/'
        T.mk_dir(outdir,force=True)
        fdir = self.datadir + '/srad/nc11/'
        for fi in T.listdir(fdir):
            print(fi)
            if fi.startswith('.'):
                continue
            f = fdir + fi
            year = fi.split('.')[-2].split('_')[-1]
            # print(year)
            # exit()
            ncin = Dataset(f, 'r')
            ncin_xarr = xr.open_dataset(f)
            # print(ncin.variables)
            # exit()
            lat = ncin['lat']
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[1] - lat[0]
            longitude_start = lon[0]
            latitude_start = lat[0]
            time = ncin.variables['time']

            start = datetime.datetime(1900, 1, 1)
            # print(time)
            # for t in time:
            #     print(t)
            # exit()
            flag = 0
            for i in tqdm(range(len(time))):
                # print(i)
                flag += 1
                # print(time[i])
                date = start + datetime.timedelta(days=int(time[i]))
                year = str(date.year)
                # exit()
                month = '%02d' % date.month
                day = '%02d'%date.day
                date_str = year + month
                newRasterfn = outdir + date_str + '.tif'
                if os.path.isfile(newRasterfn):
                    continue
                # print(date_str)
                # exit()
                # if not date_str[:4] in valid_year:
                #     continue
                # print(date_str)
                # exit()
                # arr = ncin.variables['tmax'][i]
                arr = ncin_xarr['srad'][i]
                arr = np.array(arr)
                # print(arr)
                # grid = arr < 99999
                # arr[np.logical_not(grid)] = -999999
                newRasterfn = outdir + date_str + '.tif'
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # grid = np.ma.masked_where(grid>1000,grid)
                # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
                # plt.imshow(arr,'RdBu')
                # plt.colorbar()
                # plt.show()
                # nc_dic[date_str] = arr
                # exit()

    def resample(self):
        fdir = join(self.datadir, 'srad/tif')
        outdir = join(self.datadir, 'srad/tif_05')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=0.5)
        pass

    def per_pix(self):
        fdir = join(self.datadir, 'srad/tif_05')
        outdir = join(self.datadir, 'srad/per_pix_05')
        T.mk_dir(outdir, force=True)
        Pre_Process().data_transform(fdir, outdir)

    def extract_seasonal_period(self):
        fdir = join(self.datadir, 'srad/per_pix_05')
        outdir = join(self.datadir, 'srad/extract_seasonal_period')
        T.mk_dir(outdir, force=True)
        dict_all = T.load_npy_dir(fdir)
        period_dict = {
            'early':(4,5,6),
            'peak':(7,8),
            'late':(9,10)
        }
        for period in period_dict:
            outf = join(outdir,period)
            spatial_dict = {}
            for pix in tqdm(dict_all,desc=period):
                r,c = pix
                if r > 120:
                    continue
                vals = dict_all[pix]
                vals[vals < -9999] = np.nan
                if T.is_all_nan(vals):
                    continue
                period_months = period_dict[period]
                annual_vals = T.monthly_vals_to_annual_val(vals,period_months)
                annual_vals_anomaly = Pre_Process().z_score(annual_vals)
                spatial_dict[pix] = annual_vals_anomaly
            T.save_npy(spatial_dict,outf)


class ERA:
    def __init__(self):
        data_root = '/Volumes/NVME2T/greening_project_redo/data/'
        self.datadir = join(data_root, 'ERA')
        pass

    def run(self):
        # self.nc_to_array()
        # self.array_to_tif()
        # self.resample()
        # self.unit_convert()
        # self.per_pix()
        self.extract_late_period()

        pass

    def nc_to_array(self):
        outdir = join(self.datadir, 'array')
        T.mk_dir(outdir, force=True)
        fdir = join(self.datadir, 'nc')
        fpath = join(fdir, 'srad.nc')
        self.__nc_to_array(fpath, 'ssrd', outdir)

    def __nc_to_array(self, fname, var_name, outdir):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            time_list = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time_list = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units
        print(ncin.variables)
        exit()

        data = ncin.variables[var_name]
        for time_i in tqdm(range(len(time_list))):
            fpath = join(self.datadir, 'array', f'{time_i}.npy')
            arr = data[time_i]
            arr = np.array(arr)
            np.save(fpath, arr)

    def array_to_tif(self):
        fdir = join(self.datadir,'array')
        nc_path = join(self.datadir, 'nc/srad.nc')
        outdir = join(self.datadir,'tif')
        self.kernel1_array_to_tif(nc_path,'ssrd',outdir)

    def kernel2_array_to_tif(self,params):
        time_i,basetime_unit,basetime,time_list,xx,yy,outdir = params
        fpath = join(self.datadir, 'array', f'{time_i}.npy')
        # basetime_unit, basetime, time_list, xx, yy, data, outdir, var_name, time_i = params
        if basetime_unit == 'days':
            date = basetime + datetime.timedelta(days=int(time_list[time_i]))
        elif basetime_unit == 'years':
            date1 = basetime.strftime('%Y-%m-%d')
            base_year = basetime.year
            date2 = f'{int(base_year + time_list[time_i])}-01-01'
            delta_days = Tools().count_days_of_two_dates(date1, date2)
            date = basetime + datetime.timedelta(days=delta_days)
        elif basetime_unit == 'month' or basetime_unit == 'months':
            date1 = basetime.strftime('%Y-%m-%d')
            base_year = basetime.year
            base_month = basetime.month
            date2 = f'{int(base_year + time_list[time_i] // 12)}-{int(base_month + time_list[time_i] % 12)}-01'
            delta_days = Tools().count_days_of_two_dates(date1, date2)
            date = basetime + datetime.timedelta(days=delta_days)
        elif basetime_unit == 'seconds':
            date = basetime + datetime.timedelta(seconds=int(time_list[time_i]))
        elif basetime_unit == 'hours':
            date = basetime + datetime.timedelta(hours=int(time_list[time_i]))
        else:
            raise Exception('basetime unit not supported')
        time_str = time_list[time_i]
        mon = date.month
        year = date.year
        day = date.day
        outf_name = f'{year}{mon:02d}{day:02d}.tif'
        outpath = join(outdir, outf_name)
        if os.path.isfile(outpath):
            return
        arr = np.load(fpath)
        arr = np.array(arr)
        lon_list = []
        lat_list = []
        value_list = []
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                lon_i = xx[i][j]
                if lon_i > 180:
                    lon_i -= 360
                lat_i = yy[i][j]
                value_i = arr[i][j]
                lon_list.append(lon_i)
                lat_list.append(lat_i)
                value_list.append(value_i)
        DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)

    def kernel1_array_to_tif(self, fname, var_name, outdir,njobs=6):
        try:
            ncin = Dataset(fname, 'r')

        except:
            raise UserWarning('File not supported: ' + fname)
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time_list = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time_list = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        # print(basetime_unit)
        # print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')

        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        # data = ncin.variables[var_name]
        T.mk_dir(outdir, force=True)
        parmas_list = []
        for time_i in tqdm(range(len(time_list))):
            param = [time_i,basetime_unit,basetime,time_list,xx,yy,outdir]
            parmas_list.append(param)
        MULTIPROCESS(self.kernel2_array_to_tif,parmas_list).run(process=njobs)


    def resample(self):
        fdir = join(self.datadir, 'tif')
        outdir = join(self.datadir, 'tif_05')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath, outpath, res=0.5)

    def unit_convert(self):
        '''
        scale_factor: 623.5732836891337
        add_offset: 20432002.213358156
        _FillValue: -32767
        missing_value: -32767
        units: Jm ** -2
        :return:
        '''
        fdir = join(self.datadir, 'tif_05')
        outdir = join(self.datadir, 'tif_05_unit_convert')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            arr = ToRaster().raster2array(fpath)[0]
            arr[arr==-32767] = np.nan
            arr = (arr + 20432002.213358156)/623.5732836891337
            DIC_and_TIF().arr_to_tif(arr, outpath)

    def per_pix(self):
        fdir = join(self.datadir, 'tif_05_unit_convert')
        outdir = join(self.datadir, 'per_pix_05')
        T.mk_dir(outdir, force=True)
        Pre_Process().data_transform(fdir, outdir)

    def calculate_par(self):
        fdir = join(self.datadir, 'per_pix_05')
        outdir = join(self.datadir, 'par_05')
        T.mk_dir(outdir, force=True)
        dict_all = T.load_npy_dir(fdir)
        for pix in dict_all:
            vals = dict_all[pix]
            vals[vals < -9999] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = vals * 0.5
        pass

    def extract_seasonal_period(self):
        fdir = join(self.datadir, 'per_pix_05')
        outdir = join(self.datadir, 'extract_seasonal_period')
        T.mk_dir(outdir, force=True)
        dict_all = T.load_npy_dir(fdir)
        period_dict = {
            'early':(4,5,6),
            'peak':(7,8),
            'late':(9,10)
        }
        for period in period_dict:
            outf = join(outdir,period)
            spatial_dict = {}
            for pix in tqdm(dict_all,desc=period):
                r,c = pix
                if r > 120:
                    continue
                vals = dict_all[pix]
                vals[vals < -9999] = np.nan
                if T.is_all_nan(vals):
                    continue
                period_months = period_dict[period]
                annual_vals = T.monthly_vals_to_annual_val(vals,period_months)
                annual_vals_anomaly = Pre_Process().z_score(annual_vals)
                spatial_dict[pix] = annual_vals_anomaly
            T.save_npy(spatial_dict,outf)



class SPI:
    def __init__(self):
        self.datadir = '/Volumes/NVME2T/hotcold_drought/data/CRU/pre'
        pass

    def run(self):
        self.per_pix()
        pass

    def per_pix(self):
        fdir = join(self.datadir,'spi/1901-2020')
        outdir = join(self.datadir,'spi/1982-2018')
        T.mk_dir(outdir,force=True)
        f = join(fdir,'spi03.npy')
        # generate monthly date list
        date_list = []
        for y in range(1901,2021):
            for m in range(1,13):
                date_str = '%d-%02d'%(y,m)
                date_list.append(date_str)
        selected_date_list = []
        for y in range(1982,2019):
            for m in range(1,13):
                date_str = '%d-%02d'%(y,m)
                selected_date_list.append(date_str)
        selected_date_index = [date_list.index(date_str) for date_str in selected_date_list]

        spi_dict = T.load_npy(f)
        spatial_dict = {}
        for pix in spi_dict:
            r,c = pix
            if r > 120:
                continue
            vals = spi_dict[pix]
            vals = T.pick_vals_from_1darray(vals,selected_date_index)
            spatial_dict[pix] = vals
        T.save_npy(spatial_dict,join(outdir,'1982-2018_spi03'))

class SWE:

    def __init__(self):
        self.datadir = '/Volumes/NVME2T/greening_project_redo/data/GlobSnow_v3/SWE'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.projection_trans()
        # self.per_pix()
        # self.monthly_to_annual_tif()
        # self.per_pix_annual()
        # self.per_pix_annual_1999_2018()
        self.per_pix_annual_1999_2018_one_file()
        pass


    def kernel_nc_to_tif(self,fname,var_name,outf_name):
        try:
            ncin = Dataset(fname, 'r')
        except:
            raise UserWarning('File not supported: ' + fname)
        lat = ncin.variables['x'][:]
        lon = ncin.variables['y'][:]
        shape = np.shape(lat)
        data = ncin.variables[var_name]
        arr = data[::]
        arr = np.array(arr)
        longitude_start = lon[0]
        latitude_start = lat[0]
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        ToRaster().array2raster(outf_name, longitude_start, latitude_start, pixelWidth, pixelHeight,arr)

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            date_str = f.split('_')[0]
            fpath = join(fdir,f)
            outpath = join(outdir,date_str+'.tif')
            self.kernel_nc_to_tif(fpath,'swe',outpath)

    def projection_trans(self):
        nc_f = join(self.datadir,'nc','197901_northern_hemisphere_monthly_swe_0.25grid.nc')
        fdir = join(self.datadir, 'tif')
        outdir = join(self.datadir, 'tif_wgs84')
        ncin = Dataset(nc_f, 'r')
        crs = ncin.variables['crs']
        inRasterSRS = DIC_and_TIF().gen_srs_from_wkt(crs.spatial_ref)

        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            DIC_and_TIF().resample_reproj(fpath,outpath,0.5,srcSRS=inRasterSRS)

    def per_pix(self):
        fdir = join(self.datadir,'tif_wgs84')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir,force=True)
        flist = []
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            date_str = f.split('.')[0]
            year = date_str[:4]
            month = date_str[4:6]
            year = int(year)
            flist.append(fpath)
        Pre_Process().data_transform_with_date_list(fdir,outdir,flist)

    def get_date_list(self):
        fdir = join(self.datadir,'tif')
        date_list = []
        for f in T.listdir(fdir):
            date_str = f.split('.')[0]
            year = date_str[:4]
            month = date_str[4:6]
            year = int(year)
            month = int(month)
            date_obj = datetime.datetime(year,month,1)
            date_list.append(date_obj)
        return date_list


    def monthly_to_annual_tif(self):
        date_list = self.get_date_list()
        fdir = join(self.datadir,'per_pix')
        outdir = join(self.datadir,'tif_annual')
        T.mk_dir(outdir)
        spatial_dict = T.load_npy_dir(fdir)
        annual_vals_dict = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            vals[vals < 0] = np.nan
            if T.is_all_nan(vals):
                continue
            annual_dict = {}
            for i,val in enumerate(vals):
                date = date_list[i]
                year = date.year
                month = date.month
                if month >= 9:
                    year += 1
                if year not in annual_dict:
                    annual_dict[year] = []
                annual_dict[year].append(val)
            annual_sum = {}
            for year in annual_dict:
                vals = annual_dict[year]
                vals = np.array(vals)
                annual_sum[year] = np.nansum(vals)
            annual_vals_dict[pix] = annual_sum
        df = T.dic_to_df(annual_vals_dict,'pix')
        cols = df.columns.tolist()
        cols.remove('pix')
        for c in cols:
            spatial_dict_i = T.df_to_spatial_dic(df,c)
            outf = join(outdir,str(c)+'.tif')
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_i)
            arr = np.array(arr,dtype=np.float32)
            arr[arr<=0] = np.nan
            DIC_and_TIF().arr_to_tif(arr,outf)

    def per_pix_annual(self):
        fdir = join(self.datadir,'tif_annual')
        outdir = join(self.datadir,'per_pix_annual')
        T.mk_dir(outdir)
        flist = []
        for f in T.listdir(fdir):
            year = f.split('.')[0]
            fpath = join(fdir,f)
            year = int(year)
            if year <= 1981:
                continue
            flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,flist)

    def per_pix_annual_1999_2018(self):
        fdir = join(self.datadir, 'tif_annual')
        outdir = join(self.datadir, 'per_pix_annual_1999-2018')
        T.mk_dir(outdir)
        flist = []
        for f in T.listdir(fdir):
            year = f.split('.')[0]
            fpath = join(fdir, f)
            year = int(year)
            if year < 1999:
                continue
            flist.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, flist)

    def per_pix_annual_1999_2018_one_file(self):
        fdir = join(self.datadir, 'per_pix_annual_1999-2018')
        outdir = join(self.datadir, 'per_pix_annual_1999-2018_one_file')
        T.mk_dir(outdir)
        outf = join(outdir,'SWE_1999-2018.npy')
        spatial_dict = T.load_npy_dir(fdir)
        T.save_npy(spatial_dict,outf)



def main():
    # LAI().run()
    # seasonal_split_ly_NDVI()
    # GEE_AVHRR_LAI()).run()
    # GEE_MODIS_LAI().run()
    # GLC2000().run()
    # CCI_SM().run()
    # LAI_4g().run()
    MODIS_LAI().run()
    # LAI_3g().run()
    # VODCA().run()
    # LAI_4g_v101().run()
    # MODIS_LAI_BU_CMG().run()
    # TRENDY().run()
    # VODCA_GPP().run()
    # VOD_AMSRU().run()
    # FLUX_Matt().run()
    # Terraclimate().run()
    # ERA().run()
    # SPI().run()
    # SWE().run()
    # check_cci_sm()
    # f = '/Volumes/NVME2T/greening_project_redo/data/GEE_AVHRR_LAI/per_pix_clean/per_pix_dic_005.npy'
    # dic = T.load_npy(f)
    # for pix in dic:
    #     vals = dic[pix]
    #     print(len(vals))
    #     plt.plot(vals)
    #     plt.show()
    # Resample().run()
    # check_per_pix_data()
    pass



if __name__ == '__main__':
    main()
