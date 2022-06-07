# coding=utf-8
import zipfile

from __init__ import *

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
# 'LAI_3g': {
# 'path':join(data_root, 'LAI_3g/per_pix'),
# 'unit': 'm2/m2',
# 'start_year':1982,
# },
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
        self.datadir = join(data_root,'LAI4g')
        pass

    def run(self):
        self.resample()
        pass

    def resample(self):
        fdir = join(self.datadir,'gimms_lai4g_tiff')
        outdir = join(self.datadir,'tif_resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_resample = T.resample_nan(array,0.5,pixelWidth)
            array_resample[array_resample == 0] = np.nan
            array_resample = array_resample[::-1]
            DIC_and_TIF().arr_to_tif(array_resample,outf)


class LAI_3g:

    def __init__(self):
        self.datadir = join(data_root,'LAI3g')
        pass

    def run(self):
        self.tbl_to_tif()
        pass

    def __mat_to_tif(self,mat_f,outf):
        print(mat_f)
        mat_f_r = scipy.io.matlab.mio.loadmat(mat_f)
        print(mat_f_r)
        exit()
        # print(mat_f_r.keys())
        matrix = mat_f_r['outmat']
        # lai_arr = matrix[0][0][0]
        # lai_arr = matrix[0][0][1]
        # lai_arr = matrix[0][0][2]
        # lai_arr = matrix[0][0][3]
        lai_arr = matrix[0][0][4]
        lai_arr = np.array(lai_arr)
        lai_arr[lai_arr > 200] = np.nan
        lai_arr = lai_arr / 10.
        DIC_and_TIF(pixelsize=0.05).arr_to_tif(lai_arr,outf)

    def tbl_to_tif(self):

        fdir = join(self.datadir,'avhrrbulai_v02')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        # T.open_path_and_file(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f.replace('.mat','.tif'))
            self.__mat_to_tif(join(fdir,f),outf)

        pass

    def tbl_to_tif1(self):

        fdir = join(self.datadir,'avhrrbulai_v02')
        for f in T.listdir(fdir):
            print(f)

    def resample(self):
        fdir = join(self.datadir,'gimms_lai4g_tiff')
        outdir = join(self.datadir,'tif_resample')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_resample = T.resample_nan(array,0.5,pixelWidth)
            array_resample[array_resample == 0] = np.nan
            array_resample = array_resample[::-1]
            DIC_and_TIF().arr_to_tif(array_resample,outf)


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



class MODIS_LAI_BU_CMG:

    def __init__(self):
        self.datadir = join(data_root,'BU_MCD_LAI_CMG')

    def run(self):
        # self.resample_bill()
        # self.rename()
        # self.monthly_compose()
        self.per_pix()
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

def main():
    # LAI().run()
    # seasonal_split_ly_NDVI()
    # GEE_AVHRR_LAI()).run()
    # GEE_MODIS_LAI().run()
    # GLC2000().run()
    # CCI_SM().run()
    # LAI_4g().run()
    # LAI_3g().run()
    # VODCA().run()
    # LAI_4g_v101().run()
    # MODIS_LAI_BU_CMG().run()
    # check_cci_sm()
    # f = '/Volumes/NVME2T/greening_project_redo/data/GEE_AVHRR_LAI/per_pix_clean/per_pix_dic_005.npy'
    # dic = T.load_npy(f)
    # for pix in dic:
    #     vals = dic[pix]
    #     print(len(vals))
    #     plt.plot(vals)
    #     plt.show()
    # Resample().run()
    check_per_pix_data()
    pass



if __name__ == '__main__':
    main()
