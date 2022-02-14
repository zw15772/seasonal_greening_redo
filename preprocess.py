# coding=utf-8
import zipfile
from __init__ import *
# import ee

global_start_year = 1982
global_season_dic = {
    'early':(3,4,5),
    'peak':(6,7,8),
    'late':(9,10,11),
}
class LAI:

    def __init__(self):
        self.datadir = join(data_root,'LAI')
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





def main():
    # LAI().run()
    # seasonal_split_ly_NDVI()
    # GEE_AVHRR_LAI().run()
    GLC2000().run()
    # read_abl()
    # f = '/Volumes/NVME2T/greening_project_redo/data/GEE_AVHRR_LAI/per_pix_clean/per_pix_dic_005.npy'
    # dic = T.load_npy(f)
    # for pix in dic:
    #     vals = dic[pix]
    #     print(len(vals))
    #     plt.plot(vals)
    #     plt.show()


    pass



if __name__ == '__main__':
    main()
