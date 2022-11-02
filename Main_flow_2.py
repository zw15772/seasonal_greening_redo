# coding=utf-8
import matplotlib.pyplot as plt

from __init__ import *
result_root_this_script = join(results_root, 'Main_flow_2')
global_land_tif = join(this_root,'conf/land.tif')
global_start_year = 1982
global_season_dic = [
    'early',
    'peak',
    'late',
]

class Phenology_plot_LAI3g:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Phenology_plot_LAI3g',
                                                                                       result_root_this_script)
        pass

    def run(self):
        self.plot_phenology_every_n_years()
        pass

    def plot_phenology_every_n_years(self):
        dff = Dataframe_daily().dff
        df = T.load_df(dff)
        df = self.__clean_df(df)
        self._plot_phenology_every_n_years_lc_kp(df,n=5)
        self._plot_phenology_every_n_years_lc_kp(df,n=10)
        self._plot_phenology_every_n_years_lc_kp(df,n=15)
        self._plot_phenology_every_n_years_lc(df,n=5)
        self._plot_phenology_every_n_years_lc(df,n=10)
        self._plot_phenology_every_n_years_lc(df,n=15)
        self._plot_phenology_every_n_years_kp(df,n=5)
        self._plot_phenology_every_n_years_kp(df,n=10)
        self._plot_phenology_every_n_years_kp(df,n=15)
        pass

    def __clean_df(self,df):

        df = df[df['lat']>=30]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]

        return df

    def _plot_phenology_every_n_years_lc_kp(self,df,n):
        outdir = join(self.this_class_png,'plot_phenology_every_n_years/lc_kp',str(n))
        T.mk_dir(outdir,force=True)

        lc_col = 'landcover_GLC'
        koppen_col = 'koppen'
        cross_df_dict = T.cross_select_dataframe(df,lc_col,koppen_col)
        for df_key in cross_df_dict:
            df_i = cross_df_dict[df_key]
            self.__plot_phenology_every_n_years(df_i,n,df_key)
            df_key = list(df_key)
            outf = join(outdir,'_'.join(df_key)+'.pdf')
            plt.savefig(outf)
            plt.close()

    def _plot_phenology_every_n_years_lc(self,df,n):
        outdir = join(self.this_class_png,'plot_phenology_every_n_years/lc',str(n))
        T.mk_dir(outdir,force=True)
        dff = Dataframe_daily().dff
        df = T.load_df(dff)
        df = self.__clean_df(df)
        lc_col = 'landcover_GLC'
        cross_df_dict = T.cross_select_dataframe(df,lc_col)
        for df_key in cross_df_dict:
            df_i = cross_df_dict[df_key]
            self.__plot_phenology_every_n_years(df_i,n,df_key)
            outf = join(outdir,df_key+'.pdf')
            plt.savefig(outf)
            plt.close()

    def _plot_phenology_every_n_years_kp(self,df,n):
        outdir = join(self.this_class_png,'plot_phenology_every_n_years/kp',str(n))
        T.mk_dir(outdir,force=True)
        dff = Dataframe_daily().dff
        df = T.load_df(dff)
        df = self.__clean_df(df)
        koppen_col = 'koppen'
        cross_df_dict = T.cross_select_dataframe(df,koppen_col)
        for df_key in cross_df_dict:
            df_i = cross_df_dict[df_key]
            self.__plot_phenology_every_n_years(df_i,n,df_key)
            outf = join(outdir,df_key+'.pdf')
            plt.savefig(outf)
            plt.close()

    def __plot_phenology_every_n_years(self,df,n,df_key):   # plot phenology patterns DOY
        bin_n = np.arange(0,40,step=n)
        bin_n = list(bin_n)
        bin_n.pop(-1)
        bin_n.append(37)

        lai_mean_curve_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            LAI3g_matrix = row['LAI3g']
            for bin_i in range(len(bin_n)-1):
                bin_i_start = bin_n[bin_i]
                bin_i_end = bin_n[bin_i+1]
                LAI3g_matrix_bin = LAI3g_matrix[bin_i_start:bin_i_end]
                LAI3g_matrix_bin_mean = np.nanmean(LAI3g_matrix_bin,axis=0)
                key = str(global_start_year+bin_i_start)+'-'+str(global_start_year+bin_i_end-1)
                if not key in lai_mean_curve_dict:
                    lai_mean_curve_dict[key] = []
                lai_mean_curve_dict[key].append(LAI3g_matrix_bin_mean)
        colors = sns.color_palette('Spectral', len(lai_mean_curve_dict))
        color_flag = 0
        for key in lai_mean_curve_dict:
            matrix = np.array(lai_mean_curve_dict[key])
            matrix_mean = np.nanmean(matrix,axis=0)
            plt.plot(matrix_mean,label=key,color=colors[color_flag])
            color_flag += 1
        plt.legend()
        plt.title(str(df_key))
        plt.xlabel('DOY')
        plt.ylabel('LAI 3g (m2/m2)')


class Phenology_plot_AMSRU_VOD:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Phenology_plot_AMSRU_VOD',
                                                                                       result_root_this_script)
        self.datadir = join(data_root, 'AMSRU_VOD')
        self.year_range = list(range(2003, 2016))

    def run(self):
        # self.dataframe()
        self.plot_phenology_every_n_years(5)

    def dataframe(self):
        fdir = join(self.datadir, 'tif_per_pix')
        outdir = join(self.this_class_arr, 'dataframe')
        T.mkdir(outdir)
        # T.open_path_and_file(outdir)
        outf = join(outdir, 'dataframe.df')
        date_dir = join(self.datadir, 'dateobj')
        dateobj_dict = {}
        for year in T.listdir(fdir):
            # dict_year = T.load_npy_dir(join(fdir, year))
            dateobj_list = np.load(join(date_dir, year + '.npy'), allow_pickle=True)
            dateobj_dict[year] = dateobj_list

        spatial_dict = {}
        spatial_dict_date = {}
        for year in T.listdir(fdir):
            dict_year = T.load_npy_dir(join(fdir, year))
            print(year)
            for pix in dict_year:
                r,c = pix
                if r > 120:
                    continue
                vals = dict_year[pix]
                vals[vals<0] = np.nan
                if T.is_all_nan(vals):
                    continue
                if not pix in spatial_dict:
                    spatial_dict[pix] = []
                    spatial_dict_date[pix] = []
                spatial_dict[pix].append(vals)
                spatial_dict_date[pix].append(dateobj_dict[year])
        spatial_dict_all = {'VOD':spatial_dict,'date':spatial_dict_date}
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Dataframe_func(df).df
        T.save_df(df, outf)
        T.df_to_excel(df,outf)

    def plot_phenology_every_n_years(self,n):
        bin_n = np.arange(self.year_range[0],self.year_range[-1], step=n)
        bin_n = list(bin_n)
        bin_n.append(self.year_range[-1])
        print(bin_n)
        dff = join(self.this_class_arr,'dataframe' ,'dataframe.df')
        df = T.load_df(dff)
        for bin_i in range(len(bin_n)-1):
            bin_i_start = bin_n[bin_i]
            bin_i_end = bin_n[bin_i+1]
            select_year = list(range(bin_i_start,bin_i_end))
            print(select_year)
            #
            vod_mean_all = []
            for i,row in tqdm(df.iterrows(),total=len(df)):
                vod = row['VOD']
                date = row['date']
                vod_dict = {}
                for year in select_year:
                    try:
                        date_this_year = date[year-self.year_range[0]]
                        vod_this_year = vod[year-self.year_range[0]]
                    except:
                        # print('error',date)
                        # print('error',vod)
                        continue
                    doy_list = T.date_to_DOY(date_this_year)
                    values_dict = dict(zip(doy_list,vod_this_year))
                    for doy in values_dict:
                        if not doy in vod_dict:
                            vod_dict[doy] = []
                        vod_dict[doy].append(values_dict[doy])
                    # print(vod_dict)
                # exit()
                doy_list = list(vod_dict.keys())
                doy_list.sort()
                vod_mean = []
                for doy in range(1,366):
                    if not doy in vod_dict:
                        vod_mean.append(np.nan)
                    else:
                        vod_mean.append(np.nanmean(vod_dict[doy]))
                vod_mean_all.append(vod_mean)
            # vod_mean_all = np.array(vod_mean_all)
            print(np.shape(vod_mean_all))
            vod_mean_all_mean = np.nanmean(vod_mean_all,axis=0)
            plt.plot(vod_mean_all_mean,label=str(bin_i_start)+'-'+str(bin_i_end-1))

        plt.legend()
        plt.title('AMSRU VOD')
        plt.xlabel('DOY')
        plt.ylabel('VOD')
        plt.show()


class Seasonal_variables:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Seasonal_variables',
                                                                                       result_root_this_script)
        pass

    def run(self):
        self.pick_seasonal_values()
        # self.pick_seasonal_values_VODCAGPP()
        # self.pick_seasonal_values_VOD_AMSRU()
        # self.calculate_anomaly()
        # self.calculate_std_anomaly()
        pass


    def __daily_phenology(self):
        phenology_dir = join(data_root,'lai3g_pheno')

        early_start_f = join(phenology_dir, 'early_start.npy')
        early_end_f = join(phenology_dir, 'early_end.npy')
        late_start_f = join(phenology_dir, 'late_start.npy')
        late_end_f = join(phenology_dir, 'late_end.npy')

        early_start_dict = T.load_npy(early_start_f)
        early_end_dict = T.load_npy(early_end_f)
        late_start_dict = T.load_npy(late_start_f)
        late_end_dict = T.load_npy(late_end_f)

        early_dict = {}
        peak_dict = {}
        late_dict = {}

        for pix in tqdm(early_start_dict):
            early_start = early_start_dict[pix]
            early_end = early_end_dict[pix]
            late_start = late_start_dict[pix]
            late_end = late_end_dict[pix]

            mean_early_start = np.nanmean(early_start)
            mean_early_end = np.nanmean(early_end)
            mean_late_start = np.nanmean(late_start)
            mean_late_end = np.nanmean(late_end)

            early = range(int(mean_early_start),int(mean_early_end))
            peak = range(int(mean_early_end),int(mean_late_start))
            late = range(int(mean_late_start),int(mean_late_end))
            early = list(early)
            peak = list(peak)
            late = list(late)
            early = np.array(early)
            peak = np.array(peak)
            late = np.array(late)

            early_dict[pix] = early
            peak_dict[pix] = peak
            late_dict[pix] = late
        return early_dict,peak_dict,late_dict

    def __date_list_to_DOY(self,date_list):
        '''
        :param date_list: list of datetime objects
        :return:
        '''
        start_year = date_list[0].year
        start_date = datetime.datetime(start_year, 1, 1)
        date_delta = date_list - start_date + datetime.timedelta(days=1)
        DOY = [date.days for date in date_delta]
        return DOY

    def pick_seasonal_values(self):
        fdir = join(data_root,'daily_X')
        outdir = join(self.this_class_arr,'pick_seasonal_values')
        T.mk_dir(outdir,force=True)
        early_dict, peak_dict, late_dict = self.__daily_phenology()

        for folder in T.listdir(fdir):
            print('loading',folder)
            outdir_i = join(outdir,folder,'origin')
            T.mk_dir(outdir_i,force=True)
            vals_dict = T.load_npy_dir(join(fdir,folder))
            early_vals_dict = {}
            peak_vals_dict = {}
            late_vals_dict = {}
            for pix in tqdm(vals_dict):
                vals_list = vals_dict[pix]
                if not pix in early_dict:
                    continue
                early = early_dict[pix]
                peak = peak_dict[pix]
                late = late_dict[pix]
                early_vals_list = []
                peak_vals_list = []
                late_vals_list = []
                for vals in vals_list:
                    early_vals = T.pick_vals_from_1darray(vals,early)
                    peak_vals = T.pick_vals_from_1darray(vals,peak)
                    late_vals = T.pick_vals_from_1darray(vals,late)

                    early_mean = np.nanmean(early_vals)
                    peak_mean = np.nanmean(peak_vals)
                    late_mean = np.nanmean(late_vals)

                    early_vals_list.append(early_mean)
                    peak_vals_list.append(peak_mean)
                    late_vals_list.append(late_mean)
                early_vals_list = np.array(early_vals_list)
                peak_vals_list = np.array(peak_vals_list)
                late_vals_list = np.array(late_vals_list)

                early_vals_dict[pix] = early_vals_list
                peak_vals_dict[pix] = peak_vals_list
                late_vals_dict[pix] = late_vals_list
            early_outdir_i = join(outdir_i,'early')
            peak_outdir_i = join(outdir_i,'peak')
            late_outdir_i = join(outdir_i,'late')
            T.mk_dir(early_outdir_i,force=True)
            T.mk_dir(peak_outdir_i,force=True)
            T.mk_dir(late_outdir_i,force=True)
            T.save_distributed_perpix_dic(early_vals_dict,early_outdir_i)
            T.save_distributed_perpix_dic(peak_vals_dict,peak_outdir_i)
            T.save_distributed_perpix_dic(late_vals_dict,late_outdir_i)

    def pick_seasonal_values_VODCAGPP(self):
        folder = 'VODCA_GPP'
        fdir = join(data_root,folder,'tif_05_per_pix')
        outdir = join(self.this_class_arr,'pick_seasonal_values')
        date_list = np.load(join(data_root,folder,'date_obj_list.npy'),allow_pickle=True)
        years_list = [date.year for date in date_list]
        years_list_unique = list(set(years_list))
        years_list_unique.sort()
        T.mk_dir(outdir,force=True)
        early_dict, peak_dict, late_dict = self.__daily_phenology()

        print('loading',folder)
        outdir_i = join(outdir,folder,'origin')
        T.mk_dir(outdir_i,force=True)
        vals_dict = T.load_npy_dir(fdir)
        early_vals_dict = {}
        peak_vals_dict = {}
        late_vals_dict = {}
        for pix in tqdm(vals_dict):
            vals = vals_dict[pix]
            if T.is_all_nan(vals):
                continue
            vals[vals < 0] = np.nan
            vals[vals > 10] = np.nan
            # plt.plot(date_list,vals)
            # plt.scatter(date_list,vals)
            # plt.show()
            # print(vals)
            # exit()
            if not pix in early_dict:
                continue
            early = early_dict[pix]
            peak = peak_dict[pix]
            late = late_dict[pix]
            early_vals_list = []
            peak_vals_list = []
            late_vals_list = []
            for year in years_list_unique:
                year_index = [i for i in range(len(years_list)) if years_list[i]==year]
                date_this_year = date_list[year_index]
                vals_this_year = vals[year_index]
                DOY_this_year = self.__date_list_to_DOY(date_this_year)
                early_intersect = T.intersect(DOY_this_year,early)
                peak_intersect = T.intersect(DOY_this_year,peak)
                late_intersect = T.intersect(DOY_this_year,late)

                early_picked = [DOY_this_year.index(i) for i in early_intersect]
                peak_picked = [DOY_this_year.index(i) for i in peak_intersect]
                late_picked = [DOY_this_year.index(i) for i in late_intersect]

                early_picked.sort()
                peak_picked.sort()
                late_picked.sort()

                early_date_picked = [date_this_year[i] for i in early_picked]
                peak_date_picked = [date_this_year[i] for i in peak_picked]
                late_date_picked = [date_this_year[i] for i in late_picked]

                early_vals_picked = [vals_this_year[i] for i in early_picked]
                peak_vals_picked = [vals_this_year[i] for i in peak_picked]
                late_vals_picked = [vals_this_year[i] for i in late_picked]

                early_mean = np.nanmean(early_vals_picked)
                peak_mean = np.nanmean(peak_vals_picked)
                late_mean = np.nanmean(late_vals_picked)

                early_vals_list.append(early_mean)
                peak_vals_list.append(peak_mean)
                late_vals_list.append(late_mean)
            early_vals_list = np.array(early_vals_list)
            peak_vals_list = np.array(peak_vals_list)
            late_vals_list = np.array(late_vals_list)
            early_vals_dict[pix] = early_vals_list
            peak_vals_dict[pix] = peak_vals_list
            late_vals_dict[pix] = late_vals_list
        early_outdir_i = join(outdir_i,'early')
        peak_outdir_i = join(outdir_i,'peak')
        late_outdir_i = join(outdir_i,'late')
        T.mk_dir(early_outdir_i,force=True)
        T.mk_dir(peak_outdir_i,force=True)
        T.mk_dir(late_outdir_i,force=True)
        T.save_distributed_perpix_dic(early_vals_dict,early_outdir_i)
        T.save_distributed_perpix_dic(peak_vals_dict,peak_outdir_i)
        T.save_distributed_perpix_dic(late_vals_dict,late_outdir_i)

    def pick_seasonal_values_VOD_AMSRU(self):
        folder = 'AMSRU_VOD'
        fdir = join(data_root,folder,'tif_per_pix')
        outdir = join(self.this_class_arr,'pick_seasonal_values')
        T.mk_dir(outdir,force=True)
        early_dict, peak_dict, late_dict = self.__daily_phenology()
        outdir_i = join(outdir,folder,'origin')
        T.mk_dir(outdir_i,force=True)
        early_vals_dict = {}
        peak_vals_dict = {}
        late_vals_dict = {}

        for year in T.listdir(fdir):
            fdir_year = join(fdir,year)
            date_this_year = np.load(join(data_root, folder, f'dateobj/{year}.npy'), allow_pickle=True)
            DOY_this_year = self.__date_list_to_DOY(date_this_year)
            vals_dict_year = T.load_npy_dir(fdir_year,condition='')
            for pix in tqdm(vals_dict_year,desc=year):
                if not pix in early_dict:
                    continue
                early = early_dict[pix]
                peak = peak_dict[pix]
                late = late_dict[pix]
                vals_this_year = vals_dict_year[pix]
                vals_this_year[vals_this_year < 0] = np.nan
                if T.is_all_nan(vals_this_year):
                    continue
                early_intersect = T.intersect(DOY_this_year, early)
                peak_intersect = T.intersect(DOY_this_year, peak)
                late_intersect = T.intersect(DOY_this_year, late)

                early_picked = [DOY_this_year.index(i) for i in early_intersect]
                peak_picked = [DOY_this_year.index(i) for i in peak_intersect]
                late_picked = [DOY_this_year.index(i) for i in late_intersect]

                early_picked.sort()
                peak_picked.sort()
                late_picked.sort()

                early_date_picked = [date_this_year[i] for i in early_picked]
                peak_date_picked = [date_this_year[i] for i in peak_picked]
                late_date_picked = [date_this_year[i] for i in late_picked]

                early_vals_picked = [vals_this_year[i] for i in early_picked]
                peak_vals_picked = [vals_this_year[i] for i in peak_picked]
                late_vals_picked = [vals_this_year[i] for i in late_picked]

                # plt.plot(early_date_picked,early_vals_picked)
                # plt.plot(peak_date_picked,peak_vals_picked)
                # plt.plot(late_date_picked,late_vals_picked)
                # plt.scatter(early_date_picked, early_vals_picked)
                # plt.scatter(peak_date_picked, peak_vals_picked)
                # plt.scatter(late_date_picked, late_vals_picked)
                # plt.show()

                early_mean = np.nanmean(early_vals_picked)
                peak_mean = np.nanmean(peak_vals_picked)
                late_mean = np.nanmean(late_vals_picked)

                if not pix in early_vals_dict:
                    early_vals_dict[pix] = []
                    peak_vals_dict[pix] = []
                    late_vals_dict[pix] = []
                early_vals_dict[pix].append(early_mean)
                peak_vals_dict[pix].append(peak_mean)
                late_vals_dict[pix].append(late_mean)
        early_outdir_i = join(outdir_i, 'early')
        peak_outdir_i = join(outdir_i, 'peak')
        late_outdir_i = join(outdir_i, 'late')
        T.mk_dir(early_outdir_i, force=True)
        T.mk_dir(peak_outdir_i, force=True)
        T.mk_dir(late_outdir_i, force=True)
        T.save_distributed_perpix_dic(early_vals_dict, early_outdir_i)
        T.save_distributed_perpix_dic(peak_vals_dict, peak_outdir_i)
        T.save_distributed_perpix_dic(late_vals_dict, late_outdir_i)

    def calculate_std_anomaly(self):
        fdir = join(self.this_class_arr,'pick_seasonal_values')
        for var_i in T.listdir(fdir):
            if not 'AMSRU' in var_i:
                continue
            fdir_i = join(fdir,var_i,'origin')
            outdir_i = join(fdir,var_i,'std_anomaly')
            for season_i in T.listdir(fdir_i):
                fdir_i_i = join(fdir_i,season_i)
                outdir_i_i = join(outdir_i,season_i)
                T.mk_dir(outdir_i_i,force=True)
                vals_dict = T.load_npy_dir(fdir_i_i)
                anomaly_dict = {}
                for pix in tqdm(vals_dict):
                    vals = vals_dict[pix]
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    anomaly = (vals-mean)/std
                    anomaly_dict[pix] = anomaly
                T.save_distributed_perpix_dic(anomaly_dict,outdir_i_i)

    def calculate_anomaly(self):
        fdir = join(self.this_class_arr,'pick_seasonal_values')
        for var_i in T.listdir(fdir):
            if not 'AMSRU' in var_i:
                continue
            fdir_i = join(fdir,var_i,'origin')
            outdir_i = join(fdir,var_i,'anomaly')
            for season_i in T.listdir(fdir_i):
                fdir_i_i = join(fdir_i,season_i)
                outdir_i_i = join(outdir_i,season_i)
                T.mk_dir(outdir_i_i,force=True)
                vals_dict = T.load_npy_dir(fdir_i_i)
                anomaly_dict = {}
                for pix in tqdm(vals_dict):
                    vals = vals_dict[pix]
                    mean = np.nanmean(vals)
                    anomaly = vals-mean
                    anomaly_dict[pix] = anomaly
                T.save_distributed_perpix_dic(anomaly_dict,outdir_i_i)
                # exit()

class Trend:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Trend',
                                                                                       result_root_this_script)
        pass

    def run(self):
        # self.LAI3g_df()
        # self.VODCA_GPP_df()
        # self.AMSRU_VOD_df()
        # self.Temp_df()
        # self.phenology_df()
        # self.trend_tif('VODCA_GPP')
        # self.trend_tif('AMSRU_VOD')
        # self.trend_tif('LAI3g')
        # self.trend_tif('temp')
        # self.late_Temp_trend_two_periods()
        # self.plot_late_Temp_trend_two_periods_tif()
        # self.check_wen_data()
        self.check_era_and_terraclimate()
        pass

    def LAI3g_df(self):
        fdir = join(Seasonal_variables().this_class_arr,'pick_seasonal_values/LAI3g')
        outdir = join(self.this_class_arr,'LAI3g')
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early','peak','late']
        result_df = []
        for period in period_list:
            fdir_i = join(fdir,'origin',period)
            dict_i = T.load_npy_dir(fdir_i)
            result_dict = {}
            for pix in dict_i:
                vals = dict_i[pix]
                a, b, r, p = T.nan_line_fit(list(range(len(vals))),vals)
                result_dict_i = {f'{period}_a':a,f'{period}_b':b,f'{period}_r':r,f'{period}_p':p}
                result_dict[pix] = result_dict_i
            df_i = T.dic_to_df(result_dict,'pix')
            result_df.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df,result_df,'pix')
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def VODCA_GPP_df(self):
        fdir = join(Seasonal_variables().this_class_arr,'pick_seasonal_values/VODCA_GPP')
        outdir = join(self.this_class_arr,'VODCA_GPP')
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early','peak','late']
        result_df = []
        for period in period_list:
            fdir_i = join(fdir,'origin',period)
            dict_i = T.load_npy_dir(fdir_i)
            result_dict = {}
            for pix in dict_i:
                vals = dict_i[pix]
                a, b, r, p = T.nan_line_fit(list(range(len(vals))),vals)
                result_dict_i = {f'{period}_a':a,f'{period}_b':b,f'{period}_r':r,f'{period}_p':p}
                result_dict[pix] = result_dict_i
            df_i = T.dic_to_df(result_dict,'pix')
            result_df.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df,result_df,'pix')
        df = Dataframe_func(df).df
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def AMSRU_VOD_df(self):
        fdir = join(Seasonal_variables().this_class_arr,'pick_seasonal_values/AMSRU_VOD')
        outdir = join(self.this_class_arr,'AMSRU_VOD')
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early','peak','late']
        result_df = []
        for period in period_list:
            fdir_i = join(fdir,'origin',period)
            dict_i = T.load_npy_dir(fdir_i)
            result_dict = {}
            for pix in dict_i:
                vals = dict_i[pix]
                a, b, r, p = T.nan_line_fit(list(range(len(vals))),vals)
                result_dict_i = {f'{period}_a':a,f'{period}_b':b,f'{period}_r':r,f'{period}_p':p}
                result_dict[pix] = result_dict_i
            df_i = T.dic_to_df(result_dict,'pix')
            result_df.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df,result_df,'pix')
        df = Dataframe_func(df).df
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def Temp_df(self):
        fdir = join(Seasonal_variables().this_class_arr,'pick_seasonal_values/temp')
        outdir = join(self.this_class_arr,'temp')
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early','peak','late']
        result_df = []
        for period in period_list:
            fdir_i = join(fdir,'origin',period)
            dict_i = T.load_npy_dir(fdir_i)
            result_dict = {}
            for pix in dict_i:
                vals = dict_i[pix]
                a, b, r, p = T.nan_line_fit(list(range(len(vals))),vals)
                result_dict_i = {f'{period}_a':a,f'{period}_b':b,f'{period}_r':r,f'{period}_p':p}
                result_dict[pix] = result_dict_i
            df_i = T.dic_to_df(result_dict,'pix')
            result_df.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df,result_df,'pix')
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def phenology_df(self):
        phenology_dir = join(data_root,'lai3g_pheno')
        outdir = join(self.this_class_arr, 'phenology')
        T.mk_dir(outdir, force=True)
        T.open_path_and_file(outdir)

        early_start_f = join(phenology_dir, 'early_start.npy')
        early_end_f = join(phenology_dir, 'early_end.npy')
        late_start_f = join(phenology_dir, 'late_start.npy')
        late_end_f = join(phenology_dir, 'late_end.npy')
        early_length_f = join(phenology_dir, 'early_length.npy')
        mid_length_f = join(phenology_dir, 'mid_length.npy')
        late_length_f = join(phenology_dir, 'late_length.npy')
        growing_season_length_f = join(phenology_dir, 'growing_season_length.npy')

        early_start_dict = T.load_npy(early_start_f)
        early_end_dict = T.load_npy(early_end_f)
        late_start_dict = T.load_npy(late_start_f)
        late_end_dict = T.load_npy(late_end_f)
        early_length_dict = T.load_npy(early_length_f)
        mid_length_dict = T.load_npy(mid_length_f)
        late_length_dict = T.load_npy(late_length_f)
        growing_season_length_dict = T.load_npy(growing_season_length_f)

        all_dict = {'early_start':early_start_dict,'early_end':early_end_dict,'late_start':late_start_dict,'late_end':late_end_dict,
                    'early_length':early_length_dict,'mid_length':mid_length_dict,'late_length':late_length_dict,'growing_season_length':growing_season_length_dict}

        df_list = []
        for key in all_dict:
            dict_i = all_dict[key]
            trend_dict = {}
            for pix in tqdm(dict_i,desc=key):
                vals = dict_i[pix]
                a, b, r, p = T.nan_line_fit(list(range(len(vals))),vals)
                trend_dict_i = {f'{key}_a':a,f'{key}_b':b,f'{key}_r':r,f'{key}_p':p}
                trend_dict[pix] = trend_dict_i
            df_i = T.dic_to_df(trend_dict,'pix')
            df_list.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df,df_list,'pix')
        df = Dataframe_func(df).df
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def trend_tif(self,variable):
        outdir = join(self.this_class_tif,variable)
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        dff = join(self.this_class_arr,variable,'dataframe.df')
        df = T.load_df(dff)
        T.print_head_n(df,5)
        period_list = ['early','peak','late']
        for period in period_list:
            a_dict = T.df_to_spatial_dic(df, f'{period}_a')
            p_dict = T.df_to_spatial_dic(df, f'{period}_p')
            outf_a = join(outdir, f'{variable}_{period}_a.tif')
            outf_p = join(outdir, f'{variable}_{period}_p.tif')
            arr_a = DIC_and_TIF().pix_dic_to_tif(a_dict, outf_a)
            arr_p = DIC_and_TIF().pix_dic_to_tif(p_dict, outf_p)

    def late_Temp_trend_two_periods(self):
        fdir = join(Seasonal_variables().this_class_arr, 'pick_seasonal_values/temp')
        outdir = join(self.this_class_arr, 'temp')
        T.mk_dir(outdir, force=True)
        # T.open_path_and_file(outdir)
        # year_range_list = [(1982,2001),(2002,2020)]
        year_range_list = [(2002,2020)]
        period_list = ['early', 'peak', 'late']
        result_df = []
        for year_range in year_range_list:
            for period in period_list:
                fdir_i = join(fdir, 'origin', period)
                dict_i = T.load_npy_dir(fdir_i)
                result_dict = {}
                for pix in dict_i:
                    vals = dict_i[pix]
                    vals_i = []
                    for year in range(year_range[0],year_range[1]+1):
                        print(year)
                        vals_i.append(vals[year-1982])
                    exit()

                    a, b, r, p = T.nan_line_fit(list(range(len(vals_i))), vals_i)
                    result_dict_i = {f'{period}_{year_range[0]}-{year_range[1]}_a':a,
                                     f'{period}_{year_range[0]}-{year_range[1]}_b':b,
                                     f'{period}_{year_range[0]}-{year_range[1]}_r':r,
                                     f'{period}_{year_range[0]}-{year_range[1]}_p':p}
                    result_dict[pix] = result_dict_i
                df_i = T.dic_to_df(result_dict, 'pix')
                result_df.append(df_i)
        df = pd.DataFrame()
        df = T.join_df_list(df, result_df, 'pix')
        outf = join(outdir, 'dataframe.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def plot_late_Temp_trend_two_periods_tif(self):
        variable = 'temp'
        outdir = join(self.this_class_tif, variable)
        T.mk_dir(outdir, force=True)
        T.open_path_and_file(outdir)
        dff = join(self.this_class_arr, variable, 'dataframe.df')
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        year_range_list = [(1982, 2001), (2002, 2020)]
        # year_range_list = [(2002,2020)]
        for year_range in year_range_list:
            period_list = ['early', 'peak', 'late']
            for period in period_list:
                a_dict = T.df_to_spatial_dic(df, f'{period}_{year_range[0]}-{year_range[1]}_a')
                p_dict = T.df_to_spatial_dic(df, f'{period}_{year_range[0]}-{year_range[1]}_p')
                outf_a = join(outdir, f'{variable}_{period}_{year_range[0]}-{year_range[1]}_a.tif')
                outf_p = join(outdir, f'{variable}_{period}_{year_range[0]}-{year_range[1]}_p.tif')
                arr_a = DIC_and_TIF().pix_dic_to_tif(a_dict, outf_a)
                arr_p = DIC_and_TIF().pix_dic_to_tif(p_dict, outf_p)
        pass


    def check_wen_data(self):
        fdir = join(temporary_root,'temperature_epl_wen/temperature')
        outdir = join(temporary_root,'temperature_epl_wen/temperature_trend')
        T.mk_dir(outdir,force=True)
        period_list = ['early','peak','late']
        year_range_list = [(1982,2000),(2001,2018)]
        for year_range in year_range_list:
            print(year_range)
            for period in period_list:
                outdir_period = join(outdir,period)
                T.mk_dir(outdir_period,force=True)
                fpath = join(fdir,f'during_{period}_Temp.npy')
                dict_i = T.load_npy(fpath)
                spatial_dict_a = {}
                spatial_dict_p = {}
                for pix in dict_i:
                    vals = dict_i[pix]
                    if len(vals) != 37:
                        continue
                    vals_i = []
                    for year in range(year_range[0],year_range[1]+1):
                        vals_i.append(vals[year-1982])
                    a,b,r,p = T.nan_line_fit(list(range(len(vals_i))),vals_i)
                    spatial_dict_a[pix] = a
                    spatial_dict_p[pix] = p
                outf_a = join(outdir_period,f'{period}_{year_range[0]}-{year_range[1]}_a.tif')
                outf_p = join(outdir_period,f'{period}_{year_range[0]}-{year_range[1]}_p.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_a,outf_a)
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_p,outf_p)

    def check_era_and_terraclimate(self):
        outdir = join(temporary_root,'check_era_and_terraclimate')
        T.mk_dir(outdir,force=True)
        # T.open_path_and_file(outdir)
        arr_background = np.array([1]*720*360).reshape(360,720)
        DIC_and_TIF(originX=-180.,pixelsize=0.5).arr_to_tif(arr_background,join(outdir,'background.tif'))
        exit()
        f_terra = '/Volumes/NVME2T/greening_project_redo/data/terraclimate/srad/extract_seasonal_period/late.npy'
        f_era = '/Volumes/NVME2T/greening_project_redo/data/ERA/extract_seasonal_period/late.npy'
        f_wen = '/Volumes/NVME2T/greening_project_redo/data/1982_2018_final_wen/during_late_PAR_zscore.npy'

        dict_terra = T.load_npy(f_terra)
        dict_era = T.load_npy(f_era)
        dict_wen = T.load_npy(f_wen)

        spatial_dict1 = {}
        spatial_dict2 = {}
        spatial_dict3 = {}
        for pix in tqdm(dict_terra):
            vals_terra = dict_terra[pix]
            if not pix in dict_era:
                continue
            if not pix in dict_wen:
                continue
            vals_era = dict_era[pix]
            vals_wen = dict_wen[pix]
            try:
                r1,p = stats.pearsonr(vals_terra, vals_era)
            except:
                r1 = np.nan
            try:
                r2,p = stats.pearsonr(vals_terra,vals_wen)
            except:
                r2 = np.nan
            try:
                r3,p = stats.pearsonr(vals_era,vals_wen)
            except:
                r3 = np.nan
            spatial_dict1[pix] = r1
            spatial_dict2[pix] = r2
            spatial_dict3[pix] = r3
        DIC_and_TIF().pix_dic_to_tif(spatial_dict1,join(outdir,'terra_era.tif'))
        DIC_and_TIF().pix_dic_to_tif(spatial_dict2,join(outdir,'terra_wen.tif'))
        DIC_and_TIF().pix_dic_to_tif(spatial_dict3,join(outdir,'era_wen.tif'))

        pass



class Time_series:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Time_series',
                                                                                       result_root_this_script)

    def run(self):
        self.foo()
        pass

    def foo(self):
        variable = 'VODCA_GPP'
        # variable = 'LAI3g'
        # variable = 'AMSRU_VOD'
        fdir = join(Seasonal_variables().this_class_arr,f'pick_seasonal_values/{variable}')
        # outdir = join(self.this_class_arr,variable)
        # T.mk_dir(outdir,force=True)
        # T.open_path_and_file(outdir)
        period_list = ['early','peak','late']
        vals_dict = {}
        for period in period_list:
            fdir_i = join(fdir,'origin',period)
            # fdir_i = join(fdir,'std_anomaly',period)
            dict_i = T.load_npy_dir(fdir_i)
            vals_dict[period] = dict_i
        df = T.spatial_dics_to_df(vals_dict)
        df = Dataframe_func(df).df
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        plt.figure()
        for period in period_list:
            arr = df[period].tolist()
            arr_new = []
            for i in arr:
                # print(len(i))
                if not len(i) == 13:
                    continue
                arr_new.append(i)
            early_arr_mean = np.nanmean(arr_new,axis=0)
            plt.plot(early_arr_mean,label=period,linewidth=4)
        plt.legend()
        plt.title(variable)
        plt.show()

class Sankey_plot:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot',
                                                                                       result_root_this_script)

        self.dff = self.this_class_arr + 'LAI3g.df'
        pass

    def run(self):
        df = self.__gen_df_init()
        # T.print_head_n(df,10)
        # df = self.build_df()
        # df = Dataframe_func(df).df
        # df = self.classify_trend(df)
        self.plot_spatial_pix(df)
        self.plot_sankey(df)
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)
        pass



    def __load_df(self,):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self, ):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __clean_df(self, df):
        # kp = T.get_df_unique_val_list(df, 'koppen')
        # lc = T.get_df_unique_val_list(df, 'landcover_GLC')
        # print(lc)
        #
        df = df[df['row'] < 120]
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['max_trend'] < 10]
        #
        df = df[df['landcover_GLC'] != 'Crop']
        #
        # df = df[df['landcover_GLC'] == 'Grass']
        # df= df[df['during_late_LAI3g_trend'] <0.05]
        return df


    def build_df(self):
        trend_dff = join(Trend().this_class_arr,'LAI3g/dataframe.df')
        trend_df = T.load_df(trend_dff)
        period_list = ['early', 'peak', 'late']
        all_dict_result = {}
        for period in period_list:
            trend_dict = T.df_to_spatial_dic(trend_df, f'{period}_a')
            p_dict = T.df_to_spatial_dic(trend_df, f'{period}_p')
            all_dict_result[f'{period}_trend'] = trend_dict
            all_dict_result[f'{period}_p'] = p_dict
        df = T.spatial_dics_to_df(all_dict_result)
        return df


    def classify_trend(self,df):
        # reindex dataframe
        df = df.reset_index()
        df = df.dropna()

        period_list = ['early', 'peak', 'late']
        for period in period_list:
            trend_col_name = f'{period}_trend'
            p_col_name = f'{period}_p'
            classified_col_name = f'{period}_classified'
            for i,row in tqdm(df.iterrows(),total=df.shape[0],desc=f'{period}'):
                if row[p_col_name]>0.05:
                    df.loc[i,classified_col_name] = f'{period}_not-significant'
                else:
                    if row[trend_col_name]>0:
                        df.loc[i,classified_col_name] = f'{period}_positive'
                    else:
                        df.loc[i,classified_col_name] = f'{period}_negative'
        return df

    def plot_spatial_pix(self, df):
        tif_template = global_land_tif
        self.plot_df_spatial_pix(df, tif_template)
        plt.show()

    def plot_df_spatial_pix(self,df,global_land_tif):
        pix_list = df['pix'].tolist()
        pix_list = list(set(pix_list))
        spatial_dict = {pix:1 for pix in pix_list}
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.imshow(arr, interpolation='nearest', cmap='jet', )


    def plot_sankey(self,df):
        # T.open_path_and_file(outdir)
        period_list = ['early', 'peak', 'late']
        # period_list = ['late']
        level_list = ['positive','not-significant','negative']
        early_status_list = [f'early_{var}' for var in level_list]
        peak_status_list = [f'peak_{var}' for var in level_list]
        late_status_list = [f'late_{var}' for var in level_list]

        node_list = early_status_list + peak_status_list + late_status_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        # node_color_list = [color_dict[var_] for var_ in var_list]
        # node_color_list = node_color_list * 3
        # print(node_color_list)
        # exit()
        early_var_col = 'early_classified'
        peak_var_col = 'peak_classified'
        late_var_col = 'late_classified'

        source = []
        target = []
        value = []
        # color_list = []
        # node_list_anomaly_value_mean = []
        # anomaly_value_list = []
        node_list_with_ratio = []
        node_name_list = []
        for early_status in early_status_list:
            # print(early_status)
            # exit()
            df_early = df[df[early_var_col] == early_status]
            ratio = len(df_early)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{early_status}')
        for peak_status in peak_status_list:
            df_peak = df[df[peak_var_col] == peak_status]
            ratio = len(df_peak)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{peak_status}')
        for late_status in late_status_list:
            df_late = df[df[late_var_col] == late_status]
            ratio = len(df_late)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{late_status}')
        for early_status in early_status_list:
            df_early = df[df[early_var_col] == early_status]
            early_count = len(df_early)
            for peak_status in peak_status_list:
                df_peak = df_early[df_early[peak_var_col] == peak_status]
                peak_count = len(df_peak)
                source.append(position_dict[early_status])
                target.append(position_dict[peak_status])
                value.append(peak_count)
                for late_status in late_status_list:
                    df_late = df_peak[df_peak[late_var_col] == late_status]
                    late_count = len(df_late)
                    source.append(position_dict[peak_status])
                    target.append(position_dict[late_status])
                    value.append(late_count)
        link = dict(source=source, target=target, value=value, )
        # node = dict(label=node_list_with_ratio, pad=100,
        node = dict(label=node_name_list, pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    # x=node_x,
                    # y=node_y,
                    # color=node_color_list
                    )
        data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
        fig = go.Figure(data)
        fig.update_layout(title='Whole', font=dict(size=18))
        fig.show()


class Dataframe_func:

    def __init__(self,df):
        print('add lon lat')
        df = self.add_lon_lat(df)
        print('add landcover')
        df = self.add_GLC_landcover_data_to_df(df)
        print('add NDVI mask')
        df = self.add_NDVI_mask(df)
        print('add Aridity Index')
        df = self.add_AI_to_df(df)
        df = self.add_Humid_nonhumid(df)
        df = self.clean_df(df)
        self.df = df

    def clean_df(self,df):

        df = df[df['lat']>=30]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]

        return df
    def add_GLC_landcover_data_to_df(self, df):

        f = join(data_root,'Base_data/LC_reclass2.npy')

        val_dic=T.load_npy(f)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_Koppen_data_to_df(self, df):
        f = join(data_root, 'Base_data/Koppen/koppen_reclass_spatial_dic.npy')
        koppen_dic=T.load_npy(f)
        koppen_list=[]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df
    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'Base_data/NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):  #
        P_PET_fdir = join(data_root, 'Base_data/aridity_P_PET_dic')
        p_pet_dic = self.P_PET_ratio(P_PET_fdir)
        df = T.add_spatial_dic_to_df(df, p_pet_dic, 'AI')
        return df

    def P_PET_class(self,df):
        dic = T.df_to_spatial_dic(df, 'AI')
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Arid'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Semi Arid'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Semi Humid'
                # label = 2
            dic_reclass[pix] = label
        return dic_reclass

    def add_Humid_nonhumid(self,df):
        P_PET_dic_reclass = self.P_PET_class(df)
        df = T.add_spatial_dic_to_df(df,P_PET_dic_reclass,'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Dry'
        return df


    def P_PET_ratio(self, P_PET_fdir):
        outf = join(temporary_root, 'P_PET_ratio_dic.npy')
        if os.path.exists(outf):
            return T.load_npy(outf)
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        T.save_npy(dic_long_term, outf)
        return dic_long_term
    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df

class Dataframe_daily:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe_daily',
                                                                                       result_root_this_script)
        # self.mode = 'anomaly_detrend'
        # self.mode = 'std_anomaly_detrend'
        # self.mode = 'origin'
        # outdir = join(self.this_class_arr, 'Dynamic_pheno', self.mode)
        # T.mk_dir(outdir, force=True)
        self.dff = join(self.this_class_arr, 'data_frame.df')
        pass

    def run(self):
        df = self.__gen_df_init()

        # df = self.add_variables()
        # df = self.add_LAI3g()
        # df = self.add_sos_eos(df)
        self.get_dateobj(df)

        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)
        pass


    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff


    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df



    def add_LAI3g(self):
        fdir = join(data_root,'daily_X/LAI3g')
        val_dict = T.load_npy_dir(fdir)
        LAI3g_new_dict = {}
        for pix in val_dict:
            vals = val_dict[pix]
            if len(vals) == 0:
                continue
            vals_new = []
            for val_i in vals:
                val_i = np.array(val_i)
                vals_new.append(val_i)
            vals_new = np.array(vals_new)
            LAI3g_new_dict[pix] = vals_new
        spatial_dict_all = {'LAI3g':LAI3g_new_dict}
        df = T.spatial_dics_to_df(spatial_dict_all)
        return df

    def add_sos_eos(self,df):
        fdir = join(data_root,'lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        result_dict = {}
        for pix in tqdm(sos_dict):
            sos = sos_dict[pix]
            eos = eos_dict[pix]
            sos_mean = np.nanmean(sos)
            eos_mean = np.nanmean(eos)
            sos_std = np.nanstd(sos)
            eos_std = np.nanstd(eos)
            sos_anomaly = sos - sos_mean
            eos_anomaly = eos - eos_mean
            sos_std_anomaly = sos_anomaly / sos_std
            eos_std_anomaly = eos_anomaly / eos_std
            result_dict_i = {
                             'sos_std_anomaly':sos_std_anomaly,'eos_std_anomaly':eos_std_anomaly,
                             'sos_anomaly':sos_anomaly,'eos_anomaly':eos_anomaly,'sos':sos,'eos':eos,}
            result_dict[pix] = result_dict_i
        df_pheno = T.dic_to_df(result_dict,'pix')
        df = T.join_df_list(df,[df_pheno],'pix')

        return df

    def get_dateobj(self,df):
        outf = join(self.this_class_arr,'dateobj.npy')
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            LAI3g = row['LAI3g']
            LAI3g = np.array(LAI3g)
            LAI3g_date_list = []
            # LAI3g_val_list = []
            for j in range(len(LAI3g)):
                LAI3g_i = LAI3g[j]
                LAI3g_i = np.array(LAI3g_i)
                # LAI3g_val_list.append(LAI3g_i)
                year = global_start_year + j
                date_start = datetime.datetime(year,1,1)
                date_list = []
                value_list = []
                for k in range(len(LAI3g[j])):
                    date = date_start + datetime.timedelta(days=k)
                    date_list.append(date)
                date_list = np.array(date_list)
                LAI3g_date_list.append(date_list)
            LAI3g_date_list = np.array(LAI3g_date_list)
            np.save(outf,LAI3g_date_list)
            break


    def add_variables(self):
        var_mode = self.mode
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal')
        period_list = ['early','peak','late']

        data_dict = {}
        col_list = []

        for var in T.listdir(fdir):
            print(var)
            for period in period_list:
                fdir_i = join(fdir,var,var_mode,period)
                dict_i = T.load_npy_dir(fdir_i)
                col_name = f'{var}_{period}'
                col_list.append(col_name)
                data_dict[col_name] = dict_i
        df = T.spatial_dics_to_df(data_dict)
        return df


class Carryover_AMSRU_VOD_GPP:
    def __init__(self):

        pass

    def run(self):
        # self.AMSRU_VOD()
        # self.VODCA_GPP()
        self.VODCA_GPP_timeseries()
        # self.LAI3g()
        pass
    def LAI3g(self):
        lai_start_year = 1982
        year_list = list(range(lai_start_year,2019))
        dateobj_list = [datetime.datetime(year,1,1) for year in year_list]
        fdir_LAI3g = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','LAI3g')
        period_list = ['early','peak','late']
        result_dict = {}
        for period in period_list:
            fdir_LAI3g_i = join(fdir_LAI3g,'std_anomaly',period)
            LAI3g_dict = T.load_npy_dir(fdir_LAI3g_i)
            result_dict.update({f'{period}_LAI3g':LAI3g_dict})
        df = T.spatial_dics_to_df(result_dict)
        T.print_head_n(df,10)
        enhanced_vod_list = []
        not_enhanced_vod_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            late_lai = row['late_LAI3g']
            if type(late_lai) == float:
                continue
            early_lai = row['early_LAI3g']
            peak_lai = row['peak_LAI3g']

            early_peak_lai_mean = (early_lai + peak_lai)/2
            # annual_lai_mean = (early_lai + peak_lai + late_lai)/3
            annual_lai_mean = late_lai
            early_peak_lai_mean = early_peak_lai_mean[:-1] # remove last value
            annual_lai_mean = annual_lai_mean[1:] # remove first value
            enhanced_early_peak = early_peak_lai_mean > 0
            not_enhanced_early_peak = early_peak_lai_mean < 0
            late_vod_enhanced = copy.copy(annual_lai_mean)
            late_vod_not_enhanced = copy.copy(annual_lai_mean)
            late_vod_enhanced[~enhanced_early_peak] = np.nan
            late_vod_not_enhanced[~not_enhanced_early_peak] = np.nan
            # late_vod_enhanced_mean = np.nanmean(late_vod_enhanced)
            # late_vod_not_enhanced_mean = np.nanmean(late_vod_not_enhanced)
            for enhanecd in late_vod_enhanced:
                if np.isnan(enhanecd):
                    continue
                enhanced_vod_list.append(enhanecd)
            for not_enhanced in late_vod_not_enhanced:
                if np.isnan(not_enhanced):
                    continue
                not_enhanced_vod_list.append(not_enhanced)
            # enhanced_vod_list.append(late_vod_enhanced_mean)
            # not_enhanced_vod_list.append(late_vod_not_enhanced_mean)
        plt.hist(enhanced_vod_list,bins=50,alpha=0.3,label='enhanced',density=True,range=(-3,3))
        plt.hist(not_enhanced_vod_list,bins=50,alpha=0.3,label='not enhanced',density=True,range=(-3,3))
        plt.legend()
        plt.title('LAI3g')
        plt.show()

    def AMSRU_VOD(self):
        lai_start_year = 1982
        vod_start_year = 2003
        year_list = list(range(vod_start_year,2016))
        dateobj_list = [datetime.datetime(year,1,1) for year in year_list]
        fdir_VOD = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','AMSRU_VOD')
        fdir_LAI3g = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','LAI3g')
        period_list = ['early','peak','late']
        result_dict = {}
        for period in period_list:
            fdir_VOD_i = join(fdir_VOD,'std_anomaly',period)
            fdir_LAI3g_i = join(fdir_LAI3g,'std_anomaly',period)
            VOD_dict = T.load_npy_dir(fdir_VOD_i)
            LAI3g_dict = T.load_npy_dir(fdir_LAI3g_i)
            result_dict.update({f'{period}_VOD':VOD_dict})
            result_dict.update({f'{period}_LAI3g':LAI3g_dict})
        df = T.spatial_dics_to_df(result_dict)
        enhanced_vod_list = []
        not_enhanced_vod_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            late_vod = row['late_VOD']
            if type(late_vod) == float:
                continue
            early_lai = row['early_LAI3g']
            peak_lai = row['peak_LAI3g']

            early_peak_lai_mean = (early_lai + peak_lai)/2
            early_peak_lai_mean_cut = early_peak_lai_mean[(2003-1982):]
            early_peak_lai_mean_cut = early_peak_lai_mean_cut[:-3] # 2003-2015
            if not len(early_peak_lai_mean_cut) == len(late_vod):
                continue
            enhanced_early_peak = early_peak_lai_mean_cut > 0
            not_enhanced_early_peak = early_peak_lai_mean_cut < 0
            late_vod_enhanced = copy.copy(late_vod)
            late_vod_not_enhanced = copy.copy(late_vod)
            late_vod_enhanced[~enhanced_early_peak] = np.nan
            late_vod_not_enhanced[~not_enhanced_early_peak] = np.nan
            late_vod_enhanced_mean = np.nanmean(late_vod_enhanced)
            late_vod_not_enhanced_mean = np.nanmean(late_vod_not_enhanced)
            enhanced_vod_list.append(late_vod_enhanced_mean)
            not_enhanced_vod_list.append(late_vod_not_enhanced_mean)
        plt.hist(enhanced_vod_list,bins=50,alpha=0.3,label='enhanced',density=True,range=(-2,2))
        plt.hist(not_enhanced_vod_list,bins=50,alpha=0.3,label='not enhanced',density=True,range=(-2,2))
        plt.legend()
        plt.title('AMSRU VOD')
        plt.show()

    def VODCA_GPP(self):
        lai_start_year = 1982
        vod_start_year = 1988
        lai_end_year = 1982 + 37 - 1
        vod_end_year = 1988 + 33 - 1
        year_list = list(range(vod_start_year,2019))
        dateobj_list = [datetime.datetime(year,1,1) for year in year_list]
        fdir_VOD = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','VODCA_GPP')
        fdir_LAI3g = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','LAI3g')
        period_list = ['early','peak','late']
        result_dict = {}
        for period in period_list:
            fdir_VOD_i = join(fdir_VOD,'std_anomaly',period)
            fdir_LAI3g_i = join(fdir_LAI3g,'std_anomaly',period)
            VOD_dict = T.load_npy_dir(fdir_VOD_i)
            LAI3g_dict = T.load_npy_dir(fdir_LAI3g_i)
            result_dict.update({f'{period}_GPP':VOD_dict})
            result_dict.update({f'{period}_LAI3g':LAI3g_dict})
        df = T.spatial_dics_to_df(result_dict)
        df = Dataframe_func(df).df
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        # plt.show()
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        for lc in lc_list:
            df_lc = df[df['landcover_GLC'] == lc]
            enhanced_vod_list = []
            not_enhanced_vod_list = []
            plt.figure()
            for i,row in tqdm(df_lc.iterrows(),total=len(df_lc)):
                late_vod = row['late_GPP']
                if type(late_vod) == float:
                    continue
                early_lai = row['early_LAI3g']
                peak_lai = row['peak_LAI3g']
                early_vod = row['early_GPP']
                peak_vod = row['peak_GPP']

                early_peak_lai_mean = (early_lai + peak_lai)/2
                early_peak_lai_mean_cut = early_peak_lai_mean[(vod_start_year-lai_start_year):]
                late_vod = late_vod[:len(early_peak_lai_mean_cut)]
                early_vod = early_vod[:len(early_peak_lai_mean_cut)]
                peak_vod = peak_vod[:len(early_peak_lai_mean_cut)]
                annual_vod = (early_vod + peak_vod + late_vod)/3
                # early_peak_lai_mean_cut = early_peak_lai_mean_cut[:-1] # remove last year
                # annual_vod = annual_vod[1:] # remove first year
                # print(len(early_peak_lai_mean_cut))
                # print(len(late_vod))
                # exit()
                # # early_peak_lai_mean_cut = early_peak_lai_mean_cut[:-3] # 2003-2015
                # if not len(early_peak_lai_mean_cut) == len(late_vod):
                #     continue
                enhanced_early_peak = early_peak_lai_mean_cut > 0
                not_enhanced_early_peak = early_peak_lai_mean_cut < 0
                late_vod_enhanced = copy.copy(annual_vod)
                late_vod_not_enhanced = copy.copy(annual_vod)
                late_vod_enhanced[~enhanced_early_peak] = np.nan
                late_vod_not_enhanced[~not_enhanced_early_peak] = np.nan
                # late_vod_enhanced_mean = np.nanmean(late_vod_enhanced)
                # late_vod_not_enhanced_mean = np.nanmean(late_vod_not_enhanced)
                for i in late_vod_enhanced:
                    if np.isnan(i):
                        continue
                    enhanced_vod_list.append(i)
                for i in late_vod_not_enhanced:
                    if np.isnan(i):
                        continue
                    not_enhanced_vod_list.append(i)
                # enhanced_vod_list.append(late_vod_enhanced_mean)
                # not_enhanced_vod_list.append(late_vod_not_enhanced_mean)
            plt.hist(enhanced_vod_list,bins=50,alpha=0.3,label='enhanced',density=True,range=(-3,3))
            plt.hist(not_enhanced_vod_list,bins=50,alpha=0.3,label='not enhanced',density=True,range=(-3,3))
            plt.legend()
            plt.title(f'VODCA_GPP {lc}')
        plt.show()


    def VODCA_GPP_timeseries(self):
        lai_start_year = 1982
        vod_start_year = 1988
        lai_end_year = 1982 + 37 - 1
        vod_end_year = 1988 + 33 - 1
        year_list = list(range(vod_start_year,2019))
        dateobj_list = [datetime.datetime(year,1,1) for year in year_list]
        fdir_VOD = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','VODCA_GPP')
        fdir_LAI3g = join(Seasonal_variables().this_class_arr,'pick_seasonal_values','LAI3g')
        period_list = ['early','peak','late']
        result_dict = {}
        for period in period_list:
            fdir_VOD_i = join(fdir_VOD,'std_anomaly',period)
            fdir_LAI3g_i = join(fdir_LAI3g,'std_anomaly',period)
            VOD_dict = T.load_npy_dir(fdir_VOD_i)
            LAI3g_dict = T.load_npy_dir(fdir_LAI3g_i)
            result_dict.update({f'{period}_GPP':VOD_dict})
            result_dict.update({f'{period}_LAI3g':LAI3g_dict})
        df = T.spatial_dics_to_df(result_dict)
        df = Dataframe_func(df).df
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        # plt.show()
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # for lc in lc_list:
        # df_lc = df[df['landcover_GLC'] == lc]
        df_lc = df
        enhanced_vod_dict = {}
        enhanced_lai_dict = {}
        not_enhanced_vod_dict = {}
        plt.figure()
        for i,row in tqdm(df_lc.iterrows(),total=len(df_lc)):
            late_vod = row['late_GPP']
            if type(late_vod) == float:
                continue
            early_lai = row['early_LAI3g']
            peak_lai = row['peak_LAI3g']
            late_lai = row['late_LAI3g']
            early_vod = row['early_GPP']
            peak_vod = row['peak_GPP']

            early_peak_lai_mean = (early_lai + peak_lai)/2
            early_peak_lai_mean_cut = early_peak_lai_mean[(vod_start_year-lai_start_year):]
            late_lai_cut = late_lai[(vod_start_year-lai_start_year):]
            late_vod = late_vod[:len(early_peak_lai_mean_cut)]
            early_vod = early_vod[:len(early_peak_lai_mean_cut)]
            peak_vod = peak_vod[:len(early_peak_lai_mean_cut)]
            annual_vod = (early_vod + peak_vod + late_vod)/3
            # annual_vod = late_vod
            # annual_vod = early_vod
            # annual_vod = annual_vod
            # early_peak_lai_mean_cut = early_peak_lai_mean_cut[:-1] # remove last year
            # annual_vod = annual_vod[1:] # remove first year
            # print(len(early_peak_lai_mean_cut))
            # print(len(late_vod))
            # exit()
            # # early_peak_lai_mean_cut = early_peak_lai_mean_cut[:-3] # 2003-2015
            # if not len(early_peak_lai_mean_cut) == len(late_vod):
            #     continue
            enhanced_early_peak = early_peak_lai_mean_cut > 0
            not_enhanced_early_peak = early_peak_lai_mean_cut < 0
            late_vod_enhanced = copy.copy(annual_vod)
            late_vod_not_enhanced = copy.copy(annual_vod)
            # early_peak_lai_mean_cut_value = copy.copy(early_peak_lai_mean_cut)
            late_lai_cut_value = copy.copy(late_lai_cut)
            late_vod_enhanced[~enhanced_early_peak] = np.nan
            late_vod_not_enhanced[~not_enhanced_early_peak] = np.nan
            # early_peak_lai_mean_cut_value[~enhanced_early_peak] = np.nan
            late_lai_cut_value[~enhanced_early_peak] = np.nan
            # late_vod_enhanced_mean = np.nanmean(late_vod_enhanced)
            # late_vod_not_enhanced_mean = np.nanmean(late_vod_not_enhanced)
            # for i in late_vod_enhanced:
            #     if not i in enhanced_vod_dict:
            #         enhanced_vod_dict[i] = []
            #     # enhanced_vod_dict
            #     # enhanced_vod_list.append(i)
            for i in range(len(late_vod_enhanced)):
                value = late_vod_enhanced[i]
                if not i in enhanced_vod_dict:
                    enhanced_vod_dict[i] = []
                else:
                    enhanced_vod_dict[i].append(value)
            for i in range(len(late_lai_cut_value)):
                value = late_lai_cut_value[i]
                if not i in enhanced_lai_dict:
                    enhanced_lai_dict[i] = []
                else:
                    enhanced_lai_dict[i].append(value)
        enhanced_lai_list = []
        enhanced_lai_std_list = []
        enhanced_vod_list = []
        enhanced_vod_std_list = []
        keys = list(enhanced_vod_dict.keys())
        keys.sort()
        for i in keys:
            enhanced_lai = enhanced_lai_dict[i]
            enhanced_vod = enhanced_vod_dict[i]

            enhanced_lai_mean = np.nanmean(enhanced_lai)
            enhanced_lai_std = np.nanstd(enhanced_lai)
            enhanced_vod_mean = np.nanmean(enhanced_vod)
            enhanced_vod_std = np.nanstd(enhanced_vod)

            enhanced_lai_list.append(enhanced_lai_mean)
            enhanced_lai_std_list.append(enhanced_lai_std)
            enhanced_vod_list.append(enhanced_vod_mean)
            enhanced_vod_std_list.append(enhanced_vod_std)

        plt.plot(keys,enhanced_lai_list,label='enhanced_lai')
        # Plot().plot_line_with_error_bar(keys,enhanced_lai_list,enhanced_lai_std_list)
        plt.plot(keys,enhanced_vod_list,label='enhanced_vod')
        # Plot().plot_line_with_error_bar(keys,enhanced_vod_list,enhanced_vod_std_list)
        plt.legend()
        # plt.title(lc)
        plt.show()



def main():
    # Phenology().run()
    # Phenology_plot_AMSRU_VOD().run()
    # Seasonal_variables().run()
    Trend().run()
    # Time_series().run()
    # Sankey_plot().run()
    # Dataframe_daily().run()
    # Carryover_AMSRU_VOD_GPP().run()
    pass

if __name__ == '__main__':
    main()