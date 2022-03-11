# coding=utf-8
import platform

import matplotlib.pyplot as plt
import networkx
import numpy as np

from preprocess import *
results_root_main_flow = join(results_root,'Main_flow')
import pingouin as pg
import eli5
from eli5.sklearn import PermutationImportance

global_n = 15
T.mk_dir(results_root_main_flow,force=True)


class Global_vars:
    def __init__(self):
        self.land_tif = join(this_root,'temp/tif_template.tif')
        pass

    def load_df(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = self.clean_df(df)
        return df

    def clean_df(self,df):
        df = df[df['lat'] < 60]
        df = df[df['lat'] > 30]
        df = df[df['HI_reclass']=='Non Humid']  # focus on dryland
        return df

    def cal_relative_change(self,vals):
        base = vals[:3]
        base = np.nanmean(base)
        relative_change_list = []
        for val in vals:
            change_rate = (val - base) / base
            relative_change_list.append(change_rate)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list

class Dataframe:

    def __init__(self):
        self.this_class_arr = join(results_root_main_flow,'arr/Dataframe/')
        self.dff = self.this_class_arr + 'dataframe_1982-2015.df'
        self.P_PET_fdir =data_root+ 'aridity_P_PET_dic/'
        T.mk_dir(self.this_class_arr,force=True)

    def __x_dir(self,season):
        x_dir = join(data_root,
      f'1982-2015_original_extraction_all_seasons/1982-2015_extraction_during_{season}_growing_season_static')
      # f'1982-2018_original_extraction_all_seasons/1982-2018_extraction_during_{season}_growing_season_static')
        return x_dir

    def run(self):
        df = self.__gen_df_init()
        df = self.add_data(df)
        df = self.add_lon_lat_to_df(df)
        df = self.add_Humid_nonhumid(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff, random=False)


    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff


    def __add_pix_to_df(self,df):
        pix_dic = DIC_and_TIF().void_spatial_dic()
        pix_list = []
        for pix in pix_dic:
            pix_list.append(pix)
        pix_list.sort()
        df['pix'] = pix_list
        return df

    def add_data(self,df):
        df = self.__add_pix_to_df(df)
        df = pd.DataFrame(df)
        columns = []
        for season in global_season_dic:
            x_dir = self.__x_dir(season)
            if not isdir(x_dir):
                continue
            for f in T.listdir(x_dir):
                print(season, f)
                fpath = join(x_dir,f)
                dic = T.load_npy(fpath)
                key_name = f.replace('during_','')
                key_name = key_name.replace('.npy','')
                columns.append(key_name)
                df = T.add_spatial_dic_to_df(df,dic,key_name)
        df = df.dropna(how='all',subset=columns)
        return df


    def add_ly_NDVI(self,df):
        fdir = join(data_root,'NDVI_ly/per_pix_seasonal')
        for season in global_season_dic:
            f = join(fdir,f'{season}.npy')
            dic = T.load_npy(f)
            df = T.add_spatial_dic_to_df(df,dic,f'{season}_ly_NDVI')
        return df

    def add_GEE_LAI(self,df):
        fdir = join(data_root, 'GEE_AVHRR_LAI/per_pix_seasonal')
        for season in global_season_dic:
            f = join(fdir, f'{season}.npy')
            dic = T.load_npy(f)
            df = T.add_spatial_dic_to_df(df, dic, f'{season}_GEE_AVHRR_LAI')
        return df
        pass

    def add_lon_lat_to_df(self,df):
        lon_list = []
        lat_list = []
        D = DIC_and_TIF()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            lon,lat = D.pix_to_lon_lat(pix)
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def drop_n_std(self,vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_class(self):
        outdir = join(self.this_class_arr,'P_PET_class')
        T.mk_dir(outdir)
        outf = join(outdir,'P_PET_class.npy')
        if isfile(outf):
            return T.load_npy(outf)
        dic = self.P_PET_ratio(self.P_PET_fdir)
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
        T.save_npy(dic_reclass,outf)
        return dic_reclass

    def P_PET_ratio(self,P_PET_fdir):
        # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            vals[vals == 0] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term
    def add_Humid_nonhumid(self,df):
        P_PET_dic_reclass = self.P_PET_class()
        df = T.add_spatial_dic_to_df(df,P_PET_dic_reclass,'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Non Humid'
        return df

    def add_lc(self,df):
        lc_f = join(GLC2000().datadir,'lc_dic_reclass.npy')
        lc_dic = T.load_npy(lc_f)
        df = T.add_spatial_dic_to_df(df,lc_dic,'GLC2000')
        return df

    def add_NDVI_mask(self, df):
        # ndvi_mask_tif = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
        ndvi_mask_tif = '/Volumes/NVME2T/greening_project_redo/conf/NDVI_mask.tif'
        # ndvi_mask_tif = join(NDVI().datadir,'NDVI_mask.tif')
        arr = ToRaster().raster2array(ndvi_mask_tif)[0]
        arr = T.mask_999999_arr(arr, warning=False)
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        df = T.add_spatial_dic_to_df(df, dic, 'ndvi_mask_mark')
        df = pd.DataFrame(df)
        df = df.dropna(subset=['ndvi_mask_mark'])
        return df


class Greening_phenomena:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Greening_phenomena',results_root_main_flow)
        pass

    def run(self):
        # self.trend_spatial_tif()
        # self.trend_spatial_pvalue_point_shp()
        # self.timesereis()
        # self.moving_window_mean_timeseries()
        self.moving_window_trend_timeseries()
        # self.moving_window_area_ratio_timeseries()
        pass


    def trend_spatial_tif(self):
        outdir = join(self.this_class_tif,'trend_spatial_tif')
        T.mk_dir(outdir)
        df = Global_vars().load_df()
        K = KDE_plot()
        for i in df:
            str_i = str(i)
            if not 'LAI_GIMMS' in i:
                continue
            spatial_dic_time_series = T.df_to_spatial_dic(df,i)
            trend_spatial_dic = {}
            trend_spatial_dic_p = {}
            outf_p = join(outdir,i+'_p.tif')
            outf_r = join(outdir,i+'_k.tif')
            for pix in tqdm(spatial_dic_time_series,desc=str_i):
                vals = spatial_dic_time_series[pix]
                if type(vals) == float:
                    continue
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = K.linefit(x,y)
                trend_spatial_dic[pix] = a
                trend_spatial_dic_p[pix] = p

            DIC_and_TIF().pix_dic_to_tif(trend_spatial_dic,outf_r)
            DIC_and_TIF().pix_dic_to_tif(trend_spatial_dic_p,outf_p)


    def trend_spatial_pvalue_point_shp(self):
        fdir = join(self.this_class_tif,'trend_spatial_tif')
        outdir = join(self.this_class_tif,'trend_spatial_pvalue_point_shp')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('_p.tif'):
                continue
            intif = join(fdir,f)
            outtif = join(outdir,f)
            ToRaster().resample_reproj(intif,outtif,res=2)
        for f in T.listdir(outdir):
            out_shp_f = join(outdir,f+'.shp')
            if not f.endswith('_p.tif'):
                continue
            arr = ToRaster().raster2array(join(outdir,f))[0]
            arr = T.mask_999999_arr(arr,warning=False)
            arr[arr>0.1] = np.nan
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            point_list = []
            for pix in dic:
                val = dic[pix]
                if np.isnan(val):
                    continue
                lon,lat = DIC_and_TIF(tif_template=join(outdir,f)).pix_to_lon_lat(pix)
                # print(lon,lat)
                list_i = [lon,lat,1]
                point_list.append(list_i)
            T.point_to_shp(point_list,out_shp_f)

    def timesereis(self):
        outdir = join(self.this_class_tif, 'timesereis')
        T.mk_dir(outdir)
        df = Global_vars().load_df()
        K = KDE_plot()
        for i in df:
            str_i = str(i)
            # if not 'GIMMS' in i:
            if not 'VOD' in i:
                continue
            print(i)
            spatial_dic_time_series = T.df_to_spatial_dic(df, i)
            matrix = []
            for pix in tqdm(spatial_dic_time_series, desc=str_i):
                vals = spatial_dic_time_series[pix]
                if type(vals) == float:
                    continue
                # vals = T.pick_vals_from_1darray(vals,period)
                vals = Global_vars().cal_relative_change(vals)
                matrix.append(vals)
            matrix = np.array(matrix)
            matrix_T = matrix.T
            y = []
            std_list = []
            for series in matrix_T:
                mean = np.nanmean(series)
                std = np.nanstd(series)
                y.append(mean)
                std_list.append(std)
            x = list(range(len(y)))
            plt.plot(x,y)
            # Plot().plot_line_with_error_bar(x,y,std_list)
        plt.show()

        pass

    def moving_window_mean_timeseries(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early': 'g',
            'peak': 'r',
            'late': 'b',
        }
        fdir = join(Moving_window().this_class_arr, 'mean')
        dff = join(fdir, 'mean_relative_change.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df, HI_reclass_var)
        df = df[df[HI_reclass_var] == 'Non Humid']
        # var_ ='VOD'
        var_ ='NDVI'
        # window_list = self.__get_window_list(df, 'LAI_GIMMS')
        window_list = self.__get_window_list(df, var_)
        # print(window_list)
        # exit()
        for xvar in Moving_window().all_var_list:
            if not var_ in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list, color=color_dic[season], label=season)
            plt.legend()
            plt.title(xvar)
        plt.show()

    def moving_window_trend_timeseries(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early': 'g',
            'peak': 'r',
            'late': 'b',
        }
        fdir = join(Moving_window().this_class_arr, 'trend')
        dff = join(fdir, 'trend.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df, HI_reclass_var)
        df = df[df[HI_reclass_var] == 'Non Humid']

        window_list = self.__get_window_list(df, 'VOD')
        # print(window_list)
        # exit()
        for xvar in Moving_window().all_var_list:
            if not 'NDVI' in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}_r'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list, color=color_dic[season], label=season)
            plt.legend()
            plt.title(xvar)
        plt.show()

    def moving_window_area_ratio_timeseries(self):
        # y_variable = 'LAI_GIMMS'
        # y_variable = 'VOD'
        y_variable = 'GIMMS_NDVI'
        dff = join(Moving_window().this_class_arr, 'trend', 'trend.df')
        df_all = T.load_df(dff)
        for season in global_season_dic:
            plt.figure()
            color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
            K = KDE_plot()
            y_var = f'{season}_{y_variable}'
            window_list = self.__get_window_list(df_all,y_var)
            for w in window_list:
                df_new = pd.DataFrame()
                df_new['r'] = df_all[f'{w}_{y_var}_r']
                df_new['p'] = df_all[f'{w}_{y_var}_p']
                df_new = df_new.dropna()
                df_p_non_sig = df_new[df_new['p'] > 0.1]
                df_p_05 = df_new[df_new['p'] <= 0.1]
                df_p_05 = df_p_05[df_p_05['p'] >= 0.05]
                df_p_sig = df_new[df_new['p'] < 0.05]
                pos_05 = df_p_05[df_p_05['r'] >= 0]
                neg_05 = df_p_05[df_p_05['r'] < 0]
                pos_sig = df_p_sig[df_p_sig['r'] >= 0]
                neg_sig = df_p_sig[df_p_sig['r'] < 0]
                non_sig_ratio = len(df_p_non_sig) / len(df_new)
                pos_05_ratio = len(pos_05) / len(df_new)
                neg_05_ratio = len(neg_05) / len(df_new)
                pos_sig_ratio = len(pos_sig) / len(df_new)
                neg_sig_ratio = len(neg_sig) / len(df_new)
                bars = [pos_sig_ratio, pos_05_ratio, non_sig_ratio, neg_05_ratio, neg_sig_ratio]

                bottom = 0
                color_flag = 0
                for b in bars:
                    plt.bar(f'{w}', b, bottom=bottom, color=color_list[color_flag])
                    bottom += b
                    color_flag += 1
            # plt.title(title)
            plt.tight_layout()
        plt.show()

    def moving_window_trend(self):

        pass
    def __get_window_list(self,df,x_var_1):
        # x_var_1 = self.x_var_list[0]
        # x_var_1 = self.y_var
        window_list = []
        for col in df:
            col = str(col)
            if x_var_1 in col:
                window = col.split('_')[0]
                window_list.append(int(window))
        window_list = T.drop_repeat_val_from_list(window_list)
        return window_list

    # def

class Partial_corr:

    def __init__(self,season):
        self.season = season
        self.__config__()
        self.this_class_arr = join(results_root_main_flow,'arr/Partial_corr/')
        T.mk_dir(self.this_class_arr)
        self.dff = join(self.this_class_arr,f'Dataframe_{season}.df')


    def __config__(self):
        self.n = global_n

        self.vars_list = [
            f'{self.season}_CO2',
            f'{self.season}_VPD',
            f'{self.season}_PAR',
            f'{self.season}_temperature',
            f'{self.season}_CCI_SM',
            # f'{self.season}_GIMMS_NDVI',
            # f'{self.season}_GEE_AVHRR_LAI',
            f'{self.season}_LAI_GIMMS',
        ]
        self.x_var_list = [f'{self.season}_CO2',
                      f'{self.season}_VPD',
                      f'{self.season}_PAR',
                      f'{self.season}_temperature',
                      f'{self.season}_CCI_SM', ]
        # self.y_var = f'{self.season}_GIMMS_NDVI'
        # self.y_var = f'{self.season}_GEE_AVHRR_LAI'
        self.y_var = f'{self.season}_LAI_GIMMS'

    def run(self):
        self.cal_p_correlation()
        # self.dic_to_df()
        pass

    def __load_df(self):
        print('loading df ...')
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = df[df['lat']>30]
        print('loaded')
        T.print_head_n(df)
        return df

    def __partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p

    def __cal_partial_correlation(self,df,x_var_list):

        partial_correlation={}
        partial_correlation_p_value = {}
        for x in x_var_list:
            # print(x)
            x_var_list_valid_new_cov=copy.copy(x_var_list)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,self.y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def __cal_anomaly(self,vals):
        mean = np.nanmean(vals)
        anomaly_list = []
        for val in vals:
            anomaly = val - mean
            anomaly_list.append(anomaly)
        anomaly_list = np.array(anomaly_list)
        return anomaly_list

    def __cal_relative_change(self,vals):
        base = vals[:3]
        base = np.nanmean(base)
        relative_change_list = []
        for val in vals:
            change_rate = (val - base) / base
            relative_change_list.append(change_rate)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list


    def cal_p_correlation(self):
        outdir = join(self.this_class_arr,f'{self.y_var}_relative_change')

        T.mk_dir(outdir,force=True)
        df = self.__load_df()
        df = df.dropna()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            outpath = join(outdir,f'window_{w+1:02d}-{self.n}.npy')
            if isfile(outpath):
                continue
            pcorr_results_dic_window_i = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=str(w)):
                pix = row.pix
                r,c = pix
                if r > 120:
                    continue
                val_dic = {}
                for x_var in self.vars_list:
                    xvals = row[x_var]
                    pick_index = list(range(w,w+self.n))
                    picked_vals = T.pick_vals_from_1darray(xvals,pick_index)
                    if not x_var == self.y_var:
                        picked_vals_anomaly = self.__cal_anomaly(picked_vals)
                    else:
                        picked_vals_anomaly = self.__cal_relative_change(picked_vals)
                    val_dic[x_var] = picked_vals_anomaly
                df_i = pd.DataFrame()
                for col in self.vars_list:
                    vals_list = []
                    vals = val_dic[col]
                    for val in vals:
                        vals_list.append(val)
                    df_i[col] = vals_list
                df_i = df_i.dropna(axis=1)
                x_var_valid = []
                for col in self.x_var_list:
                    if col in df_i:
                        x_var_valid.append(col)
                try:
                    dic_partial_corr,dic_partial_corr_p = self.__cal_partial_correlation(df_i,x_var_valid)
                    pcorr_results_dic_window_i[pix] = {'pcorr':dic_partial_corr,'p':dic_partial_corr_p}
                except:
                    continue
            T.save_npy(pcorr_results_dic_window_i,outpath)


    def dic_to_df(self):
        fpath = join(self.this_class_arr,f'Partial_corr_{self.season}/{self.y_var}')
        df_total = pd.DataFrame()
        for f in T.listdir(fpath):
            window = f.replace(f'-{self.n:02d}.npy','')
            window = window.replace('window_','')
            window = int(window)
            dic = T.load_npy(join(fpath,f))
            dic_new = {}
            for pix in dic:
                dic_new[pix] = dic[pix]['pcorr']
            df = T.dic_to_df(dic_new, 'pix')
            pix_list = df['pix']
            df_total['pix'] = pix_list
            for col in self.x_var_list:
                vals = df[col]
                new_col_name = f'{window}_{col}'
                df_total[new_col_name] = vals

        for x_var in self.x_var_list:
            mean_list = []
            std_list = []
            for w in range(99):
                window = w + 1
                col_name = f'{window}_{x_var}'
                if not col_name in df_total:
                    continue
                vals = df_total[col_name]
                vals = np.array(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                up = mean + std
                down = mean - std
                vals[vals>up] = np.nan
                vals[vals<down] = np.nan
                mean_list.append(np.nanmean(vals))
                std_list.append(std / 8)
            x = list(range(len(mean_list)))
            y = mean_list
            yerr = std_list
            plt.figure()
            plt.plot(x,y,color='r')
            plt.title(x_var)
            Plot_line().plot_line_with_gradient_error_band(x, y, yerr,max_alpha=0.8,min_alpha=0.1,pow=2,color_gradient_n=500)
        plt.show()



class Analysis:

    def __init__(self):
        self.n = global_n
        self.this_class_arr = join(results_root_main_flow, 'arr/Analysis')
        self.this_class_png = join(results_root_main_flow, 'png/Analysis')
        T.mk_dir(self.this_class_arr,force=True)
        T.mk_dir(self.this_class_png,force=True)
        pass

    def run(self):
        # self.spatial_greening_trend()
        self.greening_trend_time_series()
        # self.NDVI_CO2_VPD()
        # self.NDVI_CO2_VPD_corr_line()
        # self.NDVI_CO2_VPD_corr_pdf_max_vpd_year()
        # self.NDVI_CO2_VPD_corr_pdf_annual()
        # self.Early_Peak_Late_scatter()


        pass

    def __load_df(self):
        print('loading df ...')
        df = Global_vars().load_df()
        return df
    def __cal_anomaly(self,vals):
        mean = np.nanmean(vals)
        anomaly_list = []
        for val in vals:
            anomaly = val - mean
            anomaly_list.append(anomaly)
        anomaly_list = np.array(anomaly_list)
        return anomaly_list

    def __cal_relative_change(self,vals):
        base = vals[:3]
        base = np.nanmean(base)
        relative_change_list = []
        for val in vals:
            change_rate = (val - base) / base
            relative_change_list.append(change_rate)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list

    def spatial_greening_trend(self):
        df = self.__load_df()
        K = KDE_plot()
        for season in global_season_dic:
            # col_name = f'{season}_LAI'
            # col_name = f'{season}_GIMMS_NDVI'
            col_name = f'{season}_GEE_AVHRR_LAI'
            df_new = pd.DataFrame()
            df_new['pix'] = df['pix']
            df_new[col_name] = df[col_name]
            df_new = df_new.dropna()
            spatial_dic = {}
            for i,row in df_new.iterrows():
                pix = row.pix
                vals = row[col_name]
                x = list(range(len(vals)))
                y = vals
                try:
                    a, b, r, p = K.linefit(x,y)
                except:
                    continue
                spatial_dic[pix] = a
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            up = mean + std
            bottom = mean - std
            plt.figure()
            plt.imshow(arr,vmin=bottom,vmax=up)
            plt.colorbar()
            plt.title(f'trend {col_name}')
        del df
        plt.show()

        pass

    def greening_trend_time_series(self):
        # outdir = join(self.this_class_png,'greening_trend_time_series/anomaly')
        outdir = join(self.this_class_png,'greening_trend_time_series/relative_change')
        T.mk_dir(outdir,force=True)
        df = self.__load_df()
        # x_var = 'GEE_AVHRR_LAI'
        x_var = 'LAI_GIMMS'
        humid_list = T.get_df_unique_val_list(df,'HI_reclass')
        for area in humid_list:
            df_humid = df[df['HI_reclass']==area]
            plt.figure()
            for season in global_season_dic:
                print(season,area)
                # col_name = f'{season}_LAI'
                # col_name = f'{season}_GIMMS_NDVI'
                col_name = f'{season}_{x_var}'
                df_new = pd.DataFrame()
                df_new['pix'] = df_humid['pix']
                df_new[col_name] = df_humid[col_name]
                df_new = df_new.dropna()
                matrix = []
                for i, row in df_new.iterrows():
                    pix = row.pix
                    vals = row[col_name]
                    matrix.append(vals)
                matrix = np.array(matrix)
                matrix_T = matrix.T
                mean_list = []
                std_list = []
                for i in matrix_T:
                    mean = np.nanmean(i)
                    std = np.nanstd(i)
                    mean_list.append(mean)
                    std_list.append(std)
                # mean_list = self.__cal_anomaly(mean_list)
                mean_list = self.__cal_relative_change(mean_list)
                x_list = np.array(list(range(len(mean_list)))) + global_start_year
                plt.plot(x_list,mean_list,label=f'{season}')
                # plt.title(f'{area}-{season}')
            plt.legend()
            title = f'{area}-{x_var}'
            plt.title(title)
            plt.savefig(join(outdir,title+'.pdf'))
            plt.close()



    def NDVI_CO2_VPD(self):
        period = 'early'
        LAI_var = f'{period}_GEE_AVHRR_LAI'
        co2_var = f'{period}_CO2'
        # vpd_var = f'{period}_VPD'
        vpd_var = f'{period}_CCI_SM'
        df = self.__load_df()
        year_interval = 3
        year_list = np.arange(global_start_year,2015,year_interval)
        year_list = list(year_list)
        year_list.append(2015)
        c_list = KDE_plot().makeColours(year_list,'Reds')

        for year in tqdm(range(len(year_list))):
            if year + 1 >= len(year_list):
                continue
            start = year_list[year]
            end = year_list[year+1]
            pick_index = list(range(start,end+1))
            pick_index = np.array(pick_index) - global_start_year
            # print(pick_index)
            lai_list = []
            co2_list = []
            vpd_list = []
            for i,row in df.iterrows():
                lai = row[LAI_var]
                co2 = row[co2_var]
                vpd = row[vpd_var]
                if type(lai) == float:
                    continue
                if type(co2) == float:
                    continue
                if type(vpd) == float:
                    continue
                picked_lai = T.pick_vals_from_1darray(lai,pick_index)
                picked_co2 = T.pick_vals_from_1darray(co2,pick_index)
                picked_vpd = T.pick_vals_from_1darray(vpd,pick_index)
                lai_mean = np.nanmean(picked_lai)
                co2_mean = np.nanmean(picked_co2)
                vpd_mean = np.nanmean(picked_vpd)

                lai_list.append(lai_mean)
                co2_list.append(co2_mean)
                vpd_list.append(vpd_mean)
            df_new = pd.DataFrame()
            df_new['lai'] = lai_list
            df_new['co2'] = co2_list
            df_new['vpd'] = vpd_list
            df_new = df_new.dropna()
            n = 20
            # vpd_bin = np.linspace(0.2,3,n)
            vpd_bin = np.linspace(0.,0.6,n)
            co2_bin = np.linspace(340,420,n)
            #
            x_list = []
            y_list = []
            for i in range(len(vpd_bin)):
                if i + 1 >= len(vpd_bin):
                    continue
                df_new_selected = df_new[df_new['vpd']>vpd_bin[i]]
                df_new_selected = df_new_selected[df_new_selected['vpd']<vpd_bin[i+1]]
                lai = df_new_selected['lai']
                vpd = df_new_selected['vpd']

                lai_mean = np.nanmean(lai)
                vpd_mean = np.nanmean(vpd)

                x_list.append(vpd_mean)
                y_list.append(lai_mean)
            # print(df_new)
            # exit()
            plt.plot(x_list,y_list,label=year,color=c_list[year])

            # x_list = []
            # y_list = []
            # for i in range(len(co2_bin)):
            #     if i + 1 >= len(co2_bin):
            #         continue
            #     df_new_selected = df_new[df_new['co2'] > co2_bin[i]]
            #     df_new_selected = df_new_selected[df_new_selected['co2'] < co2_bin[i + 1]]
            #     lai = df_new_selected['lai']
            #     vpd = df_new_selected['co2']
            #
            #     lai_mean = np.nanmean(lai)
            #     vpd_mean = np.nanmean(vpd)
            #
            #     x_list.append(vpd_mean)
            #     y_list.append(lai_mean)
            # # print(df_new)
            # # exit()
            # plt.plot(x_list, y_list, label=year)


        plt.legend()
        plt.show()
            # lai_list_mean = np.nanmean(lai_list)
            # co2_list_mean = np.nanmean(co2_list)
            # vpd_list_mean = np.nanmean(vpd_list)
            # plt.show()

        # print(year_list)
        pass


    def NDVI_CO2_VPD_corr_line(self):
        period = 'early'
        co2_var = 'CO2'
        vpd_var = 'VPD'
        partial_corr_dir = join(self.this_class_arr,'save_partial_corr_moving_window')
        mean_moving_window_dir = join(self.this_class_arr,'save_mean_moving_window')

        partial_corr_df = T.load_df(join(partial_corr_dir,f'{period}.df'))
        vpd_df = T.load_df(join(mean_moving_window_dir,f'{period}_{vpd_var}.df'))
        window_list = []
        for i in partial_corr_df:
            if vpd_var in i:
                window = i.split('_')[0]
                window = int(window)
                window_list.append(window)

        for w in tqdm(window_list):
            vpd_col_name = f'{w-1}_{period}_{vpd_var}'
            corr_col_name = f'{w}_{period}_{co2_var}'
            vpd_pix = vpd_df['pix'].tolist()
            vpd_vals = vpd_df[vpd_col_name].tolist()
            corr_pix = partial_corr_df['pix'].tolist()
            corr_vals = partial_corr_df[corr_col_name].tolist()

            vpd_dic = dict(zip(vpd_pix,vpd_vals))
            corr_dic = dict(zip(corr_pix,corr_vals))
            x_list = []
            y_list = []
            for pix in vpd_dic:
                if not pix in corr_dic:
                    continue
                x_list.append(vpd_dic[pix])
                y_list.append(corr_dic[pix])

            # print(vpd_vals)
            # print(corr_vals)
            plt.figure()
            plt.scatter(x_list,y_list,alpha=0.6)
            # cmap = KDE_plot().cmap_with_transparency('Reds')
            # KDE_plot().plot_scatter(x_list,y_list,cmap=cmap)
            plt.title(f'{w}')
        plt.show()
        exit()
        pass


    def __longterm_NDVI_CO2_VPD_corr_pdf(self,df,x1,x2,y):

        x1_corr_list = []
        x2_corr_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            try:
                x1_vals = row[x1].tolist()
                x2_vals = row[x2].tolist()
                y_vals = row[y].tolist()
                x1_corr,_ = T.nan_correlation(x1_vals,y_vals)
                x2_corr,_ = T.nan_correlation(x2_vals,y_vals)
                x1_corr_list.append(x1_corr)
                x2_corr_list.append(x2_corr)
            except:
                pass
        return x1_corr_list,x2_corr_list


    def NDVI_CO2_VPD_corr_pdf_max_vpd_year(self):
        outdir = join(self.this_class_png,'NDVI_CO2_VPD_corr_pdf_max_vpd_year')
        T.mk_dir(outdir)
        HI_class = ['Humid','Non Humid']
        vpd_var = 'VPD'
        # vpd_var = 'CCI_SM'
        co2_var = 'CO2'
        y_var = 'GEE_AVHRR_LAI'
        flag = 1
        plt.figure(figsize=(16,10))
        for HI_class_i in HI_class:
            df_long_term = self.__load_df()
            df_long_term = df_long_term[df_long_term['HI_class']==HI_class_i]
            moving_window_x_dir = join(self.this_class_arr,'save_mean_moving_window')
            partial_corr_moving_window_dir = join(self.this_class_arr,'save_partial_corr_moving_window')

            for period in global_season_dic:
                vpd_corr_longterm,co2_corr_longterm=self.__longterm_NDVI_CO2_VPD_corr_pdf(df_long_term,f'{period}_{vpd_var}',f'{period}_{co2_var}',f'{period}_{y_var}')
                vpdf = f'{period}_{vpd_var}.df'
                co2f = f'{period}_{co2_var}.df'
                corrf = f'{period}.df'
                vpdfpath = join(moving_window_x_dir,vpdf)
                co2fpath = join(moving_window_x_dir,co2f)
                corrfpath = join(partial_corr_moving_window_dir,corrf)

                vpddf = T.load_df(vpdfpath)
                co2df = T.load_df(co2fpath)
                corrdf = T.load_df(corrfpath)

                vpddf = vpddf[vpddf['HI_class']==HI_class_i]
                co2df = co2df[co2df['HI_class']==HI_class_i]
                corrdf = corrdf[corrdf['HI_class']==HI_class_i]

                selected_col = []
                mean_list = []
                for i in vpddf:
                    if vpd_var in i:
                        vpd_vals = vpddf[i].tolist()
                        vpd_mean = np.nanmean(vpd_vals)
                        selected_col.append(i)
                        mean_list.append(vpd_mean)
                max_vpd_col = selected_col[np.argmax(mean_list)]
                # max_vpd_col = selected_col[np.argmin(mean_list)]
                max_window = max_vpd_col.split('_')[0]
                max_window = int(max_window)
                max_y_col = f'{max_window}_{period}_{y_var}'
                max_co2_col = f'{max_window}_{period}_{co2_var}'

                max_vpd_col_corr = f'{max_window+1}_{period}_{vpd_var}'
                max_co2_col_corr = f'{max_window+1}_{period}_{co2_var}'

                vpd_corr_vals = corrdf[max_vpd_col_corr].tolist()
                co2_corr_vals = corrdf[max_co2_col_corr].tolist()

                vpd_corr_vals = T.remove_np_nan(vpd_corr_vals)
                co2_corr_vals = T.remove_np_nan(co2_corr_vals)
                vpd_corr_longterm = T.remove_np_nan(vpd_corr_longterm)
                co2_corr_longterm = T.remove_np_nan(co2_corr_longterm)

                plt.subplot(3,4,flag)
                # SMOOTH().hist_plot_smooth(vpd_corr_vals,bins=80,alpha=0.5,label=f'min_{vpd_var}_{max_window}_year')
                SMOOTH().hist_plot_smooth(vpd_corr_vals,bins=80,alpha=0.5,label=f'max_{vpd_var}_{max_window}_year')
                SMOOTH().hist_plot_smooth(vpd_corr_longterm,bins=80,alpha=0.5,label='long term')
                plt.title(f'{HI_class_i}-{period}-{vpd_var}')
                # plt.legend()
                plt.xlim(-1,1)
                plt.ylim(0,0.05)
                flag += 1
                plt.subplot(3, 4, flag)
                # SMOOTH().hist_plot_smooth(co2_corr_vals,bins=80,alpha=0.5,label=f'min_{vpd_var}_{max_window}_year')
                SMOOTH().hist_plot_smooth(co2_corr_vals,bins=80,alpha=0.5,label=f'max_{vpd_var}_{max_window}_year')
                SMOOTH().hist_plot_smooth(co2_corr_longterm,bins=80,alpha=0.5,label='long term')
                plt.title(f'{HI_class_i}-{period}-{co2_var}')
                # plt.legend()
                plt.xlim(-1, 1)
                plt.ylim(0, 0.05)
                flag += 1
        plt.savefig(join(outdir,f'{vpd_var}-{co2_var}-{y_var}.pdf'))


    def NDVI_CO2_VPD_corr_pdf_annual(self):
        outdir = join(self.this_class_png,'NDVI_CO2_VPD_corr_pdf_annual')
        T.mk_dir(outdir)
        HI_class = ['Humid','Non Humid']
        # x_var = 'VPD'
        x_var = 'CO2'
        # x_var = 'CCI_SM'
        plt.figure(figsize=(8,4))
        flag = 1
        for HI_class_i in HI_class:
            partial_corr_moving_window_dir = join(self.this_class_arr,'save_partial_corr_moving_window')
            for period in global_season_dic:
                plt.subplot(2,3,flag)
                title = f'{HI_class_i}-{period}-{x_var}'
                corrf = f'{period}.df'
                corrfpath = join(partial_corr_moving_window_dir,corrf)
                corrdf = T.load_df(corrfpath)

                corrdf = corrdf[corrdf['HI_class']==HI_class_i]
                window_list = []
                for i in corrdf:
                    if x_var in i:
                        window = i.split('_')[0]
                        window_list.append(int(window))
                # c_list = KDE_plot().makeColours(window_list,'RdBu')
                c_list = KDE_plot().makeColours(window_list,'RdYlBu')
                for w in window_list:
                    x_col_corr = f'{w}_{period}_{x_var}'
                    x_col_corr_vals = corrdf[x_col_corr].tolist()
                    x_col_corr_vals = T.remove_np_nan(x_col_corr_vals)
                    # SMOOTH().hist_plot_smooth(x_col_corr_vals)
                    x1, y1 = SMOOTH().hist_plot_smooth(x_col_corr_vals, bins=80,
                                              # range=(0.6, 1.4),
                                              # alpha=1.,
                                              alpha=0,
                                              histtype='step',
                                                color = c_list[w-1])
                                              # zorder=z)
                    plt.plot(x1, y1, color = c_list[w-1],label=f'{w}',alpha=0.7,lw=2)
                plt.title(title)
                plt.xlabel(f'partial correlation {x_var} vs LAI')
                flag += 1
        plt.tight_layout()
        # plt.legend()
        plt.savefig(join(outdir,f'{x_var}.pdf'))
        # plt.show()


    def Early_Peak_Late_scatter(self):
        df = Global_vars().load_df()
        y_var = Moving_window().y_var
        peak_val_list = []
        late_val_list = []
        y = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            early_val = row[f'early_{y_var}']
            peak_val = row[f'peak_{y_var}']
            late_val = row[f'late_{y_var}']
            if type(early_val) == float:
                continue
            if type(peak_val) == float:
                continue
            if type(late_val) == float:
                continue

            peak_val = Pre_Process().cal_anomaly_juping(peak_val)
            late_val = Pre_Process().cal_anomaly_juping(late_val)

            for j in peak_val:
                peak_val_list.append(j)
            for j in late_val:
                late_val_list.append(j)
        # plt.scatter(peak_val_list,late_val_list,s=0.1,alpha=0.5)
        KDE_plot().plot_scatter(peak_val_list,late_val_list)
        plt.xlabel('peak_val')
        plt.ylabel('late_val')
        plt.show()
        T.print_head_n(df)
        print(df)




class Moving_window:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Moving_window_1982-2015',results_root_main_flow)
        self.__var_list()
        # self.end_year = 2018
        self.end_year = 2015
        pass

    def __load_df(self):
        df = Global_vars().load_df()
        return df

    def run(self):
        # self.print_var_list()
        # exit()
        # self.single_correlation()
        # self.single_correlation_matrix_plot()
        # self.single_correlation_pdf_plot()
        # self.single_correlation_time_series_plot()

        # self.trend()
        self.trend_time_series_plot()
        # self.trend_matrix_plot()

        # self.mean()
        # self.mean_time_series_plot()
        # self.mean_matrix_plot()

        # self.partial_correlation()
        # self.pdf_plot('partial_correlation_orign_nodetrend')
        # self.pdf_plot('partial_correlation_orign_detrend')
        # self.pdf_plot('partial_correlation_anomaly_nodetrend')
        # self.pdf_plot('partial_correlation_anomaly_detrend')
        # self.pdf_plot('partial_correlation')
        # self.multi_regression()

        # self.LAI_trend_vs_vpd_LAI_correlation()
        # self.LAI_trend_vs_vpd_LAI_correlation_all_year()

        pass

    def __var_list(self):
        # self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'PAR', 'Precip', 'SPEI3', 'VPD', 'temperature']
        self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'GIMMS_NDVI', 'NIRv', 'PAR', 'Precip', 'SPEI3', 'VOD', 'VPD', 'temperature']
        # self.y_var = 'LAI_GIMMS'
        self.y_var = 'GIMMS_NDVI'
        self.all_var_list = copy.copy(self.x_var_list)
        self.all_var_list.append(self.y_var)

    def __partial_corr_var_list(self):
        y_var = f'LAI_GIMMS'
        x_var_list = [f'CO2',
                           f'VPD',
                           f'PAR',
                           f'temperature',
                           f'CCI_SM',
                      ]
        all_vars_list = copy.copy(x_var_list)
        all_vars_list.append(y_var)

        return x_var_list,y_var,all_vars_list

    def __partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p

    def __cal_partial_correlation(self,df,x_var_list,y_var):
        partial_correlation={}
        partial_correlation_p_value = {}
        x_var_list_valid = []
        for x in x_var_list:
            if x in df:
                x_var_list_valid.append(x)
        for x in x_var_list_valid:
            x_var_list_valid_new_cov=copy.copy(x_var_list_valid)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def __cal_multi_regression(self,df,x_var_list,y_var):
        model = LinearRegression()
        x_var_new = []
        for x in x_var_list:
            if x in df:
                x_var_new.append(x)
        X = df[x_var_new]
        Y = df[y_var]
        model.fit(X,Y)
        coef = model.coef_
        result_dic = dict(zip(x_var_new,coef))
        return result_dic

    def print_var_list(self):
        df = self.__load_df()
        var_list = []
        for col in df:
            col = str(col)
            for season in global_season_dic:
                if season in col:
                    col_new = col.replace(f'{season}_','')
                    var_list.append(col_new)
        var_list = T.drop_repeat_val_from_list(var_list)
        print(var_list)


    def __gen_moving_window_index(self,start_year,end_year):
        # start_year = global_start_year
        # end_year = global_end_year
        n = global_n
        moving_window_index_list = []
        for y in range(start_year,end_year + 1):
            moving_window_index = list(range(y,y+n))
            moving_window_index_list.append(moving_window_index)
            if y+n > end_year:
                break
        return moving_window_index_list

    def add_constant_value_to_df(self,df):
        df = Dataframe().add_lon_lat_to_df(df)
        df = Dataframe().add_Humid_nonhumid(df)
        df = Dataframe().add_lc(df)
        df = Dataframe().add_NDVI_mask(df)
        return df

    def single_correlation(self):
        outdir = join(self.this_class_arr,'single_correlation')
        T.mk_dir(outdir)
        outf = join(outdir,f'single_correlation_{global_n}.df')
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}

        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.x_var_list:
                    x_var = f'{season}_{x}'
                    y_var = f'{season}_{self.y_var}'
                    xval_all = df[x_var]
                    yval_all = df[y_var]
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list,xval_all))
                    y_spatial_dic = dict(zip(df_pix_list,yval_all))
                    for pix in y_spatial_dic:
                        xval = x_spatial_dic[pix]
                        yval = y_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        if type(yval) == float:
                            continue
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        yval_pick = T.pick_vals_from_1darray(yval,window_index)
                        r,p = T.nan_correlation(xval_pick,yval_pick)
                        key_r = f'{n}_{x_var}_r'
                        key_p = f'{n}_{x_var}_p'
                        results_dic[pix][key_r] = r
                        results_dic[pix][key_p] = p

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def partial_correlation(self):
        # outdir = join(self.this_class_arr,'partial_correlation_orign_nodetrend')
        # outdir = join(self.this_class_arr,'partial_correlation_orign_detrend')
        # outdir = join(self.this_class_arr,'partial_correlation_anomaly_nodetrend')
        outdir = join(self.this_class_arr,'partial_correlation_anomaly_detrend')
        T.mk_dir(outdir)
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        x_var_list,y_var,all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df,'pix')

        for n in range(len(moving_window_index_list)):
            outf = join(outdir, f'{n:02d}-{len(moving_window_index_list)}_partial_correlation.df')
            window_index = np.array(moving_window_index_list[n], dtype=int)
            window_index = window_index - global_start_year
            results_dic = {}
            for pix in pix_list:
                results_dic[pix] = {}
            for season in global_season_dic:
                key = f'{n:02d}_{season}'
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{n+1}/{len(moving_window_index_list)}'):
                    pix = row.pix
                    df_for_partial_corr = pd.DataFrame()
                    for x_var in x_var_list:
                        x_col_name = f'{season}_{x_var}'
                        xval = row[x_col_name]
                        if type(xval) == float:
                            continue
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        xval_pick = Pre_Process().cal_anomaly_juping(xval_pick)
                        xval_pick = T.detrend_vals(xval_pick)
                        df_for_partial_corr[x_var] = xval_pick
                    y_col_name = f'{season}_{self.y_var}'
                    yval = row[y_col_name]
                    if type(yval) == float:
                        continue
                    yval_pick = T.pick_vals_from_1darray(yval, window_index)
                    yval_pick = Pre_Process().cal_anomaly_juping(yval_pick)
                    yval_pick = T.detrend_vals(yval_pick)

                    df_for_partial_corr[self.y_var] = yval_pick
                    df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                    partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_for_partial_corr,x_var_list,y_var)
                    results_dic[pix][key] = partial_correlation

            df_result = T.dic_to_df(results_dic,'pix')
            T.print_head_n(df_result)
            # print(df_result)
            df_result = self.add_constant_value_to_df(df_result)
            T.save_df(df_result,outf)
            T.df_to_excel(df_result,outf)


    def multi_regression(self):
        outdir = join(self.this_class_arr,'multi_regression')
        T.mk_dir(outdir)
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        x_var_list,y_var,all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df,'pix')

        for n in range(len(moving_window_index_list)):
            outf = join(outdir, f'{n:02d}-{len(moving_window_index_list)}_partial_correlation.df')
            window_index = np.array(moving_window_index_list[n], dtype=int)
            window_index = window_index - global_start_year
            results_dic = {}
            for pix in pix_list:
                results_dic[pix] = {}
            for season in global_season_dic:
                key = f'{n:02d}_{season}'
                # if not season == 'early':
                #     continue
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{n+1}/{len(moving_window_index_list)}'):
                    pix = row.pix
                    df_for_partial_corr = pd.DataFrame()
                    for x_var in x_var_list:
                        x_col_name = f'{season}_{x_var}'
                        xval = row[x_col_name]
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        df_for_partial_corr[x_var] = xval_pick
                    y_col_name = f'{season}_{self.y_var}'
                    yval = row[y_col_name]
                    if type(yval) == float:
                        continue
                    # print(yval)
                    yval_pick = T.pick_vals_from_1darray(yval, window_index)
                    df_for_partial_corr[self.y_var] = yval_pick
                    df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                    multi_reg = self.__cal_multi_regression(df_for_partial_corr,x_var_list,y_var)
                    results_dic[pix][key] = multi_reg
            # print(results_dic)
            # exit()
            df_result = T.dic_to_df(results_dic,'pix')
            T.print_head_n(df_result)
            # print(df_result)
            df_result = self.add_constant_value_to_df(df_result)
            T.save_df(df_result,outf)
            T.df_to_excel(df_result,outf)

    def __get_window_list(self,df,x_var_1):
        # x_var_1 = self.x_var_list[0]
        # x_var_1 = self.y_var
        window_list = []
        for col in df:
            col = str(col)
            if x_var_1 in col:
                window = col.split('_')[0]
                window_list.append(int(window))
        window_list = T.drop_repeat_val_from_list(window_list)
        return window_list

    def single_correlation_matrix_plot(self):
        outdir = join(self.this_class_png,'plot_single_correlation')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'single_correlation/single_correlation.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list = self.x_var_list
        window_list = self.__get_window_list(df,var_list[0])
        # variable = var_list[0]
        for variable in var_list:
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}_r'].tolist()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap='RdBu_r',vmin=-0.3,vmax=0.3)
                        # plt.scatter(x_list, y, c=z, s=120, marker='s', cmap='RdBu_r')
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                # plt.show()
                plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                plt.close()


    def trend(self):
        outdir = join(self.this_class_arr,'trend')
        T.mk_dir(outdir)
        outf = join(outdir,'trend.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year,self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.all_var_list:
                    x_var = f'{season}_{x}'
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        # print(window_index)
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        except:
                            continue
                        # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                        try:
                            a, b, r, p = K.linefit(list(range(len(xval_pick))),xval_pick)
                            key_r = f'{n}_{x_var}_r'
                            key_p = f'{n}_{x_var}_p'
                            results_dic[pix][key_r] = a
                            results_dic[pix][key_p] = p
                        except:
                            continue

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def trend_matrix_plot(self):
        outdir = join(self.this_class_png,'trend_matrix_plot')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'trend/trend.df')
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list = self.all_var_list
        window_list = self.__get_window_list(df,'VPD')
        # variable = var_list[0]
        for variable in var_list:
            # if not 'VPD' in variable:
            #     continue
            print(variable)
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            # print(variable)
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        # print(df_HI)
                        # print(window_list)
                        # exit()
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}_r'].tolist()
                            # print(df_w)
                            # exit()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        if 'CCI' in variable:
                            cmap = 'RdBu'
                        else:
                            cmap = 'RdBu_r'
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap=cmap)
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                # plt.show()
                plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                plt.close()

    def trend_time_series_plot(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        fdir = join(self.this_class_arr, 'trend')
        dff = join(fdir, 'trend.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df,HI_reclass_var)
        df = df[df[HI_reclass_var]=='Non Humid']
        title = 'Non Humid'

        # df = df[df[HI_reclass_var]=='Humid']
        # title = 'Humid'

        # print(HI_reclass_list)
        # exit()
        lc_var = 'GLC2000'
        lc_list = T.get_df_unique_val_list(df,lc_var)
        window_list = self.__get_window_list(df,'VOD')
        # print(window_list)
        # exit()
        for xvar in self.all_var_list:
            if not 'VOD' in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}_r'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list,color=color_dic[season],label=season)
            plt.legend()
            plt.title(xvar)
        plt.title(title)
        plt.show()
        exit()
        # df_all = df[df['HI_reclass']==humid]
        # # df
        # K = KDE_plot()
        # val_length = 37
        # y_var = f'{season}_{y_variable}'
        # window_list = []
        # for col in df_all:
        #     if f'{y_variable}' in str(col):
        #         window = col.split('_')[0]
        #         window_list.append(window)
        # window_list=T.drop_repeat_val_from_list(window_list)
        #
        # mean_list = []
        # std_list = []
        # for j in window_list:
        #     y_val = df_all[f'{j}_{season}_{y_variable}_trend'].to_list()
        #     y_val_mean = np.nanmean(y_val)
        #     y_val_std = np.nanstd(y_val)
        #     mean_list.append(y_val_mean)
        #     std_list.append(y_val_std)
        #     # print(df_co2)
        #     # matrix.append(y_list)
        # # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        # y_list = mean_list
        # x_list = range(len(y_list))
        # plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # # plt.imshow(matrix)
        # # plt.colorbar()
        # plt.xlabel('year')
        # plt.ylabel(y_variable)
        # plt.title(humid)
        # # plt.show()


    def mean(self):
        outdir = join(self.this_class_arr, 'mean')
        T.mk_dir(outdir)
        outf = join(outdir, 'mean_relative_change.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year, self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.all_var_list:
                    x_var = f'{season}_{x}'
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix'].tolist()
                    if not len(xval_all) == len(df_pix_list):
                        raise UserWarning
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        xval = Global_vars().cal_relative_change(xval)
                        window_index = np.array(moving_window_index_list[n], dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval, window_index)
                            mean_val = np.nanmean(xval_pick)
                            # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                            key = f'{n}_{x_var}'
                            results_dic[pix][key] = mean_val
                        except:
                            continue
        df_result = T.dic_to_df(results_dic, 'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

        pass

    def mean_time_series_plot(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        fdir = join(self.this_class_arr, 'mean')
        dff = join(fdir, 'mean.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df,HI_reclass_var)
        df = df[df[HI_reclass_var]=='Non Humid']
        title = 'Non Humid'

        # df = df[df[HI_reclass_var]=='Humid']
        # title = 'Humid'

        # print(HI_reclass_list)
        # exit()
        lc_var = 'GLC2000'
        lc_list = T.get_df_unique_val_list(df,lc_var)
        window_list = self.__get_window_list(df,'LAI_GIMMS')
        # print(window_list)
        # exit()
        for xvar in self.all_var_list:
            if not 'LAI' in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list,color=color_dic[season],label=season)
            plt.legend()
            plt.title(xvar)
        plt.title(title)
        plt.show()
        exit()
        # df_all = df[df['HI_reclass']==humid]
        # # df
        # K = KDE_plot()
        # val_length = 37
        # y_var = f'{season}_{y_variable}'
        # window_list = []
        # for col in df_all:
        #     if f'{y_variable}' in str(col):
        #         window = col.split('_')[0]
        #         window_list.append(window)
        # window_list=T.drop_repeat_val_from_list(window_list)
        #
        # mean_list = []
        # std_list = []
        # for j in window_list:
        #     y_val = df_all[f'{j}_{season}_{y_variable}_trend'].to_list()
        #     y_val_mean = np.nanmean(y_val)
        #     y_val_std = np.nanstd(y_val)
        #     mean_list.append(y_val_mean)
        #     std_list.append(y_val_std)
        #     # print(df_co2)
        #     # matrix.append(y_list)
        # # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        # y_list = mean_list
        # x_list = range(len(y_list))
        # plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # # plt.imshow(matrix)
        # # plt.colorbar()
        # plt.xlabel('year')
        # plt.ylabel(y_variable)
        # plt.title(humid)
        # # plt.show()


    def mean_matrix_plot(self):
        outdir = join(self.this_class_png,'mean_matrix_plot')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'mean/mean.df')
        df = T.load_df(dff)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list = self.all_var_list
        window_list = self.__get_window_list(df,'VPD')
        # variable = var_list[0]
        for variable in var_list:
            # if not 'VPD' in variable:
            #     continue
            print(variable)
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            # print(variable)
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        # print(df_HI)
                        # print(window_list)
                        # exit()
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}'].tolist()
                            # print(df_w)
                            # exit()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        if 'CCI' in variable:
                            cmap = 'RdBu'
                        else:
                            cmap = 'RdBu_r'
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap=cmap,vmin=0,vmax=20)
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                plt.show()
                # plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                # plt.close()

    def single_correlation_pdf_plot(self):
        outdir = join(self.this_class_png, 'single_correlation_pdf_plot')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'single_correlation/single_correlation.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list = self.x_var_list
        window_list = self.__get_window_list(df, var_list[0])
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for variable in var_list:
            print(variable)
            matrix = []
            for season in global_season_dic:
                K = KDE_plot()
                plt.figure(figsize=(5,3))
                for w in window_list:
                    vals = df[f'{w}_{season}_{variable}_r'].tolist()
                    vals = T.remove_np_nan(vals)
                    x,y = Plot().plot_hist_smooth(vals,alpha=0,bins=80)
                    plt.plot(x,y,color=gradient_color[w],label=str(w))
                plt.title(variable)
                plt.tight_layout()
                # plt.show()
                plt.savefig(join(outdir,f'{variable}-{season}-legend.pdf'))
                # plt.legend()
                # plt.savefig(join(outdir,f'legend.pdf'))
                plt.close()
                exit()
        pass


    def single_correlation_time_series_plot(self):
        # outdir = join(self.this_class_png, 'single_correlation_pdf_plot')
        # T.mk_dir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        ### check spatial ###
        # spatial_dic = T.df_to_spatial_dic(df,'ndvi_mask_mark')
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list,_,_ = self.__partial_corr_var_list()
        window_list = self.__get_window_list(df, var_list[0])
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for season in global_season_dic:
            K = KDE_plot()
            plt.figure(figsize=(5, 3))
            for variable in var_list:
                print(variable)
                x = []
                y = []
                for w in window_list:
                    vals = df[f'{w}_{season}_{variable}_r'].tolist()
                    mean = np.nanmean(vals)
                    x.append(w)
                    y.append(mean)
                plt.plot(x,y,label=variable)
            plt.title('Moving window single correlation\n'+season)
            plt.tight_layout()
            plt.legend()
        plt.show()
        pass



    def pdf_plot(self,func_name):
        outdir = join(self.this_class_png, func_name+'_pdf')
        fdir = join(self.this_class_arr,func_name)
        T.mk_dir(outdir)
        # dff = join(self.this_class_arr, f'{func_name}/{func_name}.df')
        # df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
        var_list,_,_ = self.__partial_corr_var_list()
        window_list = []
        for f in T.listdir(fdir):
            window = f.split('-')[0]
            window_list.append(window)
        window_list = T.drop_repeat_val_from_list(window_list)
        window_list = [int(i) for i in window_list]
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for variable in var_list:
            print(variable)
            matrix = []
            for season in global_season_dic:
                K = KDE_plot()
                plt.figure(figsize=(5,3))
                for w in window_list:
                    dff = f'{w:02d}-23_partial_correlation.df'
                    df = T.load_df(join(fdir,dff))
                    dic_all = df[f'{w:02d}_{season}'].tolist()
                    vals = []
                    for dic_i in dic_all:
                        # print(pix)
                        # dic_i = dic_all[pix]
                        # print(dic_i)
                        if type(dic_i) == float:
                            continue
                        if not variable in dic_i:
                            continue
                        val = dic_i[variable]
                        vals.append(val)
                    vals = T.remove_np_nan(vals)
                    x,y = Plot().plot_hist_smooth(vals,alpha=0,bins=80)
                    plt.plot(x,y,color=gradient_color[w],label=str(w))
                # plt.show()
                plt.title(variable)
                plt.tight_layout()
                plt.savefig(join(outdir,f'{variable}-{season}.pdf'))

                # plt.legend()
                # plt.savefig(join(outdir,f'legend.pdf'))
                plt.close()
                # exit()
        pass



    def LAI_trend_vs_vpd_LAI_correlation(self):
        correlation_fdir = join(self.this_class_arr,'single_correlation')
        dff_corr = join(correlation_fdir,f'single_correlation_{global_n}.df')
        df_corr = T.load_df(dff_corr)
        df_corr = Global_vars().clean_df(df_corr)
        trend_corr_dff= join(self.this_class_arr,f'trend/trend_{global_n}.df')
        df_trend = T.load_df(trend_corr_dff)
        df_trend = Global_vars().clean_df(df_trend)

        x_var_list,_,_ = self.__partial_corr_var_list()
        window_list = self.__get_window_list(df_trend,'Aridity')
        color_list = KDE_plot().makeColours(window_list, 'Spectral')
        print(window_list)
        print(df_trend)
        print(trend_corr_dff)
        for x_var in x_var_list:
            for season in global_season_dic:
                plt.figure()
                for w in window_list:
                    dic_trend = T.df_to_spatial_dic(df_trend,f'{w}_{season}_{self.y_var}_r')
                    dic_corr = T.df_to_spatial_dic(df_corr,f'{w}_{season}_{x_var}_r')
                    x = []
                    y = []
                    for pix in dic_trend:
                        if not pix in dic_corr:
                            continue
                        x.append(dic_trend[pix])
                        y.append(dic_corr[pix])
                    # KDE_plot().plot_scatter(x,y)
                    # plt.show()
                    plt.scatter(x,y,s=1,alpha=0.2,color=color_list[w])
                plt.title(f'{x_var}_{season}')
                plt.xlabel(f'{self.y_var} trend')
                plt.ylabel(f'{x_var} vs LAI Correlation')
            plt.show()

        pass


    def LAI_trend_vs_vpd_LAI_correlation_all_year(self):
        df = Global_vars().load_df()
        x_var_list,_,_ = self.__partial_corr_var_list()
        for x_var in x_var_list:
            if not 'CCI' in x_var:
                continue
            for season in global_season_dic:
                var_name = f'{season}_{x_var}'
                x_spatial_dic = T.df_to_spatial_dic(df,var_name)
                y_spatial_dic = T.df_to_spatial_dic(df,f'{season}_{self.y_var}')
                x = []
                y = []
                for pix in tqdm(x_spatial_dic):
                    xvals = x_spatial_dic[pix]
                    yvals = y_spatial_dic[pix]
                    if type(yvals) == float:
                        continue
                    if type(xvals) == float:
                        continue
                    y_k, b, r, p = KDE_plot().linefit(list(range(len(yvals))),yvals)
                    corr,_ = T.nan_correlation(xvals,yvals)
                    x.append(corr)
                    y.append(y_k)
                plt.figure()
                plt.scatter(x,y,s=1)
                plt.xlabel(f'{x_var}-{self.y_var} correlation')
                plt.ylabel(f'{self.y_var} trend')
                plt.title(season)
            plt.show()




class Moving_window_RF:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Moving_window_RF',results_root_main_flow)
        self.end_year = 2018
        self.__var_list()

    def run(self):
        self.build_moving_window_rf_dataframe()
        self.moving_window_importance()
        pass

    def __partial_corr_var_list(self):
        y_var = f'LAI_GIMMS'
        x_var_list = [
                    f'CO2',
                    f'VPD',
                    f'PAR',
                    # f'temperature',
                    f'CCI_SM',
                       ]
        all_vars_list = copy.copy(x_var_list)
        all_vars_list.append(y_var)

        return x_var_list,y_var,all_vars_list

    def __var_list(self):
        self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'PAR', 'Precip', 'SPEI3', 'VPD', 'temperature']
        self.y_var = 'LAI_GIMMS'
        self.all_var_list = copy.copy(self.x_var_list)
        self.all_var_list.append(self.y_var)

    def __gen_moving_window_index(self,start_year,end_year):
        # start_year = global_start_year
        # end_year = global_end_year
        n = global_n
        moving_window_index_list = []
        for y in range(start_year,end_year + 1):
            moving_window_index = list(range(y,y+n))
            moving_window_index_list.append(moving_window_index)
            if y+n > end_year:
                break
        return moving_window_index_list

    def train_regression(self,X,Y,selected_labels):

        rf = RandomForestRegressor(n_jobs=4, n_estimators=100, )
        X = X[selected_labels]
        X = pd.DataFrame(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        Y_train=np.array(Y_train)
        rf.fit(X_train, Y_train)
        return rf


    def permutation(self, X, Y, variable_list):
        # from sklearn import XGboost
        # from sklearn.ensemble import GradientBoostingRegressor
        X = X[variable_list]
        r_X_dic = {}
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        clf = RandomForestRegressor(n_estimators=100, n_jobs=4)
        clf.fit(X_train, Y_train)
        # importances = clf.feature_importances_
        result = permutation_importance(clf, X_test, Y_test, scoring='r2',
                                        n_repeats=10, random_state=42,
                                        n_jobs=4)

        # perm = PermutationImportance(clf, random_state=1).fit(X_test, Y_test)
        # print(perm)
        # exit()
        # eli5.show_weights(perm)
        # print(eli5.format_as_text(eli5.explain_weights(perm)))
        # exit()
        importances = result.importances_mean
        importances_dic = dict(zip(variable_list, importances))
        # print(importances_dic)
        # exit()
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        y_pred = clf.predict(X_test)
        r_model = stats.pearsonr(Y_test, y_pred)[0]
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
        score = clf.score(X_test, Y_test)
        return clf, importances_dic, r_X_dic, mse, r_model,score, Y_test, y_pred


    def build_moving_window_rf_dataframe(self):
        # outdir = join(self.this_class_arr, f'build_moving_window_rf_dataframe_{global_n}')
        outdir = join(self.this_class_arr, f'build_moving_window_rf_dataframe_anomaly_{global_n}')
        T.mk_dir(outdir)
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year, self.end_year)
        df = Global_vars().load_df()
        x_var_list, y_var, all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        for season in global_season_dic:
            outdir_i = join(outdir,f'{season}')
            T.mk_dir(outdir_i)
            for n in range(len(moving_window_index_list)):
                # outf = join(outdir, f'{n:02d}-{len(moving_window_index_list)}_partial_correlation.df')
                window_index = np.array(moving_window_index_list[n], dtype=int)
                window_index = window_index - global_start_year
                results_dic = {}
                for pix in pix_list:
                    results_dic[pix] = {}
                key = f'{n:02d}_{season}'
                df_all = pd.DataFrame()
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f'{season} {n + 1}/{len(moving_window_index_list)}'):
                    pix = row.pix
                    df_for_partial_corr = pd.DataFrame()
                    for x_var in x_var_list:
                        x_col_name = f'{season}_{x_var}'
                        xval = row[x_col_name]
                        if type(xval) == float:
                            continue
                        xval_pick = T.pick_vals_from_1darray(xval, window_index)
                        xval_pick_anomaly = Pre_Process().cal_anomaly_juping(xval_pick)
                        df_for_partial_corr[x_var] = xval_pick_anomaly
                    y_col_name = f'{season}_{self.y_var}'
                    yval = row[y_col_name]
                    if type(yval) == float:
                        continue
                    yval_pick = T.pick_vals_from_1darray(yval, window_index)
                    yval_pick_anomaly = Pre_Process().cal_anomaly_juping(yval_pick)
                    df_for_partial_corr[self.y_var] = yval_pick_anomaly
                    df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                    df_all = df_all.append(df_for_partial_corr,ignore_index=True)
                outf = join(outdir_i,key) + '.df'
                T.save_df(df_all,outf)


    def moving_window_importance(self):
        fdir = join(self.this_class_arr,f'build_moving_window_rf_dataframe_anomaly_{global_n}')
        outdir = join(self.this_class_arr,f'moving_window_importance_anomaly_{global_n}')
        T.mk_dir(outdir)
        x_var_list,y_var,all_vars_list = self.__partial_corr_var_list()
        for season in global_season_dic:
            fdir_i = join(fdir,season)
            outf = join(outdir,f'{season}.df')
            result_dic = {}
            for f in tqdm(T.listdir(fdir_i),desc=season):
                df_i = T.load_df(join(fdir_i,f))
                df_i = df_i.dropna()
                # T.print_head_n(df_i)
                col_name = f.replace('.df','')
                Y = df_i[y_var]
                X = df_i[x_var_list]
                clf, importances_dic, r_X_dic, mse, r_model,score, Y_test, y_pred = self.permutation(X,Y,x_var_list)
                importances_dic['r'] = r_model
                importances_dic['score'] = score
                importances_dic['mse'] = mse
                result_dic[col_name] = importances_dic
            result_df = T.dic_to_df(result_dic)
            T.save_df(result_df,outf)
            T.df_to_excel(result_df,outf)
        pass


class Two_period_comparison:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Two_period_comparison',results_root_main_flow)
        self.__var_list()
        self.__period()


    def run(self):
        # self.multi_regression(period=2,detrend=True)
        # self.multi_regression_tif()
        # self.delta_tif('multi_regression')
        # self.delta_pdf('multi_regression')

        # self.single_correlation(period=2,detrend=True)
        # self.single_correlation_tif()
        # self.single_correlation_delta_tif('single_correlation')
        # self.single_correlation_delta_pdf('single_correlation')

        # self.partial_correlation(period=2,detrend=True)
        # self.partial_correlation_tif()
        # self.delta_tif('partial_correlation')
        # self.delta_pdf('partial_correlation')

        # self.trend(period=2)
        # self.trend_tif()
        # self.trend_delta_tif()
        # self.seasonal_cycle()
        # self.trend_spatial_tif()
        self.trend_spatial_pvalue_point_shp()
        pass


    def __load_df(self):
        df = Global_vars().load_df()
        return df

    def __var_list(self):
        self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'PAR', 'Precip', 'SPEI3', 'VPD', 'temperature']
        self.y_var = 'LAI_GIMMS'
        self.all_var_list = copy.copy(self.x_var_list)
        self.all_var_list.append(self.y_var)

    def __period(self):
        self.period1_start = 1982
        self.period1_end = 2001
        self.period2_start = 2001
        self.period2_end = 2018
        self.data_start_year = 1982
        # self.period2_start
        period1 = list(range(self.period1_start, self.period1_end + 1))
        period2 = list(range(self.period2_start + 1, self.period2_end + 1))
        self.period1_index = np.array(period1) - global_start_year
        self.period2_index = np.array(period2) - global_start_year

    def trend_spatial_tif(self):
        outdir = join(self.this_class_tif,'trend_spatial_tif')
        T.mk_dir(outdir)
        df = Global_vars().load_df()
        K = KDE_plot()
        period_list = [self.period1_index,self.period2_index]
        flag = 1
        for period in period_list:
            outdir_i = join(outdir,f'{flag}')
            T.mk_dir(outdir_i)
            for i in df:
                str_i = str(i)
                if not 'LAI_GIMMS' in i:
                    continue
                spatial_dic_time_series = T.df_to_spatial_dic(df,i)
                trend_spatial_dic = {}
                trend_spatial_dic_p = {}
                outf_p = join(outdir_i,i+'_p.tif')
                outf_r = join(outdir_i,i+'_k.tif')
                for pix in tqdm(spatial_dic_time_series,desc=str_i):
                    vals = spatial_dic_time_series[pix]
                    if type(vals) == float:
                        continue
                    vals = T.pick_vals_from_1darray(vals,period)
                    # print(len(vals))
                    # exit()
                    x = list(range(len(vals)))
                    y = vals
                    a, b, r, p = K.linefit(x,y)
                    trend_spatial_dic[pix] = a
                    trend_spatial_dic_p[pix] = p
                DIC_and_TIF().pix_dic_to_tif(trend_spatial_dic,outf_r)
                DIC_and_TIF().pix_dic_to_tif(trend_spatial_dic_p,outf_p)
            flag += 1

    def trend_spatial_pvalue_point_shp(self):
        fdir = join(self.this_class_tif,'trend_spatial_tif')
        outdir = join(self.this_class_tif,'trend_spatial_pvalue_point_shp')
        T.mk_dir(outdir)
        flag = 1
        for folder in T.listdir(fdir):
            outdir_i = join(outdir,f'{flag}')
            T.mk_dir(outdir_i)
            fdir_i = join(fdir,folder)
            for f in T.listdir(fdir_i):
                if not f.endswith('_p.tif'):
                    continue
                intif = join(fdir_i,f)
                outtif = join(outdir_i,f)
                ToRaster().resample_reproj(intif,outtif,res=2)
            for f in T.listdir(outdir_i):
                out_shp_f = join(outdir_i,f+'.shp')
                if not f.endswith('_p.tif'):
                    continue
                arr = ToRaster().raster2array(join(outdir_i,f))[0]
                arr = T.mask_999999_arr(arr,warning=False)
                arr[arr>0.1] = np.nan
                dic = DIC_and_TIF().spatial_arr_to_dic(arr)
                point_list = []
                for pix in dic:
                    val = dic[pix]
                    if np.isnan(val):
                        continue
                    lon,lat = DIC_and_TIF(tif_template=join(outdir_i,f)).pix_to_lon_lat(pix)
                    # print(lon,lat)
                    list_i = [lon,lat,1]
                    point_list.append(list_i)
                T.point_to_shp(point_list,out_shp_f)
            flag += 1

    def __partial_corr_var_list(self):
        y_var = f'LAI_GIMMS'
        x_var_list = [f'CO2',
                       f'VPD',
                       f'PAR',
                       f'temperature',
                       f'CCI_SM', ]
        all_vars_list = copy.copy(x_var_list)
        all_vars_list.append(y_var)

        return x_var_list,y_var,all_vars_list

    def __cal_multi_regression(self,df,x_var_list,y_var):
        model = LinearRegression()
        x_var_new = []
        for x in x_var_list:
            if x in df:
                x_var_new.append(x)
        X = df[x_var_new]
        Y = df[y_var]
        model.fit(X,Y)
        coef = model.coef_
        result_dic = dict(zip(x_var_new,coef))
        return result_dic
    def __partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p

    def __cal_partial_correlation(self,df,x_var_list,y_var):
        partial_correlation={}
        partial_correlation_p_value = {}
        x_var_list_valid = []
        for x in x_var_list:
            if x in df:
                x_var_list_valid.append(x)
        for x in x_var_list_valid:
            x_var_list_valid_new_cov=copy.copy(x_var_list_valid)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def multi_regression(self,period,detrend):
        if period == 1:
            start_year = self.period1_start
            end_year = self.period1_end
        elif period == 2:
            start_year = self.period2_start
            end_year = self.period2_end
        else:
            raise UserWarning(f'period {period} is incorrect')
        if detrend==True:
            outdir = join(self.this_class_arr, 'multi_regression_detrend')
        else:
            outdir = join(self.this_class_arr, 'multi_regression')
        T.mk_dir(outdir)
        pick_index = list(range(start_year,end_year+1))

        df = self.__load_df()
        x_var_list, y_var, all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        outf = join(outdir, f'period_{period}.df')
        pick_index = np.array(pick_index, dtype=int)
        pick_index = pick_index - self.data_start_year
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        for season in global_season_dic:
            key = f'{season}'
            for i, row in tqdm(df.iterrows(), total=len(df),desc=season):
                pix = row.pix
                df_for_partial_corr = pd.DataFrame()
                for x_var in x_var_list:
                    x_col_name = f'{season}_{x_var}'
                    xval = row[x_col_name]
                    xval_pick = T.pick_vals_from_1darray(xval, pick_index)
                    if detrend == True:
                        xval_pick = T.detrend_vals(xval_pick)
                    df_for_partial_corr[x_var] = xval_pick
                y_col_name = f'{season}_{self.y_var}'
                yval = row[y_col_name]
                if type(yval) == float:
                    continue
                # print(yval)
                yval_pick = T.pick_vals_from_1darray(yval, pick_index)
                if detrend == True:
                    yval_pick = T.detrend_vals(yval_pick)
                df_for_partial_corr[self.y_var] = yval_pick
                df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                multi_reg = self.__cal_multi_regression(df_for_partial_corr, x_var_list, y_var)
                results_dic[pix][key] = multi_reg
        # print(results_dic)
        # exit()
        df_result = T.dic_to_df(results_dic, 'pix')
        T.print_head_n(df_result)
        # print(df_result)
        df_result = Moving_window().add_constant_value_to_df(df_result)
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)
        pass

    def single_correlation(self,period,detrend):
        if period == 1:
            start_year = self.period1_start
            end_year = self.period1_end
        elif period == 2:
            start_year = self.period2_start
            end_year = self.period2_end
        else:
            raise UserWarning(f'period {period} is incorrect')
        if detrend == True:
            outdir = join(self.this_class_arr, 'single_correlation_detrend')
        else:
            outdir = join(self.this_class_arr, 'single_correlation')
        T.mk_dir(outdir)
        pick_index = list(range(start_year, end_year + 1))

        df = self.__load_df()
        x_var_list, y_var, all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        outf = join(outdir, f'period_{period}.df')
        pick_index = np.array(pick_index, dtype=int)
        pick_index = pick_index - self.data_start_year
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        for season in global_season_dic:
            for x in self.x_var_list:
                x_var = f'{season}_{x}'
                y_var = f'{season}_{self.y_var}'
                xval_all = df[x_var]
                yval_all = df[y_var]
                df_pix_list = df['pix']
                x_spatial_dic = dict(zip(df_pix_list, xval_all))
                y_spatial_dic = dict(zip(df_pix_list, yval_all))
                for pix in tqdm(y_spatial_dic,desc=f'{x}_{season}'):
                    xval = x_spatial_dic[pix]
                    yval = y_spatial_dic[pix]
                    if type(xval) == float:
                        continue
                    if type(yval) == float:
                        continue
                    xval_pick = T.pick_vals_from_1darray(xval, pick_index)
                    yval_pick = T.pick_vals_from_1darray(yval, pick_index)
                    if detrend == True:
                        xval_pick = T.detrend_vals(xval_pick)
                        yval_pick = T.detrend_vals(yval_pick)
                    r, p = T.nan_correlation(xval_pick, yval_pick)
                    key_r = f'{x_var}_r'
                    key_p = f'{x_var}_p'
                    results_dic[pix][key_r] = r
                    results_dic[pix][key_p] = p

        df_result = T.dic_to_df(results_dic, 'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = Moving_window().add_constant_value_to_df(df_result)
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)
        pass


    def partial_correlation(self,period,detrend):
        if period == 1:
            start_year = self.period1_start
            end_year = self.period1_end
        elif period == 2:
            start_year = self.period2_start
            end_year = self.period2_end
        else:
            raise UserWarning(f'period {period} is incorrect')
        if detrend == True:
            outdir = join(self.this_class_arr, 'partial_correlation_detrend')
        else:
            outdir = join(self.this_class_arr, 'partial_correlation')
        T.mk_dir(outdir)
        pick_index = list(range(start_year,end_year+1))

        df = self.__load_df()
        x_var_list, y_var, all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        outf = join(outdir, f'period_{period}.df')
        pick_index = np.array(pick_index, dtype=int)
        pick_index = pick_index - self.data_start_year
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        for season in global_season_dic:
            key = f'{season}'
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{season}'):
                pix = row.pix
                df_for_partial_corr = pd.DataFrame()
                for x_var in x_var_list:
                    x_col_name = f'{season}_{x_var}'
                    xval = row[x_col_name]
                    xval_pick = T.pick_vals_from_1darray(xval,pick_index)
                    if detrend == True:
                        xval_pick = T.detrend_vals(xval_pick)
                    df_for_partial_corr[x_var] = xval_pick
                y_col_name = f'{season}_{self.y_var}'
                yval = row[y_col_name]
                if type(yval) == float:
                    continue
                # print(yval)
                yval_pick = T.pick_vals_from_1darray(yval, pick_index)
                if detrend == True:
                    yval_pick = T.detrend_vals(yval_pick)
                df_for_partial_corr[self.y_var] = yval_pick
                df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_for_partial_corr,x_var_list,y_var)
                results_dic[pix][key] = partial_correlation

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        # print(df_result)
        df_result = Moving_window().add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def trend(self,period):
        if period == 1:
            start_year = self.period1_start
            end_year = self.period1_end
        elif period == 2:
            start_year = self.period2_start
            end_year = self.period2_end
        else:
            raise UserWarning(f'period {period} is incorrect')
        outdir = join(self.this_class_arr, 'trend')
        T.mk_dir(outdir)
        pick_index = list(range(start_year, end_year + 1))

        df = self.__load_df()
        x_var_list, y_var, all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        outf = join(outdir, f'period_{period}.df')
        pick_index = np.array(pick_index, dtype=int)
        pick_index = pick_index - self.data_start_year
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for season in global_season_dic:
            for x in self.all_var_list:
                print(season,x)
                x_var = f'{season}_{x}'
                xval_all = df[x_var].tolist()
                df_pix_list = df['pix']
                x_spatial_dic = dict(zip(df_pix_list, xval_all))
                for pix in x_spatial_dic:
                    xval = x_spatial_dic[pix]
                    if type(xval) == float:
                        continue
                    # print(window_index)
                    xval_pick = T.pick_vals_from_1darray(xval, pick_index)
                    # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                    try:
                        a, b, r, p = K.linefit(list(range(len(xval_pick))), xval_pick)
                        key_r = f'{x_var}_r'
                        key_p = f'{x_var}_p'
                        results_dic[pix][key_r] = a
                        results_dic[pix][key_p] = p
                    except:
                        continue

        df_result = T.dic_to_df(results_dic, 'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = Moving_window().add_constant_value_to_df(df_result)
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

    def trend_tif(self):
        fdir = join(self.this_class_arr,'trend')
        outdir = join(self.this_class_tif,'trend')
        T.mk_dir(outdir)
        for period in [1,2]:
            outdir_i = join(outdir,f'period_{period}')
            T.mk_dir(outdir_i)
            dff = join(fdir,f'period_{period}.df')
            df = T.load_df(dff)
            for col in df:
                if col == 'pix':
                    continue
                print(period,col)
                val = df[col].tolist()
                pix = df['pix'].tolist()
                spatial_dic = dict(zip(pix,val))
                outf = join(outdir_i,f'{col}.tif')
                try:
                    DIC_and_TIF().pix_dic_to_tif(spatial_dic,outf)
                except Exception as e:
                    print(e)

        pass

    def trend_delta_tif(self):
        fdir = join(self.this_class_tif,'trend')
        outdir = join(self.this_class_tif,'trend_delta')
        T.mk_dir(outdir)
        f_list = []
        for folder in T.listdir(fdir):
            for f in T.listdir(join(fdir,folder)):
                if not f.endswith('r.tif'):
                    continue
                f_list.append(f)
            break

        for f in f_list:
            print(f)
            arrs = []
            for period in [1,2]:
                folder = join(fdir,f'period_{period}')
                fpath = join(folder,f)
                arr = ToRaster().raster2array(fpath)[0]
                arr = T.mask_999999_arr(arr,warning=False)
                arrs.append(arr)
                # plt.figure()
                # plt.imshow(arr)
            # plt.show()
            delta_arr = arrs[1] - arrs[0]
            DIC_and_TIF().arr_to_tif(delta_arr,join(outdir,f))



    def single_correlation_tif(self):
        outdir = join(self.this_class_tif,'single_correlation')
        T.mk_dir(outdir)
        for detrend in ['_detrend','']:
            fdir = join(self.this_class_arr, 'single_correlation')
            fdir = fdir + detrend
            for period in [1,2]:
                outdir_i = join(outdir,f'period_{period}')
                outdir_i = outdir_i + detrend
                T.mk_dir(outdir_i)
                dff = join(fdir,f'period_{period}.df')
                df = T.load_df(dff)
                for col in df:
                    if col == 'pix':
                        continue
                    print(period,col)
                    val = df[col].tolist()
                    pix = df['pix'].tolist()
                    spatial_dic = dict(zip(pix,val))
                    outf = join(outdir_i,f'{col}.tif')
                    try:
                        DIC_and_TIF().pix_dic_to_tif(spatial_dic,outf)
                    except Exception as e:
                        print(e)

        pass

    def delta_tif(self,folder_name):

        fdir = join(self.this_class_tif,folder_name)
        outdir = join(self.this_class_tif, folder_name+'_delta')
        T.mk_dir(outdir)
        for detrend in ['_detrend','']:
            f_list = []
            for folder in T.listdir(fdir):
                for f in T.listdir(join(fdir,folder)):
                    if not f.endswith('.tif'):
                        continue
                    f_list.append(f)
                break
            outdir_i = join(outdir,f'delta{detrend}')
            T.mk_dir(outdir_i)
            for f in f_list:
                print(f)
                arrs = []
                for period in [1,2]:
                    folder = join(fdir,f'period_{period}{detrend}')
                    fpath = join(folder,f)
                    arr = ToRaster().raster2array(fpath)[0]
                    arr = T.mask_999999_arr(arr,warning=False)
                    arrs.append(arr)
                    # plt.figure()
                    # plt.imshow(arr)
                # plt.show()
                delta_arr = arrs[1] - arrs[0]
                DIC_and_TIF().arr_to_tif(delta_arr,join(outdir_i,f))


    def __get_humid_nonhumid_dic(self):
        outdir = join(temporary_root,'get_humid_nonhumid_dic')
        T.mk_dir(outdir)
        HI_class_spatial_dic_f = join(outdir,'HI_class_spatial_dic.npy')
        HI_reclass_spatial_dic_f = join(outdir,'HI_reclass_spatial_dic.npy')
        if isfile(HI_reclass_spatial_dic_f):
            HI_class_spatial_dic = T.load_npy(HI_class_spatial_dic_f)
            HI_reclass_spatial_dic = T.load_npy(HI_reclass_spatial_dic_f)
            return HI_class_spatial_dic,HI_reclass_spatial_dic
        spatial_dic = DIC_and_TIF().void_spatial_dic_zero()
        df = T.spatial_dics_to_df({'__void__':spatial_dic})
        P_PET_dic_reclass = Dataframe().P_PET_class()
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Non Humid'

        HI_class_spatial_dic = self.df_to_spatial_dic(df,'HI_class')
        HI_reclass_spatial_dic = self.df_to_spatial_dic(df,'HI_reclass')

        T.save_npy(HI_class_spatial_dic,HI_class_spatial_dic_f)
        T.save_npy(HI_reclass_spatial_dic,HI_reclass_spatial_dic_f)
        return HI_class_spatial_dic, HI_reclass_spatial_dic

    def mask_humid_area(self,arr,humid):
        HI_class_spatial_dic, HI_reclass_spatial_dic = self.__get_humid_nonhumid_dic()
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        all_dic = {
            'val':dic,
            'HI_class':HI_class_spatial_dic,
            'HI_reclass':HI_reclass_spatial_dic,
        }
        df = T.spatial_dics_to_df(all_dic)
        df = df[df['HI_reclass']==humid]
        spatial_dic = T.df_to_spatial_dic(df,'val')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        return arr

    def single_correlation_delta_pdf(self,folder_name):
        fdir = join(self.this_class_tif, folder_name)
        # outdir = join(self.this_class_tif, folder_name + '_delta')
        # T.mk_dir(outdir)
        period_color_dic = {
            '1-Humid':'r',
            '2-Non Humid':'g',
            '1-Non Humid':'b',
            '2-Humid':'k',
        }
        for detrend in ['_detrend', '']:
            f_list = []
            for folder in T.listdir(fdir):
                for f in T.listdir(join(fdir, folder)):
                    if not f.endswith('r.tif'):
                        continue
                    f_list.append(f)
                break
            for f in f_list:
                print(f)
                plt.figure()
                for humid in ['Humid','Non Humid']:
                    for period in [1, 2]:
                        folder = join(fdir, f'period_{period}{detrend}')
                        fpath = join(folder, f)
                        arr = ToRaster().raster2array(fpath)[0]
                        arr = T.mask_999999_arr(arr, warning=False)
                        arr = self.mask_humid_area(arr,humid)
                        # plt.imshow(arr)
                        # plt.show()
                        arr_flatten = arr.flatten()
                        arr_flatten = T.remove_np_nan(arr_flatten)
                        mean = np.nanmean(arr_flatten)
                        std = np.nanstd(arr_flatten)
                        up = mean + 3*std
                        down = mean - 3*std
                        x,y = Plot().plot_hist_smooth(arr_flatten,bins=200,alpha=0,range=(down,up))
                        plt.plot(x,y,label=f'{period}-{humid}',color=period_color_dic[f'{period}-{humid}'])
                    plt.title(f)
                    plt.legend()
                plt.show()
        pass


    def delta_pdf(self,folder_name):
        fdir = join(self.this_class_tif, folder_name)
        # outdir = join(self.this_class_tif, folder_name + '_delta')
        # T.mk_dir(outdir)
        period_color_dic = {
            '1-Humid': 'r',
            '2-Non Humid': 'g',
            '1-Non Humid': 'b',
            '2-Humid': 'k',
        }
        for detrend in ['_detrend', '']:
            f_list = []
            for folder in T.listdir(fdir):
                for f in T.listdir(join(fdir, folder)):
                    if not f.endswith('.tif'):
                        continue
                    f_list.append(f)
                break
            for f in f_list:
                print(f)
                plt.figure()
                for humid in ['Humid', 'Non Humid']:
                    for period in [1, 2]:
                        folder = join(fdir, f'period_{period}{detrend}')
                        fpath = join(folder, f)
                        arr = ToRaster().raster2array(fpath)[0]
                        arr = T.mask_999999_arr(arr, warning=False)
                        arr = self.mask_humid_area(arr, humid)
                        # plt.imshow(arr)
                        # plt.show()
                        arr_flatten = arr.flatten()
                        arr_flatten = T.remove_np_nan(arr_flatten)
                        mean = np.nanmean(arr_flatten)
                        std = np.nanstd(arr_flatten)
                        up = mean + 3 * std
                        down = mean - 3 * std
                        x, y = Plot().plot_hist_smooth(arr_flatten, bins=200, alpha=0, range=(down, up))
                        plt.plot(x, y, label=f'{period}-{humid}', color=period_color_dic[f'{period}-{humid}'])
                    plt.title(f)
                    plt.legend()
                plt.show()
        pass

    def single_correlation_delta_tif(self,folder_name):

        fdir = join(self.this_class_tif,folder_name)
        outdir = join(self.this_class_tif, folder_name+'_delta')
        T.mk_dir(outdir)
        for detrend in ['_detrend','']:
            f_list = []
            for folder in T.listdir(fdir):
                for f in T.listdir(join(fdir,folder)):
                    if not f.endswith('r.tif'):
                        continue
                    f_list.append(f)
                break
            outdir_i = join(outdir,f'delta{detrend}')
            T.mk_dir(outdir_i)
            for f in f_list:
                print(f)
                arrs = []
                for period in [1,2]:
                    folder = join(fdir,f'period_{period}{detrend}')
                    fpath = join(folder,f)
                    arr = ToRaster().raster2array(fpath)[0]
                    arr = T.mask_999999_arr(arr,warning=False)
                    arrs.append(arr)
                    # plt.figure()
                    # plt.imshow(arr)
                # plt.show()
                delta_arr = arrs[1] - arrs[0]
                DIC_and_TIF().arr_to_tif(delta_arr,join(outdir_i,f))

    def partial_correlation_tif(self):
        outdir = join(self.this_class_tif,'partial_correlation')
        T.mk_dir(outdir)
        for detrend in ['_detrend','']:
            fdir = join(self.this_class_arr, 'partial_correlation')
            fdir = fdir + detrend
            for period in [1,2]:
                outdir_i = join(outdir,f'period_{period}')
                outdir_i = outdir_i + detrend
                T.mk_dir(outdir_i)
                dff = join(fdir,f'period_{period}.df')
                df = T.load_df(dff)
                for col in df:
                    if col == 'pix':
                        continue
                    print(period,col)
                    val = df[col].tolist()
                    pix = df['pix'].tolist()
                    spatial_dic = dict(zip(pix,val))
                    spatial_dic_new = {}
                    for pix in spatial_dic:
                        dic_i = spatial_dic[pix]
                        if type(dic_i) == dict:
                            spatial_dic_new[pix] = dic_i
                    # print(spatial_dic)
                    # exit()
                    try:
                        df_i = T.dic_to_df(spatial_dic_new,'pix')
                        for col_i in df_i:
                            if col_i == 'pix':
                                continue
                            pix_i = df_i['pix']
                            val_i = df_i[col_i]
                            spatial_dic_i = dict(zip(pix_i,val_i))
                            outf = join(outdir_i,f'{col}_{col_i}.tif')
                            DIC_and_TIF().pix_dic_to_tif(spatial_dic_i,outf)
                    except Exception as e:
                        print(e)

    def multi_regression_tif(self):
        outdir = join(self.this_class_tif,'multi_regression')
        T.mk_dir(outdir)
        for detrend in ['_detrend','']:
            fdir = join(self.this_class_arr, 'multi_regression')
            fdir = fdir + detrend
            for period in [1,2]:
                outdir_i = join(outdir,f'period_{period}')
                outdir_i = outdir_i + detrend
                T.mk_dir(outdir_i)
                dff = join(fdir,f'period_{period}.df')
                df = T.load_df(dff)
                for col in df:
                    if col == 'pix':
                        continue
                    print(period,col)
                    val = df[col].tolist()
                    pix = df['pix'].tolist()
                    spatial_dic = dict(zip(pix,val))
                    spatial_dic_new = {}
                    for pix in spatial_dic:
                        dic_i = spatial_dic[pix]
                        if type(dic_i) == dict:
                            spatial_dic_new[pix] = dic_i
                    try:
                        df_i = T.dic_to_df(spatial_dic_new,'pix')
                        for col_i in df_i:
                            if col_i == 'pix':
                                continue
                            pix_i = df_i['pix']
                            val_i = df_i[col_i]
                            spatial_dic_i = dict(zip(pix_i,val_i))
                            outf = join(outdir_i,f'{col}_{col_i}.tif')
                            DIC_and_TIF().pix_dic_to_tif(spatial_dic_i,outf)
                    except Exception as e:
                        print(e)

        pass


    def __get_seasonal_cycle(self,val_matrix):
        monthly_vals_matrix = []
        for i in val_matrix:
            i_T = i.T
            monthly_vals = []
            for j in i_T:
                monthly_vals.append(np.nanmean(j))
            monthly_vals_matrix.append(monthly_vals)
        monthly_vals_matrix = np.array(monthly_vals_matrix)
        monthly_vals_matrix_T = monthly_vals_matrix.T
        monthly_mean_val = []
        for i in monthly_vals_matrix_T:
            monthly_mean_val.append(np.nanmean(i))
        return monthly_mean_val

    def seasonal_cycle(self):
        # fdir = '/Users/liyang/Desktop/original_dataset/LAI_BU_dic'
        fdir = '/Users/liyang/Desktop/original_dataset/CCI_SM_2020_dic'
        # fdir = '/Users/liyang/Desktop/original_dataset/SPEI3_dic'
        # fdir = '/Users/liyang/Desktop/original_dataset/GIMMS_NDVI_dic'
        dic = T.load_npy_dir(fdir)
        df = Global_vars().load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        pix_list = set(pix_list)
        period1 = list(range(1982,2002))
        period2 = list(range(2002,2019))
        # period2 = list(range(2002,2016))
        period1_index = np.array(period1) - global_start_year
        period2_index = np.array(period2) - global_start_year

        period1_matrix = []
        period2_matrix = []
        periodall_matrix = []
        for pix in tqdm(dic):
            if not pix in pix_list:
                continue
            vals = dic[pix]
            if np.isnan(np.nanmean(vals)):
                continue
            vals = T.detrend_vals(vals)
            # vals = Pre_Process().z_score_climatology(vals)
            vals_reshape = np.reshape(vals,(-1,12))
            vals_reshape1 = T.pick_vals_from_1darray(vals_reshape,period1_index)
            vals_reshape2 = T.pick_vals_from_1darray(vals_reshape,period2_index)
            # vals_reshape1 = vals_reshape1.flatten()
            # vals_reshape2 = vals_reshape2.flatten()
            # vals_reshape1 = T.detrend_vals(vals_reshape1)
            # vals_reshape2 = T.detrend_vals(vals_reshape2)
            # vals_reshape1 = np.reshape(vals_reshape1,(-1,12))
            # vals_reshape2 = np.reshape(vals_reshape2,(-1,12))
            period1_matrix.append(vals_reshape1)
            period2_matrix.append(vals_reshape2)
            periodall_matrix.append(vals_reshape)
        seasonal_cycle1 = self.__get_seasonal_cycle(period1_matrix)
        seasonal_cycle2 = self.__get_seasonal_cycle(period2_matrix)
        seasonal_cycle_all = self.__get_seasonal_cycle(periodall_matrix)

        plt.plot(seasonal_cycle1,label='period1')
        plt.plot(seasonal_cycle2,label='period2')
        plt.plot(seasonal_cycle_all,label='period_all',color='k')
        plt.legend()
        plt.show()


class Moving_window_attribution:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Moving_window_attribution',results_root_main_flow)

        pass

    def run(self):
        pass

    def foo(self):

        pass



class Seasonal_time_series:

    def __init__(self):

        pass

    def run(self):
        self.foo()
        pass


    def foo(self):
        df = Global_vars().load_df()
        y_var = Moving_window().y_var
        T.print_head_n(df)




def main():
    # Dataframe().run()
    # Greening_phenomena().run()
    # Analysis().run()
    # Moving_window().run()
    # Moving_window_RF().run()
    # Two_period_comparison().run()
    # Partial_corr('early').run()
    # Seasonal_time_series().run()
    pass


if __name__ == '__main__':

    main()