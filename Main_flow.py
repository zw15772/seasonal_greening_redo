# coding=utf-8

from preprocess import *
results_root_main_flow = join(results_root,'Main_flow')
import pingouin as pg

T.mk_dir(results_root_main_flow,force=True)


class Dataframe:

    def __init__(self):
        self.this_class_arr = join(results_root_main_flow,'arr/Dataframe/')
        self.dff = self.this_class_arr + 'dataframe.df'
        self.P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        T.mk_dir(self.this_class_arr,force=True)

    def run(self):
        df = self.__gen_df_init()
        # df = self.add_data(df)
        # df = self.add_lon_lat_to_df(df)
        # df = self.add_Humid_nonhumid(df)
        # df = self.add_ly_NDVI(df)
        # df = self.add_GEE_LAI(df)
        df = self.add_lc(df)
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
        LAI_dir = join(LAI().datadir,'per_pix_seasonal')
        columns = []
        for season in global_season_dic:
            print(season,'LAI')
            f = join(LAI_dir,f'{season}.npy')
            dic = T.load_npy(f)
            key_name = f'{season}_LAI'
            columns.append(key_name)
            df = T.add_spatial_dic_to_df(df,dic,key_name)
        for season in global_season_dic:
            x_dir = join(data_root,
             f'1982-2015_original_extraction_all_seasons/1982-2015_extraction_during_{season}_growing_season_static')
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

    def P_PET_reclass(self,):
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
        return dic_reclass

    def P_PET_ratio(self,P_PET_fdir):
        # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            T.mask_999999_arr(vals)
            vals[vals == 0] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term
    def add_Humid_nonhumid(self,df):
        P_PET_dic_reclass = self.P_PET_reclass()
        df = T.add_spatial_dic_to_df(df,P_PET_dic_reclass,'HI_class')
        df.loc[df['HI_class'] != 'Humid', ['HI_class']] = 'Non Humid'
        return df

    def add_lc(self,df):
        lc_f = join(GLC2000().datadir,'lc_dic_reclass.npy')
        lc_dic = T.load_npy(lc_f)
        df = T.add_spatial_dic_to_df(df,lc_dic,'GLC2000')
        return df

class Partial_corr:

    def __init__(self,season):
        self.season = season
        self.__config__()
        self.this_class_arr = join(results_root_main_flow,'arr/Partial_corr/')
        T.mk_dir(self.this_class_arr)
        self.dff = join(self.this_class_arr,f'Dataframe_{season}.df')


    def __config__(self):
        self.n = 15

        self.vars_list = [
            f'{self.season}_CO2',
            f'{self.season}_VPD',
            f'{self.season}_PAR',
            f'{self.season}_temperature',
            f'{self.season}_CCI_SM',
            # f'{self.season}_GIMMS_NDVI',
            f'{self.season}_GEE_AVHRR_LAI',
        ]
        self.x_var_list = [f'{self.season}_CO2',
                      f'{self.season}_VPD',
                      f'{self.season}_PAR',
                      f'{self.season}_temperature',
                      f'{self.season}_CCI_SM', ]
        # self.y_var = f'{self.season}_GIMMS_NDVI'
        self.y_var = f'{self.season}_GEE_AVHRR_LAI'

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
        self.n = 15
        self.this_class_arr = join(results_root_main_flow, 'arr/Analysis')
        self.this_class_png = join(results_root_main_flow, 'png/Analysis')
        T.mk_dir(self.this_class_arr,force=True)
        T.mk_dir(self.this_class_png,force=True)
        pass

    def run(self):
        # self.save_trend_moving_window()
        # self.save_mean_moving_window()
        # self.add_constant_value_to_df()
        # for season in global_season_dic:
        #     print(season)
        #     self.save_partial_corr_moving_window(season)
        # self.spatial_greening_trend()
        # self.greening_trend_time_series()
        # self.trend_area()
        # self.plot_trend_area()
        ###### moving window ##########
        # outdir = join(self.this_class_png,'trend_area_ratio_moving_window')
        # outdir = join(self.this_class_png,'mean_area_ratio_moving_window')
        # # T.mk_dir(outdir)
        # # y_variable = 'GIMMS_NDVI'
        # # # y_variable = 'GEE_AVHRR_LAI'
        # y_variable = 'CCI_SM'
        # # # y_variable = 'VPD'
        # humid = 'Non Humid'
        # # humid = 'Humid'
        # df = self.__load_df()
        # for season in global_season_dic:
        # #     # self.greening_slide_trend_time_series(season,y_variable,humid)
        # #     self.trend_area_ratio_moving_window(df,season,y_variable,humid)
        # #     plt.twinx()
        #     self.greening_slide_mean_time_series(season,y_variable,humid)
        # #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(join(outdir,f'{season}_{y_variable}_{humid}.pdf'))
        #     plt.close()
        ###### moving window ##########
        # self.NDVI_CO2_VPD()
        # self.NDVI_CO2_VPD_corr_line()
        self.NDVI_CO2_VPD_corr_pdf_max_vpd_year()
        # self.NDVI_CO2_VPD_corr_pdf_annual()


        ############ MATRIX ##############
        # self.matrix_trend_moving_window()
        # self.matrix_mean_moving_window()
        #
        # season = 'early'
        # season = 'peak'
        # season = 'late'
        # self.plot_partial_corr_moving_window()
        ############ MATRIX ##############

        pass

    def __load_df(self):
        print('loading df ...')
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = df[df['lat']>30]
        print('loaded')
        T.print_head_n(df)
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
    def get_moving_window_df(self):
        df = self.__load_df()
        df_all = pd.DataFrame()
        window_list = []
        val_length = 34
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            window_mean_val_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                val_dic = {}
                vals = row[y_var]
                if type(vals) == float:
                    continue
                try:
                    picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                except:
                    picked_vals = np.nan
                picked_vals_mean = np.nanmean(picked_vals)  # mean
                # try:
                #     picked_vals_mean,b,r = K.linefit(list(range(len(picked_vals))),picked_vals)  # trend
                # except:
                #     picked_vals_mean = np.nan
                val_dic[y_var] = picked_vals_mean
                window_mean_val_dic[pix] = val_dic
            window_list.append(f'{w}')
            df_j = T.dic_to_df(window_mean_val_dic, 'pix')
            colomns = df_j.columns
            df_new = pd.DataFrame()
            df_new['pix'] = df_j['pix']
            for col in colomns:
                if col == 'pix':
                    continue
                new_col = f'{w}_{col}'
                df_new[new_col] = df_j[col]
            df_all[df_new.columns] = df_new[df_new.columns]
        del df
        pass

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
        x_var = 'GIMMS_NDVI'
        humid_list = T.get_df_unique_val_list(df,'HI_class')
        for area in humid_list:
            df_humid = df[df['HI_class']==area]
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

    def greening_slide_trend_time_series(self,season,y_variable,humid):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        df = self.__load_df()
        df = df[df['HI_class']==humid]
        # df
        K = KDE_plot()
        val_length = 34
        # season = 'early'
        # season = 'late'
        # y_var = f'{season}_GEE_AVHRR_LAI'
        y_var = f'{season}_{y_variable}'
        # y_var = 'peak_GEE_AVHRR_LAI'
        # y_var = 'late_GEE_AVHRR_LAI'
        df_all = pd.DataFrame()
        window_list = []
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            window_mean_val_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                val_dic = {}
                vals = row[y_var]
                if type(vals) == float:
                    continue
                try:
                    picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                except:
                    picked_vals = np.nan
                # picked_vals_mean = np.nanmean(picked_vals)  # mean
                try:
                    picked_vals_mean,b,r = K.linefit(list(range(len(picked_vals))),picked_vals)  # trend
                except:
                    picked_vals_mean = np.nan
                val_dic[y_var] = picked_vals_mean
                window_mean_val_dic[pix] = val_dic
            window_list.append(f'{w}')
            df_j = T.dic_to_df(window_mean_val_dic,'pix')
            colomns = df_j.columns
            df_new = pd.DataFrame()
            df_new['pix'] = df_j['pix']
            for col in colomns:
                if col == 'pix':
                    continue
                new_col = f'{w}_{col}'
                df_new[new_col] = df_j[col]
            df_all[df_new.columns] = df_new[df_new.columns]
        del df
        year_dic = {
            'year': [],
            'pix': [],
        }
        variable_vals_dic = {
            y_var:[],
        }
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df_all[col_name]
                for val in vals:
                    variable_vals_dic[var_i].append(val)
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                pix_list = df_all['pix']
                vals = df_all[col_name]
                for val in vals:
                    year_dic['year'].append(w)
                for pix in pix_list:
                    year_dic['pix'].append(pix)
                break
        df_new = pd.DataFrame()

        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        df_new['pix'] = year_dic['pix']
        df_new['year'] = year_dic['year']
        df_new = df_new.dropna()
        mean_list = []
        std_list = []
        for j in window_list:
            df_year = df_new[df_new['year'] == j]
            y_val = df_year[y_var].to_list()
            y_val_mean = np.nanmean(y_val)
            y_val_std = np.nanstd(y_val)
            mean_list.append(y_val_mean)
            std_list.append(y_val_std)
            # print(df_co2)
            # matrix.append(y_list)
        # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        y_list = mean_list
        x_list = range(len(y_list))
        plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # plt.imshow(matrix)
        # plt.colorbar()
        plt.xlabel('year')
        plt.ylabel(y_variable)
        plt.title(humid)
        # plt.show()

    def greening_slide_mean_time_series(self,season,y_variable,humid):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        fdir = join(self.this_class_arr,'save_mean_moving_window')
        fpath = join(fdir,f'{season}_{y_variable}.df')
        df_all = T.load_df(fpath)
        T.print_head_n(df_all)
        # exit()
        df_all = df_all[df_all['HI_class']==humid]
        # exit()
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        K = KDE_plot()
        val_length = 34
        year_dic = {
            'year': [],
            'pix': [],
        }
        window_list = []
        for i in df_all:
            if y_variable in i:
                window = i.split('_')[0]
                window_list.append(window)

        mean_list = []
        std_list = []
        for w in window_list:
            col_name = f'{w}_{season}_{y_variable}'
            pix_list = df_all['pix']
            vals = df_all[col_name].tolist()
            vals = np.array(vals)
            vals[vals<=0] = np.nan
            vals = T.remove_np_nan(vals)
            vals_mean = np.nanmean(vals)
            vals_std = np.nanmean(vals)
            mean_list.append(vals_mean)
            std_list.append(vals_std)
        plt.plot(mean_list)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        std_list = std_list / 8
        up = mean_list + std_list
        down = mean_list - std_list
        # plt.fill_between(range(len(mean_list)),up,down,alpha=0.3)
        # plt.show()

    def greening_slide_partial_corr_area(self,season,y_variable,humid):
        '''
        todo: next task
        :param season:
        :param y_variable:
        :param humid:
        :return:
        '''
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        df = self.__load_df()
        df = df[df['HI_class']==humid]
        # df
        K = KDE_plot()
        val_length = 34
        # season = 'early'
        # season = 'late'
        # y_var = f'{season}_GEE_AVHRR_LAI'
        y_var = f'{season}_{y_variable}'
        # y_var = 'peak_GEE_AVHRR_LAI'
        # y_var = 'late_GEE_AVHRR_LAI'
        df_all = pd.DataFrame()
        window_list = []
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            window_mean_val_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                val_dic = {}
                vals = row[y_var]
                if type(vals) == float:
                    continue
                try:
                    picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                except:
                    picked_vals = np.nan
                picked_vals_mean = np.nanmean(picked_vals)  # mean
                # try:
                #     picked_vals_mean,b,r = K.linefit(list(range(len(picked_vals))),picked_vals)  # trend
                # except:
                #     picked_vals_mean = np.nan
                val_dic[y_var] = picked_vals_mean
                window_mean_val_dic[pix] = val_dic
            window_list.append(f'{w}')
            df_j = T.dic_to_df(window_mean_val_dic,'pix')
            colomns = df_j.columns
            df_new = pd.DataFrame()
            df_new['pix'] = df_j['pix']
            for col in colomns:
                if col == 'pix':
                    continue
                new_col = f'{w}_{col}'
                df_new[new_col] = df_j[col]
            df_all[df_new.columns] = df_new[df_new.columns]
        del df
        year_dic = {
            'year': [],
            'pix': [],
        }
        variable_vals_dic = {
            y_var:[],
        }
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df_all[col_name]
                for val in vals:
                    variable_vals_dic[var_i].append(val)
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                pix_list = df_all['pix']
                vals = df_all[col_name]
                for val in vals:
                    year_dic['year'].append(w)
                for pix in pix_list:
                    year_dic['pix'].append(pix)
                break
        df_new = pd.DataFrame()

        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        df_new['pix'] = year_dic['pix']
        df_new['year'] = year_dic['year']
        df_new = df_new.dropna()
        mean_list = []
        std_list = []
        for j in window_list:
            df_year = df_new[df_new['year'] == j]
            y_val = df_year[y_var].to_list()
            y_val_mean = np.nanmean(y_val)
            y_val_std = np.nanstd(y_val)
            mean_list.append(y_val_mean)
            std_list.append(y_val_std)
            # print(df_co2)
            # matrix.append(y_list)
        # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        y_list = mean_list
        x_list = range(len(y_list))
        plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # plt.imshow(matrix)
        # plt.colorbar()
        plt.xlabel('year')
        plt.ylabel(y_variable)
        plt.title(humid)
        # plt.show()


    def trend_area(self):
        outdir = join(self.this_class_arr,'trend_area')
        T.mk_dir(outdir)
        outf = join(outdir,'trend_area.df')
        df = self.__load_df()
        columns = df.columns
        trend_dic_all = {}
        for v in columns:
            trend_spatial_dic = {}
            trend_p_spatial_dic = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=v):
                vals = row[v]
                pix = row.pix
                try:
                    if len(vals) <=30:
                        continue
                    x = list(range(len(vals)))
                    y = vals
                    r,p = T.nan_correlation(x,y)
                    trend_spatial_dic[pix] = r
                    trend_p_spatial_dic[pix] = p
                except:
                    continue
            if len(trend_spatial_dic) == 0:
                continue
            trend_dic_all[f'{v}_trend'] = trend_spatial_dic
            trend_dic_all[f'{v}_trend_p'] = trend_p_spatial_dic
        df_result = T.spatial_dics_to_df(trend_dic_all)
        T.print_head_n(df_result)
        T.save_df(df_result,outf)

    def plot_trend_area(self):
        dff = join(self.this_class_arr,'trend_area/trend_area.df')
        df = T.load_df(dff)
        variables_list = []
        for v in df:
            print(v)
            if '_p' in v:
                continue
            if not 'early_' in v:
                continue
            variable = v.replace('early_','')
            variable = variable.replace('_trend','')
            variables_list.append(variable)
        for v in variables_list:
            for period in global_season_dic:
                variable_trend = f'{period}_{v}_trend'
                variable_trend_p = f'{period}_{v}_trend_p'
                df_new = pd.DataFrame()
                df_new['r'] = df[variable_trend]
                df_new['p'] = df[variable_trend_p]
                df_new = df_new.dropna()
                df_p_non_sig = df_new[df_new['p']>0.1]
                df_p_05 = df_new[df_new['p']<=0.1]
                df_p_05 = df_p_05[df_p_05['p']>=0.05]
                df_p_sig = df_new[df_new['p']<0.05]
                pos_05 = df_p_05[df_p_05['r']>=0]
                neg_05 = df_p_05[df_p_05['r']<0]
                pos_sig = df_p_sig[df_p_sig['r']>=0]
                neg_sig = df_p_sig[df_p_sig['r']<0]
                non_sig_ratio = len(df_p_non_sig) / len(df_new)
                pos_05_ratio = len(pos_05) / len(df_new)
                neg_05_ratio = len(neg_05) / len(df_new)
                pos_sig_ratio = len(pos_sig) / len(df_new)
                neg_sig_ratio = len(neg_sig) / len(df_new)
                bars = [pos_sig_ratio,pos_05_ratio,non_sig_ratio,neg_05_ratio,neg_sig_ratio]

                bottom = 0
                for b in bars:
                    plt.bar(f'{v}-{period}',b,bottom=bottom)
                    bottom += b
            plt.show()

    def trend_area_ratio_moving_window(self,df,season,y_variable,humid):
        plt.figure()
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
        title = f'{season}_{y_variable}_{humid}'
        # df = self.__load_df()
        df = df[df['HI_class']==humid]
        K = KDE_plot()
        val_length = 34
        # season = 'early'
        # season = 'late'
        # y_var = f'{season}_GEE_AVHRR_LAI'
        y_var = f'{season}_{y_variable}'
        # y_var = 'peak_GEE_AVHRR_LAI'
        # y_var = 'late_GEE_AVHRR_LAI'
        df_all = pd.DataFrame()
        window_list = []
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            window_mean_val_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                val_dic = {}
                vals = row[y_var]
                if type(vals) == float:
                    continue
                try:
                    picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                except:
                    picked_vals = np.nan
                try:
                    x = list(range(len(picked_vals)))
                    y = picked_vals
                    r,p = T.nan_correlation(x,y)
                except:
                    r,p = np.nan,np.nan
                val_dic['r'] = r
                val_dic['p'] = p
                window_mean_val_dic[pix] = val_dic
            window_list.append(f'{w}')
            df_j = T.dic_to_df(window_mean_val_dic,'pix')
            colomns = df_j.columns
            df_new = pd.DataFrame()
            df_new['pix'] = df_j['pix']
            for col in colomns:
                if col == 'pix':
                    continue
                new_col = f'{w}_{col}'
                df_new[new_col] = df_j[col]
            df_all[df_new.columns] = df_new[df_new.columns]
        df_all = df_all.dropna()
        for w in window_list:
            df_new = pd.DataFrame()
            df_new['r'] = df_all[f'{w}_r']
            df_new['p'] = df_all[f'{w}_p']
            df_new = df_new.dropna()
            T.print_head_n(df_new)
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
                plt.bar(f'{w}', b, bottom=bottom,color=color_list[color_flag])
                bottom += b
                color_flag += 1
        plt.title(title)
        plt.tight_layout()

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

    def matrix_trend_moving_window(self):
        outdir = join(self.this_class_png,'matrix_trend_moving_window')
        T.mk_dir(outdir)
        var_list = [
            # 'CCI_SM',
            # 'CO2',
            # 'PAR',
            'VPD',
            # 'VOD',
            # 'temperature',
            # 'GIMMS_NDVI',
            # 'GEE_AVHRR_LAI',
            # 'LAI',
            # 'NIRv',
        ]
        variable = var_list[0]
        lc_list = ['Evergreen','Deciduous','Shrubs','Grass']
        HI_class_list = ['Humid','Non Humid']
        matrix = []
        for HI_class in HI_class_list:
            plt.figure()
            for season in global_season_dic:
                for lc in lc_list:
                    val_length = 34
                    K = KDE_plot()
                    df_moving_window_f = join(self.this_class_arr,'save_trend_moving_window',f'{season}_{variable}.df')
                    # df_moving_window_f = join(self.this_class_arr,'save_mean_moving_window',f'{season}_{variable}.df')
                    df_moving_window = T.load_df(df_moving_window_f)
                    df_moving_window = df_moving_window[df_moving_window['GLC2000']==lc]
                    df_moving_window = df_moving_window[df_moving_window['HI_class']==HI_class]
                    col_list = []
                    window_list = []
                    for i in df_moving_window:
                        if variable in i:
                            col_list.append(i)
                            window_list.append(i.replace(f'_{season}_{variable}',''))
                    window_list = [int(i) for i in window_list]
                    window_list.sort()
                    df_moving_window = df_moving_window.dropna(how='all',subset=col_list)
                    mean_list = []
                    x_list = []
                    for w in tqdm(window_list):
                        df_w = df_moving_window[f'{w}_{season}_{variable}'].tolist()
                        mean = np.nanmean(df_w)
                        mean_list.append(mean)
                        x_list.append(w)
                    # matrix.append(mean_list)
                    y = [f'{season}_{variable}_{lc}']*len(mean_list)
                    z = mean_list
                    plt.scatter(x_list,y,c=z,s=120,marker='s',cmap='RdBu_r')
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
            plt.tight_layout()
            plt.axis('equal')
            plt.savefig(join(outdir,f'{variable}_{HI_class}.pdf'))
            plt.close()

    def matrix_mean_moving_window(self):
        outdir = join(self.this_class_png,'matrix_mean_moving_window')
        T.mk_dir(outdir)
        var_list = [
            # 'CCI_SM',
            # 'CO2',
            # 'PAR',
            'VPD',
            # 'VOD',
            # 'temperature',
            # 'GIMMS_NDVI',
            # 'GEE_AVHRR_LAI',
            # 'LAI',
            # 'NIRv',
        ]
        variable = var_list[0]
        lc_list = ['Evergreen','Deciduous','Shrubs','Grass']
        HI_class_list = ['Humid','Non Humid']
        matrix = []
        for HI_class in HI_class_list:
            plt.figure()
            for season in global_season_dic:
                for lc in lc_list:
                    val_length = 34
                    K = KDE_plot()
                    # df_moving_window_f = join(self.this_class_arr,'save_trend_moving_window',f'{season}_{variable}.df')
                    df_moving_window_f = join(self.this_class_arr,'save_mean_moving_window',f'{season}_{variable}.df')
                    df_moving_window = T.load_df(df_moving_window_f)
                    df_moving_window = df_moving_window[df_moving_window['GLC2000']==lc]
                    df_moving_window = df_moving_window[df_moving_window['HI_class']==HI_class]
                    col_list = []
                    window_list = []
                    for i in df_moving_window:
                        if variable in i:
                            col_list.append(i)
                            window_list.append(i.replace(f'_{season}_{variable}',''))
                    window_list = [int(i) for i in window_list]
                    window_list.sort()
                    df_moving_window = df_moving_window.dropna(how='all',subset=col_list)
                    mean_list = []
                    x_list = []
                    for w in tqdm(window_list):
                        df_w = df_moving_window[f'{w}_{season}_{variable}'].tolist()
                        df_w = np.array(df_w)
                        df_w[df_w<=0] = np.nan
                        mean = np.nanmean(df_w)
                        mean_list.append(mean)
                        x_list.append(w)
                    # matrix.append(mean_list)
                    y = [f'{season}_{variable}_{lc}']*len(mean_list)
                    z = mean_list
                    plt.scatter(x_list,y,c=z,s=120,marker='s',cmap='RdBu_r')
            plt.colorbar()
            plt.title(HI_class)
            plt.tight_layout()
            plt.axis('equal')
            title = f'{variable}_{HI_class}'
            plt.title(title)
            plt.savefig(join(outdir,f'{title}.pdf'))

    def save_trend_moving_window(self):
        df = self.__load_df()
        for col in df:
            try:
                y_var = col
                outdir = join(self.this_class_arr,'save_trend_moving_window')
                T.mk_dir(outdir)
                outf = join(outdir,f'{y_var}.df')
                if isfile(outf):
                    print(f'{outf} is existed')
                    continue
                val_length = 34
                df_all = pd.DataFrame()
                window_list = []
                K = KDE_plot()
                col_list = []
                for w in tqdm(range(val_length),col):
                    if w + self.n >= val_length:
                        continue
                    pick_index = list(range(w, w + self.n))
                    window_mean_val_dic = {}
                    for i, row in df.iterrows():
                        pix = row.pix
                        val_dic = {}
                        vals = row[y_var]
                        if type(vals) == float:
                            continue
                        try:
                            picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                        except:
                            picked_vals = np.nan
                        # picked_vals_mean = np.nanmean(picked_vals)  # mean
                        try:
                            picked_vals_mean, b, r = K.linefit(list(range(len(picked_vals))), picked_vals)  # trend
                        except:
                            picked_vals_mean = np.nan
                        val_dic[y_var] = picked_vals_mean
                        window_mean_val_dic[pix] = val_dic
                    window_list.append(f'{w}')
                    df_j = T.dic_to_df(window_mean_val_dic, 'pix')
                    colomns = df_j.columns
                    df_new = pd.DataFrame()
                    df_new['pix'] = df_j['pix']
                    for col in colomns:
                        if col == 'pix':
                            continue
                        new_col = f'{w}_{col}'
                        col_list.append(new_col)
                        df_new[new_col] = df_j[col]
                    df_all[df_new.columns] = df_new[df_new.columns]
                T.save_df(df_all,outf)
            except:
                pass

        pass

    def save_mean_moving_window(self):
        df = self.__load_df()
        for col in df:
            if 'pix' in col:
                continue
            try:
                y_var = col
                outdir = join(self.this_class_arr,'save_mean_moving_window')
                T.mk_dir(outdir)
                outf = join(outdir,f'{y_var}.df')
                if isfile(outf):
                    print(f'{outf} is existed')
                    continue
                val_length = 34
                df_all = pd.DataFrame()
                window_list = []
                K = KDE_plot()
                col_list = []
                for w in tqdm(range(val_length),col):
                    if w + self.n >= val_length:
                        continue
                    pick_index = list(range(w, w + self.n))
                    window_mean_val_dic = {}
                    for i, row in df.iterrows():
                        pix = row.pix
                        val_dic = {}
                        vals = row[y_var]
                        if type(vals) == float:
                            continue
                        try:
                            picked_vals = T.pick_vals_from_1darray(vals, pick_index)
                        except:
                            picked_vals = np.nan
                        picked_vals_mean = np.nanmean(picked_vals)  # mean
                        # try:
                        #     picked_vals_mean, b, r = K.linefit(list(range(len(picked_vals))), picked_vals)  # trend
                        # except:
                        #     picked_vals_mean = np.nan
                        val_dic[y_var] = picked_vals_mean
                        window_mean_val_dic[pix] = val_dic
                    window_list.append(f'{w}')
                    df_j = T.dic_to_df(window_mean_val_dic, 'pix')
                    colomns = df_j.columns
                    df_new = pd.DataFrame()
                    df_new['pix'] = df_j['pix']
                    for col in colomns:
                        if col == 'pix':
                            continue
                        new_col = f'{w}_{col}'
                        col_list.append(new_col)
                        df_new[new_col] = df_j[col]
                    df_all[df_new.columns] = df_new[df_new.columns]
                T.save_df(df_all,outf)
            except:
                pass

        pass

    def add_constant_value_to_df(self):
        # fdir = join(self.this_class_arr,'save_mean_moving_window')
        fdir = join(self.this_class_arr,'save_trend_moving_window')
        for f in tqdm(T.listdir(fdir)):
            df_moving_window = T.load_df(join(fdir,f))
            df_moving_window = Dataframe().add_lon_lat_to_df(df_moving_window)
            df_moving_window = Dataframe().add_Humid_nonhumid(df_moving_window)
            df_moving_window = Dataframe().add_lc(df_moving_window)
            T.save_df(df_moving_window,join(fdir,f))

    def save_partial_corr_moving_window(self,season):
        fdir = join(Partial_corr(season).this_class_arr,f'{season}_GEE_AVHRR_LAI_relative_change')
        outdir = join(self.this_class_arr,'save_partial_corr_moving_window_relative_change')
        T.mk_dir(outdir)
        outf = join(outdir,f'{season}.df')
        df_all = pd.DataFrame()
        for f in T.listdir(fdir):
            window = f.replace('window_','')
            window = window.replace('-15.npy','')
            window = int(window)
            dic = T.load_npy(join(fdir,f))
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]['pcorr']
                new_dic_i = {}
                for key in dic_i:
                    new_key = f'{window}_{key}'
                    new_dic_i[new_key] = dic_i[key]
                spatial_dic[pix] = new_dic_i
            df = T.dic_to_df(spatial_dic,'pix')
            col_list = df.columns
            df_all[col_list] = df[col_list]
        col_name_list = []
        for i in df_all:
            if i=='pix':
                continue
            col_name_list.append(i)
        df_all = df_all.dropna(how='all',subset=col_name_list)
        pix_list = []
        for i,row in df_all.iterrows():
            pix_list.append(i)
        df_all['pix'] = pix_list
        df_all = Dataframe().add_lon_lat_to_df(df_all)
        df_all = Dataframe().add_Humid_nonhumid(df_all)
        df_all = Dataframe().add_lc(df_all)

        T.save_df(df_all,outf)


    def plot_partial_corr_moving_window(self):
        outdir = join(self.this_class_png,'plot_partial_corr_moving_window')
        T.mk_dir(outdir)
        var_list = [
            'CCI_SM',
            # 'CO2',
            # 'PAR',
            # 'VPD',
            # 'VOD',
            # 'temperature',
            # 'GIMMS_NDVI',
            # 'GEE_AVHRR_LAI',
            # 'LAI',
            # 'NIRv',
        ]
        variable = var_list[0]
        lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
        HI_class_list = ['Humid', 'Non Humid']
        matrix = []
        for HI_class in HI_class_list:
            plt.figure()
            for season in global_season_dic:
                for lc in lc_list:
                    val_length = 34
                    K = KDE_plot()
                    df_moving_window_f = join(self.this_class_arr, 'save_partial_corr_moving_window',
                                              f'{season}.df')
                    # df_moving_window_f = join(self.this_class_arr,'save_mean_moving_window',f'{season}_{variable}.df')
                    df_moving_window = T.load_df(df_moving_window_f)
                    df_moving_window = df_moving_window[df_moving_window['GLC2000'] == lc]
                    df_moving_window = df_moving_window[df_moving_window['HI_class'] == HI_class]
                    # T.print_head_n(df_moving_window)
                    # exit()
                    col_list = []
                    window_list = []
                    for i in df_moving_window:
                        if variable in i:
                            col_list.append(i)
                            window_list.append(i.replace(f'_{season}_{variable}', ''))
                    window_list = [int(i) for i in window_list]
                    window_list.sort()
                    df_moving_window = df_moving_window.dropna(how='all', subset=col_list)
                    mean_list = []
                    x_list = []
                    for w in tqdm(window_list):
                        df_w = df_moving_window[f'{w}_{season}_{variable}'].tolist()
                        mean = np.nanmean(df_w)
                        mean_list.append(mean)
                        x_list.append(w)
                    # matrix.append(mean_list)
                    y = [f'{season}_{variable}_{lc}_{HI_class}'] * len(mean_list)
                    z = mean_list
                    plt.scatter(x_list, y, c=z, s=120, marker='s', cmap='RdBu_r')
            plt.colorbar()
            # plt.title(HI_class)
            plt.tight_layout()
            plt.axis('equal')
            title = f'{HI_class}_{variable}'
            plt.title(title)
            plt.savefig(join(outdir,title+'.pdf'))

def main():
    # Greening().run()
    # Dataframe().run()
    Analysis().run()
    # Partial_corr('early').run()
    # Partial_corr('peak').run()
    # Partial_corr('late').run()
    pass


if __name__ == '__main__':

    main()