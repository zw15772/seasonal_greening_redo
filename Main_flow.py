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


    def cal_p_correlation(self):
        outdir = join(self.this_class_arr,f'{self.y_var}')

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
                    picked_vals_anomaly = self.__cal_anomaly(picked_vals)
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
        T.mk_dir(self.this_class_arr,force=True)
        pass

    def run(self):
        # self.save_trend_moving_window()
        # self.save_mean_moving_window()
        # self.add_constant_value_to_df()

        # self.spatial_greening_trend()
        # self.greening_trend_time_series()
        # self.trend_area()
        # self.plot_trend_area()
        ###### moving window ##########
        # y_variable = 'GIMMS_NDVI'
        # y_variable = 'GEE_AVHRR_LAI'
        # humid = 'Non Humid'
        # humid = 'Humid'
        # for season in global_season_dic:
            # self.greening_slide_trend_time_series(season,y_variable,humid)
            # self.trend_area_ratio_moving_window(season,y_variable,humid)
        #     # plt.twinx()
        #     # self.greening_slide_mean_time_series(season,y_variable,humid)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        ###### moving window ##########
        # self.NDVI_CO2_VPD()


        ############ MATRIX ##############
        self.matrix_trend_moving_window()

        season = 'early'
        season = 'peak'
        season = 'late'
        self.plot_partial_corr_moving_window(season)
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
                    a,b,r = K.linefit(x,y)
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
        df = self.__load_df()
        humid_list = T.get_df_unique_val_list(df,'HI_class')
        for area in humid_list:
            df_humid = df[df['HI_class']==area]
            for season in global_season_dic:
                print(season,area)
                # col_name = f'{season}_LAI'
                # col_name = f'{season}_GIMMS_NDVI'
                col_name = f'{season}_GEE_AVHRR_LAI'
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
                plt.figure()
                x_list = np.array(list(range(len(mean_list)))) + global_start_year
                plt.plot(x_list,mean_list)
                plt.title(f'{area}-{season}')
        plt.show()

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

    def trend_area_ratio_moving_window(self,season,y_variable,humid):
        plt.figure()
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
        title = f'{season}_{y_variable}_{humid}'
        df = self.__load_df()
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
        df = self.__load_df()
        year_interval = 10
        year_list = np.arange(global_start_year,2015,year_interval)
        year_list = list(year_list)
        year_list.append(2015)

        for i in range(len(year_list)):
            if i + 1 >= len(year_list):
                continue
            start = year_list[i]
            end = year_list[i+1]
            pick_index = list(range(start,end+1))
            for i,row in df.iterrows():
                print(row)
                exit()
                pass

            print(pick_index)
        # print(year_list)
        pass

    def matrix_trend_moving_window(self):
        var_list = [
            # 'CCI_SM',
            'CO2',
            # 'PAR',
            # 'VPD',
            # 'VOD',
            # 'temperature',
            # 'GIMMS_NDVI',
            # 'GEE_AVHRR_LAI',
            # 'LAI',
            # 'NIRv',
        ]
        lc_list = ['Evergreen','Deciduous','Shrubs','Grass']
        HI_class_list = ['Humid','Non Humid']
        matrix = []
        for HI_class in HI_class_list:
            plt.figure()
            for season in global_season_dic:
                for lc in lc_list:
                    for variable in var_list:
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
            plt.colorbar()
            plt.title(HI_class)
            plt.tight_layout()
            plt.axis('equal')
        plt.show()


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
        fdir = join(Partial_corr(season).this_class_arr,f'Partial_corr_{season}',f'{season}_GEE_AVHRR_LAI')
        outdir = join(self.this_class_arr,'save_partial_corr_moving_window')
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


    def plot_partial_corr_moving_window(self,season):
        var_list = [
            'CCI_SM',
            'CO2',
            'PAR',
            'VPD',
            # 'VOD',
            'temperature',
            # 'GIMMS_NDVI',
            # 'GEE_AVHRR_LAI',
            # 'LAI',
            # 'NIRv',
        ]
        if not season == 'late':
            return
        lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
        HI_class_list = ['Humid', 'Non Humid']
        matrix = []
        for variable in var_list:
            plt.figure()
            for HI_class in HI_class_list:
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
        plt.show()

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