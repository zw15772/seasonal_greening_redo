# coding=utf-8
from __init__ import *
this_root = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
global_start_year = 1982
land_tif = join(results_root,'/Base_data/tif_template.tif')

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe',
                                                                                       results_root)
        # self.mode = 'anomaly_detrend'
        self.mode = 'std_anomaly_detrend'
        # self.mode = 'origin' #todo: add origin values, different landcover
        outdir = join(self.this_class_arr,'Dynamic_pheno',self.mode)
        T.mk_dir(outdir,force=True)
        self.dff = join(outdir, 'data_frame.df')
        pass

    def run(self):
        df = self.__gen_df_init()
        # df = self.add_carryover(df)
        df = self.add_variables2()
        # df = self.add_sos_eos(df)
        # df = self.add_GLC_landcover_data_to_df(df)
        # df = self.add_NDVI_mask(df)
        # df = self.add_Koppen_data_to_df(df)
        # df = self.add_AI_to_df(df)
        # df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        # df = self.add_early_peak_lai(df)
        # df = self.add_early_peak_lai_anomaly(df)
        df = self.add_late_lai_anomaly(df)

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

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

    def clean_df(self,df):

        df = df[df['lat']>=30]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]

        return df



    def add_carryover(self,df):
        carryover_f = join(results_root, 'carryover', 'loga_b.npy')
        carryover_dict = T.load_npy(carryover_f)
        vals_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year = row['year']
            index = year - global_start_year
            if not pix in carryover_dict:
                vals_list.append(np.nan)
                continue
            vals = carryover_dict[pix]
            if index >= len(vals):
                vals_list.append(np.nan)
                continue
            val = vals[index]
            if type(val) == float:
                vals_list.append(np.nan)
                continue
            carryover = val['carryover']
            vals_list.append(carryover)
        df['carryover'] = vals_list
        return df


    def add_carryover1(self,df):
        start_year = 1982
        carryover_f = join(results_root, 'carryover', 'loga_b.npy')
        carryover_dict = T.load_npy(carryover_f)
        for pix in tqdm(carryover_dict):
            vals = carryover_dict[pix]
            for i in range(len(vals)):
                year = start_year + i
                val = vals[i]
                if type(val) == float:
                    continue
                a = val['a']
                b = val['b']
                carryover = val['carryover']
                df = pd.concat([df, pd.DataFrame({'pix': [pix], 'year': [year], 'early_peak_lai_anomaly': [a], 'late_lai_anomaly': [b], 'carryover': [carryover]})], ignore_index=True)
        return df


    def add_variables(self):
        fdir = r'D:\Greening\Result\detrend\detrend_anomaly\variables\\'
        period_list = ['early','peak','late']

        void_df = pd.DataFrame()
        void_dict = DIC_and_TIF().void_spatial_dic()
        pix_list = []
        for pix in void_dict:
            pix_list.append(pix)
        year_list = list(range(1982, 2022))
        pix_list_all = []
        year_list_all = []
        for pix in pix_list:
            for year in year_list:
                pix_list_all.append(pix)
                year_list_all.append(year)
        void_df['pix'] = pix_list_all
        void_df['year'] = year_list_all
        # T.print_head_n(void_df,100)
        # exit()

        data_dict = {}
        col_list = []

        for f in T.listdir(fdir):
            # detrend_PAR_peak_anomaly.npy
            col_name = f.replace('detrend_','').replace('_anomaly','').replace('.npy','')
            print(col_name)
            dict_i = T.load_npy(join(fdir,f))
            col_list.append(col_name)
            data_dict[col_name] = dict_i
        for col in data_dict:
            dict_i = data_dict[col]
            vals_list = []
            for i, row in tqdm(void_df.iterrows(), total=len(void_df), desc=col):
                pix = row['pix']
                year = row['year']
                index = year - global_start_year
                if not pix in dict_i:
                    vals_list.append(np.nan)
                    continue
                vals = dict_i[pix]
                if index >= len(vals):
                    vals_list.append(np.nan)
                    continue
                val = vals[index]
                vals_list.append(val)
            void_df[col] = vals_list
        void_df = void_df.dropna(subset=col_list, how='all')
        return void_df

        pass

    def add_variables2(self):
        var_mode = self.mode
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal')
        period_list = ['early','peak','late']

        void_df = pd.DataFrame()
        void_dict = DIC_and_TIF().void_spatial_dic()
        pix_list = []
        for pix in void_dict:
            pix_list.append(pix)
        year_list = list(range(1982,2032))
        pix_list_all = []
        year_list_all = []
        for pix in pix_list:
            for year in year_list:
                pix_list_all.append(pix)
                year_list_all.append(year)
        void_df['pix'] = pix_list_all
        void_df['year'] = year_list_all

        data_dict = {}
        col_list = []

        for var in T.listdir(fdir):
            for period in period_list:
                fdir_i = join(fdir,var,var_mode,period)
                dict_i = T.load_npy_dir(fdir_i)
                col_name = f'{var}_{period}'
                col_list.append(col_name)
                data_dict[col_name] = dict_i
        for col in data_dict:
            dict_i = data_dict[col]
            vals_list = []
            for i,row in tqdm(void_df.iterrows(),total=len(void_df),desc=col):
                pix = row['pix']
                year = row['year']
                index = year - global_start_year
                if not pix in dict_i:
                    vals_list.append(np.nan)
                    continue
                vals = dict_i[pix]
                if index >= len(vals):
                    vals_list.append(np.nan)
                    continue
                val = vals[index]
                vals_list.append(val)
            void_df[col] = vals_list
        void_df = void_df.dropna(subset=col_list,how='all')
        return void_df

    def add_sos_eos(self,df):
        fdir = join(data_root,'lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')

        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        for i,row in tqdm(df.iterrows(),total=len(df),desc='sos_eos'):
            pix = row['pix']
            year = row['year']
            if not pix in sos_dict:
                df.loc[i,'sos'] = np.nan
                df.loc[i,'eos'] = np.nan
                continue
            sos = sos_dict[pix]
            eos = eos_dict[pix]
            sos = np.array(sos)
            eos = np.array(eos)

            sos_mean = np.nanmean(sos)
            eos_mean = np.nanmean(eos)
            sos_std = np.nanstd(sos)
            eos_std = np.nanstd(eos)

            sos_std_anomaly = (sos - sos_mean) / sos_std
            eos_std_anomaly = (eos - eos_mean) / eos_std
            sos_anomaly = sos - sos_mean
            eos_anomaly = eos - eos_mean
            index = year - global_start_year
            if index >= len(sos):
                df.loc[i,'sos_anomaly'] = np.nan
                df.loc[i,'eos_anomaly'] = np.nan
                df.loc[i,'sos_std_anomaly'] = np.nan
                df.loc[i,'eos_std_anomaly'] = np.nan
                continue
            df.loc[i,'sos_anomaly'] = sos_anomaly[index]
            df.loc[i,'eos_anomaly'] = eos_anomaly[index]
            df.loc[i,'sos_std_anomaly'] = sos_std_anomaly[index]
            df.loc[i,'eos_std_anomaly'] = eos_std_anomaly[index]
        return df


    def add_variables1(self,df):
        # fdir = 'D:\Greening\Result\anomaly\1982-2020\X_daily_1982-2020\early\LAI3g'
        # fdir = join(results_root,'anomaly','1982-2020','X_daily_1982-2020','early','LAI3g')
        fdir = join(results_root,'anomaly','1982-2020','X_daily_1982-2020')
        period_list = ['early','peak','late']
        variable_list = ['CCI_SM','CO2','PAR','SPEI3','Temp','VPD']
        for period in period_list:
            for variable in variable_list:
                if variable == 'SPEI3':
                    f = join(fdir, period, 'LAI3g', f'{variable}_{period}.npy')
                else:
                    f = join(fdir,period,'LAI3g',f'{variable}_{period}_anomaly.npy')
                var_dict = T.load_npy(f)
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{period}_{variable}'):
                    pix = row['pix']
                    year = row['year']
                    index = year - global_start_year
                    if not pix in var_dict:
                        continue
                    val = var_dict[pix][index]
                    df.loc[i,f'{variable}_{period}'] = val
        return df

    def add_early_peak_lai(self, df):
        LAI3g_early = df['LAI3g_early'].tolist()
        LAI3g_peak = df['LAI3g_peak'].tolist()
        LAI3g_early = np.array(LAI3g_early,dtype=object)
        LAI3g_peak = np.array(LAI3g_peak,dtype=object)
        mean = (LAI3g_early + LAI3g_peak) / 2
        df['LAI3g_early_peak_mean'] = mean

        return df


    def add_late_lai_anomaly(self, df):
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr, 'pick_daily_seasonal')
        var_mode = 'std_anomaly_detrend'
        late_fdir = join(fdir, 'LAI3g', var_mode, 'late')
        late_dict = T.load_npy_dir(late_fdir)
        vals_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year = row['year']
            index = year - global_start_year
            if not pix in late_dict:
                vals_list.append(np.nan)
                continue
            vals = late_dict[pix]
            if index >= len(vals):
                vals_list.append(np.nan)
                continue
            val = vals[index]
            vals_list.append(val)
        df['LAI3g_late_anomaly'] = vals_list
        return df

    def add_early_peak_lai_anomaly(self, df):
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr, 'pick_daily_seasonal')
        var_mode = 'std_anomaly_detrend'
        data_dict = {}
        col_list = []

        early_fdir = join(fdir, 'LAI3g', var_mode, 'early')
        peak_fdir = join(fdir, 'LAI3g', var_mode, 'peak')
        early_dict = T.load_npy_dir(early_fdir)
        peak_dict = T.load_npy_dir(peak_fdir)
        early_peak_dict = {}
        for pix in tqdm(early_dict):
            early = early_dict[pix]
            peak = peak_dict[pix]
            early_peak_mean = (early + peak) / 2
            early_peak_dict[pix] = early_peak_mean
        vals_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year = row['year']
            index = year - global_start_year
            if not pix in early_peak_dict:
                vals_list.append(np.nan)
                continue
            vals = early_peak_dict[pix]
            if index >= len(vals):
                vals_list.append(np.nan)
                continue
            val = vals[index]
            vals_list.append(val)
        df['LAI3g_early_peak_mean_anomaly'] = vals_list
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
    def add_AI_to_df(self, df):  #
        P_PET_fdir = join(data_root, 'Base_data/aridity_P_PET_dic')
        p_pet_dic = self.P_PET_ratio(P_PET_fdir)
        df = T.add_spatial_dic_to_df(df, p_pet_dic, 'AI')
        return df
    def P_PET_ratio(self, P_PET_fdir):

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

class Dataframe_One_pix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe_One_pix',
                                                                                       results_root)
        # self.mode = 'anomaly_detrend'
        self.mode = 'std_anomaly_detrend'
        # self.mode = 'origin'
        outdir = join(self.this_class_arr, 'Dynamic_pheno', self.mode)
        T.mk_dir(outdir, force=True)
        self.dff = join(outdir, 'data_frame.df')
        pass

    def run(self):
        df = self.__gen_df_init()

        # df = self.add_variables()
        # df = self.add_sos_eos(df)
        # df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        # df = Dataframe().add_GLC_landcover_data_to_df(df)
        # df = Dataframe().add_NDVI_mask(df)
        # df = Dataframe().add_Koppen_data_to_df(df)
        # df = Dataframe().add_AI_to_df(df)
        df = self.add_early_peak_lai(df)
        # df = self.add_early_peak_lai_anomaly(df)
        # df = self.add_late_lai_anomaly(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
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

    def add_early_peak_lai_anomaly(self, df):
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr, 'pick_daily_seasonal')
        var_mode = 'std_anomaly_detrend'
        data_dict = {}
        col_list = []

        early_fdir = join(fdir, 'LAI3g', var_mode, 'early')
        peak_fdir = join(fdir, 'LAI3g', var_mode, 'peak')
        early_dict = T.load_npy_dir(early_fdir)
        peak_dict = T.load_npy_dir(peak_fdir)
        early_peak_dict = {}
        for pix in tqdm(early_dict):
            early = early_dict[pix]
            peak = peak_dict[pix]
            early_peak_mean = (early + peak) / 2
            early_peak_dict[pix] = early_peak_mean
        vals_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year = row['year']
            index = year - global_start_year
            if not pix in early_peak_dict:
                vals_list.append(np.nan)
                continue
            vals = early_peak_dict[pix]
            if index >= len(vals):
                vals_list.append(np.nan)
                continue
            val = vals[index]
            vals_list.append(val)
        df['LAI3g_early_peak_mean_anomaly'] = vals_list
        return df

class RF:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF',
                                                                                       results_root)

    def run(self):
        dff = Dataframe().dff
        # dff = r'D:\Greening\Result\Data_frame_1982-2020_daily\Data_frame_1982-2020_anomaly\Data_frame_1982-2020_anomaly_new.df'
        df = T.load_df(dff)
        # df = Dataframe().add_carryover(df)
        # df = df.dropna(subset=['carryover'],how='any')
        # print(df['eos_std_anomaly'])
        # T.print_head_n(df,5)
        # exit()
        # df = green_driver_trend_contribution.Build_dataframe().add_GLC_landcover_data_to_df(df)
        # df = green_driver_trend_contribution.Build_dataframe().add_NDVI_mask(df)
        # df = green_driver_trend_contribution.Build_dataframe().add_Koppen_data_to_df(df)
        # df = green_driver_trend_contribution.Build_dataframe().add_AI_to_df(df)
        # df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        # df = Dataframe().add_sos_eos(df)
        df = Dataframe().clean_df(df)
        # df = df[df['LAI3g_early']>=0]
        # df = df[df['LAI3g_peak']>=0]
        # df = df[df['LAI3g_early_peak']>=0]
        df = df[df['LAI3g_early_peak_mean']>=0]
        # T.print_head_n(df,5)
        # exit()
        # df = df[df['LAI3g_early_peak_mean']>=0.5]
        # df = df[df['LAI3g_early_peak_mean']>=0]
        # df = df[df['detrend_LAI3g_early_peak_anomaly']>=0]


        ## RF permutation importance in different PFT vs KP
        # max_var_dict = self.do_rf_lc_kp(df)
        # self.plot_PFT_LC_imp(df)
        # self.plot_PFT_LC_max_imp(df,max_var_dict)

        ## RF permutation importance in different PFT
        # max_var_dict = self.do_rf_lc(df)
        # self.plot_rf_lc_imp(df)


        ## Multiregression beta
        # self.do_multireg(df)
        # self.plot_multireg()

        # RF PDP
        self.do_rf_pdp(df)
        # threshold_list = [0,0.5,1]
        # for threshold in threshold_list:
        #     print(threshold)
        #     self.do_rf_pdp(df,threshold)

        # self.eos_and_late_lai(df)
        pass



    def x_variables(self):
        x_list = ['SPEI3_peak','SPEI3_late','VPD_peak','temp_peak',
                  'VPD_late','temp_late','sos_std_anomaly','LAI3g_early_peak_mean']
        # x_list = ['LAI3g_early_peak_mean', 'SPEI3_peak', 'SPEI3_late', 'VPD_peak', 'temp_peak',
        #           'VPD_late', 'temp_late', 'sos_std_anomaly']
        # x_list = ['LAI3g_early_peak_mean']
        # x_list = ['CO2_peak','SPEI3_peak', 'VPD_peak', 'Temp_peak',
        #           'VPD_late', 'Temp_late']
        # x_list = ['detrend_CO2_peak_anomaly','detrend_SPEI3_peak','detrend_VPD_peak_anomaly','detrend_Temp_peak_anomaly','detrend_VPD_late_anomaly','detrend_Temp_late_anomaly']
        # x_list = ['detrend_CO2_peak_anomaly','detrend_SPEI3_peak','detrend_VPD_peak_anomaly','detrend_Temp_peak_anomaly','detrend_VPD_late_anomaly','detrend_Temp_late_anomaly']
        return x_list

    def y_variable(self):
        # y = 'carryover'
        y = 'LAI3g_late'
        # y = 'LAI3g_late_anomaly'
        # y = 'eos_anomaly'
        # y = 'eos_std_anomaly'
        # y = 'detrend_LAI3g_late_anomaly'
        return y


    def do_rf_lc(self,df):
        outdir = join(self.this_class_arr, 'do_rf_lc')
        T.mk_dir(outdir, force=True)
        x_list = self.x_variables()
        y = self.y_variable()
        outf = join(outdir, f'RF_lc_{y}.npy')

        cross_df_dict = T.cross_select_dataframe(df,'landcover_GLC')
        imp_result_dict_all = {}
        for lc in cross_df_dict:
            df_i = cross_df_dict[lc]
            print(lc)
            X = df_i[x_list]
            Y = df_i[y]
            if len(X) < 100:
                continue
            imp_result_dict = self.train_classfication_permutation_importance(X,Y,x_list,y)
            imp_result_dict_all[lc] = imp_result_dict
            # importances_mean = imp_result_dict['importances_mean']
            # importances_mean_dict = dict(zip(x_list,importances_mean))
            # max_var = T.pick_max_key_val_from_dict(importances_mean_dict)
            # max_var_dict[(lc,kp)] = max_var
        T.save_npy(imp_result_dict_all,outf)
        return imp_result_dict_all

    def plot_rf_lc_imp(self,df):

        fdir = join(self.this_class_arr, 'do_rf_lc')
        y = self.y_variable()
        f = join(fdir, f'RF_lc_{y}.npy')
        imp_result_dict_all = T.load_npy(f)
        cross_df_dict = T.cross_select_dataframe(df, 'landcover_GLC')
        x_list = self.x_variables()
        x_list_dict = dict(zip(x_list, list(range(len(x_list)))))
        print(x_list_dict)
        flag = 1
        for lc in cross_df_dict:
            if not lc in imp_result_dict_all:
                continue
            imp_result_dict = imp_result_dict_all[lc]
            df_i = cross_df_dict[lc]
            # print(imp_result_dict)
            # exit()
            # pix_value = imp_result_dict['r']
            importances_mean = imp_result_dict['importances_mean']
            importances_std = imp_result_dict['importances_std']
            r = imp_result_dict['r']
            r = f'{r:.2f}'
            plt.subplot(2, 2, flag)
            flag += 1
            plt.bar(x_list, importances_mean, yerr=importances_std, color='r', alpha=0.5)
            if flag >= 4:
                plt.xticks(rotation=90)
            else:
                plt.xticks([])
            plt.title(f'{lc} r:{r}\nsample size:{len(df_i)}')
            plt.ylim(0, 0.85)
        plt.tight_layout()
        plt.show()

        pass
        pass

    def do_rf_lc_kp(self,df):
        outdir = join(self.this_class_arr, 'RF_lc_kp')
        T.mk_dir(outdir, force=True)
        x_list = self.x_variables()
        y = self.y_variable()
        outf = join(outdir, f'RF_lc_kp_{y}.npy')

        cross_df_dict = T.cross_select_dataframe(df,'landcover_GLC','koppen')
        imp_result_dict_all = {}
        for lc,kp in cross_df_dict:
            df_i = cross_df_dict[(lc,kp)]
            print(lc,kp)
            X = df_i[x_list]
            Y = df_i[y]
            if len(X) < 100:
                continue
            imp_result_dict = self.train_classfication_permutation_importance(X,Y,x_list,y)
            imp_result_dict_all[(lc,kp)] = imp_result_dict
            # importances_mean = imp_result_dict['importances_mean']
            # importances_mean_dict = dict(zip(x_list,importances_mean))
            # max_var = T.pick_max_key_val_from_dict(importances_mean_dict)
            # max_var_dict[(lc,kp)] = max_var
        T.save_npy(imp_result_dict_all,outf)
        return imp_result_dict_all

    def plot_PFT_LC_imp(self,df):
        fdir = join(self.this_class_arr, 'RF_lc_kp')
        y = self.y_variable()
        f = join(fdir, f'RF_lc_kp_{y}.npy')
        imp_result_dict_all = T.load_npy(f)
        cross_df_dict = T.cross_select_dataframe(df, 'landcover_GLC', 'koppen')
        x_list = self.x_variables()
        x_list_dict = dict(zip(x_list, list(range(len(x_list)))))
        print(x_list_dict)
        flag = 1
        for lc, kp in cross_df_dict:
            if not (lc, kp) in imp_result_dict_all:
                continue
            imp_result_dict = imp_result_dict_all[(lc, kp)]
            df_i = cross_df_dict[(lc, kp)]
            # print(imp_result_dict)
            # exit()
            # pix_value = imp_result_dict['r']
            importances_mean = imp_result_dict['importances_mean']
            importances_std = imp_result_dict['importances_std']
            r = imp_result_dict['r']
            r = f'{r:.2f}'
            plt.subplot(4, 7, flag)
            flag += 1
            plt.bar(x_list, importances_mean, yerr=importances_std, color='r', alpha=0.5)
            if flag >= 22:
                plt.xticks(rotation=90)
            else:
                plt.xticks([])
            plt.title(f'{lc}_{kp} r:{r}\nsample size:{len(df_i)}')
            plt.ylim(0, 0.85)
        plt.tight_layout()
        plt.show()

        pass

    def plot_PFT_LC_max_imp(self,df,imp_result_dict_all):
        cross_df_dict = T.cross_select_dataframe(df,'landcover_GLC','koppen')
        x_list = self.x_variables()
        x_list_dict = dict(zip(x_list,list(range(len(x_list)))))
        print(x_list_dict)
        spatial_dict = {}
        for lc,kp in cross_df_dict:
            df_i = cross_df_dict[(lc,kp)]
            if not (lc,kp) in imp_result_dict_all:
                continue
            imp_result_dict = imp_result_dict_all[(lc,kp)]
            # pix_value = imp_result_dict['r']
            pix_value = imp_result_dict['max_var']
            pix_list = df_i['pix'].tolist()
            pix_list = set(pix_list)
            for pix in pix_list:
                spatial_dict[pix] = pix_value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()


    def rf(self,X_input,Y_input):
        # rf = RandomForestRegressor(n_jobs=4,n_estimators=100,)
        rf = RandomForestRegressor(n_jobs=7,n_estimators=100,)
        rf.fit(X_input, Y_input)
        return rf

    def train_classfication_permutation_importance(self,X_input,Y_input,x_list,y,isplot=False):
        # X = X.loc[:, ~X.columns.duplicated()]
        # outfir_fig_dir=self.this_class_arr+'train_classfication_fig_driver/'  #修改
        # Tools().mk_dir(outfir_fig_dir)

        rf = RandomForestRegressor(n_jobs=6,n_estimators=100,)
        X_input=X_input[x_list]
        X_input=pd.DataFrame(X_input)
        df_new = pd.concat([X_input,Y_input], axis=1)
        df_new = df_new.dropna()
        X = df_new[x_list]
        Y = df_new[y]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        rf.fit(X_train,Y_train)
        score = rf.score(X_test,Y_test)

        pred=rf.predict(X_test)
        r = stats.pearsonr(pred,Y_test)[0]


        result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
                                        random_state=42)
        x_list_dict = dict(zip(x_list, list(range(len(x_list)))))
        importances_mean = result.importances_mean
        imp_dict = dict(zip(x_list, importances_mean))
        max_key = T.pick_max_key_val_from_dict(imp_dict)
        max_V = x_list_dict[max_key]
        result['score'] = score
        result['r'] = r
        result['max_var'] = max_V

        if isplot:
            perm_sorted_idx = result.importances_mean.argsort()
            selected_labels_sorted = []
            for i in perm_sorted_idx:
                selected_labels_sorted.append(x_list[i])
            tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
            tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            ax1.barh(tree_indices,
                     rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
            print(rf.feature_importances_[tree_importance_sorted_idx])
            ax1.set_yticks(tree_indices)
            ax1.set_yticklabels(selected_labels_sorted)
            ax1.set_ylim((0, len(rf.feature_importances_)))
            ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                        labels=selected_labels_sorted)
            print(result.importances_mean)
            plt.title('late') # 修改
            fig.tight_layout()
            plt.show()  #如果是存数据就不能有show
        now = datetime.datetime.now()
        a = now.strftime('%Y-%m-%d-%H-%M')
        return result

    def do_multireg(self,df):
        outdir = join(self.this_class_arr,'multireg')
        T.mk_dir(outdir)
        outf = join(outdir,'result.df')
        x_list = self.x_variables()
        y = self.y_variable()
        pix_list = T.get_df_unique_val_list(df,'pix')
        result_dict = {}
        for pix in tqdm(pix_list):
            df_pix = df[df['pix']==pix]
            # sort df_pix
            df_pix = df_pix.sort_values(by=['year'])
            df_pix = df_pix[df_pix['LAI3g_early_peak_mean']>0]
            X = df_pix[x_list]
            Y = df_pix[y]
            df_i = pd.concat([X,Y],axis=1)
            df_i = df_i.dropna()
            X = df_i[x_list]
            Y = df_i[y]
            try:
                result = self.multireg(X,Y,x_list)
                result_dict[pix] = result
            except:
                continue
        df_result = T.dic_to_df(result_dict,key_col_str='pix',col_order=x_list)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)


    def multireg(self,X,Y,x_list):
        linear_model = LinearRegression()
        linear_model.fit(X, Y)
        coef_ = np.array(linear_model.coef_)
        coef_dic = dict(zip(x_list, coef_))
        return coef_dic

    def plot_multireg(self):
        outdir = join(self.this_class_arr,'multireg','tif')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'multireg', 'result.df')
        df = T.load_df(dff)
        col_list = df.columns.tolist()
        col_list.remove('pix')
        for col in col_list:
            dic_i = T.df_to_spatial_dic(df, col)
            plt.figure()
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_i)
            outf = join(outdir, 'multireg_'+col + '.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

        pass

    def __plot_PDP(self,col_name, data, model):
        df = self.__get_PDPvalues(col_name, data, model)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (6,5)
        # fig, ax = plt.subplots()
        # ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
        plt.plot(df[col_name], df['PDs'], lw = 2)
        plt.ylabel(self.y_variable())
        plt.xlabel(col_name)
        # plt.tight_layout()

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        # print('------------------')
        # print(col_name)
        # print('------------------')
        # print(sequence)
        # print(Y_pdp)
        # print(Y_pdp_std)
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})


    def do_rf_pdp(self,df):
        outdir = join(self.this_class_png,'rf_pdp_png',Dataframe().mode)
        T.mk_dir(outdir,force=True)

        x_list = self.x_variables()
        y = self.y_variable()
        outdir_i = join(outdir,y)
        # if os.path.isdir(outdir_i):
        #     return
        T.mk_dir(outdir_i)
        # T.open_path_and_file(outdir_i)

        # cross_df_dict = T.cross_select_dataframe(df, 'landcover_GLC', 'koppen')
        cross_df_dict = T.cross_select_dataframe(df, 'landcover_GLC')
        imp_result_dict_all = {}
        for lc in cross_df_dict:
            df_lc = cross_df_dict[lc]
            print(lc)
            X = df_lc[x_list]
            Y = df_lc[y]
            df_i = pd.concat([X, Y], axis=1)
            df_i = df_i.dropna()
            X = df_i[x_list]
            Y = df_i[y]
            if len(X) < 100:
                continue
            model = self.rf(X, Y)
            print(model)
            flag = 1
            plt.figure(figsize=(10, 10))
            for x_var in x_list:
                plt.subplot(3, 3, flag)
                self.__plot_PDP(x_var, X, model)
                flag += 1

            plt.suptitle(lc)
            plt.tight_layout()
            outf = join(outdir_i,lc+'.pdf')
            plt.savefig(outf)
            plt.close()

            # plot importance
            # importances = model.feature_importances_
            # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            # forest_importances = pd.Series(importances, index=x_list)
            # fig, ax = plt.subplots()
            # forest_importances.plot.bar(yerr=std, ax=ax)
            # ax.set_title("Feature importances using MDI")
            # ax.set_ylabel("Mean decrease in impurity")
            # fig.tight_layout()
            # plt.show()
        # return imp_result_dict_all
        pass

class Carryover_calculation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Carryover_calculation',
                                                                                       results_root)
        pass

    def log_ab(self):  # carryover algrithm

        # dic_mask_lc_file = 'C:/Users/pcadmin/Desktop/Data/Base_data/LC_reclass2.npy'
        dic_mask_lc_file = join(results_root,'Base_data','LC_reclass2.npy')
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        # tiff_mask_landcover_change = 'C:/Users/pcadmin/Desktop/Data/Base_data/lc_trend/max_trend.tif'
        tiff_mask_landcover_change = join(results_root,'Base_data','lc_trend','max_trend.tif')

        # array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
        #     tiff_mask_landcover_change)
        # array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        # array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        f_early_peak = results_root + f'anomaly/1982-2020/Y_daily_1982-2020/early_peak/LAI3g_early_peak_anomaly.npy'
        f_early_peak = join(results_root,'anomaly','1982-2020','Y_daily_1982-2020','early_peak','LAI3g_early_peak_anomaly.npy')
        # f_early_peak = results_root + r'detrend\detrend_anomaly\Y\detrend_LAI3g_early_peak_anomaly.npy'
        f_late= results_root + f'anomaly/1982-2020/Y_daily_1982-2020/late/LAI3g_late_anomaly.npy'
        # f_late= results_root + r'detrend\detrend_anomaly\Y\detrend_LAI3g_late_anomaly.npy'
        outdir= results_root +f'carryover/'
        outf = outdir+f'loga_b.npy'
        Tools().mk_dir(outdir, force=True)



        dic_early_peak = dict(np.load(f_early_peak, allow_pickle=True, ).item())
        dic_late = dict(np.load(f_late, allow_pickle=True, ).item())

        log_dic = {}
        dic_spatial_count={}
        for pix in tqdm(dic_early_peak):
            r,c=pix
            if r>120:
                continue
            if dic_mask_lc[pix] == 'Crop':
                continue
            if pix not in dic_late:
                continue
            val_early_peak = dic_early_peak[pix]
            val_early_peak=np.array(val_early_peak)

            val_late = dic_late[pix]
            val_late = np.array(val_late)

            if np.nanmean(val_late)==np.nan:
                continue
            if np.nanmean(val_early_peak)==np.nan:
                continue

            val_early_peak[val_early_peak<0]=np.nan


            log_time_series = []
            valid_count = 0

            for i in range(len(val_early_peak)):
                a = val_early_peak[i]
                b = val_late[i]
                if np.isnan(a):
                    log_time_series.append(np.nan)
                    continue

                if np.isnan(b):
                    log_time_series.append(np.nan)
                    continue
                # print(val_late[pos])

                # carryover = (np.log(i/(abs(val_late[pos]))))*(val_late[pos]/np.abs(val_late[pos]))
                # carryover = np.log(abs(a)/abs(b))*b/np.abs(b)
                carryover = -np.log((a-b)/(a+b))
                # carryover = -np.log(abs((a-b)/(a+b)))
                if np.isnan(carryover):
                    log_time_series.append(np.nan)
                    continue
                # carryover = (a-b)/(a+b)
                # carryover = np.log((a-b)/abs(b))*b/np.abs(b)
                # carryopver = np.log(i / (abs(val_late[pos])))
                valid_count += 1

                log_time_series.append({'carryover': carryover, 'a': a, 'b': b})
            log_dic[pix] = log_time_series
            # print(log_time_series)

            dic_spatial_count[pix] = valid_count


        # hist_list = []
        # for pix in log_dic:
        #     hist = log_dic[pix]
        #     for i in hist:
        #         hist_list.append(i)
        #
        # plt.hist(hist_list, bins=80)
        # plt.show()

        # 看看空间有多少有有效值

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
        plt.imshow(arr, cmap='jet',interpolation='nearest',vmax=20)
        plt.colorbar()
        plt.title('')
        plt.show()

        # plt.plot(result_dic[pix])
        # plt.show()

        np.save(outf, log_dic)

    def check_carryover(self):
        fdir = results_root + f'carryover/'
        hants_dir = r'C:\Users\pcadmin\Desktop\Data\DIC_Daily\LAI3g\\'
        f = fdir + f'loga_b.npy'
        carryover_dic = T.load_npy(f)
        hants_dict = T.load_npy_dir(hants_dir)
        for pix in carryover_dic:
            print(pix)
            r,c = pix
            if r<40:
                continue
            # print(carryover_dic[pix])
            # print(len(carryover_dic[pix]))
            carryover_list = carryover_dic[pix]
            hants = hants_dict[pix]
            hants = np.array(hants)
            # hants_flatten = hants.flatten()
            # hants_flatten_detrend = T.detrend_vals(hants_flatten)
            # plt.plot(hants_flatten_detrend)
            # plt.show()
            # hants = np.reshape(hants_flatten_detrend, hants.shape)
            # hants = T.detrend_vals(hants)
            hants_mean = np.nanmean(hants,axis=0)
            # print(carryover_list)
            # plt.plot(hants_mean)
            # plt.show()
            #
            # exit()
            for i in range(len(carryover_list)):
                carryover_dic_i = carryover_list[i]
                if type(carryover_dic_i)==float:
                    continue
                print(carryover_dic_i)
                carryover_i = carryover_dic_i['carryover']
                a_i = carryover_dic_i['a']
                b_i = carryover_dic_i['b']
                # if carryover_i<0:
                if np.isnan(carryover_i):
                    continue
                # if carryover_i>0:
                #     continue
                hants_i = hants[i]

                print(carryover_i)
                plt.plot(hants_i,color='r')
                plt.plot(hants_mean,'--', color='gray')
                plt.title(f'carryover:{carryover_i:.2f}\na:{a_i:.4f}\nb:{b_i:.4f}')
                plt.tight_layout()
                plt.show()


        pass

    def check_carryover_matrix(self):
        fdir = results_root + f'carryover/'
        # hants_dir = r'C:\Users\pcadmin\Desktop\Data\DIC_Daily\LAI3g\\'
        f = fdir + f'loga_b.npy'
        carryover_dic = T.load_npy(f)
        # hants_dict = T.load_npy_dir(hants_dir)
        a_list = []
        b_list = []
        carryover_list_all = []
        for pix in tqdm(carryover_dic):
            # print(pix)
            r, c = pix
            # if r < 40:
            #     continue
            carryover_list = carryover_dic[pix]

            for i in range(len(carryover_list)):
                carryover_dic_i = carryover_list[i]
                if type(carryover_dic_i) == float:
                    continue
                # print(carryover_dic_i)
                carryover_i = carryover_dic_i['carryover']
                a_i = carryover_dic_i['a']
                b_i = carryover_dic_i['b']
                a_list.append(a_i)
                b_list.append(b_i)
                carryover_list_all.append(carryover_i)
        df = pd.DataFrame({'a':a_list,'b':b_list,'carryover':carryover_list_all})
        col = 'carryover'
        # a_bin = np.linspace(0,1,40)
        a_bin = np.linspace(-1,1,200)
        b_bin = np.linspace(-1,1,200)
        df_group_a,bins_list_str_a = T.df_bin(df,'a',a_bin)
        matrix = []
        bins_list_str_b = ''
        for name_a,df_group_a_i in df_group_a:
            df_group_b,bins_list_str_b = T.df_bin(df_group_a_i,'b',b_bin)
            temp = []
            for name_b,df_group_b_i in df_group_b:
                vals = df_group_b_i[col].tolist()
                mean = np.nanmean(vals)
                temp.append(mean)
            matrix.append(temp)
        matrix = np.array(matrix)
        # plt.imshow(matrix,cmap='jet',vmin=-10,vmax=10)
        plt.imshow(matrix,cmap='RdBu',vmin=-4,vmax=4)
        # plt.imshow(matrix,cmap='jet')
        plt.colorbar()
        bins_list_str_b = [eval(i.replace(']',')')) for i in bins_list_str_b]
        bins_list_str_b = [f'{i[0]:0.2f}' for i in bins_list_str_b]
        bins_list_str_a = [eval(i.replace(']',')')) for i in bins_list_str_a]
        bins_list_str_a = [f'{i[0]:0.2f}' for i in bins_list_str_a]
        # print(bins_list_str_b)
        # exit()
        plt.xticks(np.arange(len(bins_list_str_b))[::5],bins_list_str_b[::5],rotation=90)
        plt.yticks(np.arange(len(bins_list_str_a))[::5],bins_list_str_a[::5])
        plt.xlabel('LAI anomaly in Late GS')
        plt.ylabel('LAI anomaly in Early and Peak GS')
        plt.title(col)
        plt.tight_layout()
        plt.show()


    def check_carryover_hist(self):
        fdir = results_root + f'carryover/'
        # hants_dir = r'C:\Users\pcadmin\Desktop\Data\DIC_Daily\LAI3g\\'
        f = fdir + f'loga_b.npy'
        carryover_dic = T.load_npy(f)
        # hants_dict = T.load_npy_dir(hants_dir)
        carryover_list_all = []
        for pix in tqdm(carryover_dic):
            carryover_list = carryover_dic[pix]

            for i in range(len(carryover_list)):
                carryover_dic_i = carryover_list[i]
                if type(carryover_dic_i) == float:
                    continue
                # print(carryover_dic_i)
                carryover_i = carryover_dic_i['carryover']
                carryover_list_all.append(carryover_i)
        plt.hist(carryover_list_all,bins=300,range=(-5,5))
        plt.show()

    def check_matrix_events_number(self):
        from matplotlib.colors import LogNorm
        fdir = results_root + f'carryover/'
        # hants_dir = r'C:\Users\pcadmin\Desktop\Data\DIC_Daily\LAI3g\\'
        f = fdir + f'loga_b.npy'
        carryover_dic = T.load_npy(f)
        # hants_dict = T.load_npy_dir(hants_dir)
        a_list = []
        b_list = []
        carryover_list_all = []
        for pix in tqdm(carryover_dic):
            # print(pix)
            r, c = pix
            # if r < 40:
            #     continue
            carryover_list = carryover_dic[pix]

            for i in range(len(carryover_list)):
                carryover_dic_i = carryover_list[i]
                if type(carryover_dic_i) == float:
                    continue
                # print(carryover_dic_i)
                carryover_i = carryover_dic_i['carryover']
                a_i = carryover_dic_i['a']
                b_i = carryover_dic_i['b']
                a_list.append(a_i)
                b_list.append(b_i)
                carryover_list_all.append(carryover_i)
        df = pd.DataFrame({'a': a_list, 'b': b_list, 'carryover': carryover_list_all})
        col = 'carryover'
        # a_bin = np.linspace(0,1,40)
        a_bin = np.linspace(-1, 1, 200)
        b_bin = np.linspace(-1, 1, 200)
        df_group_a, bins_list_str_a = T.df_bin(df, 'a', a_bin)
        matrix = []
        bins_list_str_b = ''
        for name_a, df_group_a_i in df_group_a:
            df_group_b, bins_list_str_b = T.df_bin(df_group_a_i, 'b', b_bin)
            temp = []
            for name_b, df_group_b_i in df_group_b:
                vals = df_group_b_i[col].tolist()
                # mean = np.nanmean(vals)
                mean = len(vals)
                temp.append(mean)
            matrix.append(temp)
        matrix = np.array(matrix)
        # plt.imshow(matrix,cmap='jet',vmin=-10,vmax=10)
        # plt.imshow(matrix, cmap='RdBu', vmin=-4, vmax=4)
        plt.imshow(matrix, cmap='Blues', norm=LogNorm())
        # plt.imshow(matrix,cmap='jet')
        plt.colorbar()
        bins_list_str_b = [eval(i.replace(']', ')')) for i in bins_list_str_b]
        bins_list_str_b = [f'{i[0]:0.2f}' for i in bins_list_str_b]
        bins_list_str_a = [eval(i.replace(']', ')')) for i in bins_list_str_a]
        bins_list_str_a = [f'{i[0]:0.2f}' for i in bins_list_str_a]
        # print(bins_list_str_b)
        # exit()
        plt.xticks(np.arange(len(bins_list_str_b))[::5], bins_list_str_b[::5], rotation=90)
        plt.yticks(np.arange(len(bins_list_str_a))[::5], bins_list_str_a[::5])
        plt.xlabel('LAI anomaly in Late GS')
        plt.ylabel('LAI anomaly in Early and Peak GS')
        plt.title(col)
        plt.tight_layout()
        plt.show()
        pass

class Carryover_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Carryover_analysis',
                                                                                       results_root)
        pass

    def run(self):
        # self.eos_ratio()
        self.eos_ratio_dynamic()
        # self.eos_pdf()
        # self.late_ratio()
        # self.late_lai_ratio_dynamic()
        # self.late_lai_pdf()
        # self.carryover_x_threshold()
        # self.plot_carryover_x_threshold()
        pass

    # def add_early_peak_mean_to_df(self,df):
    #     f = join(data_root, 'x_variables_detrend/X_predictors/detrend_LAI3g_early_peak_anomaly.npy')
    #     # f = join(fdir, 'detrend_LAI3g_early_peak_anomaly.npy')
    #     dict_early_peak = T.load_npy(f)
    #     df = T.add_spatial_dic_to_df(df, dict_early_peak, 'early_peak_mean')
    #     return df


    def eos_ratio(self):
        outdir = join(self.this_class_arr, 'eos_ratio')
        T.mk_dir(outdir, force=True)
        # fdir = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/lai3g_pheno'
        fdir = join(results_root, 'carryover_ML/lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        dict_all = {'sos':sos_dict,'eos':eos_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = green_driver_trend_contribution.Build_dataframe().add_GLC_landcover_data_to_df(df)
        df = green_driver_trend_contribution.Build_dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = self.add_early_peak_mean_to_df(df)
        df = df[df['NDVI_MASK']==1]
        df = df[df['landcover_GLC']!='Crop']
        df = df[df['lat']>=30]
        # landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        for i,row in df.iterrows():
            sos_list = row['sos']
            eos_list = row['eos']
            early_peak_mean = row['early_peak_mean']
            if type(early_peak_mean) == float:
                continue
            mean_sos = np.mean(sos_list)
            mean_eos = np.mean(eos_list)
            std_sos = np.std(sos_list)
            std_eos = np.std(eos_list)
            advanced_sos_list = []
            for j in range(len(sos_list)):
                if sos_list[j]<mean_sos: # advanced SOS
                    advanced_sos_list.append(j)
                # if early_peak_mean[j]>0: # early peak greener than the average
                #     advanced_sos_list.append(j)
                # advanced_sos_list.append(j)
            advanced_sos_list = np.array(advanced_sos_list)
            advanced_eos_list = []
            for j in advanced_sos_list:
                eos = eos_list[j]
                if eos<mean_eos:
                    advanced_eos_list.append(eos-mean_eos)
            adv_eos_ratio = len(advanced_eos_list)/len(advanced_sos_list)
            mean_adv_eos = np.mean(advanced_eos_list)
            df.loc[i,'advanced_eos_ratio'] = adv_eos_ratio
            df.loc[i,'mean_advanced_eos'] = mean_adv_eos
            df.loc[i,'std_sos'] = std_sos / mean_eos
            df.loc[i,'std_eos'] = std_eos / mean_eos
        df = Dataframe().Dataframe_build.add_AI_to_df(df)
        AI_bins = np.linspace(0.1,1.6,21)
        # lat_bins = np.linspace(30,90,31)
        df_group,bin_names = T.df_bin(df, 'AI', AI_bins)
        # df_group,bin_names = T.df_bin(df, 'lat', lat_bins)
        x_list = []
        y_list = []
        yerr_list = []
        flag = 0
        for name,df_group_i in df_group:
            vals = df_group_i['advanced_eos_ratio'].tolist()
            mean = np.nanmean(vals)
            # err = np.nanstd(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(flag)
            y_list.append(mean)
            yerr_list.append(err)
            flag += 1
        plt.errorbar(x_list,y_list,yerr=yerr_list,fmt='o')
        plt.xlabel('AI')
        plt.ylabel('advanced EOS ratio')
        plt.title('AI vs advanced EOS ratio')
        plt.xticks(x_list,bin_names,rotation=90)
        plt.tight_layout()
        plt.show()
        advanced_eos_ratio_dict = T.df_to_spatial_dic(df,'advanced_eos_ratio')
        advanced_eos_ratio_arr = DIC_and_TIF().pix_dic_to_spatial_arr(advanced_eos_ratio_dict)
        plt.imshow(advanced_eos_ratio_arr,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()

    def eos_ratio_dynamic(self):
        outdir = join(self.this_class_tif, 'eos_ratio_dynamic')
        T.mk_dir(outdir, force=True)
        T.open_path_and_file(outdir)
        # fdir = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/lai3g_pheno'
        fdir = join(data_root, 'lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        lai3g_early_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/early')
        lai3g_peak_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/peak')
        lai3g_late_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/late')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        lai3g_early_dict = T.load_npy_dir(lai3g_early_dir)
        lai3g_peak_dict = T.load_npy_dir(lai3g_peak_dir)
        lai3g_late_dict = T.load_npy_dir(lai3g_late_dir)
        dict_all = {'sos':sos_dict,'eos':eos_dict,'LAI3g_late':lai3g_late_dict,'LAI3g_peak':lai3g_peak_dict,'LAI3g_early':lai3g_early_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = Dataframe().add_GLC_landcover_data_to_df(df)
        df = Dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = Dataframe().add_early_peak_lai(df)
        df = df[df['NDVI_MASK']==1]
        df = df[df['landcover_GLC']!='Crop']
        df = df[df['lat']>=30]

        # landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        for i,row in df.iterrows():
            # sos_list = row['sos']
            eos_list = row['eos']
            eos_mean = np.nanmean(eos_list)
            eos_anomaly = []
            for eos in eos_list:
                eos_anomaly.append(eos-eos_mean)
            early_peak_mean = row['LAI3g_early_peak_mean']
            # late_lai = row['LAI3g_late']

            # exit()
            if type(early_peak_mean) == float:
                continue
            if type(eos_list) == float:
                continue
            # print(len(early_peak_mean))
            # print(len(late_lai))
            if not len(early_peak_mean) == len(eos_list):
                continue

            enhanced_early_peak_mean = []
            for j in range(len(early_peak_mean)):
                if early_peak_mean[j]>0:
                    enhanced_early_peak_mean.append(j)

            advanced_eos_list = []
            for j in enhanced_early_peak_mean:
                if eos_anomaly[j]>0: # eos_anomaly less than 0
                    advanced_eos_list.append(j)
            ratio = len(advanced_eos_list)/len(enhanced_early_peak_mean)
            df.loc[i,'ratio'] = ratio
        df = Dataframe().add_AI_to_df(df)

        ratio_dict = T.df_to_spatial_dic(df,'ratio')
        ratio_arr = DIC_and_TIF().pix_dic_to_spatial_arr(ratio_dict)
        DIC_and_TIF().arr_to_tif(ratio_arr,join(outdir,'ratio.tif'))
        # exit()
        ratio_arr[ratio_arr<-999] = np.nan
        plt.imshow(ratio_arr,cmap='jet_r',interpolation='nearest',vmin=0.5,vmax=1)
        plt.colorbar()
        plt.figure()
        AI_bins = np.linspace(0.1,3.0,21)
        # lat_bins = np.linspace(30,90,31)
        df_group,bin_names = T.df_bin(df, 'AI', AI_bins)
        # df_group,bin_names = T.df_bin(df, 'lat', lat_bins)
        x_list = []
        y_list = []
        yerr_list = []
        flag = 0
        for name,df_group_i in df_group:
            vals = df_group_i['ratio'].tolist()
            mean = np.nanmean(vals)
            # err = np.nanstd(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(flag)
            y_list.append(mean)
            yerr_list.append(err)
            flag += 1
        plt.errorbar(x_list,y_list,yerr=yerr_list,fmt='o')
        plt.xlabel('AI')
        plt.ylabel('ratio')
        plt.title('AI vs ratio')
        plt.xticks(x_list,bin_names,rotation=90)
        plt.tight_layout()
        plt.show()

    def late_lai_ratio_dynamic(self):
        outdir = join(self.this_class_tif, 'late_lai_ratio_dynamic')
        T.mk_dir(outdir, force=True)
        T.open_path_and_file(outdir)
        # fdir = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/lai3g_pheno'
        fdir = join(data_root, 'lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        lai3g_early_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/early')
        lai3g_peak_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/peak')
        lai3g_late_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal/LAI3g/anomaly_detrend/late')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        lai3g_early_dict = T.load_npy_dir(lai3g_early_dir)
        lai3g_peak_dict = T.load_npy_dir(lai3g_peak_dir)
        lai3g_late_dict = T.load_npy_dir(lai3g_late_dir)
        dict_all = {'sos':sos_dict,'eos':eos_dict,'LAI3g_late':lai3g_late_dict,'LAI3g_peak':lai3g_peak_dict,'LAI3g_early':lai3g_early_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = Dataframe().add_GLC_landcover_data_to_df(df)
        df = Dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = Dataframe().add_early_peak_lai(df)
        df = df[df['NDVI_MASK']==1]
        df = df[df['landcover_GLC']!='Crop']
        df = df[df['lat']>=30]

        # landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        for i,row in df.iterrows():
            # sos_list = row['sos']
            # eos_list = row['eos']
            early_peak_mean = row['LAI3g_early_peak_mean']
            late_lai = row['LAI3g_late']

            # exit()
            if type(early_peak_mean) == float:
                continue
            if type(late_lai) == float:
                continue
            # print(len(early_peak_mean))
            # print(len(late_lai))
            if not len(early_peak_mean) == len(late_lai):
                continue

            enhanced_early_peak_mean = []
            for j in range(len(early_peak_mean)):
                if early_peak_mean[j]>0:
                    enhanced_early_peak_mean.append(j)

            late_lai_list = []
            for j in enhanced_early_peak_mean:
                if late_lai[j]>0: # late lai greater than 0
                    late_lai_list.append(j)
            ratio = len(late_lai_list)/len(enhanced_early_peak_mean)
            df.loc[i,'ratio'] = ratio
        df = Dataframe().add_AI_to_df(df)

        ratio_dict = T.df_to_spatial_dic(df,'ratio')
        ratio_arr = DIC_and_TIF().pix_dic_to_spatial_arr(ratio_dict)
        DIC_and_TIF().arr_to_tif(ratio_arr,join(outdir,'ratio.tif'))
        # exit()
        ratio_arr[ratio_arr<-999] = np.nan
        plt.imshow(ratio_arr,cmap='jet_r',interpolation='nearest',vmin=0.5,vmax=1)
        plt.colorbar()
        plt.figure()
        AI_bins = np.linspace(0.1,3.0,21)
        # lat_bins = np.linspace(30,90,31)
        df_group,bin_names = T.df_bin(df, 'AI', AI_bins)
        # df_group,bin_names = T.df_bin(df, 'lat', lat_bins)
        x_list = []
        y_list = []
        yerr_list = []
        flag = 0
        for name,df_group_i in df_group:
            vals = df_group_i['ratio'].tolist()
            mean = np.nanmean(vals)
            # err = np.nanstd(vals)
            err,_,_ = T.uncertainty_err(vals)
            x_list.append(flag)
            y_list.append(mean)
            yerr_list.append(err)
            flag += 1
        plt.errorbar(x_list,y_list,yerr=yerr_list,fmt='o')
        plt.xlabel('AI')
        plt.ylabel('ratio')
        plt.title('AI vs ratio')
        plt.xticks(x_list,bin_names,rotation=90)
        plt.tight_layout()
        plt.show()

    def late_lai_pdf(self):
        # fdir = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/lai3g_pheno'
        fdir = join(results_root, 'carryover_ML/lai3g_pheno')
        sos_f = join(fdir, 'early_start.npy')
        eos_f = join(fdir, 'late_end.npy')
        lai3g_late_dir = join(results_root, 'carryover_ML/Detrend_pick_seasonal/LAI3g/anomaly_detrend/late')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        lai3g_late_dict = T.load_npy_dir(lai3g_late_dir)
        dict_all = {'sos': sos_dict, 'eos': eos_dict, 'LAI3g_late': lai3g_late_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = green_driver_trend_contribution.Build_dataframe().add_GLC_landcover_data_to_df(df)
        df = green_driver_trend_contribution.Build_dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = self.add_early_peak_mean_to_df(df)
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['lat'] >= 30]

        # landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        late_lai_list_enhanced = []
        late_lai_list_low = []
        for i, row in df.iterrows():
            # sos_list = row['sos']
            # eos_list = row['eos']
            early_peak_mean = row['early_peak_mean']
            late_lai = row['LAI3g_late']

            # exit()
            if type(early_peak_mean) == float:
                continue
            if type(late_lai) == float:
                continue
            # print(len(early_peak_mean))
            # print(len(late_lai))
            if not len(early_peak_mean) == len(late_lai):
                continue

            enhanced_early_peak_mean = []
            low_early_peak_mean = []
            for j in range(len(early_peak_mean)):
                if early_peak_mean[j] > 0:
                    enhanced_early_peak_mean.append(j)
                else:
                    low_early_peak_mean.append(j)

            for j in enhanced_early_peak_mean:
                # if late_lai[j] > 0:  # late lai greater than 0
                late_lai_list_enhanced.append(late_lai[j])
            for j in low_early_peak_mean:
                late_lai_list_low.append(late_lai[j])
        plt.hist(late_lai_list_enhanced, bins=200, alpha=0.5, label='enhanced', density=True)
        plt.hist(late_lai_list_low, bins=200, alpha=0.5, label='low', density=True)
        plt.legend()
        plt.show()


    def eos_pdf(self):
        # fdir = '/Volumes/NVME2T/greening_project_redo/Result/carryover_ML/lai3g_pheno'
        fdir = join(results_root, 'carryover_ML/lai3g_pheno')
        sos_f = join(fdir, 'early_start.npy')
        eos_f = join(fdir, 'late_end.npy')
        lai3g_late_dir = join(results_root, 'carryover_ML/Detrend_pick_seasonal/LAI3g/anomaly_detrend/late')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        lai3g_late_dict = T.load_npy_dir(lai3g_late_dir)
        dict_all = {'sos': sos_dict, 'eos': eos_dict, 'LAI3g_late': lai3g_late_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = green_driver_trend_contribution.Build_dataframe().add_GLC_landcover_data_to_df(df)
        df = green_driver_trend_contribution.Build_dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = self.add_early_peak_mean_to_df(df)
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['lat'] >= 30]

        # landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        late_lai_list_enhanced = []
        late_lai_list_low = []
        for i, row in df.iterrows():
            # sos_list = row['sos']
            # eos_list = row['eos']
            early_peak_mean = row['early_peak_mean']
            eos = row['eos']

            # exit()
            if type(early_peak_mean) == float:
                continue
            if type(eos) == float:
                continue
            # print(len(early_peak_mean))
            # print(len(late_lai))
            if not len(early_peak_mean) == len(eos):
                continue
            eos_mean = np.mean(eos)
            eos_anomaly = []
            for j in range(len(eos)):
                eos_anomaly.append(eos[j] - eos_mean)
            enhanced_early_peak_mean = []
            low_early_peak_mean = []
            for j in range(len(early_peak_mean)):
                if early_peak_mean[j] > 0:
                    enhanced_early_peak_mean.append(j)
                else:
                    low_early_peak_mean.append(j)

            for j in enhanced_early_peak_mean:
                # if late_lai[j] > 0:  # late lai greater than 0
                late_lai_list_enhanced.append(eos_anomaly[j])
            for j in low_early_peak_mean:
                late_lai_list_low.append(eos_anomaly[j])
        plt.hist(late_lai_list_enhanced, bins=200, alpha=0.5, label='enhanced', density=True,range=(-40,40))
        plt.hist(late_lai_list_low, bins=200, alpha=0.5, label='low', density=True,range=(-40,40))
        plt.legend()
        plt.show()

    def eos_ratio_to_tif(self):
        outdir = join(self.this_class_arr, 'eos_ratio')
        T.mk_dir(outdir, force=True)
        fdir = r'C:\Users\pcadmin\Desktop\Data\phenology_dic\LAI3g'
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        dict_all = {'sos':sos_dict,'eos':eos_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = green_driver_trend_contribution.Build_dataframe().add_GLC_landcover_data_to_df(df)
        df = green_driver_trend_contribution.Build_dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = df[df['NDVI_MASK']==1]
        df = df[df['landcover_GLC']!='Crop']
        df = df[df['lat']>=30]
        landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(landcover_GLC_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif=land_tif)
        # plt.show()
        for i,row in df.iterrows():
            sos_list = row['sos']
            eos_list = row['eos']
            mean_sos = np.mean(sos_list)
            mean_eos = np.mean(eos_list)
            std_sos = np.std(sos_list)
            std_eos = np.std(eos_list)
            advanced_sos_list = []
            for j in range(len(sos_list)):
                if sos_list[j]<mean_sos:
                    advanced_sos_list.append(j)
            advanced_sos_list = np.array(advanced_sos_list)
            advanced_eos_list = []
            for j in advanced_sos_list:
                eos = eos_list[j]
                if eos<mean_eos:
                    advanced_eos_list.append(eos-mean_eos)
            adv_eos_ratio = len(advanced_eos_list)/len(advanced_sos_list)
            mean_adv_eos = np.mean(advanced_eos_list)
            df.loc[i,'advanced_eos_ratio'] = adv_eos_ratio
            df.loc[i,'mean_advanced_eos'] = mean_adv_eos
            df.loc[i,'std_sos'] = std_sos / mean_eos
            df.loc[i,'std_eos'] = std_eos / mean_eos
        eos_ratio_dict = T.df_to_spatial_dic(df,'advanced_eos_ratio')
        mean_eos_dict = T.df_to_spatial_dic(df,'mean_advanced_eos')
        std_sos_dict = T.df_to_spatial_dic(df,'std_sos')
        std_eos_dict = T.df_to_spatial_dic(df,'std_eos')
        eos_ratio_arr = DIC_and_TIF().pix_dic_to_spatial_arr(eos_ratio_dict)
        mean_eos_arr = DIC_and_TIF().pix_dic_to_spatial_arr(mean_eos_dict)
        std_sos_arr = DIC_and_TIF().pix_dic_to_spatial_arr(std_sos_dict)
        std_eos_arr = DIC_and_TIF().pix_dic_to_spatial_arr(std_eos_dict)
        DIC_and_TIF().arr_to_tif(eos_ratio_arr, join(outdir, 'eos_ratio.tif'))
        DIC_and_TIF().arr_to_tif(mean_eos_arr, join(outdir, 'adv_mean_eos.tif'))
        DIC_and_TIF().arr_to_tif(std_sos_arr, join(outdir, 'std_sos.tif'))
        DIC_and_TIF().arr_to_tif(std_eos_arr, join(outdir, 'std_eos.tif'))

        pass

    def carryover_x_threshold(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        x_list = RF().x_variables()
        y = RF().y_variable()
        outdir = join(self.this_class_arr, 'carryover_x_threshold',y)
        T.mk_dir(outdir, force=True)

        pix_list = T.get_df_unique_val_list(df, 'pix')
        flag = 0
        for x in x_list:
            outf = join(outdir, x + '.df')

            result_dict = {}
            flag += 1
            if os.path.isfile(outf):
                continue
            for pix in tqdm(pix_list, desc=f'{flag}/{len(x_list)}'):
                df_pix = df[df['pix'] == pix]
                # sort df_pix
                df_pix = df_pix.sort_values(by=['year'])
                df_pix = df_pix[df_pix['LAI3g_early_peak_mean'] > 0]
                # X = df_pix[x_list]
                # Y = df_pix[y]
                X = df_pix[x]
                Y = df_pix[y]
                # calculate the slope and intercept
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
                    x0 = -(intercept / slope)
                    result_dict[pix] = {'x0': x0, 'slope': slope, 'intercept': intercept, 'r_value': r_value,'p_value': p_value}
                except:
                    pass
            df_result = T.dic_to_df(result_dict,'pix')
            T.save_df(df_result,outf)
            T.df_to_excel(df_result,outf)

    def plot_carryover_x_threshold(self):
        y = RF().y_variable()
        fdir = join(self.this_class_arr, 'carryover_x_threshold', y)
        outdir = join(self.this_class_arr, 'carryover_x_threshold_tif', y)
        T.mk_dir(outdir, force=True)
        x_list = RF().x_variables()
        for x in x_list:
            f = join(fdir, x + '.df')
            df = T.load_df(f)
            spatial_dict_x0 = T.df_to_spatial_dic(df, 'x0')
            spatial_dict_p = T.df_to_spatial_dic(df, 'p_value')
            arr_x0 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_x0)
            arr_p = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_p)
            outf_x0 = join(outdir, x + '_x0.tif')
            outf_p = join(outdir, x + '_p.tif')
            DIC_and_TIF().arr_to_tif(arr_x0, outf_x0)
            DIC_and_TIF().arr_to_tif(arr_p, outf_p)
            # exit()

        pass


class Variables_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Variables_analysis',
                                                                                       results_root)
        pass

    def run(self):
        # self.obs_x_and_y_response_function()
        # self.obs_x_and_y_response_function_bin_y()
        # self.obs_x_and_y_response_function_bin_y_lc()
        # self.obs_x_and_y_response_function_matrix()
        # self.obs_x_and_y_response_function_matrix_PFTs()
        # self.x_variables_to_y_correlation()
        # self.eos_and_late_lai()
        # self.eos_and_late_lai_lc()
        # self.pdf()
        self.relative_timeseries_one_pix()
        # self.relative_timeseries_all_pix()
        pass


    def obs_x_and_y_response_function(self):
        n = 41
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        df = df[df['LAI3g_early_peak_mean'] >= 0]
        x_variables_list = RF().x_variables()
        y = RF().y_variable()
        lc_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        # T.print_head_n(df, 5)
        # exit()

        # for lc in lc_list:
        flag = 1
        # df_lc = df[df['landcover_GLC'] == lc]
        plt.figure(figsize=(12, 8))
        for x in x_variables_list:
            x_bins = np.linspace(df[x].min(), df[x].max(), n)
            df_group, bin_names = T.df_bin(df, x, x_bins)
            mean_list = []
            std_list = []
            vals_count_list = []
            for name, df_group_i in df_group:
                Y = df_group_i[y].values
                if len(Y) == 0:
                    mean_list.append(np.nan)
                    std_list.append(np.nan)
                    vals_count_list.append(np.nan)
                    continue
                greater_than_zero = Y < 0
                Y = Y[greater_than_zero]
                # Y_mean = len(Y) / len(df_group_i)
                count = len(Y)
                vals_count_list.append(count/len(df))
                # exit()
                Y_mean = np.nanmean(Y)
                # Y_std = np.nanstd(Y)
                # Y_std,_,_ = T.uncertainty_err(Y)
                mean_list.append(Y_mean)
                # std_list.append(Y_std)
            mean_list = np.array(mean_list)
            std_list = np.array(std_list)
            plt.subplot(3, 3, flag)
            # Plot().plot_line_with_error_bar(x_bins[1:], mean_list, std_list)
            # Plot().plot_line_with_gradient_error_band(x_bins[1:], mean_list, std_list,pow=8,color_gradient_n=500)
            plt.plot(x_bins[1:], mean_list, '--', color='r')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.twinx()
            plt.bar(x_bins[1:], vals_count_list, width=0.1, color='b', alpha=0.5)
            # plt.ylim(min(mean_list) - 0.2, max(mean_list) + 0.2)
            # plt.ylim(0, 1)
            # plt.title(x)
            # plt.xticks(rotation=90)

            # plt.hlines(0, -3, 3, linestyles='--', colors='k')
            # plt.vlines(0, min(mean_list) - 0.2, max(mean_list) + 0.2, linestyles='--', colors='k')
            flag += 1
        plt.tight_layout()
        # plt.suptitle(lc)
        plt.show()

    def obs_x_and_y_response_function_bin_y(self):
        outdir = join(self.this_class_png, 'obs_x_and_y_response_function_bin_y')
        T.mk_dir(outdir, force=True)
        n = 41
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        df = df[df['LAI3g_early_peak_mean_anomaly'] >= 0]
        x_variables_list = RF().x_variables()
        y = RF().y_variable()
        flag = 1
        plt.figure(figsize=(12, 8))
        for x in x_variables_list:
            y_bins = np.linspace(df[y].min(), df[y].max(), n)
            df_group, bin_names = T.df_bin(df, y, y_bins)
            mean_list = []
            std_list = []
            vals_count_list = []
            for name, df_group_i in df_group:
                X = df_group_i[x].values
                if len(X) == 0:
                    mean_list.append(np.nan)
                    std_list.append(np.nan)
                    vals_count_list.append(np.nan)
                    continue
                # greater_than_zero = Y < 0
                # Y = Y[greater_than_zero]
                # Y_mean = len(Y) / len(df_group_i)
                count = len(X)
                vals_count_list.append(count/len(df))
                # exit()
                X_mean = np.nanmean(X)
                # Y_std = np.nanstd(Y)
                X_std,_,_ = T.uncertainty_err(X)
                mean_list.append(X_mean)
                std_list.append(X_std)
            mean_list = np.array(mean_list)
            std_list = np.array(std_list)
            plt.subplot(3, 3, flag)
            Plot().plot_line_with_error_bar(y_bins[1:], mean_list, std_list)
            # Plot().plot_line_with_gradient_error_band(x_bins[1:], mean_list, std_list,pow=8,color_gradient_n=500)
            plt.plot(y_bins[1:], mean_list, '--', color='r')
            # plt.hlines(0, df[y].min(), df[y].max(), linestyles='--', colors='k')
            plt.vlines(0, np.nanmin(mean_list), np.nanmax(mean_list), linestyles='--', colors='k')
            plt.xlabel(y)
            plt.ylabel(x)
            plt.ylim(np.nanmin(mean_list), np.nanmax(mean_list))
            plt.twinx()
            plt.bar(y_bins[1:], vals_count_list, color='b', alpha=0.3, width=(df[y].max() - df[y].min()) / (1.5 * n))
            # plt.ylim(min(mean_list) - 0.2, max(mean_list) + 0.2)
            # plt.ylim(0, 1)
            # plt.title(x)
            # plt.xticks(rotation=90)

            # plt.hlines(0, -3, 3, linestyles='--', colors='k')
            # plt.vlines(0, min(mean_list) - 0.2, max(mean_list) + 0.2, linestyles='--', colors='k')
            flag += 1
        plt.tight_layout()
        # plt.suptitle(lc)
        outf = join(outdir, 'pdf.pdf')
        plt.savefig(outf)
        plt.close()
        # plt.show()

        pass

    def obs_x_and_y_response_function_bin_y_lc(self):
        lc_var = 'koppen'
        # lc_var = 'landcover_GLC'
        outdir = join(self.this_class_png,'obs_x_and_y_response_function_bin_y_lc',lc_var)
        T.mk_dir(outdir,force=True)
        T.open_path_and_file(outdir)
        n = 41
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        df = df[df['LAI3g_early_peak_mean_anomaly'] >= 0]
        x_variables_list = RF().x_variables()
        y = RF().y_variable()
        lc_list = T.get_df_unique_val_list(df, lc_var)

        for lc in lc_list:
            flag = 1
            df_lc = df[df[lc_var] == lc]
            df_len = len(df_lc)
            plt.figure(figsize=(12, 8))
            for x in x_variables_list:
                y_bins = np.linspace(df_lc[y].min(), df_lc[y].max(), n)
                df_group, bin_names = T.df_bin(df_lc, y, y_bins)
                mean_list = []
                std_list = []
                vals_count_list = []
                for name, df_group_i in df_group:
                    X = df_group_i[x].values
                    if len(X) == 0:
                        mean_list.append(np.nan)
                        std_list.append(np.nan)
                        vals_count_list.append(np.nan)
                        continue
                    # greater_than_zero = Y < 0
                    # Y = Y[greater_than_zero]
                    # Y_mean = len(Y) / len(df_group_i)
                    count = len(X)
                    vals_count_list.append(count/len(df_lc))
                    # exit()
                    X_mean = np.nanmean(X)
                    # Y_std = np.nanstd(Y)
                    X_std,_,_ = T.uncertainty_err(X)
                    mean_list.append(X_mean)
                    std_list.append(X_std)
                mean_list = np.array(mean_list)
                std_list = np.array(std_list)
                plt.subplot(3, 3, flag)
                Plot().plot_line_with_error_bar(y_bins[1:], mean_list, std_list)
                # Plot().plot_line_with_gradient_error_band(x_bins[1:], mean_list, std_list,pow=8,color_gradient_n=500)
                plt.plot(y_bins[1:], mean_list, '--', color='r')
                # plt.hlines(0, df[y].min(), df[y].max(), linestyles='--', colors='k')
                plt.vlines(0, np.nanmin(mean_list), np.nanmax(mean_list), linestyles='--', colors='k')
                plt.xlabel(y)
                plt.ylabel(x)
                plt.ylim(np.nanmin(mean_list), np.nanmax(mean_list))
                plt.twinx()
                plt.bar(y_bins[1:], vals_count_list, color='b', alpha=0.3, width=(df_lc[y].max() - df_lc[y].min()) / (1.5 * n))
                # plt.ylim(min(mean_list) - 0.2, max(mean_list) + 0.2)
                # plt.ylim(0, 1)
                # plt.title(x)
                # plt.xticks(rotation=90)

                # plt.hlines(0, -3, 3, linestyles='--', colors='k')
                # plt.vlines(0, min(mean_list) - 0.2, max(mean_list) + 0.2, linestyles='--', colors='k')
                flag += 1
            plt.suptitle(f'{lc} df_len:{df_len}')
            plt.tight_layout()
            outf = join(outdir, lc + '.pdf')
            plt.savefig(outf)
            plt.close()
        # plt.show()

        pass

    def obs_x_and_y_response_function_matrix(self):
        outdir = join(self.this_class_arr, 'obs_x_and_y_response_function_matrix')
        T.mkdir(outdir,force=True)
        n = 21
        dff = r'D:\Greening\Result\carryover_ML\Dataframe\juping\data_frame.df'
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        # df = df[df['LAI3g_early_peak_mean'] >= 0]
        y = 'LAI3g_early_peak_mean'
        x_variables_list = RF().x_variables()
        col = RF().y_variable()
        y_vals = df[y].values
        # y_bins = np.linspace(0, max(y_vals), n)
        y_bins = np.linspace(min(y_vals), max(y_vals), n)

        lc_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        plt.figure(figsize=(16, 12))
        flag = 1

        for x in x_variables_list:
            x_vals = df[x].values
            x_bins = np.linspace(min(x_vals), max(x_vals),n)

            df_group_y, bin_names_y = T.df_bin(df, y, y_bins)
            matrix = []
            for name_y, df_group_y_i in df_group_y:

                df_group_x, bin_names_x = T.df_bin(df_group_y_i, x, x_bins)
                temp = []
                for name_x, df_group_x_i in df_group_x:
                    Y = df_group_x_i[col].values
                    Y_mean = np.nanmean(Y)
                    Y_std = np.nanstd(Y)
                    temp.append(Y_mean)
                    # Y_std,_,_ = T.uncertainty_err(Y)
                matrix.append(temp)
            matrix = np.array(matrix)
            plt.subplot(3, 3, flag)
            # plt.title(x)
            plt.imshow(matrix, cmap='RdBu', interpolation='nearest', aspect='auto',vmin=-0.4,vmax=0.4)
            # plt.imshow(matrix, cmap='RdBu', interpolation='nearest', aspect='auto')
            plt.xticks(rotation=90)
            plt.xticks(range(len(bin_names_x))[::5], bin_names_x[::5])
            plt.yticks(range(len(bin_names_y))[::5], bin_names_y[::5])
            plt.xlabel(x)
            plt.ylabel(y)
            # plt.hlines(0, -3, 3, linestyles='--', colors='k')
            # plt.vlines(0, min(mean_list) - 0.2, max(mean_list) + 0.2, linestyles='--', colors='k')
            flag += 1
            plt.colorbar()
            # plt.show()
        plt.tight_layout()
        # plt.suptitle(lc)
        # plt.show()
        plt.savefig(join(outdir,'obs_x_and_y_response_function_matrix.pdf'))
        plt.close()
        pass

    def obs_x_and_y_response_function_matrix_PFTs(self):
        outdir = join(self.this_class_arr, 'obs_x_and_y_response_function_matrix_PFTs')
        T.mkdir(outdir,force=True)
        n = 21
        dff = r'D:\Greening\Result\carryover_ML\Dataframe\detrended\data_frame.df'
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        lc_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        # df = df[df['LAI3g_early_peak_mean'] >= 0]
        y = 'LAI3g_early_peak_mean'
        x_variables_list = RF().x_variables()
        col = RF().y_variable()
        x_bins = np.linspace(-3, 3, n)
        y_bins = np.linspace(0, 3, n)
        lc_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        # plt.figure(figsize=(12, 8))
        for lc in lc_list:
            df_lc = df[df['landcover_GLC'] == lc]
            flag = 1
            plt.figure(figsize=(16, 12))

            for x in x_variables_list:
                df_group_y, bin_names_y = T.df_bin(df_lc, y, y_bins)
                matrix = []
                for name_y, df_group_y_i in df_group_y:
                    df_group_x, bin_names_x = T.df_bin(df_group_y_i, x, x_bins)
                    temp = []
                    for name_x, df_group_x_i in df_group_x:
                        Y = df_group_x_i[col].values
                        Y_mean = np.nanmean(Y)
                        Y_std = np.nanstd(Y)
                        temp.append(Y_mean)
                        # Y_std,_,_ = T.uncertainty_err(Y)
                    matrix.append(temp)
                matrix = np.array(matrix)
                plt.subplot(3, 3, flag)
                # plt.title(x)
                plt.imshow(matrix, cmap='RdBu', interpolation='nearest', aspect='auto',vmin=-1,vmax=1)
                plt.xticks(rotation=90)
                plt.xticks(range(len(bin_names_x))[::5], bin_names_x[::5])
                plt.yticks(range(len(bin_names_y))[::5], bin_names_y[::5])
                plt.xlabel(x)
                plt.ylabel(y)
                # plt.hlines(0, -3, 3, linestyles='--', colors='k')
                # plt.vlines(0, min(mean_list) - 0.2, max(mean_list) + 0.2, linestyles='--', colors='k')
                flag += 1
                plt.colorbar()
                # plt.show()
            plt.suptitle(lc)

            plt.tight_layout()
            # plt.show()
            plt.savefig(join(outdir,f'{lc}.pdf'))
            plt.close()
            pass

    def x_variables_to_y_correlation(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        df = df[df['LAI3g_early_peak_mean'] >= 0]
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        AI_bins = np.linspace(0.1, 2., 21)
        x_variables_list = RF().x_variables()
        y = RF().y_variable()

        for lc in lc_list:
            df_lc = df[df['landcover_GLC']==lc]
            df_group, bin_names = T.df_bin(df_lc, 'AI', AI_bins)
            # df_group, bin_names = T.df_bin(df, 'AI', AI_bins)
            matrix = []
            sig_row = []
            sig_col = []
            row = 0
            for x in x_variables_list:
                temp = []
                col = 0
                for name,df_group_i in df_group:
                    X = df_group_i[x].values
                    Y = df_group_i[y].values
                    r,p = T.nan_correlation(X,Y)
                    temp.append(r)
                    if p < 0.01:
                        sig_row.append(row)
                        sig_col.append(col)
                    col += 1
                row += 1
                matrix.append(temp)
            matrix = np.array(matrix)
            plt.figure()
            plt.imshow(matrix, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
            plt.colorbar()

            plt.scatter(sig_col, sig_row, c='k', s=40, marker='*')
            plt.xticks(np.arange(len(bin_names)), bin_names, rotation=90)
            plt.yticks(np.arange(len(x_variables_list)), x_variables_list)
            plt.title(f'{y}-{lc}')
            # plt.title(f'{y}')
            plt.tight_layout()
        plt.show()

        pass

    def eos_and_late_lai_lc(self):
        # df = df[df['landcover_GLC']=='EF']
        outdir = join(self.this_class_png, 'eos_and_late_lai_lc')
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        # df = df[df['LAI3g_early_peak_mean'] >= 0]
        lc_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        for lc in lc_list:
            df_lc = df[df['landcover_GLC'] == lc]
            x = 'eos_std_anomaly'
            y = 'LAI3g_late'
            X = df_lc[x].values
            Y = df_lc[y].values
            sns.jointplot(x=X, y=Y, kind="hex", color="#4CB391",gridsize=100,xlim=(-3,3),ylim=(-3,3))
            plt.xlabel(x)
            plt.ylabel(y)
            r,p = T.nan_correlation(X,Y)
            plt.title(f'{lc} r={r:.2f} p={p:.2f}')
            plt.tight_layout()
            outf = join(outdir,f'{lc}_not_pick.pdf')
            plt.savefig(outf)
            plt.close()


    def eos_and_late_lai(self):
        fdir = join(data_root, 'lai3g_pheno')
        sos_f = join(fdir, 'early_start.npy')
        eos_f = join(fdir, 'late_end.npy')
        lai3g_late_dir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr, 'pick_daily_seasonal/LAI3g/std_anomaly_detrend/late')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        lai3g_late_dict = T.load_npy_dir(lai3g_late_dir)
        dict_all = {'sos': sos_dict, 'eos': eos_dict, 'LAI3g_late': lai3g_late_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = Dataframe().add_GLC_landcover_data_to_df(df)
        df = Dataframe().add_NDVI_mask(df)
        df = T.add_lon_lat_to_df(df)
        df = Carryover_analysis().add_early_peak_mean_to_df(df)
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['lat'] >= 30]

        late_lai_all = []
        eos_all = []
        for i, row in df.iterrows():
            # sos_list = row['sos']
            # eos_list = row['eos']
            early_peak_mean = row['early_peak_mean']
            eos = row['eos']
            late_lai = row['LAI3g_late']

            # exit()
            if type(early_peak_mean) == float:
                continue
            if type(eos) == float:
                continue
            # print(len(early_peak_mean))
            # print(len(late_lai))
            if not len(early_peak_mean) == len(eos):
                continue
            eos_mean = np.nanmean(eos)
            eos_std = np.nanstd(eos)
            eos_anomaly = []
            for j in range(len(eos)):
                eos_anomaly.append((eos[j] - eos_mean) / eos_std)
            enhanced_early_peak_mean = []
            low_early_peak_mean = []
            for j in range(len(early_peak_mean)):
                if early_peak_mean[j] > 0:
                    enhanced_early_peak_mean.append(j)
                else:
                    low_early_peak_mean.append(j)

            for j in enhanced_early_peak_mean:
                # if late_lai[j] > 0:  # late lai greater than 0
                late_lai_all.append(late_lai[j])
                eos_all.append(eos_anomaly[j])
        # plt.scatter(eos_all, late_lai_all)
        # KDE_plot().plot_scatter(late_lai_all, eos_all)
        KDE_plot().plot_scatter_hex(late_lai_all, eos_all,xlim=(-3,3),ylim=(-3,3))
        r, p = T.nan_correlation(late_lai_all, eos_all)
        plt.xlabel('LAI3g_late')
        plt.ylabel('eos_std_anomaly')
        print(r, p)
        plt.tight_layout()
        plt.show()

    def pdf(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        df = Dataframe().clean_df(df)
        x_list = RF().x_variables()
        y = RF().y_variable()
        # df = df[df['LAI3g_early_peak_mean'] >= 0]
        # df2 = df[df['LAI3g_early_peak_mean'] < 0]
        df = df[df['LAI3g_early'] >= 0]
        df = df[df['LAI3g_peak'] >= 0]
        # df = df[df['year'] == 2015]
        Y = df[y].values
        plt.hist(Y, bins=900)
        plt.show()
        for x in x_list:
            df1 = df[df[y] >= 0]
            df2 = df[df[y] < 0]
            # T.print_head_n(df1, 5)
            # exit()
            vals1 = df1[x].values
            vals2 = df2[x].values
            # T.ANOVA_test()
            plt.figure()
            plt.hist(vals1, bins=100, alpha=0.5, label='LAI3g_lai>=0', density=True)
            plt.hist(vals2, bins=100, alpha=0.5, label='LAI3g_lai<0', density=True)
            plt.legend()
            plt.title(x)
            plt.show()


    def relative_timeseries_one_pix(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        picked_pix = []
        for pix in pix_list:
            r,c = pix
            if r<80:
                continue
            picked_pix.append(pix)
        for pix in tqdm(picked_pix):
            df_pix = df[df['pix'] == pix]
            lon = df_pix['lon'].values[0]
            lat = df_pix['lat'].values[0]
            glc = df_pix['landcover_GLC'].values[0]
            koppen = df_pix['koppen'].values[0]
            print(pix, lon, lat)

            if lat > 60:
                continue
            df_pix = df_pix.sort_values(by=['year'])
            pick_index = df_pix['LAI3g_early_peak_mean']< -0.5
            pick_index_1 = df_pix['LAI3g_early_peak_mean']> 0.5
            pick_index2 = df_pix['LAI3g_late']< -0.5
            pick_index3 = df_pix['LAI3g_late']> 0.5
            pick_index4 = df_pix['SPEI3_peak']< -1.5
            pick_index5 = df_pix['SPEI3_late']< -1.5
            df_pix['LAI3g_early_peak_mean1'] = np.nan
            df_pix['LAI3g_early_peak_mean1'][pick_index] = -9999
            df_pix['LAI3g_early_peak_mean1'][pick_index_1] = 9999
            df_pix['LAI3g_late1'] = np.nan
            df_pix['LAI3g_late1'][pick_index2] = -9999
            df_pix['LAI3g_late1'][pick_index3] = 9999
            # df_pix['SPEI3_peak1'] = np.nan
            # df_pix['SPEI3_late1'] = np.nan
            df_pix['SPEI3_peak'][pick_index4] = -9999
            df_pix['SPEI3_late'][pick_index5] = -9999
            x_list = RF().x_variables()
            x_list.append('LAI3g_early_peak_mean1')
            x_list.append('LAI3g_late')
            x_list.append('LAI3g_late1')
            # x_list.append('SPEI3_peak1')
            # x_list.append('SPEI3_late1')
            X = df_pix[x_list].values
            df_i = pd.DataFrame(X, columns=x_list)
            # plt.plot(Y, label='LAI3g_lai')
            plt.figure(figsize=(14, 7))
            plt.imshow(df_i.T, aspect='auto', cmap='RdBu',vmin=-4,vmax=4)
            plt.colorbar()
            plt.yticks(range(len(x_list)), x_list)
            # plt.title('lon: {}, lat: {}'.format(lon, lat))
            plt.title('lon: {}, lat: {}\nglc: {}, koppen: {}'.format(lon, lat, glc, koppen))
            plt.tight_layout()
            plt.show()

    def relative_timeseries_all_pix(self):
        mode = 'std_anomaly_detrend'
        data_dir = Pick_detrended_seasonal_variables_dynamic().this_class_arr
        data_dir = join(data_dir, 'pick_daily_seasonal')
        x_list = RF().x_variables()
        y = RF().y_variable()
        all_variables = x_list.copy()
        all_variables.append(y)

        # for

        print(all_variables)
        exit()
        pass


class Detrend_variables:

    def __init__(self):

        pass

    def run(self):
        # self.LAI3g()
        # self.climate_variables()
        self.detrend_anomaly()
        self.detrend_std_anomaly()
        pass

    def LAI3g(self):
        lai_dir = r'C:\Users\pcadmin\Desktop\Data\DIC_Daily\LAI3g\\'
        outdir = r'C:\Users\pcadmin\Desktop\Data\Detrend\LAI3g\\'
        T.mk_dir(outdir,force=True)
        lai_dict = T.load_npy_dir(lai_dir)
        detrend_lai_dict = {}
        for pix in tqdm(lai_dict):
            lai_list = lai_dict[pix]
            if len(lai_list)==0:
                continue
            lai_list = np.array(lai_list)
            lai_list_flatten = lai_list.flatten()
            lai_detrend_flatten = T.detrend_vals(lai_list_flatten)
            lai_detrend = lai_detrend_flatten.reshape(len(lai_list),len(lai_list[0]))
            detrend_lai_dict[pix] = lai_detrend
        T.save_distributed_perpix_dic(detrend_lai_dict,outdir,n=1000)


    def climate_variables(self):

        fdir = r'E:\interpolation_climate_drivers_to_daily\\'
        outdir = r'C:\Users\pcadmin\Desktop\Data\Detrend\\'
        for f in T.listdir(fdir):
            print('loading',f)
            outdir_i = join(outdir,f.replace('_daily_dic.npy',''))
            T.mk_dir(outdir_i,force=True)
            var_dict = T.load_npy(join(fdir,f))
            detrend_var_dict = {}
            for pix in tqdm(var_dict):
                var_list = var_dict[pix]
                var_list = np.array(var_list)
                var_detrend = T.detrend_vals(var_list)
                var_detrend_matrix = var_detrend.reshape(-1,365)
                detrend_var_dict[pix] = var_detrend_matrix
            T.save_distributed_perpix_dic(detrend_var_dict,outdir_i,n=1000)

    def detrend_anomaly(self):
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal')
        for var in T.listdir(fdir):
            fdir_i = join(fdir,var,'anomaly')
            outdir_i = join(fdir,var,'anomaly_detrend')
            T.mk_dir(outdir_i,force=True)
            for season in T.listdir(fdir_i):
                print('loading',var,season)
                fdir_i_i = join(fdir_i,season)
                outdir_i_i = join(outdir_i,season)
                T.mk_dir(outdir_i_i,force=True)
                val_dict = T.load_npy_dir(fdir_i_i)
                detrend_dict = T.detrend_dic(val_dict)
                T.save_distributed_perpix_dic(detrend_dict,outdir_i_i)

    def detrend_std_anomaly(self):
        fdir = join(Pick_detrended_seasonal_variables_dynamic().this_class_arr,'pick_daily_seasonal')
        for var in T.listdir(fdir):
            fdir_i = join(fdir,var,'std_anomaly')
            outdir_i = join(fdir,var,'std_anomaly_detrend')
            T.mk_dir(outdir_i,force=True)
            for season in T.listdir(fdir_i):
                print('loading',var,season)
                fdir_i_i = join(fdir_i,season)
                outdir_i_i = join(outdir_i,season)
                T.mk_dir(outdir_i_i,force=True)
                val_dict = T.load_npy_dir(fdir_i_i)
                detrend_dict = T.detrend_dic(val_dict)
                T.save_distributed_perpix_dic(detrend_dict,outdir_i_i)



class Pick_detrended_seasonal_variables:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Pick_detrended_seasonal_variables',
                                                                                       results_root)
        pass

    def run(self):
        self.daily_phenology()
        # self.pick_seasonal_values()
        # self.calculate_anomaly()
        # self.calculate_anomaly_juping()
        pass


    def daily_phenology(self):
        phenology_dir = r'C:\Users\pcadmin\Desktop\Data\phenology_dic\LAI3g\\'

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


    def pick_seasonal_values(self):
        fdir = r'C:\Users\pcadmin\Desktop\Data\Detrend\\'
        outdir = r'C:\Users\pcadmin\Desktop\Data\Detrend_pick_seasonal\\'
        T.mk_dir(outdir,force=True)
        early_dict, peak_dict, late_dict = self.daily_phenology()

        for folder in T.listdir(fdir):
            print('loading',folder)
            outdir_i = join(outdir,folder)
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

        pass

    def calculate_anomaly(self):
        fdir = r'C:\Users\pcadmin\Desktop\Data\Detrend_pick_seasonal\\'
        for var_i in T.listdir(fdir):
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
                    std = np.nanstd(vals)
                    anomaly = (vals-mean)/std
                    anomaly_dict[pix] = anomaly
                T.save_distributed_perpix_dic(anomaly_dict,outdir_i_i)
                # exit()

    def calculate_anomaly_juping(self):
        fdir = r'C:\Users\pcadmin\Desktop\Data\Detrend_pick_seasonal\\'
        for var_i in T.listdir(fdir):
            fdir_i = join(fdir,var_i,'origin')
            outdir_i = join(fdir,var_i,'anomaly_juping')
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


class Pick_detrended_seasonal_variables_dynamic:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Pick_detrended_seasonal_variables_dynamic',
                                                                                       results_root)

    def run(self):
        # self.daily_phenology()
        # self.pick_daily_seasonal()
        # self.check_seasonal_values()
        self.calculate_anomaly()
        self.calculate_std_anomaly()
        pass

    def daily_phenology(self):
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
            early_range_all_years = []
            peak_range_all_years = []
            late_range_all_years = []
            for i in range(len(early_start)):
                early_start_i = early_start[i]
                early_end_i = early_end[i]
                late_start_i = late_start[i]
                late_end_i = late_end[i]

                early_range = (early_start_i,early_end_i)
                peak_range = (early_end_i,late_start_i)
                late_range = (late_start_i,late_end_i+1)

                early_range_all_years.append(early_range)
                peak_range_all_years.append(peak_range)
                late_range_all_years.append(late_range)
            early_dict[pix] = early_range_all_years
            peak_dict[pix] = peak_range_all_years
            late_dict[pix] = late_range_all_years
        return early_dict,peak_dict,late_dict


    def longterm_phenology(self,all_years_pheno):

        start_list = []
        end_list = []
        for start,end in all_years_pheno:
            start_list.append(start)
            end_list.append(end)
        mean_start = np.nanmean(start_list)
        mean_end = np.nanmean(end_list)
        return (int(mean_start),int(mean_end))


    def pick_daily_seasonal(self):
        fdir = join(data_root,'daily_X')
        outdir = join(self.this_class_arr,'pick_daily_seasonal')
        T.mk_dir(outdir,force=True)
        early_dict, peak_dict, late_dict = self.daily_phenology()

        for folder in T.listdir(fdir):
            print('loading',folder)
            outdir_i = join(outdir,folder,'origin')
            T.mk_dir(outdir_i,force=True)
            vals_dict = T.load_npy_dir(join(fdir,folder),condition='')
            early_vals_dict = {}
            peak_vals_dict = {}
            late_vals_dict = {}
            for pix in tqdm(vals_dict):
                vals_list = vals_dict[pix]
                if not pix in early_dict:
                    continue
                early_all_years = early_dict[pix]
                peak_all_years = peak_dict[pix]
                late_all_years = late_dict[pix]

                early_vals_list = []
                peak_vals_list = []
                late_vals_list = []
                for i,vals in enumerate(vals_list):
                    if i >= len(early_all_years):
                        # continue
                        early_range = self.longterm_phenology(early_all_years)
                        peak_range = self.longterm_phenology(peak_all_years)
                        late_range = self.longterm_phenology(late_all_years)
                    else:
                        early_range = early_all_years[i]
                        peak_range = peak_all_years[i]
                        late_range = late_all_years[i]
                    early = list(range(early_range[0],early_range[1]))
                    peak = list(range(peak_range[0],peak_range[1]))
                    late = list(range(late_range[0],late_range[1]))
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
            # exit()

        pass

    def calculate_anomaly(self):
        fdir = join(self.this_class_arr,'pick_daily_seasonal')
        for var_i in T.listdir(fdir):
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
                    std = np.nanstd(vals)
                    anomaly = vals-mean
                    anomaly_dict[pix] = anomaly
                T.save_distributed_perpix_dic(anomaly_dict,outdir_i_i)

    def calculate_std_anomaly(self):
        fdir = join(self.this_class_arr,'pick_daily_seasonal')
        for var_i in T.listdir(fdir):
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
                # exit()


    def check_seasonal_values(self):
        fdir = join(self.this_class_arr,'pick_daily_seasonal','CCI_SM','early')
        vals_dict = T.load_npy_dir(fdir)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(vals_dict)
        plt.imshow(arr)
        plt.show()


def main():
    # Dataframe().run()
    # Dataframe_One_pix().run()
    # RF().run()
    # Carryover_calculation().run()
    # Carryover_analysis().run()
    Variables_analysis().run()
    # Detrend_variables().run()
    # Pick_detrended_seasonal_variables().run()
    # Pick_detrended_seasonal_variables_dynamic().run()
    pass

if __name__ == '__main__':
    main()