# coding=utf-8
import tarfile

import Main_flow_2
import semopy
import lytools
from __init__ import *
land_tif = '/Volumes/NVME2T/greening_project_redo/conf/land.tif'
result_root_this_script = '/Users/liyang/Desktop/detrend_zscore_test_factors/results'


class Earlier_positive_anomaly:

    def __init__(self):
        self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/X'
        pass

    def run(self):
        x_dir = join(self.datadir,'corresponding_factors')
        y_f = join(self.datadir,'detrend_during_early_peak_MODIS_LAI_zscore.npy')
        # outdir = join(self.datadir,'earlier_positive_anomaly')
        # T.mkdir(outdir)
        dict_y = T.load_npy(y_f)
        condition_list = [-1,1]
        for x_var in T.listdir(x_dir):
            arr_list = []
            for condition in condition_list:
                dict_x = T.load_npy(join(x_dir,x_var))
                spatial_dict = {}
                for pix in dict_y:
                    if not pix in dict_x:
                        continue
                    y = dict_y[pix] # earlier greenness
                    x = dict_x[pix]
                    y = np.array(y)
                    x = np.array(x)
                    if condition == -1:
                        y_gt_0 = y < -0.5
                    else:
                        y_gt_0 = y > 0.5
                    x_selected = x[y_gt_0]
                    mean = np.nanmean(x_selected)
                    spatial_dict[pix] = mean
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                arr_list.append(arr)
            arr1 = arr_list[0]
            arr2 = arr_list[1]
            arr_sum = arr2 + arr1
            plt.imshow(arr_sum, cmap='jet',vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(x_var+'_'+'sum')

            plt.figure()
            plt.imshow(arr1, cmap='jet',vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(x_var+'_'+'browning')

            plt.figure()
            plt.imshow(arr2, cmap='jet',vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(x_var+'_'+'greening')

            arr1_flatten = arr1.flatten()
            arr2_flatten = arr2.flatten()
            plt.figure()
            plt.hist(arr1_flatten,bins=100,alpha=0.5,label='browning',density=True)
            plt.hist(arr2_flatten,bins=100,alpha=0.5,label='greening',density=True)
            plt.legend()
            plt.title(x_var+'_'+'hist')
            plt.show()

        pass


class Dataframe:
    def __init__(self):
        # self.datadir = Earlier_positive_anomaly().datadir
        # self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/data_moving_window'
        self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_moving_window'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Dataframe_detrend',
            result_root_this_script)
        # self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
        #     'Dataframe_with_trend',
        #     result_root_this_script)
        T.mkdir(self.this_class_arr)
        self.dff = join(self.this_class_arr, 'Dataframe.df')

    def run(self):
        variable_list = self.get_variable_list()
        df = self.__gen_df_init()
        df = self.add_variables(df)
        df = Main_flow_2.Dataframe_func(df).df
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def add_variables(self,df):
        fdir = self.datadir
        dict_all = {}
        for f in T.listdir(fdir):
            print(f)
            key = f.split('.')[0]
            dict_i = T.load_npy(join(fdir,f))
            dict_all[key] = dict_i
        df = T.spatial_dics_to_df(dict_all)
        return df

    def get_variable_list(self):
        fdir = self.datadir
        variable_list = []
        for f in T.listdir(fdir):
            key = f.split('.')[0]
            variable_list.append(key)
        # variable_list.append('HI_reclass')
        # variable_list.append('limited_area')
        # variable_list.append('landcover_GLC')
        return variable_list

    def add_tif(self,df):
        f = '/Users/liyang/Desktop/detrend_zscore_test_factors/during_late_MODIS_LAI_trend.tif'
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'late_trend')
        return df

class Dataframe_per_value:
    def __init__(self):
        # self.datadir = Earlier_positive_anomaly().datadir
        # self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/data_moving_window'
        self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_moving_window'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Dataframe_per_value',
            result_root_this_script)
        T.mkdir(self.this_class_arr)
        self.dff = join(self.this_class_arr, 'Dataframe.df')
        # self.dff = join(self.this_class_arr, 'Dataframe_with_trend.df')

    def run(self):
        variable_list = self.get_variable_list()
        start_year = 2000
        end_year = 2018
        # variable_list =
        df = self.__gen_df_init()
        df = self.add_variables(df)
        df = lytools.Dataframe_per_value(df,variable_list,start_year,end_year).df
        df = Main_flow_2.Dataframe_func(df).df
        df = self.add_tif(df)

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

        pass
    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def add_variables(self,df):
        fdir = self.datadir
        dict_all = {}
        for f in T.listdir(fdir):
            print(f)
            key = f.split('.')[0]
            dict_i = T.load_npy(join(fdir,f))
            dict_all[key] = dict_i
        df = T.spatial_dics_to_df(dict_all)
        return df

    def get_variable_list(self):
        fdir = self.datadir
        variable_list = []
        for f in T.listdir(fdir):
            key = f.split('.')[0]
            variable_list.append(key)
        # variable_list.append('HI_reclass')
        # variable_list.append('limited_area')
        # variable_list.append('landcover_GLC')
        return variable_list

    def add_tif(self,df):
        f = '/Users/liyang/Desktop/detrend_zscore_test_factors/during_late_MODIS_LAI_trend.tif'
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'late_trend')
        return df

class SEM:

    def __init__(self):
        '''
        This class is used to calculate the structural equation model
        '''
        pass


    def run(self):
        self.per_value_SEM()
        pass


    def per_value_SEM(self):
        dff = Dataframe_per_value().dff
        dff_1 = Dataframe().dff
        df = T.load_df(dff)
        df1 = T.load_df(dff_1)
        late_lai_spatial_dict = T.df_to_spatial_dic(df1,'during_late_MODIS_LAI_zscore')
        # late_lai_spatial_dict = T.df_to_spatial_dic(df1,'during_early_peak_MODIS_LAI_zscore')
        browning_pix_list = []
        for pix in late_lai_spatial_dict:
            vals = late_lai_spatial_dict[pix]
            if type(vals) == float:
                continue
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            # if p > 0.05:
                # continue
            # if a > 0:
            #     continue
            browning_pix_list.append(pix)
        browning_pix_list = set(browning_pix_list)
        T.print_head_n(df, 5)
        cols = df.columns
        for c in cols:
            print(c)
        result_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['year']
            if pix not in browning_pix_list:
                continue
            key = (pix,year)
            result_i = row.to_dict()
            result_dict[key] = result_i
        df_new = T.dic_to_df(result_dict,'pix_year')
        df_new = Main_flow_2.Dataframe_func(df_new).df
        df_new = df_new[df_new['lat']>50]
        df_new = df_new[df_new['lon']>0]

        self.build_model(df_new)
        pass

    def trend_SEM(self):
        dff = Dataframe().dff
        # dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        T.print_head_n(df,5)
        cols = df.columns
        for c in cols:
            print(c)
        exit()
        x_list = self.variables()
        all_trend_dict = {}
        for x in x_list:
            spatial_dict = {}
            for i,row in tqdm(df.iterrows(),total = len(df),desc=x):
                vals = row[x]
                pix = row['pix']
                if type(vals) == float:
                    continue
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
                    if x == 'during_late_MODIS_LAI_zscore':
                        if a > 0:
                            continue
                    # if p > 0.05:
                    #     continue
                except:
                    continue
                spatial_dict[pix] = a
            all_trend_dict[x] = spatial_dict
        df_trend = T.spatial_dics_to_df(all_trend_dict)
        df_trend = df_trend.dropna(subset=['during_late_MODIS_LAI_zscore'])
        # df_trend = df_trend.dropna()
        # T.print_head_n(df_trend,5)
        # exit()
        df_trend = Main_flow_2.Dataframe_func(df_trend).df
        self.build_model(df_trend)
        # self.plot_scatter(df)

    def variables(self):
        x_list = [
            'during_early_peak_MODIS_LAI_zscore',
            'during_early_Temp_zscore',
            'during_early_SPI3',
            'during_peak_SPEI3_zscore',
            'during_late_Temp_zscore',
            'during_late_MODIS_LAI_zscore',
            'during_peak_CCI_SM_zscore',
            'during_peak_Temp_zscore',
            'during_peak_SPI3',
            'EOS_zscore',
            'during_late_SPEI3_zscore',
            'during_peak_VPD_zscore',
        ]
        return x_list

    def model_description_detrend(self,ltd_region):
        desc_water_limited = '''
                # regressions
                detrend_during_early_peak_MODIS_LAI_zscore ~ detrend_during_early_Temp_zscore + detrend_during_early_SPI3
                detrend_during_peak_CCI_SM_zscore ~ detrend_during_early_SPI3 + detrend_during_early_Temp_zscore + detrend_during_early_peak_MODIS_LAI_zscore
                detrend_during_late_MODIS_LAI_zscore ~ detrend_during_peak_CCI_SM_zscore + detrend_during_early_peak_MODIS_LAI_zscore
                # residual correlations
                '''
        desc_energy_limited = '''
                # regressions
                detrend_during_early_peak_MODIS_LAI_zscore ~ detrend_during_early_Temp_zscore + detrend_during_early_SPI3
                detrend_during_peak_SPEI3 ~ detrend_during_early_SPI3 + detrend_during_early_Temp_zscore + detrend_during_early_peak_MODIS_LAI_zscore
                detrend_during_late_MODIS_LAI_zscore ~ detrend_during_peak_SPEI3 + detrend_during_early_peak_MODIS_LAI_zscore + detrend_during_late_Temp_zscore
                # residual correlations
                '''
        if ltd_region == 'water-limited':
            return desc_water_limited
        if ltd_region == 'energy-limited':
            return desc_energy_limited

        return None

    def model_description_not_detrend(self,ltd_region):
        desc_water_limited = '''
                # regressions
                during_early_peak_MODIS_LAI_zscore ~ during_early_Temp_zscore + during_early_SPI3
                during_peak_CCI_SM_zscore ~ during_early_SPI3 + during_early_Temp_zscore + during_early_peak_MODIS_LAI_zscore
                during_late_MODIS_LAI_zscore ~ during_peak_CCI_SM_zscore + during_early_peak_MODIS_LAI_zscore
                # residual correlations
                '''
        desc_energy_limited = '''
                # regressions
                during_peak_SPEI3_zscore ~ during_early_peak_MODIS_LAI_zscore
                during_late_MODIS_LAI_zscore ~ during_early_peak_MODIS_LAI_zscore + during_peak_SPEI3_zscore + during_late_Temp_zscore
                during_late_Temp_zscore ~ during_peak_SPEI3_zscore
                # residual correlations
                '''
        if ltd_region == 'water-limited':
            return desc_water_limited
        if ltd_region == 'energy-limited':
            return desc_energy_limited

        return None

    def build_model(self,df):
        '''
        :param df: a dataframe
        :return: a SEM model and output a report
        '''
        # df = df[df['late_trend']<0]

        # T.print_head_n(df)
        # exit()
        limited_area = T.get_df_unique_val_list(df,'limited_area')
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # print(limited_area)
        # exit()
        print(limited_area)
        for lc in lc_list:
            df_lc = df[df['landcover_GLC']==lc]
            for ltd in limited_area:
                desc = self.model_description_not_detrend(ltd)
                if desc == None:
                    continue
                # df_ltd = df_lc[df_lc['limited_area']==ltd]
                df_ltd = df_lc[df_lc['limited_area']==ltd]
                # DIC_and_TIF().plot_df_spatial_pix(df_ltd, land_tif)
                # plt.title(f'{ltd} {lc}')
                # plt.show()
                # DIC_and_TIF().plot_df_spatial_pix(df_ltd, land_tif)
                # plt.title(ltd)
                # plt.show()
                # DIC_and_TIF().plot_df_spatial_pix(df_ltd,land_tif)
                # plt.title(ltd)
                # plt.show()
                mod = semopy.Model(desc)
                res = mod.fit(df_ltd)
                # semopy.report(mod, f'SEM_result/{ltd}-{lc}')
                semopy.report(mod, f'SEM_result/{ltd}-{lc}')

    def plot_scatter(self,df):
        x = df['during_late_CO2_zscore'].values
        y = df['during_late_Temp_zscore'].values
        # KDE_plot().plot_scatter(x,y)
        # plt.xlabel('CO2')
        # plt.ylabel('Temp')
        # plt.show()
        # plt.hist(y,bins=200)
        # plt.show()

class Dataframe_moving_window:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe_moving_window',
                                                                                       result_root_this_script)
        self.dff = join(self.this_class_arr, 'data_frame_with_trend.df')
        # self.dff = join(self.this_class_arr, 'data_frame_detrend.df')
        self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/data_moving_window'
        # self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_moving_window'
        self.n = 10
        pass

    def run(self):
        valid_pix = self.valid_pix()
        df = self.add_variables()
        df = self.gen_moving_window_df()

    def gen_window(self,start_year,end_year):
        n = self.n
        year_list = list(range(start_year,end_year+1))
        window_list = []
        for i in range(len(year_list)-n+1):
            window_list.append(year_list[i:i+n])
        return window_list

    def valid_pix(self):
        outf = join(self.this_class_arr, 'valid_pix.npy')
        if isfile(outf):
            valid_pix = np.load(outf, allow_pickle=True)
            return valid_pix
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(land_tif)
        dict_all = {'1': spatial_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = Main_flow_2.Dataframe_func(df).df
        pix_list = df['pix'].values.tolist()
        pix_list = np.array(pix_list)
        np.save(join(self.this_class_arr, 'valid_pix.npy'), pix_list)
        return pix_list

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df

    def gen_var_fpath_dict(self):
        fdir = self.datadir
        variables_fpath_dict = {}
        for f in T.listdir(fdir):
            var_name = f.split('.')[0]
            fpath = join(fdir, f)
            variables_fpath_dict[var_name] = fpath
        return variables_fpath_dict

    def add_variables(self):
        var_fpath_dict = self.gen_var_fpath_dict()
        # print(var_fpath_dict)
        # exit()
        valid_pix = self.valid_pix()
        dict_all = {}
        for var_name in var_fpath_dict:
            fpath = var_fpath_dict[var_name]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_new = {}
            for pix in tqdm(spatial_dict,desc=var_name):
                if not pix in valid_pix:
                    continue
                vals = spatial_dict[pix]
                if len(vals) == 0:
                    continue
                spatial_dict_new[pix] = vals
            dict_all[var_name] = spatial_dict_new
        df = T.spatial_dics_to_df(dict_all)
        df = Main_flow_2.Dataframe_func(df).df
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def gen_moving_window_df(self):
        start_year = 2000
        end_year = 2018
        outdir = join(self.this_class_arr, 'moving_window_with_trend')
        T.mkdir(outdir,force=True)
        df = T.load_df(self.dff)
        variables_dict = self.gen_var_fpath_dict()
        variables_list = []
        for v in variables_dict:
            variables_list.append(v)
        window_list = self.gen_window(start_year,end_year)
        for w in range(len(window_list)):
            window = window_list[w]
            window = np.array(window)
            window = window - start_year
            window_str = f'{w:02d}'
            outdir_i = join(outdir,window_str)
            T.mkdir(outdir_i,force=True)
            outf = join(outdir_i,f'{window_str}.df')
            # if isfile(outf):
            #     continue
            dict_all = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=window_str):
                pix = row['pix']
                dict_i = {}
                for x in variables_list:
                    vals = row[x]
                    if type(vals) == float:
                        continue
                    if len(vals) != end_year - start_year + 1:
                        continue
                    if T.is_all_nan(vals):
                        continue
                    vals = T.pick_vals_from_1darray(vals,window)
                    dict_i[x] = vals
                    dict_i['limited_area'] = row['limited_area']
                    dict_i['HI_reclass'] = row['HI_reclass']
                dict_all[pix] = dict_i
            df_out = T.dic_to_df(dict_all,key_col_str='pix')
            T.save_df(df_out,outf)
            T.df_to_excel(df_out,outf)

class Moving_window_single_correlation:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Moving_window_single_correlation',
                                                                                       result_root_this_script)
        self.datadir = '/Users/liyang/Desktop/detrend_zscore_test_factors'

    def run(self):
        # self.run_corr()
        # self.plot_corr()
        # self.moving_window_trend()
        # self.over_all_corr()
        self.plot_bivariate_map()
        pass

    def run_corr(self):
        fdir = join(Dataframe_moving_window().this_class_arr,'moving_window_with_trend')
        outdir_father = join(self.this_class_arr,'corr_with_trend')
        is_detrend = False
        x_var_list = self.x_variables(is_detrend)
        y_var = self.y_variable(is_detrend)
        T.mkdir(outdir_father,force=True)
        for window in T.listdir(fdir):
            fpath = join(fdir,window,f'{window}.df')
            outdir_i = join(outdir_father,window)
            T.mkdir(outdir_i)
            outf_p_corr = join(outdir_i,f'{window}_corr.df')
            outf_p_corr_p_value = join(outdir_i,f'{window}_corr_p_value.df')
            df_i = T.load_df(fpath)
            cols = df_i.columns.tolist()
            for c in cols:
                print(c)
            exit()
            corr_result = {}
            corr_result_p_value = {}
            for i,row in tqdm(df_i.iterrows(),total=len(df_i)):
                # print(row)
                row_dic = row.to_dict()
                pix = row['pix']
                corr_result_i = {}
                corr_result_p_value_i = {}
                for x in x_var_list:
                    x_vals = row[x]
                    y_vals = row[y_var]
                    if type(x_vals) == float or type(y_vals) == float:
                        continue
                    r,p = T.nan_correlation(x_vals,y_vals)
                    corr_result_i[x] = r
                    corr_result_i['limited_area'] = row_dic['limited_area']
                    corr_result_i['HI_reclass'] = row_dic['HI_reclass']
                    corr_result_p_value_i[x] = p
                    corr_result_p_value_i['limited_area'] = row_dic['limited_area']
                    corr_result_p_value_i['HI_reclass'] = row_dic['HI_reclass']
                corr_result[pix] = corr_result_i
                corr_result_p_value[pix] = corr_result_p_value_i
            df_p_corr = T.dic_to_df(corr_result, 'pix')
            df_p_corr_p_value = T.dic_to_df(corr_result_p_value, 'pix')
            T.save_df(df_p_corr, outf_p_corr)
            T.save_df(df_p_corr_p_value, outf_p_corr_p_value)
            T.df_to_excel(df_p_corr, join(outdir_i, f'{window}_corr.xlsx'))
            T.df_to_excel(df_p_corr_p_value, join(outdir_i, f'{window}_corr_p_value.xlsx'))

    def y_variable(self,is_detrend):
        # return 'during_early_peak_late_MODIS_LAI_zscore'
        # return 'detrend_during_early_peak_MODIS_LAI_zscore'
        if is_detrend:
            # return 'detrend_during_early_peak_MODIS_LAI_zscore'
            return 'detrend_during_late_MODIS_LAI_zscore'
        else:
            # return 'during_early_peak_MODIS_LAI_zscore'
            return 'during_late_MODIS_LAI_zscore'

    def x_variables(self,is_detrend):
        if is_detrend:
            x_variables = [
                'detrend_during_early_peak_late_CCI_SM_zscore',
                'detrend_during_peak_CCI_SM_zscore',
                'detrend_during_early_peak_late_CO2_zscore',
                # 'detrend_during_late_CCI_SM_zscore',
                # '----',
            ]
        else:
            x_variables = [
                'during_early_peak_late_CCI_SM_zscore',
                'during_peak_CCI_SM_zscore',
                'during_early_peak_late_CO2_zscore',
                # 'during_late_CCI_SM_zscore',
                # '----',
            ]
        return x_variables

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
        for x in x_var_list:
            # print(x)
            x_var_list_valid_new_cov=copy.copy(x_var_list)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value


    def plot_corr(self):
        is_detrend = False
        outdir = join(self.this_class_png,'corr_with_trend')
        # outdir = join(self.this_class_png,'p_corr_detrend')
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        fdir = join(self.this_class_arr,'corr_with_trend')
        # fdir = join(self.this_class_arr,'p_corr_detrend')
        # ltd_list = ('Non-sig', 'energy-limited', 'water-limited')
        # ltd_list = ('energy-limited', 'water-limited')
        ltd_list = ('Dryland', 'Humid')
        for ltd in ltd_list:
            x_list = self.x_variables(is_detrend)
            for x in x_list:
                window_number = len(T.listdir(fdir))
                color_list = T.gen_colors(window_number)
                plt.figure(figsize=(4, 4))
                for window in T.listdir(fdir):
                    fpath = join(fdir,window,f'{window}_corr.df')
                    df = T.load_df(fpath)

                    # df = Main_flow_2.Dataframe_func(df).df
                    # df = df[df['limited_area']==ltd]
                    df = df[df['HI_reclass']==ltd]
                    # limited_area_list = T.get_df_unique_val_list(df, 'HI_reclass')
                    # print(limited_area_list)
                    # exit()
                    # T.print_head_n(df)
                    # exit()
                    xvals = df[x].values
                    xx,yy = Plot().plot_hist_smooth(xvals,bins=80,alpha=0)
                    plt.plot(xx,yy,label=window,color=color_list[int(window)])
                plt.title(f'{ltd}\n{x}\n{self.y_variable(is_detrend)}')
                outf = join(outdir,f'{ltd}_{x}_legend.pdf')
                plt.legend()
                plt.tight_layout()
                plt.savefig(outf)
                plt.close()

    def moving_window_trend(self):
        is_detrend = False

        outdir = join(self.this_class_tif, 'moving_window_trend')
        T.mkdir(outdir, force=True)
        # T.open_path_and_file(outdir)
        if is_detrend:
            fdir = join(Dataframe_moving_window().this_class_arr, 'moving_window_detrend')
            outdir = join(self.this_class_tif, 'moving_window_trend_detrend')
        else:
            fdir = join(Dataframe_moving_window().this_class_arr, 'moving_window_with_trend')
            outdir = join(self.this_class_tif, 'moving_window_trend_with_trend')
        x_list = self.x_variables(is_detrend)
        for x in x_list:
            window_number = len(T.listdir(fdir))
            T.mkdir(outdir, force=True)
            arrs = []
            for window in T.listdir(fdir):
                fpath = join(fdir, window, f'{window}.df')
                df = T.load_df(fpath)
                spatial_dict_x = T.df_to_spatial_dic(df, x)
                spatial_dict_y = T.df_to_spatial_dic(df, self.y_variable(is_detrend))
                corr_dict = {}
                for pix in tqdm(spatial_dict_y,desc=f'{x}-{window}'):
                    if not pix in spatial_dict_x:
                        continue
                    x_vals = spatial_dict_x[pix]
                    y_vals = spatial_dict_y[pix]
                    if type(x_vals) == float:
                        continue
                    if type(y_vals) == float:
                        continue
                    try:
                        a,b,r,p = T.nan_line_fit(x_vals,y_vals)
                        corr_dict[pix] = r
                    except:
                        continue
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(corr_dict)
                arrs.append(arr)

            trend_spatial_dict = {}
            trend_p_spatial_dict = {}
            for r in range(len(arrs[0])):
                for c in range(len(arrs[0][0])):
                    ts = []
                    pix = (r,c)
                    for i in range(len(arrs)):
                        val = arrs[i][r][c]
                        ts.append(val)
                    if T.is_all_nan(ts):
                        continue
                    a,b,_,p = T.nan_line_fit(list(range(len(ts))),ts)
                    trend_spatial_dict[pix] = a
                    trend_p_spatial_dict[pix] = p
            outf_trend = join(outdir, f'{x}.tif')
            outf_p = join(outdir, f'{x}_p.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_spatial_dict, outf_trend)
            DIC_and_TIF().pix_dic_to_tif(trend_p_spatial_dict, outf_p)

    def over_all_corr(self):
        is_detrend = False
        if is_detrend:
            datadir = join(self.datadir,'data_for_SEM_detrend/data_moving_window')
            fdir = join(Dataframe_moving_window().this_class_arr, 'moving_window_detrend')
            outdir = join(self.this_class_tif, 'over_all_corr_detrend')
            y_variable = self.y_variable(is_detrend) + '.npy'
        else:
            datadir = join(self.datadir, 'data_for_SEM_with_trend/data_moving_window')
            fdir = join(Dataframe_moving_window().this_class_arr, 'moving_window_with_trend')
            outdir = join(self.this_class_tif, 'over_all_corr_with_trend')
            y_variable = self.y_variable(is_detrend) + '.npy'

        spatial_dict_y_f = join(datadir,y_variable)

        T.mkdir(outdir, force=True)
        spatial_dict_y = T.load_npy(spatial_dict_y_f)
        x_var_list = self.x_variables(is_detrend)
        for x in x_var_list:
            outf = join(outdir, f'{x}.tif')
            spatial_dict_f = join(datadir, f'{x}.npy')
            spatial_dict = T.load_npy(spatial_dict_f)
            corr_dict = {}
            for pix in tqdm(spatial_dict,desc=f'{x}'):
                x_vals = spatial_dict[pix]
                if not pix in spatial_dict_y:
                    continue
                y_vals = spatial_dict_y[pix]
                if len(y_vals) == 0:
                    continue
                if len(x_vals) == 0:
                    continue
                a, b, r, p = T.nan_line_fit(x_vals, y_vals)
                corr_dict[pix] = r
            DIC_and_TIF().pix_dic_to_tif(corr_dict, outf)

    def plot_bivariate_map(self):
        over_all_corr_dir = join(self.this_class_tif, 'over_all_corr_with_trend')
        moving_window_trend_dir = join(self.this_class_tif, 'moving_window_trend_with_trend')
        outdir = join(self.this_class_tif, 'plot_bivariate_map_with_trend')
        T.mkdir(outdir, force=True)

        for f in T.listdir(over_all_corr_dir):
            if not f.endswith('.tif'):
                continue
            outf = join(outdir, f)
            overall_corr_f = join(over_all_corr_dir, f)
            moving_window_trend_f = join(moving_window_trend_dir, f)
            x_label = 'Overall Correlation'
            y_label = 'Moving Window Trend'
            min1,max1 = -0.5,0.5
            min2,max2 = -0.05,0.05

            XYmap().plot_bivariate_map(overall_corr_f, moving_window_trend_f,x_label,y_label,min1,max1,min2,max2,outf)


class Single_corr:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Single_corr_detrend',
            result_root_this_script)
        pass

    def run(self):
        self.to_tif()
        # self.to_df()
        # self.cal_trend()
        # self.statistic()

    def to_df(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        x_list = self.x_list_detrend()
        y_var = self.y_var_detrend()
        outdir = join(self.this_class_arr, 'single_corr_df')
        T.mk_dir(outdir, force=True)
        y_dict = T.df_to_spatial_dic(df, y_var)
        result_dict = {}
        for x_var in x_list:
            spatial_dict = {}
            x_dict = T.df_to_spatial_dic(df, x_var)
            for pix in x_dict:
                if not pix in y_dict:
                    continue
                x_val = x_dict[pix]
                y_val = y_dict[pix]
                try:
                    r, p = T.nan_correlation(x_val, y_val)
                except:
                    continue
                spatial_dict[pix] = r
            result_dict[x_var] = spatial_dict
        df_out = T.spatial_dics_to_df(result_dict)
        outf = join(outdir, f'{y_var}.df')
        T.save_df(df_out, outf)
        T.df_to_excel(df_out,outf)

    def to_tif(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df,5)
        x_list = self.x_list_detrend()
        y_var = self.y_var_detrend()
        outdir = join(self.this_class_tif,'single_corr',y_var)
        T.mk_dir(outdir,force=True)
        y_dict = T.df_to_spatial_dic(df, y_var)
        for x_var in x_list:
            if not 'CO2' in x_var:
                continue
            print(x_var)
            spatial_dict = {}
            x_dict = T.df_to_spatial_dic(df,x_var)
            for pix in x_dict:
                if not pix in y_dict:
                    continue
                x_val = x_dict[pix]
                y_val = y_dict[pix]
                try:
                    r,p = T.nan_correlation(x_val,y_val)
                    if r > 0.6:
                        # plt.scatter(x_val,y_val)
                        plt.plot(x_val,c='r')
                        plt.twinx()
                        plt.plot(y_val,c='g')
                        plt.show()
                except:
                    continue
                spatial_dict[pix] = r
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
            # outf = join(outdir,x_var+'.tif')
            # DIC_and_TIF().arr_to_tif(arr,outf)
        pass

    def x_list_with_trend(self):
        x_list_ = [
            'during_peak_SPEI3_zscore',
            'during_early_peak_late_CO2_zscore',
            'during_late_PAR_zscore',
            'during_late_Temp_zscore',
            'during_late_VPD_zscore',
        ]
        return x_list_

    def x_list_detrend(self):
        x_list_ = [
            'detrend_during_peak_SPEI3_zscore',
            'detrend_during_early_peak_late_CO2_zscore',
            'detrend_during_late_PAR_zscore',
            'detrend_during_late_Temp_zscore',
            'detrend_during_late_VPD_zscore',
        ]
        return x_list_

    def y_var(self):
        return 'during_late_MODIS_LAI_zscore'
        # return 'EOS_zscore'

    def y_var_detrend(self):
        return 'detrend_during_late_MODIS_LAI_zscore'
        # return 'EOS_zscore'

    def cal_trend(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        # x_list = self.y_var()
        x_list = ['during_early_peak_MODIS_LAI_zscore','during_late_MODIS_LAI_zscore']
        outdir = join(self.this_class_arr, 'trend_df')
        T.mk_dir(outdir, force=True)
        result_dict = {}
        for x_var in x_list:
            spatial_dict_a = {}
            spatial_dict_p = {}
            x_dict = T.df_to_spatial_dic(df, x_var)
            for pix in x_dict:
                x_val = x_dict[pix]
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(x_val))),x_val)
                except:
                    continue
                spatial_dict_a[pix] = a
                spatial_dict_p[pix] = p
            result_dict[x_var] = spatial_dict_a
            result_dict[x_var+'_p'] = spatial_dict_p
        df_out = T.spatial_dics_to_df(result_dict)
        # DIC_and_TIF().plot_df_spatial_pix(df_out,land_tif)
        # plt.show()
        outf = join(outdir, f'trend.df')
        T.save_df(df_out, outf)
        T.df_to_excel(df_out, outf)

        pass


    def statistic(self):
        fdir = join(self.this_class_arr,'single_corr_df')
        dff = join(fdir,f'{self.y_var()}.df')

        # condition1 earlier greening
        trend_dff = join(self.this_class_arr, 'trend_df', 'trend.df')
        trend_df = T.load_df(trend_dff)
        # trend_df = trend_df[trend_df[f'during_early_peak_MODIS_LAI_zscore_p'] < 0.05]
        trend_df = trend_df[trend_df['during_early_peak_MODIS_LAI_zscore'] > 0]
        selected_pix1 = trend_df['pix'].tolist()
        selected_pix1 = set(selected_pix1)

        # condition2 late browning / greening
        # trend_df = trend_df[trend_df[f'during_late_MODIS_LAI_zscore_p'] < 0.05]
        trend_df = trend_df[trend_df['during_late_MODIS_LAI_zscore'] < 0]
        # trend_df = trend_df[trend_df['during_late_MODIS_LAI_zscore'] > 0]
        selected_pix2 = trend_df['pix'].tolist()
        selected_pix2 = set(selected_pix2)
        # selected_pix2 = selected_pix1


        intersect = selected_pix1.intersection(selected_pix2)
        intersect = set(intersect)
        print(len(intersect))
        # exit()

        df = T.load_df(dff)
        df = Main_flow_2.Dataframe_func(df).df
        x_list = self.x_list()
        HI_reclass_list = T.get_df_unique_val_list(df,'HI_reclass')
        print(HI_reclass_list)
        val_list = []
        err_list = []
        x_list_plot = []
        color_list = []
        for HI_reclass in HI_reclass_list:
            df_HI = df[df['HI_reclass']==HI_reclass]
            print(len(df_HI))
            df_HI = df_HI[df_HI['pix'].isin(intersect)]
            # print(len(df_HI))
            # exit()
            for x in x_list:
                x_tick = f'{x}_{HI_reclass}'
                vals = df_HI[x].values
                vals_mean = np.nanmean(vals)
                val_list.append(vals_mean)
                err = np.nanstd(vals)
                err_list.append(err)
                x_list_plot.append(x_tick)
                if HI_reclass == 'Dryland':
                    color_list.append('red')
                elif HI_reclass == 'Humid':
                    color_list.append('blue')
        plt.barh(x_list_plot,val_list,xerr=err_list,color=color_list)
        # plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()


class Bivarite_plot:

    def __init__(self):

        pass

    def run(self):
        self.plot()
        pass

    def plot(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        earlier = 'during_early_peak_MODIS_LAI_zscore'
        late = 'during_late_MODIS_LAI_zscore'
        spatial_dict = {}
        val_dict = {
            '1-1':2,
            '1-0':1,
            '0-1':-1,
            '0-0':-2,
        }
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            x = row[earlier]
            y = row[late]
            if type(x) == float:
                continue
            if type(y) == float:
                continue
            trend_x,_,_,_ = T.nan_line_fit(list(range(len(x))),x)
            trend_y,_,_,_ = T.nan_line_fit(list(range(len(y))),y)
            if trend_x > 0 and trend_y > 0:
                val = '1-1'
            elif trend_x > 0 and trend_y < 0:
                val = '1-0'
            elif trend_x < 0 and trend_y > 0:
                val = '0-1'
            elif trend_x < 0 and trend_y < 0:
                val = '0-0'
            else:
                continue
            val_ = val_dict[val]
            spatial_dict[pix] = val_
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='jet')
        plt.colorbar()
        plt.show()


class Scatter_plot:

    def __init__(self):

        pass

    def run(self):
        dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        cols = df.columns
        for c in cols:
            print(c)
        limited_area = 'water-limited'
        # limited_area = 'energy-limited'
        df = df[df['limited_area']==limited_area]
        print(len(df))
        # during_early_peak_CCI_SM_zscore
        # during_late_CCI_SM_zscore
        # during_late_MODIS_LAI_zscore
        # during_early_peak_MODIS_LAI_zscore
        # cci =
        # T.df_bin()
        # early_sm_var = 'during_early_peak_CCI_SM_zscore'
        # early_sm_var = 'detrend_during_peak_CCI_SM_zscore'
        early_sm_var = 'during_early_peak_SPEI3_zscore'
        # early_sm_var = 'detrend_during_early_peak_SPI3'
        # early_sm_var = 'detrend_during_late_VPD_zscore'
        # early_sm_var = 'during_late_VPD_zscore'
        # early_sm_var = 'during_peak_VPD_zscore'
        # early_sm_var = 'detrend_during_early_peak_SPEI3'
        late_lai_var = 'during_late_MODIS_LAI_zscore'
        early_lai_var = 'during_early_peak_MODIS_LAI_zscore'

        early_sm = df[early_sm_var].values.tolist()
        late_lai = df[late_lai_var].values.tolist()
        earlier_lai = df[early_lai_var].values.tolist()
        max_bin = 0
        min_bin = -2
        n = 7
        percentile_list = np.linspace(0,1,n)
        early_sm = T.remove_np_nan(early_sm)
        late_lai = T.remove_np_nan(late_lai)
        earlier_lai = T.remove_np_nan(earlier_lai)
        earlier_lai = np.array(earlier_lai)
        earlier_lai = earlier_lai[earlier_lai>0]
        earlier_lai = earlier_lai[earlier_lai<1.]
        # sm_bins = np.percentile(early_sm,percentile_list*100)
        # late_lai_bins = np.percentile(late_lai,percentile_list*100)
        # earlier_lai_bins = np.percentile(earlier_lai,percentile_list*100)
        sm_bins = np.linspace(min_bin,max_bin,n)
        # late_lai_bins = np.linspace(min_bin,max_bin,n)
        earlier_lai_bins = np.linspace(0,1,n)
        df_group_sm,bins_list_str_sm = T.df_bin(df,early_sm_var,sm_bins)
        matrix = []
        x_ticks = []
        y_ticks = []
        for name_sm,df_group_i_sm in df_group_sm:
            # print(name_sm)
            df_group_earlier_lai,bins_list_str_earlier_lai = T.df_bin(df_group_i_sm,early_lai_var,earlier_lai_bins)
            y_ticks.append(name_sm.left)
            temp = []
            x_ticks = []
            for name_earlier_lai,df_group_i_earlier_lai in df_group_earlier_lai:
                x_ticks.append(name_earlier_lai.left)
                vals = df_group_i_earlier_lai[late_lai_var]
                vals_mean = np.nanmean(vals)
                temp.append(vals_mean)
            temp = np.array(temp)
            matrix.append(temp)
        matrix = np.array(matrix)
        plt.imshow(matrix,cmap='RdBu',vmin=-0.4,vmax=0.4)
        plt.xticks(range(len(x_ticks)),x_ticks,rotation=90)
        plt.yticks(range(len(y_ticks)),y_ticks)
        plt.xlabel(early_lai_var)
        plt.ylabel(early_sm_var)
        plt.colorbar()
        plt.title(limited_area)
        plt.tight_layout()
        plt.show()


class RF_per_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF_per_value',
                                                                                       result_root_this_script)
        pass

    def run(self):

        self.run_RF()
        self.plot_RF_results()
        # self.run_pdp()
        # self.plot_pdp()
        # self.run_RF(is_detrend=False)
        # self.plot_RF_results(is_detrend=True)
        # self.plot_RF_results(is_detrend=False)

    def run_RF(self):
        outdir = join(self.this_class_arr,'permutation_importance')
        T.mkdir(outdir)
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        df = df[df['during_early_peak_MODIS_LAI_zscore']>0]
        T.print_head_n(df)
        cols = df.columns
        for c in cols:
            print(c)
        limited_area = T.get_df_unique_val_list(df,'limited_area')
        limited_area = ['energy-limited', 'water-limited']
        for ltd in limited_area:
            print(ltd)
            df_ltd = df[df['limited_area']==ltd]
            X = df_ltd[x_var_list]
            Y = df_ltd[y_var]
            result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var)
            outf = join(outdir,f'{ltd}')
            T.save_npy(result,outf)


    def plot_RF_results(self):
        y = self.y_variable()
        fdir = join(self.this_class_arr,'permutation_importance')
        outdir = join(self.this_class_png,f'importance')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        x_list = self.x_variables()
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result = T.load_npy(fpath)
            print(result)
            # exit()
            importances_mean = result['importances_mean']
            # err = result['importances_std']
            plt.figure(figsize=(4, 4))
            plt.barh(x_list,importances_mean)
            plt.title(f)
            # plt.xlim(0,0.4)
            plt.tight_layout()
            # plt.show()
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf)
            plt.close()
        # exit()
        pass

    def y_variable(self):
        return 'during_late_MODIS_LAI_zscore'

    def x_variables(self):
        x_list = [
            'during_early_peak_SPEI3_zscore',
            'during_late_Temp_zscore',
            'during_late_CO2_zscore',
            'during_late_PAR_zscore',
            'during_late_VPD_zscore',
            'during_early_peak_SPI3',
            'during_early_peak_MODIS_LAI_zscore',
        ]
        return x_list

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
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})


    def train_classfication_permutation_importance(self,X_input,Y_input,x_list,y):
        # X = X.loc[:, ~X.columns.duplicated()]
        # outfir_fig_dir=self.this_class_arr+'train_classfication_fig_driver/'  #修改
        # Tools().mk_dir(outfir_fig_dir)
        print('training...')
        rf = RandomForestRegressor(n_jobs=6,n_estimators=100,)
        X_input=X_input[x_list]
        X_input=pd.DataFrame(X_input)
        df_new = pd.concat([X_input,Y_input], axis=1)
        df_new = df_new.dropna()
        X = df_new[x_list]
        Y = df_new[y]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        rf.fit(X_train,Y_train)
        print('training finished')
        score = rf.score(X_test,Y_test)
        coef = rf.feature_importances_

        pred=rf.predict(X_test)
        r = stats.pearsonr(pred,Y_test)[0]

        print('permuation importance...')
        result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
                                        random_state=42)
        # result = {}
        # result['importances_mean'] = coef
        print('permuation importance finished')
        x_list_dict = dict(zip(x_list, list(range(len(x_list)))))
        importances_mean = result['importances_mean']
        imp_dict = dict(zip(x_list, importances_mean))
        max_key = T.pick_max_key_val_from_dict(imp_dict)
        max_V = x_list_dict[max_key]
        result['score'] = score
        result['r'] = r
        result['max_var'] = max_key

        return result

    def run_pdp(self):
        outdir = join(self.this_class_arr, 'pdp')
        T.mkdir(outdir)
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        df = df[df['during_early_peak_MODIS_LAI_zscore'] > 0]
        T.print_head_n(df)
        cols = df.columns
        for c in cols:
            print(c)
        limited_area = T.get_df_unique_val_list(df, 'limited_area')
        limited_area = ['energy-limited', 'water-limited']
        for ltd in limited_area:
            print(ltd)
            df_ltd = df[df['limited_area'] == ltd]
            X = df_ltd[x_var_list]
            Y = df_ltd[y_var]
            result = self.partial_dependence_plots(df_ltd,x_var_list, y_var)
            outf = join(outdir, f'{ltd}')
            T.save_npy(result, outf)

    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=4) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def plot_pdp(self):
        fdir = join(self.this_class_arr,'pdp')
        for f in T.listdir(fdir):
            result_dict = T.load_npy(join(fdir,f))
            for key in result_dict:
                print(key)
                vals = result_dict[key]
                x = vals['x']
                y = vals['y']
                y_std = vals['y_std']
                plt.figure()
                Plot().plot_line_with_error_bar(x,y,y_std)
                plt.plot(x,y,c='r',zorder=1)
                plt.title(f'{f}')
                plt.ylabel(self.y_variable())
                plt.xlabel(key)
                plt.tight_layout()
            plt.show()
        pass



def main():
    # Earlier_positive_anomaly().run()
    # Dataframe().run()
    # Dataframe_per_value().run()
    # SEM().run()
    # Dataframe_moving_window().run()
    # Moving_window_single_correlation().run()
    Single_corr().run()
    # Bivarite_plot().run()
    # Scatter_plot().run()
    # RF_per_value().run()
    pass


if __name__ == '__main__':
    main()