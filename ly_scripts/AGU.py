# coding=utf-8
import re
import xymap
import xycmap
import pymannkendall as mk
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
        # self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
        #     'Dataframe_detrend',
        #     result_root_this_script)
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Dataframe_with_trend',
            result_root_this_script)
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
        df = Dataframe_per_value_transform(df,variable_list,start_year,end_year).df
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
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        cols = df.columns
        for c in cols:
            print(c)
        # selected_pix = self.filter_df_with_browning_pix() # late browning
        selected_pix = self.filter_pix_with_flatten_strong_weak()
        # exit()
        result_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['year']
            if pix not in selected_pix:
                continue
            key = (pix,year)
            result_i = row.to_dict()
            result_dict[key] = result_i
        df_new = T.dic_to_df(result_dict,'pix_year')
        df_new = Main_flow_2.Dataframe_func(df_new).df
        self.build_model(df_new)
        pass

    def filter_pix_with_flatten_strong_weak(self):
        fdir = join(CarryoverInDryYear().this_class_tif,'flatten_plot_strong_weak')
        f = 'MODIS_LAI.tif'
        fpath = join(fdir,f)
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        selected_value = [8,9,12,13]
        selected_value = set(selected_value)
        selected_pix = []
        spatial_dict_1 = {}
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            if not val in selected_value:
                continue
            selected_pix.append(pix)
            spatial_dict_1[pix] = val
        selected_pix = set(selected_pix)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_1)
        # plt.imshow(arr)
        # DIC_and_TIF().plot_back_ground_arr(land_tif)
        # plt.show()
        return selected_pix

    def filter_df_with_browning_pix(self):
        dff_1 = Dataframe().dff
        df1 = T.load_df(dff_1)
        late_lai_spatial_dict = T.df_to_spatial_dic(df1, 'during_late_MODIS_LAI_zscore')
        browning_pix_list = []
        for pix in late_lai_spatial_dict:
            vals = late_lai_spatial_dict[pix]
            if type(vals) == float:
                continue
            a, b, r, p = T.nan_line_fit(list(range(len(vals))), vals)
            browning_pix_list.append(pix)
        browning_pix_list = set(browning_pix_list)
        return browning_pix_list

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
        # desc_energy_limited = '''
        #         # regressions
        #         during_late_CCI_SM_zscore ~ during_early_peak_MODIS_LAI_zscore
        #         during_late_MODIS_LAI_zscore ~ during_early_peak_MODIS_LAI_zscore + during_late_CCI_SM_zscore + during_late_VPD_zscore
        #         during_late_VPD_zscore ~ during_late_CCI_SM_zscore
        #         # residual correlations
        #         '''
        desc_energy_limited = '''
                # regressions
                during_peak_CCI_SM_zscore ~ during_early_peak_MODIS_LAI_zscore
                during_late_MODIS_LAI_zscore ~ during_early_peak_MODIS_LAI_zscore + during_peak_CCI_SM_zscore
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
        # for lc in lc_list:
        #     df_lc = df[df['landcover_GLC']==lc]
        for ltd in limited_area:
            desc = self.model_description_not_detrend(ltd)
            if desc == None:
                continue
            # df_ltd = df_lc[df_lc['limited_area']==ltd]
            df_ltd = df[df['limited_area']==ltd]
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
            semopy.report(mod, f'SEM_result/{ltd}')

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
        # self.plot_bivariate_map()
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


class Bivarite_plot_partial_corr:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Bivarite_plot_partial_corr',
            result_root_this_script)
        pass

    def run(self):
        # self.earlier_and_late_greening_trend_binary()
        # self.earlier_and_late_greening_trend_binary_with_p()
        # self.partial_corr_dict_to_tif()
        # self.partial_corr_window_trend()
        self.long_term_corr_dict_to_tif()
        self.bivarite_plot()
        pass


    def partial_corr_dict_to_tif(self):
        father_outdir = join(self.this_class_tif, 'partial_corr_tif')
        T.mk_dir(father_outdir, force=True)
        father_dir = '/Users/liyang/Desktop/detrend_zscore_test_factors/partial_window'
        moving_window_dir = join(father_dir,'moving_window')
        long_term_dir = join(father_dir,'long_term')
        window_list = ['05','10']
        is_detrend_list = ['detrend','with-trend']
        scenario_list = ['scenario1','scenario2']

        for detrend in is_detrend_list:
            for window in window_list:
                for scenario in scenario_list:
                    print(f'{detrend}_{window}_{scenario}')
                    outdir = join(father_outdir, f'{detrend}_{window}_{scenario}')
                    T.mk_dir(outdir, force=True)
                    window_dir = join(moving_window_dir,f'{detrend}_{window}_{scenario}')
                    if not isdir(window_dir):
                        continue
                    for f in T.listdir(window_dir):
                        if f.endswith('_p_value.npy'):
                            continue
                        p_list = re.findall(r'zscore_.*?_', f)[0]
                        w = p_list.split('_')[1]
                        fpath = join(window_dir,f)
                        dict_i = T.load_npy(fpath)
                        variables_list = {}
                        for pix in dict_i:
                            dict_j = dict_i[pix]
                            for var_i in dict_j:
                                if not var_i in variables_list:
                                    variables_list[var_i] = 1
                                else:
                                    continue
                        # print(variables_list)
                        for var_i in variables_list:
                            outdir_i = join(outdir,var_i)
                            T.mk_dir(outdir_i,force=True)
                            outf = join(outdir_i,f'{w}.tif')
                            spatial_dict = {}
                            for pix in dict_i:
                                dict_j = dict_i[pix]
                                if not var_i in dict_j:
                                    continue
                                val = dict_j[var_i]
                                spatial_dict[pix] = val
                            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def partial_corr_window_trend(self):
        father_fdir = join(self.this_class_tif,'partial_corr_tif')
        father_outdir = join(self.this_class_tif,'partial_corr_trend')
        T.mk_dir(father_outdir,force=True)
        for folder in T.listdir(father_fdir):
            fdir = join(father_fdir,folder)
            outdir = join(father_outdir,folder)
            for var_i in T.listdir(fdir):
                var_i_dir = join(fdir,var_i)
                print(f'{folder}-{var_i}')
                T.mk_dir(outdir,force=True)
                outf = join(outdir,f'{var_i}.tif')
                void_spatial_dict = DIC_and_TIF().void_spatial_dic()
                for f in T.listdir(var_i_dir):
                    w = f.split('.')[0]
                    fpath = join(var_i_dir,f)
                    spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
                    for pix in spatial_dict:
                        val = spatial_dict[pix]
                        void_spatial_dict[pix].append(val)
                trend_spatial_dict = {}
                for pix in void_spatial_dict:
                    vals = void_spatial_dict[pix]
                    vals = np.array(vals)
                    vals[vals<-9999] = np.nan
                    if T.is_all_nan(vals):
                        continue
                    a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
                    trend_spatial_dict[pix] = a
                DIC_and_TIF().pix_dic_to_tif(trend_spatial_dict,outf)

    def long_term_corr_dict_to_tif(self):
        father_dir = '/Users/liyang/Desktop/detrend_zscore_test_factors/partial_window/long_term'
        father_outdir = join(self.this_class_tif,'long_term_corr_tif')
        T.mk_dir(father_outdir,force=True)
        for folder in T.listdir(father_dir):
            fdir = join(father_dir,folder)
            outdir = join(father_outdir,folder)
            T.mk_dir(outdir,force=True)
            for f in T.listdir(fdir):
                fpath = join(fdir,f)
                if 'p_value' in f:
                    continue
                dict_i = T.load_npy(fpath)
                variables_list = {}
                for pix in dict_i:
                    dict_j = dict_i[pix]
                    for var_i in dict_j:
                        if not var_i in variables_list:
                            variables_list[var_i] = 1
                        else:
                            continue
                variables_list = list(variables_list.keys())
                # variables_list = [i.split('\\')[-1] for i in variables_list]
                for var_i in variables_list:
                    var_i_out = var_i.split('\\')[-1]
                    outf = join(outdir, var_i_out + '.tif')
                    spatial_dict = {}
                    for pix in dict_i:
                        dict_j = dict_i[pix]
                        if not var_i in dict_j:
                            continue
                        val = dict_j[var_i]
                        spatial_dict[pix] = val
                    DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)


    def earlier_and_late_greening_trend_binary(self):
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


    def bivarite_plot(self):
        fdir_moving_window = join(self.this_class_tif,'partial_corr_trend')
        fdir_long_term = join(self.this_class_tif,'long_term_corr_tif')
        outdir = join(self.this_class_tif,'bivarite_plot')
        T.mk_dir(outdir,force=True)

        long_term_path_dict = {}
        for folder in T.listdir(fdir_long_term):
            folder_path = join(fdir_long_term,folder)
            for f in T.listdir(folder_path):
                var_i = f.split('.')[0]
                fpath = join(folder_path,f)
                long_term_path_dict[var_i] = fpath
        for folder in T.listdir(fdir_moving_window):
            folder_path = join(fdir_moving_window,folder)
            outdir_i = join(outdir,folder)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(folder_path):
                fpath = join(folder_path,f)
                var_i = f.split('.')[0]
                if not var_i in long_term_path_dict:
                    continue
                long_term_path = long_term_path_dict[var_i]
                # tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf
                tif1 = fpath
                tif2 = long_term_path
                x_label = 'moving_window_trend'
                y_label = 'long_term_trend'
                min1 = -0.01
                max1 = 0.01
                min2 = -0.3
                max2 = 0.3
                outf = join(outdir_i,var_i+'.tif')
                xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf)



class Scatter_plot:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Scatter_plot',
            result_root_this_script)
        pass

    def run(self):
        # self.matrix_plot()
        self.matrix_plot_every_trendy_model()
        # self.earlier_and_late_greening_trend_binary_with_p()
        pass

    def Trendy_dataframe(self):
        outdir = join(self.this_class_arr,'Trendy_dataframe')
        T.mk_dir(outdir,force=True)
        dff = join(outdir,'Trendy_dataframe.df')
        xfdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/for_matrix_plot'
        yfdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/for_matrix_plot_trendy/late'
        y_earlier_fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/for_matrix_plot_trendy/early_peak'

        x_list = []
        y_list = []
        earlier_list = []
        for f in T.listdir(xfdir):
            var_i = f.split('.')[0]
            x_list.append(var_i)
        for f in T.listdir(yfdir):
            var_i = f.split('.')[0]
            y_list.append(var_i)
        for f in T.listdir(y_earlier_fdir):
            var_i = f.split('.')[0]
            earlier_list.append(var_i)

        if isfile(dff):
            df = T.load_df(dff)
            return df,x_list,y_list,earlier_list
        dict_all = {}
        var_list = []
        for f in T.listdir(xfdir):
            fpath = join(xfdir,f)
            dict_i = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = dict_i
            var_list.append(var_i)
        for f in T.listdir(yfdir):
            fpath = join(yfdir,f)
            dict_i = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = dict_i
            var_list.append(var_i)
        for f in T.listdir(y_earlier_fdir):
            fpath = join(y_earlier_fdir,f)
            dict_i = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = dict_i
            var_list.append(var_i)
        df = T.spatial_dics_to_df(dict_all)
        start_year = 2000
        end_year = 2018
        df = Dataframe_per_value_transform(df, var_list, start_year, end_year).df
        df = Main_flow_2.Dataframe_func(df).df
        T.save_df(df,dff)
        T.df_to_excel(df,join(outdir,'Trendy_dataframe.xlsx'))
        return df,x_list,y_list,earlier_list


    def matrix_plot(self):
        dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        outdir = join(self.this_class_png,'matrix_plot_modis')
        T.mk_dir(outdir,force=True)
        # df = self.Trendy_dataframe()
        # df = self.filter_df_with_trend_p(df)
        # df = self.filter_df_with_gs_SPEI3(df)
        # mode = '1'
        # mode = '2'
        # df = self.filter_df_with_trend_class(df,mode=mode)
        #  1_-2 8
        #  1_-1 9
        #  2_-2 12
        #  2_-1 13
        mark_in_list = ['2_-2','2_-1','1_-2','1_-1']
        # mark_in_list = ['2_-2','2_-1']
        # mark_in_list = ['1_-2','1_-1']
        # title = '|'.join(mark_in_list)
        # title = ''
        # df = self.filter_df_with_strong_weak(df,['2_-2','2_-1'])
        # df = self.filter_df_with_strong_weak(df,mark_in_list)
        # df = self.filter_df_with_strong_weak_trendy(df,mark_in_list)
        # DIC_and_TIF().plot_df_spatial_pix(df,land_tif)
        # plt.show()
        cols = df.columns
        for c in cols:
            print(c)
        # limited_area = 'water-limited'
        # limited_area = 'energy-limited'
        # df = df[df['limited_area']==limited_area]
        # limited_area = 'All'
        print(len(df))
        # during_early_peak_CCI_SM_zscore
        # during_late_CCI_SM_zscore
        # during_late_MODIS_LAI_zscore
        # during_early_peak_MODIS_LAI_zscore
        # cci =
        # T.df_bin()
        # early_sm_var = 'during_early_peak_CCI_SM_zscore'
        # early_sm_var = 'detrend_during_peak_CCI_SM_zscore'
        # early_sm_var = 'during_early_peak_SPEI3_zscore'
        # early_sm_var = 'detrend_during_early_peak_SPI3'
        # early_sm_var = 'detrend_during_late_VPD_zscore'
        # early_sm_var = 'during_late_VPD_zscore'
        early_sm_var = 'during_late_Temp_zscore'
        # early_sm_var = 'during_peak_VPD_zscore'
        # early_sm_var = 'detrend_during_early_peak_SPEI3'
        late_lai_var = 'during_late_MODIS_LAI_zscore'
        # early_lai_var = 'during_early_peak_MODIS_LAI_zscore'
        # early_lai_var = 'during_early_peak_MODIS_LAI_zscore'
        early_lai_var = 'during_peak_CCI_SM_zscore'
        # late_lai_var = 'during_late_Trendy_ensemble_zscore'
        # early_lai_var = 'during_early_peak_Trendy_ensemble_zscore'
        df = df.dropna(subset=[early_sm_var,late_lai_var])
        # df = df[df['during_early_peak_MODIS_LAI_zscore']>0]
        # df = df[df['limited_area']=='energy-limited']
        # df = df[df['limited_area']=='water-limited']
        early_sm = df[early_sm_var].values.tolist()
        late_lai = df[late_lai_var].values.tolist()
        earlier_lai = df[early_lai_var].values.tolist()

        max_bin = 2
        min_bin = -2
        n = 7
        percentile_list = np.linspace(0,1,n)
        early_sm = T.remove_np_nan(early_sm)
        late_lai = T.remove_np_nan(late_lai)
        # plt.hist(early_sm,bins=80)
        # plt.figure()
        # plt.hist(late_lai,bins=80)
        # plt.show()
        earlier_lai = T.remove_np_nan(earlier_lai)
        earlier_lai = np.array(earlier_lai)
        # earlier_lai = earlier_lai[earlier_lai>0]
        # earlier_lai = earlier_lai[earlier_lai<1.]
        # sm_bins = np.percentile(early_sm,percentile_list*100)
        # late_lai_bins = np.percentile(late_lai,percentile_list*100)
        # earlier_lai_bins = np.percentile(earlier_lai,percentile_list*100)
        sm_bins = np.linspace(min_bin,max_bin,n)
        # late_lai_bins = np.linspace(min_bin,max_bin,n)
        earlier_lai_bins = np.linspace(-2,2,n)
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
        # plt.title(f'{limited_area} {mode}')
        plt.title(late_lai_var)
        plt.tight_layout()
        outf = join(outdir,f'{late_lai_var}.pdf')
        plt.savefig(outf)
        plt.close()


    def matrix_plot_every_trendy_model(self):
        df,x_list,y_list,_ = self.Trendy_dataframe()
        outdir = join(self.this_class_png,'matrix_plot_every_trendy_model')
        T.mk_dir(outdir)
        # mark_in_list = ['2_-2','2_-1','1_-2','1_-1']
        # mark_in_list = ['2_-2','2_-1']
        # mark_in_list = ['1_-2','1_-1']
        # title = '|'.join(mark_in_list)
        for y_var in y_list:
            cols = df.columns
            # early_sm_var = 'during_late_Temp_zscore'
            early_sm_var = 'during_late_VPD_zscore'
            late_lai_var = y_var
            filter_var = y_var.replace('late','early_peak')
            print(filter_var)
            early_lai_var = 'during_peak_CCI_SM_zscore'
            df = df.dropna(subset=[early_sm_var,late_lai_var])
            df = df[df[filter_var]>0]

            early_sm = df[early_sm_var].values.tolist()
            late_lai = df[late_lai_var].values.tolist()
            earlier_lai = df[early_lai_var].values.tolist()

            max_bin = 2
            min_bin = -2
            n = 7
            percentile_list = np.linspace(0,1,n)
            early_sm = T.remove_np_nan(early_sm)
            late_lai = T.remove_np_nan(late_lai)
            # plt.hist(early_sm,bins=80)
            # plt.figure()
            # plt.hist(late_lai,bins=80)
            # plt.show()
            earlier_lai = T.remove_np_nan(earlier_lai)
            earlier_lai = np.array(earlier_lai)
            # earlier_lai = earlier_lai[earlier_lai>0]
            # earlier_lai = earlier_lai[earlier_lai<1.]
            # sm_bins = np.percentile(early_sm,percentile_list*100)
            # late_lai_bins = np.percentile(late_lai,percentile_list*100)
            # earlier_lai_bins = np.percentile(earlier_lai,percentile_list*100)
            sm_bins = np.linspace(min_bin,max_bin,n)
            # late_lai_bins = np.linspace(min_bin,max_bin,n)
            earlier_lai_bins = np.linspace(-2,2,n)
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
            # plt.title(f'{limited_area} {mode}')
            plt.title(y_var)
            plt.tight_layout()
            outf = join(outdir,y_var+'.pdf')
            plt.savefig(outf)
            plt.close()
            # plt.show()

    def earlier_and_late_greening_trend_binary_with_p(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df, 5)
        # exit()
        earlier = 'during_early_peak_MODIS_LAI_zscore'
        late = 'during_late_MODIS_LAI_zscore'
        spatial_dict = {}
        val_dict = {
            '1-1':2,
            '1-0':1,
            '0-1':-1,
            '0-0':-2,
        }
        mode_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            x = row[earlier]
            y = row[late]
            if type(x) == float:
                continue
            if type(y) == float:
                continue
            trend_x,_,_,p_x = T.nan_line_fit(list(range(len(x))),x)
            trend_y,_,_,p_y = T.nan_line_fit(list(range(len(y))),y)
            r,p = stats.pearsonr(list(range(len(x))),x)
            if p_x > 0.1:
                continue
            # if p_y > 0.1:
            #     continue
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
        DIC_and_TIF().plot_back_ground_arr_north_sphere(land_tif)
        plt.imshow(arr,cmap='jet',aspect='auto')
        plt.colorbar()
        plt.show()

    def filter_df_with_trend_p(self,df_in):
        dff = Dataframe().dff
        df = T.load_df(dff)
        earlier = 'during_early_peak_MODIS_LAI_zscore'
        late = 'during_late_MODIS_LAI_zscore'
        p_list = []
        trend_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            x = row[earlier]
            y = row[late]
            if type(x) == float:
                p_list.append(np.nan)
                trend_list.append(np.nan)
                continue
            if type(y) == float:
                p_list.append(np.nan)
                trend_list.append(np.nan)
                continue
            trend_x,_,_,p_x = T.nan_line_fit(list(range(len(x))),x)
            trend_y,_,_,p_y = T.nan_line_fit(list(range(len(y))),y)
            p_list.append(p_y)
            trend_list.append(trend_y)
        p_list = np.array(p_list)
        trend_list = np.array(trend_list)
        df['p'] = p_list
        df['trend'] = trend_list
        # df = df[df['p']<0.1]
        # df = df[df['trend']<0]
        df = df[df['trend']>0]
        spatial_dict_p = T.df_to_spatial_dic(df,'p')
        spatial_dict_trend = T.df_to_spatial_dic(df,'trend')
        df_out = T.add_spatial_dic_to_df(df_in,spatial_dict_trend,'p')
        df_out = T.add_spatial_dic_to_df(df_in,spatial_dict_trend,'trend')
        df_out = df_out.dropna(subset=['p','trend'])
        # DIC_and_TIF().plot_df_spatial_pix(df_out,land_tif)
        # plt.show()
        return df_out

    def filter_df_with_gs_SPEI3(self,df_in):
        spei_var = 'during_early_peak_late_SPEI3_zscore'
        df_in = df_in[df_in[spei_var]<0]
        # df_in = df_in[df_in[spei_var]>0]
        return df_in

    def filter_df_with_trend_class(self,df_in,mode='2'):

        tif = '/Users/liyang/Desktop/detrend_zscore_test_factors/results/tif/CarryoverInDryYear/bivariate_plot_earlier_and_late_classification_from_tif/MODIS_LAI.tif'
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)
        df_in = T.add_spatial_dic_to_df(df_in,spatial_dict,'trend_class')
        df_in = df_in.dropna(subset=['trend_class'])
        trend_mark_dict = {'-1--1': 0, '-1-0': 1, '-1-1': 2, '0--1': 3, '0-0': 4, '0-1': 5, '1--1': 6, '1-0': 7,
                           '1-1': 8}
        if mode == '1':
            df_in = df_in[df_in['trend_class'] == 8]
            return df_in
        if mode == '2':
            df_in1 = df_in[df_in['trend_class'] == 7]
            df_in2 = df_in[df_in['trend_class'] == 6]
            df_in = pd.concat([df_in1,df_in2])
            return df_in
    def filter_df_with_strong_weak(self,df_in,mark_in_list):
        '''
        :param mark_in:
        1_-2 8
        1_-1 9
        2_-2 12
        2_-1 13
        :param df_in:
        :return:
        '''
        fdir = join(CarryoverInDryYear().this_class_tif, 'flatten_plot_strong_weak')
        # T.open_path_and_file(fdir)
        # exit()
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        trend_mark_dict_reverse = T.reverse_dic(trend_mark_dict)
        # print(trend_mark_dict_reverse)
        # exit()
        # mark_list = {'1--1','1-0','1-1'}
        mark_list_group = [8, 9, 12, 13]
        f = 'MODIS_LAI.tif'
        product = f.split('.')[0]
        fpath = join(fdir, f)
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df_in = T.add_spatial_dic_to_df(df_in, spatial_dict, 'strong_weak')
        df_in = df_in.dropna(subset=['strong_weak'])
        df_list = []
        for mark_in in mark_in_list:
            df_mark = df_in[df_in['strong_weak'] == trend_mark_dict[mark_in]]
            df_list.append(df_mark)
        df_out = pd.concat(df_list)
        return df_out

        pass

    def filter_df_with_strong_weak_trendy(self,df_in,mark_in_list):
        '''
        :param mark_in:
        1_-2 8
        1_-1 9
        2_-2 12
        2_-1 13
        :param df_in:
        :return:
        '''
        fdir = join(CarryoverInDryYear().this_class_tif, 'flatten_plot_strong_weak')
        # T.open_path_and_file(fdir)
        # exit()
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        trend_mark_dict_reverse = T.reverse_dic(trend_mark_dict)
        # print(trend_mark_dict_reverse)
        # exit()
        # mark_list = {'1--1','1-0','1-1'}
        mark_list_group = [8, 9, 12, 13]
        f = 'Trendy_ensemble.tif'
        product = f.split('.')[0]
        fpath = join(fdir, f)
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
        df_in = T.add_spatial_dic_to_df(df_in, spatial_dict, 'strong_weak')
        df_in = df_in.dropna(subset=['strong_weak'])
        df_list = []
        for mark_in in mark_in_list:
            df_mark = df_in[df_in['strong_weak'] == trend_mark_dict[mark_in]]
            df_list.append(df_mark)
        df_out = pd.concat(df_list)
        return df_out

        pass


class RF_per_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF_per_value',
                                                                                       result_root_this_script)
        pass

    def run(self):

        # self.detrend_trendy_ensemble()
        # self.cal_trend()
        # self.run_RF()
        # self.plot_RF_results()
        # self.run_pdp()
        self.plot_pdp()
        # self.run_RF(is_detrend=False)
        # self.plot_RF_results(is_detrend=True)
        # self.plot_RF_results(is_detrend=False)

    def run_RF(self):
        outdir = join(self.this_class_arr, 'permutation_importance')
        T.mkdir(outdir)
        mode = 'modis'
        if mode == 'modis':
            x_var_list = self.x_variables_modis()
            y_var = self.y_variable_modis()
            flatten_strong_weak_f = '/Users/liyang/Desktop/detrend_zscore_test_factors/results/tif/CarryoverInDryYear/flatten_plot_strong_weak/MODIS_LAI.tif'
        else:
            return
        df = self.gen_dataframe(mode)
        vpd_trend_dict, co2_trend_dict = self.cal_trend()
        flatten_strong_weak_dict = DIC_and_TIF().spatial_tif_to_dic(flatten_strong_weak_f)

        df = T.add_spatial_dic_to_df(df, vpd_trend_dict, 'late_vpd_trend')
        df = T.add_spatial_dic_to_df(df, co2_trend_dict, 'late_co2_trend')
        df = T.add_spatial_dic_to_df(df, flatten_strong_weak_dict, 'flatten_strong_weak')
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        selected_value = [8, 9, 12, 13]
        cols = list(df.columns)
        x_var_list.append('late_vpd_trend')
        x_var_list.append('late_co2_trend')
        df = df[df['detrend_during_early_peak_MODIS_LAI_zscore'] > 0]
        T.print_head_n(df)
        cols = df.columns
        for c in cols:
            print(c)
        # for flatten_value in selected_value:
        #     print(flatten_value)
        #     df_selected = df[df['flatten_strong_weak'] == flatten_value]
        #     X = df_selected[x_var_list]
        #     Y = df_selected[y_var]
        #     result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var)
        #     result['x_var_list'] = x_var_list
        #     # print(result)
        #     # exit()
        #     outf = join(outdir, f'{mode} {flatten_value}')
        #     T.save_npy(result, outf)

        # compose
        X = df[x_var_list]
        Y = df[y_var]
        result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var)
        result['x_var_list'] = x_var_list
        # print(result)
        # exit()
        outf = join(outdir, f'{mode} all')
        T.save_npy(result, outf)

    def gen_dataframe(self,mode):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_for_pdp'
        if mode == 'modis':
            x_var_list = self.x_variables_modis()
            y_var = self.y_variable_modis()
        else:
            x_var_list = self.x_variables_trendy()
            y_var = self.y_variable_trendy()
        all_dict = {}
        for x_var in x_var_list:
            fpath = join(fdir,x_var+'.npy')
            spatial_dict = T.load_npy(fpath)
            all_dict[x_var] = spatial_dict
        fpath = join(fdir,y_var+'.npy')
        spatial_dict = T.load_npy(fpath)
        all_dict[y_var] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        start_year = 2000
        end_year = 2018
        var_list = x_var_list + [y_var]
        df = Dataframe_per_value_transform(df,var_list,start_year,end_year).df
        df = Main_flow_2.Dataframe_func(df).df
        return df


    def plot_RF_results(self):
        y = self.y_variable_modis()
        fdir = join(self.this_class_arr,'permutation_importance')
        outdir = join(self.this_class_png,f'importance')
        T.mkdir(outdir)
        # T.open_path_and_file(outdir)
        # x_list = self.x_variables()
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result = T.load_npy(fpath)
            x_list = result['x_var_list']
            importances_mean = result['importances_mean']
            # err = result['importances_std']
            plt.figure(figsize=(8, 8))
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

    def y_variable_modis(self):
        return 'detrend_during_late_MODIS_LAI_zscore'

    def y_variable_trendy(self):
        return 'detrend_during_late_Trendy_ensemble_zscore'

    def detrend_trendy_ensemble(self):
        f_earlier = '/Users/liyang/Desktop/detrend_zscore_test_factors/zscore/Trendy_ensemble/during_early_peak_Trendy_ensemble_zscore.npy'
        f_late = '/Users/liyang/Desktop/detrend_zscore_test_factors/zscore/Trendy_ensemble/during_late_Trendy_ensemble_zscore.npy'
        outf_earlier = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_for_pdp/detrend_during_early_peak_Trendy_ensemble_zscore.npy'
        outf_late = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/data_for_pdp/detrend_during_late_Trendy_ensemble_zscore.npy'
        earlier_spatial_dict = T.load_npy(f_earlier)
        late_spatial_dict = T.load_npy(f_late)
        earlier_spatial_dict_detrend = T.detrend_dic(earlier_spatial_dict)
        late_spatial_dict_detrend = T.detrend_dic(late_spatial_dict)
        T.save_npy(earlier_spatial_dict_detrend,outf_earlier)
        T.save_npy(late_spatial_dict_detrend,outf_late)
        pass

    def cal_trend(self):
        f_vpd = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/ALL/during_late_VPD_zscore.npy'
        f_co2 = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_with_trend/ALL/during_late_CO2_zscore.npy'
        vpd_spatial_dict = T.load_npy(f_vpd)
        co2_spatial_dict = T.load_npy(f_co2)
        vpd_trend_dict = {}
        co2_trend_dict = {}
        for pix in vpd_spatial_dict:
            vpd = vpd_spatial_dict[pix]
            co2 = co2_spatial_dict[pix]
            vpd_trend,b,r,p = T.nan_line_fit(list(range(len(vpd))),vpd)
            co2_trend,b,r,p = T.nan_line_fit(list(range(len(co2))),co2)
            vpd_trend_dict[pix] = vpd_trend
            co2_trend_dict[pix] = co2_trend
        return vpd_trend_dict,co2_trend_dict

    def x_variables_modis(self):
        x_list = [
            'detrend_during_early_peak_SPEI3_zscore',
            'detrend_during_late_Temp_zscore',
            'detrend_during_late_VPD_zscore',
            'detrend_during_early_peak_MODIS_LAI_zscore',
        ]
        return x_list

    def x_variables_trendy(self):
        x_list = [
            'detrend_during_early_peak_SPEI3_zscore',
            'detrend_during_late_Temp_zscore',
            'detrend_during_late_VPD_zscore',
            'detrend_during_early_peak_Trendy_ensemble_zscore',
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
        # mode = 'modis'
        mode = 'trendy'
        if mode == 'modis':
            x_var_list = self.x_variables_modis()
            y_var = self.y_variable_modis()
            flatten_strong_weak_f = '/Users/liyang/Desktop/detrend_zscore_test_factors/results/tif/CarryoverInDryYear/flatten_plot_strong_weak/MODIS_LAI.tif'
            earlier_var = 'detrend_during_early_peak_MODIS_LAI_zscore'
        else:
            x_var_list = self.x_variables_trendy()
            y_var = self.y_variable_trendy()
            flatten_strong_weak_f = '/Users/liyang/Desktop/detrend_zscore_test_factors/results/tif/CarryoverInDryYear/flatten_plot_strong_weak/Trendy_ensemble.tif'
            earlier_var = 'detrend_during_early_peak_Trendy_ensemble_zscore'

        df = self.gen_dataframe(mode)
        vpd_trend_dict, co2_trend_dict = self.cal_trend()
        flatten_strong_weak_dict = DIC_and_TIF().spatial_tif_to_dic(flatten_strong_weak_f)

        df = T.add_spatial_dic_to_df(df, vpd_trend_dict, 'late_vpd_trend')
        df = T.add_spatial_dic_to_df(df, co2_trend_dict, 'late_co2_trend')
        df = T.add_spatial_dic_to_df(df, flatten_strong_weak_dict, 'flatten_strong_weak')
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        selected_value = [8, 9, 12, 13]
        cols = list(df.columns)
        x_var_list.append('late_vpd_trend')
        x_var_list.append('late_co2_trend')
        df = df[df[earlier_var] > 0]
        T.print_head_n(df)
        cols = df.columns
        for c in cols:
            print(c)
        # exit()
        # for flatten_value in selected_value:
        #     print(flatten_value)
        #     df_selected = df[df['flatten_strong_weak'] == flatten_value]
        #     X = df_selected[x_var_list]
        #     Y = df_selected[y_var]
        #     result = self.partial_dependence_plots(df_selected,x_var_list, y_var)
        #     outf = join(outdir, f'{mode} {flatten_value}')
        #     T.save_npy(result, outf)
        X = df[x_var_list]
        Y = df[y_var]
        result = self.partial_dependence_plots(df,x_var_list, y_var)
        outf = join(outdir, f'{mode} all')
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
        y_var = self.y_variable_modis()
        fdir = join(self.this_class_arr,'pdp')
        for f in T.listdir(fdir):
            if not 'trendy' in f:
            # if not 'modis all' in f:
                continue
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
                plt.ylabel(y_var)
                plt.xlabel(key)
                plt.ylim(-1,2)
                plt.tight_layout()
            plt.show()
        pass



class CarryoverInDryYear:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('CarryoverInDryYear',
                                                                                       result_root_this_script)
        pass

    def run(self):

        # self.matrix()
        # self.bivariate_plot_earlier_and_late()
        # self.bivariate_plot_earlier_and_late_classification()
        # self.bivariate_plot_earlier_and_late_classification_from_tif()
        # self.statistic_bivariate_plot_earlier_and_late_classification_from_tif()
        # self.long_term_trend()
        # self.bivariate_plot_earlier_and_late_strong_weak_classification()
        # self.bivariate_plot_strong_weak()
        # self.flatten_plot_strong_weak()
        # self.statistic_flatten_plot_strong_weak()
        # self.time_sereis()
        # self.earlier_green()
        self.statistic_earlier_green()

        pass

    def matrix(self):
        dff = Dataframe_per_value().dff
        df = T.load_df(dff)
        # T.print_head_n(df)
        cols = df.columns
        # for c in cols:
        #     print(c)
        gs_spei_var = 'during_early_peak_late_SPEI3_zscore' # condition1 GS water
        earlier_green_var = 'during_early_peak_MODIS_LAI_zscore' # condition2 earlier Greenness
        lai_var = 'during_late_MODIS_LAI_zscore'
        # lai_var = 'during_early_peak_MODIS_LAI_zscore'
        # sceanrio1: Humid and Green
        # df = df[df[gs_spei_var] > 0]
        # df = df[df[earlier_green_var] > 0]
        # sceanrio_name = 'Humid and Green'

        # scenario2: Humid and Brown
        # df = df[df[gs_spei_var] > 0]
        # df = df[df[earlier_green_var] < 0]
        # sceanrio_name = 'Humid and Brown'

        # scenario3: Dry and Green
        # df = df[df[gs_spei_var] < 0]
        # df = df[df[earlier_green_var] > 0]
        # sceanrio_name = 'Dry and Green'

        # scenario4: Dry and Brown
        df = df[df[gs_spei_var] < 0]
        df = df[df[earlier_green_var] < 0]
        sceanrio_name = 'Dry and Brown'

        df_group = df.groupby(by=['pix'])
        spatial_dict = {}
        for pix,df_i in tqdm(df_group):
            vals = df_i[lai_var].values
            mean = np.nanmean(vals)
            spatial_dict[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='RdBu')
        plt.colorbar()
        plt.title(f'{sceanrio_name}')
        plt.show()

    def bivariate_plot_earlier_and_late(self):
        outtifdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late')
        outf = join(outtifdir,'bivariate_plot_earlier_and_late.tif')
        T.mk_dir(outtifdir)
        dff = Dataframe().dff
        df = T.load_df(dff)
        T.print_head_n(df)
        earlier_var = 'during_early_peak_MODIS_LAI_zscore'
        late_var = 'during_late_MODIS_LAI_zscore'
        earlier_tif = self.__calculate_trend_tif(df,earlier_var,outtifdir)
        late_tif = self.__calculate_trend_tif(df,late_var,outtifdir)
        earlier_min = -0.2
        earlier_max = 0.2
        late_min = -0.2
        late_max = 0.2
        xymap.Bivariate_plot().plot_bivariate_map(earlier_tif,late_tif,earlier_var,late_var,earlier_min,earlier_max,late_min,late_max,outf)

    def __calculate_trend_tif(self,df,var_i,outdir):
        outf = join(outdir,f'{var_i}.tif')
        if isfile(outf):
            return outf
        spatial_dict = T.df_to_spatial_dic(df,var_i)
        spatial_dict_trend = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            if type(vals) == float:
                continue
            if T.is_all_nan(vals):
                continue
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            spatial_dict_trend[pix] = a
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_trend,outf)
        return outf

    def bivariate_plot_earlier_and_late_classification(self):
        outtifdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_classification')
        outf = join(outtifdir,'classification.tif')
        T.mk_dir(outtifdir)
        dff = Dataframe().dff
        df = T.load_df(dff)
        # T.print_head_n(df)
        earlier_var = 'during_early_peak_MODIS_LAI_zscore'
        late_var = 'during_late_MODIS_LAI_zscore'
        df = self.__calculate_trend_to_df(df,earlier_var)
        df = self.__calculate_trend_to_df(df,late_var)
        # compose trend mark
        trend_mark_compose = []
        for i,row in df.iterrows():
            earlier_trend = row[f'{earlier_var}_trend']
            late_trend = row[f'{late_var}_trend']
            if np.isnan(earlier_trend) or np.isnan(late_trend):
                trend_mark_compose.append(np.nan)
                continue
            earlier_trend = int(earlier_trend)
            late_trend = int(late_trend)
            earlier_trend = str(earlier_trend)
            late_trend = str(late_trend)
            trend_mark = earlier_trend + '-' + late_trend
            trend_mark_compose.append(trend_mark)
        df['trend_mark'] = trend_mark_compose
        trend_mark_unique = T.get_df_unique_val_list(df,'trend_mark')
        trend_mark_dict = {key:j for j,key in enumerate(trend_mark_unique)}
        trend_mark_compose_value = []
        for i, row in df.iterrows():
            trend_mark = row['trend_mark']
            if type(trend_mark) == float:
                trend_mark_compose_value.append(np.nan)
                continue
            trend_mark_value = trend_mark_dict[trend_mark]
            trend_mark_compose_value.append(trend_mark_value)
        df['trend_mark_value'] = trend_mark_compose_value
        print(trend_mark_dict)
        spatial_dict = T.df_to_spatial_dic(df,'trend_mark_value')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def __calculate_trend_to_df(self,df,var_i):
        spatial_dict = T.df_to_spatial_dic(df,var_i)
        trend_mark_list = []
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            if type(vals) == float:
                trend_mark_list.append(np.nan)
                continue
            if T.is_all_nan(vals):
                trend_mark_list.append(np.nan)
                continue
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            if p > 0.1:
                trend_mark = 0
            else:
                if a > 0:
                    trend_mark = 1
                else:
                    trend_mark = -1
            trend_mark_list.append(trend_mark)
        df[f'{var_i}_trend'] = trend_mark_list
        return df

    def bivariate_plot_earlier_and_late_classification_from_tif(self):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/Figure1'
        outtifdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_classification_from_tif')
        T.mk_dir(outtifdir)
        all_data_dict = {}
        for f in T.listdir(fdir):
            # print(f)
            if not f.endswith('.tif'):
                continue
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(join(fdir,f))
            var_name = f.split('.')[0]
            all_data_dict[var_name] = spatial_dict
        df = T.spatial_dics_to_df(all_data_dict)
        cols = list(df.columns)
        # for c in cols:
        #     print(c)
        var_list = ['Trendy_ensemble','MODIS_LAI','LAI3g']
        period_list = ['during_early_peak','during_late']
        for var_i in var_list:
            outf = join(outtifdir, f'{var_i}.tif')
            earlier_trend_var = f'{period_list[0]}_{var_i}_trend'
            earlier_p_var = f'{period_list[0]}_{var_i}_p_value'
            late_trend_var = f'{period_list[1]}_{var_i}_trend'
            late_p_var = f'{period_list[1]}_{var_i}_p_value'
            df = self.__add_trend_to_df(df,earlier_trend_var,earlier_p_var)
            df = self.__add_trend_to_df(df,late_trend_var,late_p_var)
            # compose trend mark
            trend_mark_compose = []
            for i, row in df.iterrows():
                earlier_trend = row[f'{earlier_trend_var}_mark']
                late_trend = row[f'{late_trend_var}_mark']
                if np.isnan(earlier_trend) or np.isnan(late_trend):
                    trend_mark_compose.append(np.nan)
                    continue
                earlier_trend = int(earlier_trend)
                late_trend = int(late_trend)
                earlier_trend = str(earlier_trend)
                late_trend = str(late_trend)
                trend_mark = earlier_trend + '-' + late_trend
                trend_mark_compose.append(trend_mark)
            df['trend_mark'] = trend_mark_compose
            trend_mark_unique = T.get_df_unique_val_list(df, 'trend_mark')
            trend_mark_dict = {'-1--1': 0, '-1-0': 1, '-1-1': 2, '0--1': 3, '0-0': 4, '0-1': 5, '1--1': 6, '1-0': 7, '1-1': 8}
            trend_mark_compose_value = []
            for i, row in df.iterrows():
                trend_mark = row['trend_mark']
                if type(trend_mark) == float:
                    trend_mark_compose_value.append(np.nan)
                    continue
                trend_mark_value = trend_mark_dict[trend_mark]
                trend_mark_compose_value.append(trend_mark_value)
            df['trend_mark_value'] = trend_mark_compose_value
            print(trend_mark_dict)
            spatial_dict = T.df_to_spatial_dic(df, 'trend_mark_value')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)


    def __add_trend_to_df(self,df,a_var,p_var):
        trend_mark_list = []
        for i,row in df.iterrows():
            a = row[a_var]
            p = row[p_var]
            if p > 0.1:
                trend_mark = 0
            else:
                if a > 0:
                    trend_mark = 1
                else:
                    trend_mark = -1
            trend_mark_list.append(trend_mark)
        df[f'{a_var}_mark'] = trend_mark_list
        return df

    def statistic_bivariate_plot_earlier_and_late_classification_from_tif(self):
        fdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_classification_from_tif')
        trend_mark_dict = {'-1--1': 0, '-1-0': 1, '-1-1': 2, '0--1': 3, '0-0': 4, '0-1': 5, '1--1': 6, '1-0': 7,
                           '1-1': 8}
        trend_mark_dict_reverse = T.reverse_dic(trend_mark_dict)
        mark_list = {'1--1','1-0','1-1'}
        result_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            product = f.split('.')[0]
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_i = {}
            for mark in mark_list:
                total = 0
                count = 0
                for pix in spatial_dict:
                    val = spatial_dict[pix]
                    if np.isnan(val):
                        continue
                    val = int(val)
                    mark_i = trend_mark_dict_reverse[val][0]
                    if mark_i == mark:
                        count += 1
                    total += 1
                ratio = count/total * 100
                dict_i[mark] = ratio
            result_dict[product] = dict_i
        df = pd.DataFrame(result_dict)
        # plot
        # df = df.T
        df.plot(kind='bar',stacked=False)
        plt.xticks(rotation=0)
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()

    def long_term_trend(self):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/origin_data'
        outdir = join(self.this_class_tif,'long_term_trend_mk')
        T.mk_dir(outdir)
        spatial_dict_all = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        cols = df.columns
        cols = list(cols)
        cols.remove('pix')
        trend_str_dict = {'increasing':1,'decreasing':-1,'no trend':0}
        for c in cols:
            outf = join(outdir,f'{c}.tif')
            spatial_dict = T.df_to_spatial_dic(df,c)
            spatial_dict_trend = {}
            for pix in tqdm(spatial_dict):
                vals = spatial_dict[pix]
                if type(vals) == float:
                    continue
                a,b,r,_ = T.nan_line_fit(list(range(len(vals))),vals)
                result = mk.original_test(vals)
                trend_str = result[0]
                trend_value = trend_str_dict[trend_str]
                p_value = result[2]
                spatial_dict_trend[pix] = trend_value
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_trend,outf)

    def bivariate_plot_earlier_and_late_strong_weak_classification(self):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/Figure1'
        outtifdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_strong_weak_classification')
        T.mk_dir(outtifdir)
        all_data_dict = {}
        for f in T.listdir(fdir):
            # print(f)
            if not f.endswith('.tif'):
                continue
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(join(fdir,f))
            var_name = f.split('.')[0]
            all_data_dict[var_name] = spatial_dict
        df = T.spatial_dics_to_df(all_data_dict)
        cols = list(df.columns)
        # for c in cols:
        #     print(c)
        # exit()
        var_list = ['Trendy_ensemble','MODIS_LAI','LAI3g']
        period_list = ['during_early_peak','during_late']
        for var_i in var_list:
            outf = join(outtifdir, f'{var_i}.tif')
            earlier_trend_var = f'{period_list[0]}_{var_i}_trend'
            earlier_p_var = f'{period_list[0]}_{var_i}_p_value'
            late_trend_var = f'{period_list[1]}_{var_i}_trend'
            late_p_var = f'{period_list[1]}_{var_i}_p_value'
            df = self.__add_strong_weak_to_df(df,earlier_trend_var,earlier_p_var)
            df = self.__add_strong_weak_to_df(df,late_trend_var,late_p_var)
            spatial_dict_earlier = T.df_to_spatial_dic(df,earlier_trend_var+'_mark')
            spatial_dict_late = T.df_to_spatial_dic(df,late_trend_var+'_mark')
            outf_earlier = join(outtifdir,f'{earlier_trend_var}.tif')
            outf_late = join(outtifdir,f'{late_trend_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_earlier,outf_earlier)
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_late,outf_late)

    def __add_strong_weak_to_df(self,df,a_var,p_var):
        trend_mark_list = []
        for i, row in df.iterrows():
            a = row[a_var]
            p = row[p_var]
            if a > 0:
                if p < 0.05:
                    trend_mark = 2
                else:
                    trend_mark = 1
            else:
                if p < 0.05:
                    trend_mark = -2
                else:
                    trend_mark = -1
            trend_mark_list.append(trend_mark)
        df[f'{a_var}_mark'] = trend_mark_list
        return df

    def bivariate_plot_strong_weak(self):
        fdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_strong_weak_classification')
        outtifdir = join(self.this_class_tif,'bivariate_plot_strong_weak')
        T.mk_dir(outtifdir)
        var_list = ['Trendy_ensemble','MODIS_LAI','LAI3g']
        period_list = ['during_early_peak','during_late']
        for var_i in var_list:
            outf = join(outtifdir, f'{var_i}.tif')
            earlier_trend_var = f'{period_list[0]}_{var_i}_trend'
            late_trend_var = f'{period_list[1]}_{var_i}_trend'
            earlier_trend_var_mark_path = join(fdir,f'{earlier_trend_var}.tif')
            late_trend_var_mark_path = join(fdir,f'{late_trend_var}.tif')
            # tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf
            tif1 = earlier_trend_var_mark_path
            tif2 = late_trend_var_mark_path
            x_label = 'early peak'
            y_label = 'late peak'
            min1 = -2
            max1 = 2
            min2 = -2
            max2 = 2
            n = (5,5)
            n_plot = (5,5)
            # c00 = hex_to_Color('#e8e8e8')
            # c10 = hex_to_Color('#be64ac')
            # c01 = hex_to_Color('#5ac8c8')
            # c11 = hex_to_Color('#3b4994')
            corner_colors = ("#AF0000", '#FF9D00', "#FF9D00", "#007511")
            zcmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
            # plt.imshow(zcmap)
            # plt.show()
            xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf,n=n,n_legend=n_plot,zcmap=zcmap)

    def flatten_plot_strong_weak(self):
        fdir = join(self.this_class_tif,'bivariate_plot_earlier_and_late_strong_weak_classification')
        outtifdir = join(self.this_class_tif,'flatten_plot_strong_weak')
        T.mk_dir(outtifdir)
        var_list = ['Trendy_ensemble','MODIS_LAI','LAI3g']
        period_list = ['during_early_peak','during_late']
        for var_i in var_list:
            outf = join(outtifdir, f'{var_i}.tif')
            earlier_trend_var = f'{period_list[0]}_{var_i}_trend'
            late_trend_var = f'{period_list[1]}_{var_i}_trend'
            earlier_trend_var_mark_path = join(fdir,f'{earlier_trend_var}.tif')
            late_trend_var_mark_path = join(fdir,f'{late_trend_var}.tif')
            spatial_dict_earlier = DIC_and_TIF().spatial_tif_to_dic(earlier_trend_var_mark_path)
            spatial_dict_late = DIC_and_TIF().spatial_tif_to_dic(late_trend_var_mark_path)
            dict_all = {'earlier':spatial_dict_earlier,'late':spatial_dict_late}
            df = T.spatial_dics_to_df(dict_all)
            value_list = [-2,-1,1,2]
            value_dict = {}
            flag = 0
            for v1 in value_list:
                for v2 in value_list:
                    compose = f'{v1}_{v2}'
                    value_dict[compose] = flag
                    flag += 1
            print(value_dict)
            df_value_dict = pd.DataFrame.from_dict(value_dict,orient='index')
            print(df_value_dict)
            exit()
            spatial_dict = {}
            for i,row in df.iterrows():
                pix = row['pix']
                earlier = row['earlier']
                late = row['late']
                earlier = str(int(earlier))
                late = str(int(late))
                relass = earlier+'_'+late
                value = value_dict[relass]
                spatial_dict[pix] = value
            outf = join(outtifdir, f'{var_i}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def statistic_flatten_plot_strong_weak(self):
        fdir = join(self.this_class_tif,'flatten_plot_strong_weak')
        trend_mark_dict = {'-2_-2': 0, '-2_-1': 1, '-2_1': 2, '-2_2': 3, '-1_-2': 4, '-1_-1': 5,
                           '-1_1': 6, '-1_2': 7, '1_-2': 8, '1_-1': 9, '1_1': 10, '1_2': 11,
                           '2_-2': 12, '2_-1': 13, '2_1': 14, '2_2': 15}
        # trend_mark_dict_reverse = T.reverse_dic(trend_mark_dict)
        # print(trend_mark_dict_reverse)
        # exit()
        # mark_list = {'1--1','1-0','1-1'}
        mark_list_group = [8,9,10,11,12,13,14,15]
        result_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            product = f.split('.')[0]
            fpath = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_i = {}
            for marks in mark_list_group:
                total = 0
                count = 0
                for pix in spatial_dict:
                    val = spatial_dict[pix]
                    if np.isnan(val):
                        continue
                    val = int(val)
                    # print(marks)
                    # if val in marks:
                    if val == marks:
                        count += 1
                    total += 1
                ratio = count/total * 100
                dict_i[str(marks)] = ratio
            result_dict[product] = dict_i
        df = pd.DataFrame(result_dict)
        # plot
        # df = df.T
        df.plot(kind='bar',stacked=False)
        plt.xticks(rotation=0)
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()

    def time_sereis(self):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/zscore/selected'
        outdir = join(self.this_class_tif,'time_series')
        T.mk_dir(outdir)
        dict_all = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        cols = df.columns
        # for c in cols:
        #     print(c)
        # exit()
        df = Main_flow_2.Dataframe_func(df).df
        limited_area_list = ['water-limited','energy-limited']
        # limited_area_list = T.get_df_unique_val_list(df,'limited_area')
        # print(limited_area_list)
        # exit()
        T.print_head_n(df,5)
        period_list = ['during_early_peak','during_late']
        product_list = ['Trendy_ensemble','MODIS_LAI','LAI3g']
        for product in product_list:
            for ltd in limited_area_list:
                df_ltd = df[df['limited_area']==ltd]
                for period in period_list:
                    var = f'{period}_{product}_zscore'
                    spatial_dict = T.df_to_spatial_dic(df_ltd,var)
                    all_value = []
                    for pix in spatial_dict:
                        val = spatial_dict[pix]
                        if type(val) == float:
                            continue
                        all_value.append(val)
                    all_value = np.array(all_value)
                    all_value_mean = []
                    all_value_ci = []
                    for i in range(all_value.shape[1]):
                        val = all_value[:,i]
                        val_mean = np.nanmean(val)
                        err,_,_ = T.uncertainty_err(val)
                        # err = np.nanstd(val)
                        all_value_mean.append(val_mean)
                        all_value_ci.append(err)
                    all_value_mean = np.array(all_value_mean)
                    all_value_ci = np.array(all_value_ci)
                    plt.plot(all_value_mean)
                    plt.fill_between(range(len(all_value_mean)),all_value_mean-all_value_ci,all_value_mean+all_value_ci,alpha=0.5,label=f'{period}')
                plt.title(f'{product} {ltd}')
                plt.legend()
                outf = join(outdir,f'{product}_{ltd}.pdf')
                plt.savefig(outf)
                plt.close()

    def earlier_green(self):
        fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/zscore/selected'
        outdir = join(self.this_class_tif,'earlier_green')
        T.mk_dir(outdir)
        dff = Dataframe().dff
        df = T.load_df(dff)
        cols = df.columns
        earlier_lai_var = 'during_early_peak_MODIS_LAI_zscore'
        # late_sm_var = 'during_late_CCI_SM_zscore'
        # late_sm_var = 'during_peak_CCI_SM_zscore'
        # late_sm_var = 'during_late_MODIS_LAI_zscore'
        # late_sm_var = 'during_late_VPD_zscore'
        late_sm_var = 'during_late_Temp_zscore'
        for c in cols:
            print(c)
        lai_spatial_dict = T.df_to_spatial_dic(df,earlier_lai_var)
        sm_spatial_dict = T.df_to_spatial_dic(df,late_sm_var)
        spatial_dict = {}
        for pix in tqdm(lai_spatial_dict):
            lai = lai_spatial_dict[pix]
            if type(lai) == float:
                continue
            sm = sm_spatial_dict[pix]
            if type(sm) == float:
                continue
            lai = np.array(lai)
            sm = np.array(sm)
            lai_gt_0 = lai > 0
            sm_selected = sm[lai_gt_0]
            sm_mean = np.nanmean(sm_selected)
            spatial_dict[pix] = sm_mean
        outf = join(outdir,f'{late_sm_var}.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def statistic_earlier_green(self):
        # fdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/zscore/selected'
        fdir = join(self.this_class_tif,'earlier_green')
        outdir = join(self.this_class_png,'statistic_earlier_green')
        T.mk_dir(outdir)
        dict_all = {}
        var_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            if not fpath.endswith('.tif'):
                continue
            var_i = f.split('.')[0]
            var_list.append(var_i)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(dict_all)
        df = Main_flow_2.Dataframe_func(df).df
        limited_area_list = ['water-limited','energy-limited']
        for var_i in var_list:
            plt.figure()
            for ltd in limited_area_list:
                df_ltd = df[df['limited_area']==ltd]
                vals = df_ltd[var_i].values
                x,y = Plot().plot_hist_smooth(vals,alpha=0,bins=80)
                # plt.plot(x,y,label=f'{ltd}')
                plt.fill(x, y,zorder=-9,label=f'{ltd}',alpha=0.5)
                plt.ylim(0,0.05)
                plt.xlim(-1.3,1.3)
            plt.title(f'{var_i}')
            plt.legend()
            outf = join(outdir,f'{var_i}.pdf')
            plt.savefig(outf)
            plt.close()


class ResponseFunction:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('ResponseFunction',
                                                                                       result_root_this_script)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        self.Yfdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/Y/late'
        self.Yfdir_earlier = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/Y/early_peak'
        self.Xfdir = '/Users/liyang/Desktop/detrend_zscore_test_factors/data_for_SEM_detrend/X/corresponding_factors'
        pass

    def run(self):
        # self.build_df()
        self.plot_response_func()
        pass

    def build_df(self):
        Yfdir = self.Yfdir
        Xfdir = self.Xfdir
        Yfdir_earlier = self.Yfdir_earlier
        dict_all = {}
        var_list = []
        for f in tqdm(T.listdir(Yfdir),desc='Y'):
            fpath = join(Yfdir,f)
            spatial_dict = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = spatial_dict
            var_list.append(var_i)
        for f in tqdm(T.listdir(Xfdir),desc='X'):
            fpath = join(Xfdir,f)
            spatial_dict = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = spatial_dict
            var_list.append(var_i)
        for f in tqdm(T.listdir(Yfdir_earlier),desc='Y_earlier'):
            fpath = join(Yfdir_earlier,f)
            spatial_dict = T.load_npy(fpath)
            var_i = f.split('.')[0]
            dict_all[var_i] = spatial_dict
            var_list.append(var_i)
        df = T.spatial_dics_to_df(dict_all)
        df = Dataframe_per_value_transform(df,var_list,2000,2018).df
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

    def get_x_list(self):
        xfdir = self.Xfdir
        x_list = []
        for f in T.listdir(xfdir):
            if not f.endswith('.npy'):
                continue
            x_list.append(f.split('.')[0])
        return x_list

    def get_y_list(self):
        yfdir = self.Yfdir
        y_list = []
        for f in T.listdir(yfdir):
            if not f.endswith('.npy'):
                continue
            y_list.append(f.split('.')[0])
        return y_list

    def get_y_earlier_list(self):
        yfdir = self.Yfdir_earlier  # 早期的
        y_list = []
        y_name_dict = {}
        for f in T.listdir(yfdir):
            if not f.endswith('.npy'):
                continue
            y_var = f.split('.')[0]
            model_name = y_var.replace('_lai_zscore', '')
            model_name = model_name.replace('detrend_during_early_peak_', '')
            y_name_dict[model_name] = y_var
            y_list.append(f.split('.')[0])
        return y_list,y_name_dict

    def plot_response_func(self):
        outdir = join(self.this_class_png,'response_func')
        T.mk_dir(outdir)
        df = T.load_df(self.dff)
        cols = df.columns
        # for c in cols:
        #     print(c)
        # exit()
        x_list = self.get_x_list()
        y_list = self.get_y_list()
        y_list_earlier,y_dict_earlier = self.get_y_earlier_list()
        # x_list.remove('EOS_zscore')
        # x_list.remove('SWE_zscore')
        # x_list = ['during_peak_CCI_SM_zscore']
        x_bins = np.arange(-2,2,0.1)
        for x_var in tqdm(x_list):
            # print(x_var)
            x_vals = df[x_var].values
            df_group,bins_list_str = T.df_bin(df,x_var,x_bins)
            plt.figure(figsize=(10,10))
            for y_var in y_list:
                # print(y_var)
                model_name = y_var.replace('_lai_zscore','')
                model_name = model_name.replace('detrend_during_late_','')
                y_var_earlier = y_dict_earlier[model_name]
                mean_list = []
                x_label_list = []
                for name,df_group_i in df_group:
                    # df_group_i = df_group_i[df_group_i[y_var_earlier] > 0]
                    vals = df_group_i[y_var].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                    x_label_list.append(name.left)
                if 'MODIS' in y_var:
                    c = 'k'
                    lw = 4
                elif 'LAI3g' in y_var:
                    c = 'g'
                    lw = 3
                elif 'ensemble' in y_var:
                    c = 'r'
                    lw = 3
                else:
                    c=None
                    lw = 1
                plt.plot(x_label_list,mean_list,label=f'{y_var}',c=c,lw=lw)
                y_var_label = y_var.replace('detrend_during_late_','')
                y_var_label = y_var_label.replace('_lai_zscore','')
                # plt.text(x_label_list[-1],mean_list[-1],f'{y_var_label}')
            # plt.legend()
            plt.xlabel(f'{x_var}')
            plt.ylabel('Late LAI anomaly')
            plt.ylim(-0.8,0.8)
            outf = join(outdir,f'{x_var}.pdf')
            plt.savefig(outf,dpi=200)
            plt.close()
            # plt.show()

def main():
    # Earlier_positive_anomaly().run()
    # Dataframe().run()
    # Dataframe_per_value().run()
    # SEM().run()
    # Dataframe_moving_window().run()
    # Moving_window_single_correlation().run()
    # Single_corr().run()
    # Bivarite_plot_partial_corr().run()
    # Scatter_plot().run()
    # RF_per_value().run()
    CarryoverInDryYear().run()
    # ResponseFunction().run()
    pass


if __name__ == '__main__':
    main()