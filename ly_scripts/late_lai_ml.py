# coding=utf-8
import time

import torch
import torch.hub
import xycmap
import PIL.Image as Image
from __init__ import *
result_root_this_script = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/'
global_start_year = 1982
land_tif = join('/Volumes/NVME2T/greening_project_redo/conf/land.tif')
import Main_flow_2
data_root = '/Volumes/NVME2T/greening_project_redo/data/'

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe',
                                                                                       result_root_this_script)
        self.dff = join(self.this_class_arr, 'data_frame.df')
        pass

    def run(self):
        df = self.__gen_df_init()

        df = self.add_variables()
        df = Main_flow_2.Dataframe_func(df).df
        df = self.add_sos_eos(df)
        df = self.add_early_peak_lai(df)
        df = self.add_tipping_point(df)
        df = self.add_photo_period(df)

        # self.to_wen(df)

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def to_wen(self,df):
        col1 = 'after_2003_T_late_trend'
        dict_1 = T.df_to_spatial_dic(df,col1)
        T.open_path_and_file(self.this_class_arr)
        outf = join(self.this_class_arr,'after_2003_T_late_trend.npy')
        T.save_npy(dict_1,outf)
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

    def add_sos_eos(self,df):
        fdir = join(data_root,'lai3g_pheno')
        sos_f = join(fdir,'early_start.npy')
        eos_f = join(fdir,'late_end.npy')
        late_start_f = join(fdir,'late_start.npy')
        sos_dict = T.load_npy(sos_f)
        eos_dict = T.load_npy(eos_f)
        late_start_dict = T.load_npy(late_start_f)
        # exit()
        result_dict = {}
        for pix in tqdm(sos_dict):
            sos = sos_dict[pix]
            eos = eos_dict[pix]
            late_start = late_start_dict[pix]
            late_start = np.array(late_start)
            # print(sos)
            # print(eos)
            # print(late_start)
            # exit()
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
                             'sos_anomaly':sos_anomaly,'eos_anomaly':eos_anomaly,'sos':sos,'eos':eos,'late_start':late_start}
            result_dict[pix] = result_dict_i
        df_pheno = T.dic_to_df(result_dict,'pix')
        df = T.join_df_list(df,[df_pheno],'pix')

        return df


    def add_variables(self):
        fdir = join(data_root,'1982_2018_final_wen')
        variable_list = []
        for f in T.listdir(fdir):
            fname = f.split('.')[0]
            fname = fname.replace('_zscore','')
            fname = fname.replace('during_','')
            period = fname.split('_')[0]
            fname = fname.replace(period+'_','')
            variable_list.append(fname)
        variable_list = list(set(variable_list))
        variable_list.sort()
        period_list = ['early','peak','late']
        data_dict = {}
        for period in period_list:
            for var_i in variable_list:
                key = f'{period}_{var_i}'
                fname = f'during_{period}_{var_i}_zscore.npy'
                fpath = join(fdir,fname)
                dict_i = T.load_npy(fpath)
                col_name = f'{var_i}_{period}'
                data_dict[col_name] = dict_i
        df = T.spatial_dics_to_df(data_dict)
        return df

    def add_early_peak_lai(self, df):
        mean_list = []
        for i,row in df.iterrows():
            lai3g_early = row['LAI3g_early']
            lai3g_peak = row['LAI3g_peak']
            lai3g_late = row['LAI3g_late']
            lai3g_early = np.array(lai3g_early)
            lai3g_peak = np.array(lai3g_peak)
            lai3g_late = np.array(lai3g_late)
            mean = (lai3g_early + lai3g_peak) / 2
            mean_list.append(mean)

        # LAI3g_early = df['LAI3g_early'].tolist()
        # LAI3g_peak = df['LAI3g_peak'].tolist()
        # LAI3g_early = np.array(LAI3g_early,dtype=object)
        # LAI3g_peak = np.array(LAI3g_peak,dtype=object)
        # mean = (LAI3g_early + LAI3g_peak) / 2
        df['LAI3g_earlier'] = mean_list

        return df

    def add_tipping_point(self,df):
        col = 'Temp_late'
        mark_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_vals = row[col]
            T_vals_second_period = T_vals[2003-1982:]
            a,b,r,p = T.nan_line_fit(list(range(len(T_vals_second_period))),T_vals_second_period)
            # if p > 0.05:
            #     mark = 0
            # else:
            if r > 0:
                mark = 1
            elif r < 0:
                mark = -1
            else:
                mark = 0
            mark_list.append(mark)
        mark_list = np.array(mark_list)
        df['after_2003_T_late_trend'] = mark_list
        return df


    def __cal_photo_period(self,doy,lat):
        # Define parameters
        light_intensity = 2.206 * 10 ** -3  # cal cm^-2 min^-1
        lattitude_rad = math.radians(lat)  # Convert the user supplied lattitude to radians
        B = -4.76 - 1.03 * math.log(light_intensity)  # Calculate angle of sun below horizon
        alpha_rad = math.radians(90 + B)  # Calculate zenith distance
        M_rad = math.radians(0.985600 * doy - 3.251)  # Calculate mean anomaly of the sun
        Lambda_rad = math.radians(
            math.degrees(M_rad) + 1.916 * math.sin(M_rad) + 0.020 * math.sin(2 * M_rad) + 282.565)  # calculate lambda
        Delta_rad = math.asin(0.39779 * math.sin(Lambda_rad))  # calculate angle of declination
        day_length = 2 / 15 * math.degrees(math.acos(
            math.cos(alpha_rad) * 1 / math.cos(lattitude_rad) * 1 / math.cos(Delta_rad) - math.tan(
                lattitude_rad) * math.tan(Delta_rad)))  # calculate day length in hours..
        return day_length

    def add_photo_period(self,df):
        photo_period_list_all = []

        for i,row in tqdm(df.iterrows(),total=len(df)):
            lat = row['lat']
            sos = row['sos']
            eos = row['eos']
            photo_period_sum_list = []
            for year in range(len(sos)):
                start = sos[year]
                end = eos[year]
                photo_period_list = []
                for doy in range(start,end+1):
                    try:
                        photo_period = self.__cal_photo_period(doy,lat)
                    except:
                        photo_period = np.nan
                    photo_period_list.append(photo_period)
                photo_period_sum = np.nansum(photo_period_list)
                photo_period_sum_list.append(photo_period_sum)
            photo_period_sum_list = np.array(photo_period_sum_list)
            photo_period_list_all.append(photo_period_sum_list)
        df['photo_period'] = photo_period_list_all
        return df
class Dataframe_moving_window:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe_moving_window',
                                                                                       result_root_this_script)
        self.dff = join(self.this_class_arr, 'data_frame.df')
        self.datadir = '/Volumes/NVME2T/greening_project_redo/data/original_20221107'
        self.n = 15
        pass

    def run(self):
        # valid_pix = self.valid_pix()
        # df = self.add_variables()
        df = self.gen_moving_window_df()

    def gen_window(self):
        n = self.n
        year_list = list(range(1982,2019))
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
        period_list = ['early','peak','late']
        variables_fpath_dict = {}
        y_fdir = join(self.datadir, 'LAI3g')
        for period in period_list:
            x_fdir = join(self.datadir,'X',period)
            for f in T.listdir(x_fdir):
                var_name = f.split('.')[0].replace('during_','')
                fpath = join(x_fdir,f)
                variables_fpath_dict[var_name] = fpath
            var_name = f'{period}_LAI3g'
            fpath = join(y_fdir, f'during_{period}_LAI3g.npy')
            variables_fpath_dict[var_name] = fpath
        var_name_early_peak = 'early_peak_LAI3g'
        fpath_early_peak = join(y_fdir, f'during_early_peak_LAI3g.npy')
        variables_fpath_dict[var_name_early_peak] = fpath_early_peak
        return variables_fpath_dict

    def add_variables(self):
        var_fpath_dict = self.gen_var_fpath_dict()
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
        outdir = join(self.this_class_arr, 'moving_window')
        T.mkdir(outdir,force=True)
        df = T.load_df(self.dff)
        variables_dict = self.gen_var_fpath_dict()
        variables_list = []
        for v in variables_dict:
            variables_list.append(v)
        window_list = self.gen_window()
        for w in tqdm(range(len(window_list))):
            window = window_list[w]
            window = np.array(window)
            window = window - 1982
            window_str = f'{w:02d}'
            outdir_i = join(outdir,window_str)
            T.mkdir(outdir_i,force=True)
            outf = join(outdir_i,f'{window_str}.df')
            if isfile(outf):
                continue
            dict_all = {}
            for i,row in df.iterrows():
                pix = row['pix']
                dict_i = {}
                for x in variables_list:
                    vals = row[x]
                    if type(vals) == float:
                        continue
                    if len(vals) != 37:
                        continue
                    if T.is_all_nan(vals):
                        continue
                    vals = T.pick_vals_from_1darray(vals,window)
                    dict_i[x] = vals
                dict_all[pix] = dict_i
            df_out = T.dic_to_df(dict_all,key_col_str='pix')
            df_out = Main_flow_2.Dataframe_func(df_out).df
            T.save_df(df_out,outf)
            T.df_to_excel(df_out,outf)


class Dataframe_per_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Dataframe_per_value',
                                                                                       result_root_this_script)
        self.datadir = All_variables_single_corr().datadir
        self.start_year = 2000


    def run(self):
        # self.detrend()
        self.not_detrend()
        pass


    def add_MAT_MAP(self,df):
        df = All_variables_single_corr().add_MAT_MAP_to_df(df)
        return df

    def add_sand_fraction(self,df):
        sand_tif = join(All_variables_single_corr().datadir,'T_SAND_resample.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(sand_tif)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'sand')
        return df

    def detrend(self):
        outdir = join(self.this_class_arr, 'detrend')
        T.mkdir(outdir, force=True)
        # x_list_current_year_fpath, x_list_previous_year_fpath = All_variables_single_corr().x_variables()
        all_x_variables = All_variables_single_corr().all_x_variables()
        # print(all_x_variables.keys())
        # exit()
        y_spatial_dict = All_variables_single_corr().y_variable()
        df = T.spatial_dics_to_df({'y': y_spatial_dict})
        df = Main_flow_2.Dataframe_func(df).df
        limited_area_list = T.get_df_unique_val_list(df, 'limited_area')
        limited_area_pix_dict = {}
        for ltd in limited_area_list:
            df_ltd = df[df['limited_area'] == ltd]
            pix_list = df_ltd['pix'].values.tolist()
            pix_list = set(pix_list)
            limited_area_pix_dict[ltd] = pix_list
        for ltd in limited_area_pix_dict:
            print(ltd)
            outf = join(outdir, f'{ltd}.df')
            selected_pix_list = limited_area_pix_dict[ltd]
            data_dict = {}
            for x_var in all_x_variables:
                # if 'CO2' in x_var:
                #     continue
                if 'SWE' in x_var:
                    spatial_dict = T.load_npy_dir(all_x_variables[x_var])
                    spatial_dict_new = {}
                    for pix in spatial_dict:
                        vals = spatial_dict[pix]
                        vals[vals < -999] = np.nan
                        spatial_dict_new[pix] = vals
                    spatial_dict = spatial_dict_new
                else:
                    spatial_dict = T.load_npy(all_x_variables[x_var])
                year_list = []
                all_vals_list = []
                all_pix_list = []
                for pix in tqdm(selected_pix_list, desc=x_var):
                    if not pix in spatial_dict:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        continue
                    x_val = spatial_dict[pix]
                    if x_var in ['current_during_late_CO2',
                                 'current_during_late_PAR',
                                 'current_during_early_peak_MODIS_LAI',
                                 'current_SWE',
                                 'pre_during_early_peak_late_SPEI3',
                                 'current_during_early_peak_late_SPI3',
                                 ]:
                        x_val = x_val[1:]
                    elif x_var in [

                    ]:
                        x_val = list(x_val)
                        x_val = x_val[1:]
                        x_val.append(np.nan)
                        x_val = np.array(x_val)
                    elif x_var in [

                    ]:
                        x_val = x_val[1:]
                    elif x_var in [
                        'current_early_start',
                    ]:
                        x_val = x_val
                    else:
                        continue
                    if not len(x_val) == 19:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        # all_pix_list.append(pix)
                        continue
                    x_val = Pre_Process().z_score(x_val)

                    if type(x_val) == int:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        continue
                    x_val = T.detrend_vals(x_val)
                    for i, val in enumerate(x_val):
                        year = self.start_year + i
                        year_list.append(year)
                        all_vals_list.append(val)
                        all_pix_list.append(pix)
                data_dict[x_var] = all_vals_list
                data_dict['year'] = year_list
                data_dict['pix'] = all_pix_list
            all_vals_list = []
            for pix in tqdm(selected_pix_list):
                x_val = y_spatial_dict[pix]
                x_val = Pre_Process().z_score(x_val)
                x_val = T.detrend_vals(x_val)
                for i, val in enumerate(x_val):
                    all_vals_list.append(val)
            data_dict['late_MODIS_LAI'] = all_vals_list
            df_out = pd.DataFrame(data_dict)
            df_out = self.add_MAT_MAP(df_out)
            df_out = self.add_sand_fraction(df_out)
            # df_out = df_out.dropna()
            # T.print_head_n(df_out, 10)
            T.save_df(df_out, outf)
            T.df_to_excel(df_out, outf)
        pass

    def not_detrend(self):
        outdir = join(self.this_class_arr,'not_detrend')
        T.mkdir(outdir, force=True)
        # x_list_current_year_fpath, x_list_previous_year_fpath = All_variables_single_corr().x_variables()
        all_x_variables = All_variables_single_corr().all_x_variables()
        print(all_x_variables.keys())
        # exit()
        y_spatial_dict = All_variables_single_corr().y_variable()
        df = T.spatial_dics_to_df({'y': y_spatial_dict})
        df = Main_flow_2.Dataframe_func(df).df
        limited_area_list = T.get_df_unique_val_list(df, 'limited_area')
        limited_area_pix_dict = {}
        for ltd in limited_area_list:
            df_ltd = df[df['limited_area'] == ltd]
            pix_list = df_ltd['pix'].values.tolist()
            pix_list = set(pix_list)
            limited_area_pix_dict[ltd] = pix_list
        for ltd in limited_area_pix_dict:
            print(ltd)
            outf = join(outdir, f'{ltd}.df')
            selected_pix_list = limited_area_pix_dict[ltd]
            data_dict = {}
            max_length = 0
            for x_var in all_x_variables:
                # if 'CO2' in x_var:
                #     continue
                if 'SWE' in x_var:
                    spatial_dict = T.load_npy_dir(all_x_variables[x_var])
                    spatial_dict_new = {}
                    for pix in spatial_dict:
                        vals = spatial_dict[pix]
                        vals[vals < -999] = np.nan
                        spatial_dict_new[pix] = vals
                    spatial_dict = spatial_dict_new
                else:
                    spatial_dict = T.load_npy(all_x_variables[x_var])
                year_list = []
                all_vals_list = []
                all_pix_list = []
                for pix in tqdm(selected_pix_list, desc=x_var):
                    if not pix in spatial_dict:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        continue
                    x_val = spatial_dict[pix]
                    if x_var in ['current_during_late_CO2',
                                 'current_during_late_PAR',
                                 'current_during_late_Temp',
                                 'current_SWE',
                                 'current_during_early_peak_late_SPEI3',
                                 'current_during_early_peak_late_SPI3',
                                 ]:
                        x_val = x_val[1:]
                    elif x_var in [

                    ]:
                        x_val = list(x_val)
                        x_val = x_val[1:]
                        x_val.append(np.nan)
                        x_val = np.array(x_val)
                    elif x_var in [

                    ]:
                        x_val = x_val[1:]
                    elif x_var in [
                        'current_early_start','current_during_early_peak_MODIS_LAI',
                    ]:
                        x_val = x_val
                    else:
                        continue
                    if not len(x_val) == 19:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        # all_pix_list.append(pix)
                        continue
                    x_val = Pre_Process().z_score(x_val)

                    if type(x_val) == int:
                        year_list.extend(list(range(self.start_year, 2019)))
                        all_vals_list.extend([np.nan] * 19)
                        continue
                    for i, val in enumerate(x_val):
                        year = self.start_year + i
                        year_list.append(year)
                        all_vals_list.append(val)
                        all_pix_list.append(pix)
                data_dict[x_var] = all_vals_list
                data_dict['year'] = year_list
                if len(all_pix_list) >= max_length:
                    data_dict['pix'] = all_pix_list
                    max_length = len(all_pix_list)
            all_vals_list = []
            for pix in tqdm(selected_pix_list):
                x_val = y_spatial_dict[pix]
                x_val = Pre_Process().z_score(x_val)
                for i, val in enumerate(x_val):
                    all_vals_list.append(val)
            data_dict['late_MODIS_LAI'] = all_vals_list
            data_dict_new = {}
            for var_i in data_dict:
                print(var_i,len(data_dict[var_i]))
                vals = data_dict[var_i]
                if not len(vals) >= max_length:
                    continue
                data_dict_new[var_i] = vals
            # exit()
            df_out = pd.DataFrame(data_dict_new)
            T.print_head_n(df_out, 10)
            df_out = self.add_MAT_MAP(df_out)
            df_out = self.add_sand_fraction(df_out)
            # df_out = df_out.dropna()
            # T.print_head_n(df_out, 10)
            T.save_df(df_out, outf)
            T.df_to_excel(df_out, outf)


class RF_per_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF_per_value',
                                                                                       result_root_this_script)
        pass

    def run(self):

        # self.run_RF(is_detrend=True)
        # self.run_RF(is_detrend=False)
        # self.plot_RF_results(is_detrend=True)
        self.plot_RF_results(is_detrend=False)

    def run_RF(self,is_detrend=True):
        x_var_list = self.x_variables(is_detrend)
        y_var = self.y_variable()
        if is_detrend:
            df_dir = join(Dataframe_per_value().this_class_arr,'detrend')
            outdir = join(self.this_class_arr, 'detrend')
        else:
            df_dir = join(Dataframe_per_value().this_class_arr, 'not_detrend')
            outdir = join(self.this_class_arr,'not_detrend')
        T.mkdir(outdir)
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            fpath = join(df_dir,f)
            df = T.load_df(fpath)
            X = df[x_var_list]
            Y = df[y_var]
            # T.print_head_n(Y,10)
            # exit()
            # print(outf)
            # plt.figure()
            # DIC_and_TIF().plot_df_spatial_pix(df_HI,land_tif)
            # plt.title(f'{year_range[0]}-{year_range[1]}_{HI_reclass}')
            result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var)
            outf = join(outdir,f'{f}')
            T.save_npy(result,outf)


    def plot_RF_results(self,is_detrend=True):
        y = self.y_variable()
        if is_detrend:
            fdir = join(self.this_class_arr,f'detrend')
            outdir = join(self.this_class_png,f'detrend')
        else:
            fdir = join(self.this_class_arr,f'not_detrend')
            outdir = join(self.this_class_png,f'not_detrend')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        x_list = self.x_variables(is_detrend)
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
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf)
            plt.close()
        # exit()
        pass

    def y_variable(self):
        return 'late_MODIS_LAI'

    def x_variables(self,is_detrend):
        if is_detrend == True:
            x_list = [
                'current_early_start',
                'current_early_start',
                'current_early_start',
                'current_early_start',
                'current_early_start',
                'current_early_start',
            ]
        else:
            x_list = [
                'current_early_start',
                'current_during_early_peak_late_SPEI3',
                'current_during_late_Temp',
                'current_during_late_PAR',
                'current_during_late_CO2',
                'current_during_early_peak_late_SPI3',
                'current_SWE',
                'current_during_early_peak_MODIS_LAI',
                'MAT',
                'MAP',
                'sand',
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

class RF:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF',
                                                                                       result_root_this_script)
        # self.year_range_list = ((1982, 2000), (2001, 2018), (1982, 2018))
        self.year_range_list = [(1982, 2018),]
        pass

    def run(self):
        dff = Dataframe().dff
        df = T.load_df(dff)

        self.gen_new_df(df)
        self.run_RF()
        self.plot_RF_results()

    def run_RF(self):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        df_dir = join(self.this_class_arr,f'new_df_{y_var}')
        outdir = join(self.this_class_arr,f'RF_results_{y_var}')
        T.mkdir(outdir)

        HI_reclass_list = ['Humid','Dry']
        T_trend_list = [-1,1]
        year_range_list = self.year_range_list
        for year_range in year_range_list:
            fpath = join(df_dir,f'{year_range[0]}-{year_range[1]}.df')
            df = T.load_df(fpath)
            # xx = df['photo_period'].tolist()
            # yy = df['eos_anomaly'].tolist()
            # KDE_plot().plot_scatter_hex(xx, yy, ylim=(-20, 20), xlim=(1500, 4000))
            # plt.show()
            for HI_reclass in HI_reclass_list:
                df_HI = df[df['HI_reclass']==HI_reclass]
                for T_trend in T_trend_list:
                    df_T_trend = df_HI[df_HI['after_2003_T_late_trend']==T_trend]
                    X = df_T_trend[x_var_list]
                    Y = df_T_trend[y_var]
                    outf = join(outdir,f'{year_range[0]}-{year_range[1]}_{HI_reclass}_{T_trend}.npy')
                    # print(outf)
                    # plt.figure()
                    # DIC_and_TIF().plot_df_spatial_pix(df_HI,land_tif)
                    # plt.title(f'{year_range[0]}-{year_range[1]}_{HI_reclass}')
                    result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var,isplot=False)
                    T.save_npy(result,outf)

    def run_RF1(self):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        df_dir = join(self.this_class_arr,f'new_df_{y_var}')
        outdir = join(self.this_class_arr,f'RF_results_{y_var}')
        T.mkdir(outdir)

        HI_reclass_list = ['Humid','Dry']
        year_range_list = self.year_range_list
        for year_range in year_range_list:
            fpath = join(df_dir,f'{year_range[0]}-{year_range[1]}.df')
            df = T.load_df(fpath)
            for HI_reclass in HI_reclass_list:
                df_HI = df[df['HI_reclass']==HI_reclass]
                X = df_HI[x_var_list]
                Y = df_HI[y_var]
                outf = join(outdir,f'{year_range[0]}-{year_range[1]}_{HI_reclass}.npy')
                # print(outf)
                # plt.figure()
                # DIC_and_TIF().plot_df_spatial_pix(df_HI,land_tif)
                # plt.title(f'{year_range[0]}-{year_range[1]}_{HI_reclass}')
                result = self.train_classfication_permutation_importance(X,Y,x_var_list,y_var,isplot=False)
                T.save_npy(result,outf)

    def plot_RF_results(self):
        y = self.y_variable()
        fdir = join(self.this_class_arr,f'RF_results_{y}')
        outdir = join(self.this_class_png,f'plot_RF_results_{y}')
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
            outf = join(outdir,f'{f}.pdf')
            plt.savefig(outf)
            plt.close()
        # exit()
        pass

    def gen_new_df(self,df):

        x_variables = self.x_variables()
        y_variables = self.y_variable()
        outdir = join(self.this_class_arr, f'new_df_{y_variables}')
        T.mkdir(outdir)
        all_variables = list(x_variables) + [y_variables]
        year_range_list = self.year_range_list
        for year_range in year_range_list:
            outf = join(outdir,f'{year_range[0]}-{year_range[1]}.df')
            start_year = year_range[0]
            end_year = year_range[1]
            results_dict = {}
            results_dict = {'pix':[],'landcover_GLC':[],'lon':[],'lat':[],'year':[],'HI_reclass':[],'after_2003_T_late_trend':[]}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{year_range[0]}-{year_range[1]}'):
                pix = row['pix']
                landcover_GLC = row['landcover_GLC']
                lon = row['lon']
                lat = row['lat']
                HI_reclass = row['HI_reclass']
                after_2003_T_late_trend = row['after_2003_T_late_trend']
                # result_dict_i = {}
                for j in range(start_year-1982,end_year-1982+1):
                    results_dict['pix'].append(pix)
                    results_dict['landcover_GLC'].append(landcover_GLC)
                    results_dict['lon'].append(lon)
                    results_dict['lat'].append(lat)
                    results_dict['HI_reclass'].append(HI_reclass)
                    results_dict['after_2003_T_late_trend'].append(after_2003_T_late_trend)
                    results_dict['year'].append(j+1982)
                for var_i in all_variables:
                    vals = row[var_i]
                    try:
                        if not len(vals) == (2018 - 1982 + 1):
                            vals_new = [np.nan] * (end_year - start_year + 1)
                        else:
                            vals_new = vals[start_year-1982:end_year-1982+1]
                            vals_new = T.detrend_vals(vals_new) # detrend
                    except:
                        vals_new = [np.nan] * (end_year-start_year+1)
                    # plt.plot(vals_new,label=var_i)
                    if not var_i in results_dict:
                        results_dict[var_i] = []
                    for val in vals_new:
                        results_dict[var_i].append(val)
                # plt.legend()
                # plt.title(f'{pix}')
                # plt.show()
            df_new = pd.DataFrame(results_dict)
            T.save_df(df_new,outf)
            T.df_to_excel(df_new,outf)

    def y_variable(self):
        # return 'LAI3g_late'
        return 'eos_anomaly'

    def x_variables(self):
        # x_list = ['LAI3g_earlier','eos_anomaly','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late','CO2_late','PAR_late']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period','eos_anomaly']
        x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period']
        # x_list = ['LAI3g_earlier','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','PAR_late','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
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


    def train_classfication_permutation_importance(self,X_input,Y_input,x_list,y,isplot=False):
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


class Partial_corr:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Partial_corr',
                                                                                       result_root_this_script)
        # self.year_range_list = ((1982, 2000), (2001, 2018), (1982, 2018))
        self.year_range_list = [(1982, 2018), ]

    def run(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        self.gen_new_df(df)
        self.run_p_corr()
        pass


    def y_variable(self):
        return 'LAI3g_late'
        # return 'eos_anomaly'
    def x_variables(self):
        # x_list = ['LAI3g_earlier','eos_anomaly','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late','CO2_late','PAR_late']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period','eos_anomaly']
        x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period']
        # x_list = ['LAI3g_earlier','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','PAR_late','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        return x_list

    def run_p_corr(self):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        df_dir = join(self.this_class_arr,f'new_df_{y_var}')
        outdir = join(self.this_class_arr,f'p_corr_results_{y_var}')
        T.mkdir(outdir)

        # HI_reclass_list = ['Humid','Dry']
        HI_reclass_list = ['Non-sig','energy-limited','water-limited']
        # T_trend_list = [-1,1]
        year_range_list = self.year_range_list
        for year_range in year_range_list:
            fpath = join(df_dir,f'{year_range[0]}-{year_range[1]}.df')
            df = T.load_df(fpath)
            for HI_reclass in HI_reclass_list:
                # df_HI = df[df['HI_reclass']==HI_reclass]
                df_HI = df[df['limited_area']==HI_reclass]
                X = df_HI[x_var_list]
                Y = df_HI[y_var]
                outf = join(outdir,f'{year_range[0]}-{year_range[1]}_{HI_reclass}.npy')
                # print(outf)
                # plt.figure()
                # DIC_and_TIF().plot_df_spatial_pix(df_HI,land_tif)
                # plt.title(f'{year_range[0]}-{year_range[1]}_{HI_reclass}')
                partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_HI,x_var_list,y_var)
                print(partial_correlation)
                print(partial_correlation_p_value)
                plt.figure()
                plt.barh(x_var_list,[partial_correlation[v] for v in x_var_list])
                plt.title(f'{year_range[0]}-{year_range[1]}_{HI_reclass}')
                plt.tight_layout()
            plt.show()

    def gen_new_df(self,df):

        x_variables = self.x_variables()
        y_variables = self.y_variable()
        outdir = join(self.this_class_arr, f'new_df_{y_variables}')
        T.mkdir(outdir)
        all_variables = list(x_variables) + [y_variables]
        year_range_list = self.year_range_list
        for year_range in year_range_list:
            outf = join(outdir,f'{year_range[0]}-{year_range[1]}.df')
            start_year = year_range[0]
            end_year = year_range[1]
            results_dict = {}
            results_dict = {'pix':[],'landcover_GLC':[],'lon':[],'lat':[],'year':[],'HI_reclass':[],'after_2003_T_late_trend':[],
                            'limited_area':[]}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{year_range[0]}-{year_range[1]}'):
                # print(row)
                # exit()
                pix = row['pix']
                landcover_GLC = row['landcover_GLC']
                lon = row['lon']
                lat = row['lat']
                HI_reclass = row['HI_reclass']
                limited_area = row['limited_area']
                after_2003_T_late_trend = row['after_2003_T_late_trend']
                # result_dict_i = {}
                for j in range(start_year-1982,end_year-1982+1):
                    results_dict['pix'].append(pix)
                    results_dict['landcover_GLC'].append(landcover_GLC)
                    results_dict['lon'].append(lon)
                    results_dict['lat'].append(lat)
                    results_dict['HI_reclass'].append(HI_reclass)
                    results_dict['after_2003_T_late_trend'].append(after_2003_T_late_trend)
                    results_dict['limited_area'].append(limited_area)
                    results_dict['year'].append(j+1982)
                for var_i in all_variables:
                    vals = row[var_i]
                    try:
                        if not len(vals) == (2018 - 1982 + 1):
                            vals_new = [np.nan] * (end_year - start_year + 1)
                        else:
                            vals_new = vals[start_year-1982:end_year-1982+1]
                            vals_new = T.detrend_vals(vals_new) # detrend
                    except:
                        vals_new = [np.nan] * (end_year-start_year+1)
                    # plt.plot(vals_new,label=var_i)
                    if not var_i in results_dict:
                        results_dict[var_i] = []
                    for val in vals_new:
                        results_dict[var_i].append(val)
                # plt.legend()
                # plt.title(f'{pix}')
                # plt.show()
            df_new = pd.DataFrame(results_dict)
            T.save_df(df_new,outf)
            T.df_to_excel(df_new,outf)

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

class Partial_corr_per_pix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Partial_corr_per_pix',
                                                                                       result_root_this_script)
        # self.year_range_list = ((1982, 2000), (2001, 2018), (1982, 2018))
        self.year_range_list = [(1982, 2018), ]

    def run(self):
        dff = Dataframe().dff
        # df = T.load_df(dff)
        # self.run_p_corr(df)
        self.plot_p_corr()
        pass

    def y_variable(self):
        return 'LAI3g_late'
        # return 'eos_anomaly'
    def x_variables(self):
        x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late']
        return x_list

    def run_p_corr(self,df):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        outdir = join(self.this_class_arr,f'p_corr_results_{y_var}')
        T.mkdir(outdir)

        p_corr_result = {}
        p_corr_result_p_value = {}
        for i,row in tqdm(df.iterrows(),total=len(df),desc='p_corr'):
            df_new = pd.DataFrame()
            for x in x_var_list:
                df_new[x] = row[x]
            df_new[y_var] = row[y_var]
            try:
                partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_new,x_var_list,y_var)
                p_corr_result[row['pix']] = partial_correlation
                p_corr_result_p_value[row['pix']] = partial_correlation_p_value
                # print(partial_correlation)
                # print(partial_correlation_p_value)
                # plt.figure()
                # plt.barh(x_var_list,[partial_correlation[v] for v in x_var_list])
                # plt.tight_layout()
                # plt.show()
            except:
                continue
                # print(df_new)
                # pause()
        df_p_corr = T.dic_to_df(p_corr_result, 'pix')
        df_p_corr_p_value = T.dic_to_df(p_corr_result_p_value, 'pix')
        T.save_df(df_p_corr, join(outdir, 'p_corr_results.df'))
        T.save_df(df_p_corr_p_value, join(outdir, 'p_corr_results_p_value.df'))
        T.df_to_excel(df_p_corr, join(outdir, 'p_corr_results.xlsx'))
        T.df_to_excel(df_p_corr_p_value, join(outdir, 'p_corr_results_p_value.xlsx'))


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

    def plot_p_corr(self):
        y = self.y_variable()
        x_list = self.x_variables()
        outdir = join(self.this_class_tif, f'p_corr_results_{y}')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        fdir = join(self.this_class_arr, f'p_corr_results_{y}')
        f = join(fdir, 'p_corr_results.df')
        df = T.load_df(f)
        for x in x_list:
            print(x)
            dict_i = T.df_to_spatial_dic(df, x)
            outf = join(outdir, f'{x}.tif')
            DIC_and_TIF().pix_dic_to_tif(dict_i, outf)
        pass


class RF_per_pix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF_per_pix',
                                                                                       result_root_this_script)
        # self.year_range_list = ((1982, 2000), (2001, 2018), (1982, 2018))
        self.year_range_list = [(1982, 2018), ]

    def run(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        self.run_RF(df)
        self.plot_RF_result()
        pass


    def y_variable(self):
        return 'LAI3g_late'
        # return 'eos_anomaly'
    def x_variables(self):
        # x_list = ['LAI3g_earlier','eos_anomaly','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late','CO2_late','PAR_late']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period','eos_anomaly']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period']
        x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','CO2_late']
        # x_list = ['LAI3g_earlier','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','PAR_late','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        return x_list

    def run_RF(self,df):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        outdir = join(self.this_class_arr,f'p_corr_results_{y_var}')
        T.mkdir(outdir)

        RF_results_all = {}
        for i,row in tqdm(df.iterrows(),total=len(df),desc='RF'):
            df_new = pd.DataFrame()
            for x in x_var_list:
                df_new[x] = row[x]
            df_new[y_var] = row[y_var]
            # T.print_head_n(df_new, 5)
            RF_results = self.train_classfication_permutation_importance(df_new,x_var_list,y_var)
            RF_results_all[row['pix']] = RF_results
        T.save_npy(RF_results_all, join(outdir, 'RF_results.npy'))
        # df_RF_results = T.dic_to_df(RF_results_all, 'pix')
        # T.save_df(df_RF_results, join(outdir, 'RF_results.df'))
        # T.df_to_excel(df_RF_results, join(outdir, 'RF_results.xlsx'))


    def train_classfication_permutation_importance(self,df,x_list,y):
        rf = RandomForestRegressor()
        df = df.dropna()
        if len(df) == 0:
            return np.nan
        X = df[x_list]
        Y = df[y]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        rf.fit(X_train,Y_train)
        # print('training finished')
        score = rf.score(X_test,Y_test)
        coef = rf.feature_importances_

        pred=rf.predict(X_test)
        r = stats.pearsonr(pred,Y_test)[0]

        # print('permuation importance...')
        # result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
        #                                 random_state=42)
        result = {}
        imp_dict = dict(zip(x_list, coef))
        for key in imp_dict:
            result[key+'-importance'] = imp_dict[key]
        # result['importances_mean'] = imp_dict
        # print('permuation importance finished')
        x_list_dict = dict(zip(x_list, list(range(len(x_list)))))
        result['score'] = score
        result['r'] = r

        return result

    def plot_RF_result(self):
        outdir = join(self.this_class_tif, f'RF_results')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        fpath = join(self.this_class_arr, f'p_corr_results_{y_var}', 'RF_results.npy')
        result_dict = T.load_npy(fpath)
        for x in x_var_list:
            spatial_dict = {}
            for pix in result_dict:
                dict_i = result_dict[pix]
                if type(dict_i) == float:
                    continue
                val = dict_i[x+'-importance']
                spatial_dict[pix] = val
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, join(outdir, f'{x}.tif'))


class Correlation_per_pix:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Correlation_per_pix',
                                                                                       result_root_this_script)
        pass

    def run(self):
        dff = Dataframe().dff
        df = T.load_df(dff)
        self.run_corr(df)
        self.max_correlation()
        # self.plot_RF_result()
        pass

    def y_variable(self):
        return 'LAI3g_late'
        # return 'eos_anomaly'

    def x_variables(self):
        # x_list = ['LAI3g_earlier','eos_anomaly','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late','CO2_late','PAR_late']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period','eos_anomaly']
        # x_list = ['LAI3g_earlier','Temp_late','SPEI3_peak','SPEI3_late','VPD_late','PAR_late','photo_period']
        x_list = ['LAI3g_earlier', 'Temp_late', 'SPEI3_peak', 'SPEI3_late', 'VPD_late', 'PAR_late', 'CO2_late']
        # x_list = ['LAI3g_earlier','Temp_peak','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        # x_list = ['LAI3g_earlier','PAR_late','Temp_late','CCI_SM_peak','CCI_SM_late','VPD_late']
        return x_list

    def run_corr(self, df,isdtrend=True):
        x_var_list = self.x_variables()
        y_var = self.y_variable()
        outdir = join(self.this_class_arr, f'correlation_{y_var}','detrend' if isdtrend else 'not_detrend')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)

        RF_results_all = {}
        for x in x_var_list:
            spatial_dict = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=x):
                pix = row['pix']
                xvals = row[x]
                yvals = row[y_var]

                if type(xvals) == float:
                    continue
                if type(yvals) == float:
                    continue
                if isdtrend:
                    xvals = T.detrend_vals(xvals)
                    yvals = T.detrend_vals(yvals)
                a,b,r,p = T.nan_line_fit(xvals,yvals)
                spatial_dict[pix] = r
            if isdtrend:
                outf = join(outdir, f'detrend_{x}.tif')
            else:
                outf = join(outdir, f'{x}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict, outf)


    def max_correlation(self,isdetrend=True):
        fdir = join(self.this_class_arr, f'correlation_{self.y_variable()}', 'detrend' if isdetrend else 'not_detrend')
        outdir = join(self.this_class_arr, f'max_correlation_{self.y_variable()}', 'detrend' if isdetrend else 'not_detrend')
        T.mkdir(outdir, force=True)
        # T.open_path_and_file(outdir)
        all_dict = {}
        x_list = []
        for f in T.listdir(fdir):
            var_name = f.split('.')[0]
            dict_i = DIC_and_TIF().spatial_tif_to_dic(join(fdir,f))
            all_dict[var_name] = dict_i
            x_list.append(var_name)
        df = T.spatial_dics_to_df(all_dict)
        var_name_dict = {}
        flag = 1
        for x in x_list:
            var_name_dict[x] = flag
            flag += 1
        print(var_name_dict)
        # max var name
        max_var_name_dict = {}
        max_val_dict = {}
        for i in df.iterrows():
            row = i[1]
            pix = row['pix']
            max_val = -999
            max_var = ''
            for x in x_list:
                val = row[x]
                val = abs(val)
                if val > max_val:
                    max_val = val
                    max_var = x
            max_var_name_dict[pix] = var_name_dict[max_var]
            max_val_dict[pix] = max_val
        if isdetrend:
            outf_max_var = join(outdir, f'detrend_max_var.tif')
            outf_max_val = join(outdir, f'detrend_max_val.tif')
        else:
            outf_max_var = join(outdir, f'max_var.tif')
            outf_max_val = join(outdir, f'max_val.tif')
        DIC_and_TIF().pix_dic_to_tif(max_var_name_dict, outf_max_var)
        DIC_and_TIF().pix_dic_to_tif(max_val_dict, outf_max_val)



        pass


class Test_wen_data:

    def __init__(self):
        self.datadir = '/Users/liyang/Desktop/1982-2020_'
        pass

    def run(self):
        self.to_tif()
        # self.window_pdf()
        # self.window_step_pdf()
        pass
    def window_pdf(self):

        # fdir = join(self.datadir, 'tif','during_late_LAI3g_15_trend')
        fdir = join(self.datadir, 'tif','during_late_LAI3g_15_detrend')
        # outdir = join(self.datadir,'png','during_late_LAI3g_15_trend')
        outdir = join(self.datadir,'png','during_late_LAI3g_15_detrend')
        T.mkdir(outdir,force=True)
        folder_list = T.listdir(fdir)
        file_list = []
        for folder in folder_list:
            window = folder
            fdir_i = join(fdir, folder)
            file_list = []
            for f in T.listdir(fdir_i):
                if f.endswith('.tif'):
                    # fname = f.split('.')[0]
                    fname = f.replace('00_','')
                    file_list.append(fname)
            break
        for f in file_list:
            window_list = []
            dict_all = {}
            for folder in tqdm(folder_list, desc=f):
                fdir_i = join(fdir, folder)
                fpath = join(fdir_i, f'{folder}_{f}')
                dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
                dict_all[int(folder)] = dict_i
                window_list.append(int(folder))
            df = T.spatial_dics_to_df(dict_all)
            df = Main_flow_2.Dataframe_func(df).df
            limited_area_list = T.get_df_unique_val_list(df, 'limited_area')
            for ltd in limited_area_list:
                df_ltd = df[df['limited_area'] == ltd]
                color_list = T.gen_colors(len(window_list))
                plt.figure(figsize=(4, 4))
                for window in window_list:
                    vals = df_ltd[window].values
                    x, y = Plot().plot_hist_smooth(vals, alpha=0,bins=80)
                    plt.plot(x, y, label=f'{window}', color=color_list[window])
                plt.title(f'{ltd}\n{f}')
                plt.legend()
                outf = join(outdir, f'{ltd}_{f}.pdf')
                plt.savefig(outf)
                plt.close()

    def window_step_pdf(self):
        window_bin = np.linspace(0, 22, 11)
        print(window_bin)

        # exit()
        fdir = join(self.datadir, 'tif','during_late_LAI3g_15_detrend')
        folder_list = T.listdir(fdir)
        file_list = []
        for folder in folder_list:
            window = folder
            fdir_i = join(fdir, folder)
            file_list = []
            for f in T.listdir(fdir_i):
                if f.endswith('.tif'):
                    # fname = f.split('.')[0]
                    fname = f.replace('00_','')
                    file_list.append(fname)
            break
        for f in file_list:
            pix_list = []
            val_list = []
            window_list = []
            for folder in tqdm(folder_list, desc=f):
                fdir_i = join(fdir, folder)
                fpath = join(fdir_i, f'{folder}_{f}')
                dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
                for pix in dict_i:
                    val = dict_i[pix]
                    if np.isnan(val):
                        continue
                    pix_list.append(pix)
                    val_list.append(val)
                    window_list.append(int(folder))
            df = pd.DataFrame()
            df['pix'] = pix_list
            df['val'] = val_list
            df['window'] = window_list
            df = Main_flow_2.Dataframe_func(df).df
            limited_area_list = T.get_df_unique_val_list(df,'limited_area')
            for ltd in limited_area_list:
                df_ltd = df[df['limited_area'] == ltd]
                pdf_colname = 'val'
                bin_colname = 'window'
                bins = window_bin
                Plot().multi_step_pdf(df_ltd, pdf_colname ,bin_colname, bins)
                plt.title(f'{ltd} {f.replace(".tif","")}')
        plt.show()

    def to_tif(self):
        # fdir = '/Users/liyang/Desktop/1982-2020_/during_late_LAI3g_15_detrend'
        fdir = '/Users/liyang/Desktop/1982-2020_/during_late_LAI3g_15_trend'
        # outdir ='/Users/liyang/Desktop/1982-2020_/tif/during_late_LAI3g_15_detrend'
        outdir ='/Users/liyang/Desktop/1982-2020_/tif/during_late_LAI3g_15_trend'
        T.mkdir(outdir,force=True)
        for f in T.listdir(fdir):
            if not f.endswith('correlation.npy'):
                continue
            print(f)
            dict_i = T.load_npy(join(fdir,f))
            df = T.dic_to_df(dict_i,'pix')
            cols = df.columns.tolist()
            cols.remove('pix')
            folder = f.split('.')[0].replace('_correlation','').replace('during_late_LAI3g_','')
            outdir_i = join(outdir,folder)
            T.mkdir(outdir_i,force=True)
            for col in cols:
                dict_j = T.df_to_spatial_dic(df,col)
                DIC_and_TIF().pix_dic_to_tif(dict_j,join(outdir_i,f'{folder}_{col}.tif'))
        pass




class Moving_window_p_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Moving_window_p_correlation',
                                                                                       result_root_this_script)
        pass

    def run(self):
        self.run_p_corr()
        # self.plot_P_corr()
        pass

    def run_p_corr(self):
        fdir = join(Dataframe_moving_window().this_class_arr,'moving_window')
        outdir_father = join(self.this_class_arr,'p_corr')
        period_list = ['early','peak','late']
        for period in period_list:
            x_var_list = self.x_variables(period)
            outdir = join(outdir_father,period)
            T.mkdir(outdir,force=True)
            y_var = f'{period}_LAI3g'
            for window in T.listdir(fdir):
                fpath = join(fdir,window,f'{window}.df')
                outdir_i = join(outdir,window)
                T.mkdir(outdir_i)
                outf_p_corr = join(outdir_i,f'{window}_corr.df')
                outf_p_corr_p_value = join(outdir_i,f'{window}_corr_p_value.df')
                df_i = T.load_df(fpath)

                p_corr_result = {}
                p_corr_result_p_value = {}
                for i,row in tqdm(df_i.iterrows(),total=len(df_i),desc=window):
                    df_new = pd.DataFrame()
                    for x in x_var_list:
                        x_vals = row[x]
                        df_new[x] = x_vals
                    y_vals = row[y_var]
                    df_new[y_var] = y_vals
                    try:
                        partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_new,x_var_list,y_var)
                        p_corr_result[row['pix']] = partial_correlation
                        p_corr_result_p_value[row['pix']] = partial_correlation_p_value
                        # print(partial_correlation)
                        # print(partial_correlation_p_value)
                        # plt.figure()
                        # plt.barh(x_var_list,[partial_correlation[v] for v in x_var_list])
                        # plt.tight_layout()
                        # plt.show()
                    except:
                        continue
                        # print(df_new)
                        # pause()
                df_p_corr = T.dic_to_df(p_corr_result, 'pix')
                df_p_corr_p_value = T.dic_to_df(p_corr_result_p_value, 'pix')
                T.save_df(df_p_corr, outf_p_corr)
                T.save_df(df_p_corr_p_value, outf_p_corr_p_value)
                T.df_to_excel(df_p_corr, join(outdir_i,f'{window}_corr.xlsx'))
                T.df_to_excel(df_p_corr_p_value, join(outdir_i,f'{window}_corr_p_value.xlsx'))

    def y_variable(self):
        return 'late_LAI3g'

    def x_variables(self,period_in):
        period_list = ['early', 'peak', 'late']
        y_fdir = join(Dataframe_moving_window().datadir, 'LAI3g')
        x_variables_dict = {}
        for period in period_list:
            x_fdir = join(Dataframe_moving_window().datadir, 'X', period)
            x_list = []
            for f in T.listdir(x_fdir):
                var_name = f.split('.')[0].replace('during_', '')
                fpath = join(x_fdir, f)
                x_list.append(var_name)
            x_variables_dict[period] = x_list
        return x_variables_dict[period_in]

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


    def plot_P_corr(self):
        outdir = join(self.this_class_png,'p_corr')
        # outdir = join(self.this_class_png,'p_corr_detrend')
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early', 'peak', 'late']
        fdir = join(self.this_class_arr,'p_corr')
        # fdir = join(self.this_class_arr,'p_corr_detrend')
        ltd_list = ('Non-sig', 'energy-limited', 'water-limited')
        for ltd in ltd_list:
            for period in period_list:
                x_list = self.x_variables(period)
                fdir_i = join(fdir,period)
                outdir_i = join(outdir,period)
                T.mkdir(outdir_i,force=True)
                for x in x_list:
                    window_number = len(T.listdir(fdir_i))
                    color_list = T.gen_colors(window_number)
                    plt.figure(figsize=(4, 4))
                    for window in T.listdir(fdir_i):
                        fpath = join(fdir_i,window,f'{window}_corr.df')
                        df = T.load_df(fpath)
                        df = Main_flow_2.Dataframe_func(df).df
                        df = df[df['limited_area']==ltd]
                        # limited_area_list = T.get_df_unique_val_list(df, 'limited_area')
                        # print(limited_area_list)
                        # exit()
                        # T.print_head_n(df)
                        # exit()
                        xvals = df[x].values
                        xx,yy = Plot().plot_hist_smooth(xvals,bins=80,alpha=0)
                        plt.plot(xx,yy,label=window,color=color_list[int(window)])
                    plt.title(f'{x}')
                    outf = join(outdir_i,f'{ltd}_{x}.pdf')
                    plt.tight_layout()
                    plt.savefig(outf)
                    plt.close()

class Moving_window_single_correlation:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Moving_window_single_correlation',
                                                                                       result_root_this_script)
        pass

    def run(self):
        # self.run_corr()
        # self.plot_corr()
        # self.moving_window_trend()
        self.over_all_corr()
        pass

    def run_corr(self):
        fdir = join(Dataframe_moving_window().this_class_arr,'moving_window')
        outdir_father = join(self.this_class_arr,'corr')
        period_list = ['early','peak','late']
        for period in period_list:
            x_var_list = self.x_variables(period)
            outdir = join(outdir_father,period)
            T.mkdir(outdir,force=True)
            y_var = f'{period}_LAI3g'
            for window in T.listdir(fdir):
                fpath = join(fdir,window,f'{window}.df')
                outdir_i = join(outdir,window)
                T.mkdir(outdir_i)
                outf_p_corr = join(outdir_i,f'{window}_corr.df')
                outf_p_corr_p_value = join(outdir_i,f'{window}_corr_p_value.df')
                df_i = T.load_df(fpath)
                corr_result = {}
                corr_result_p_value = {}
                for i,row in tqdm(df_i.iterrows(),total=len(df_i)):
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
                        corr_result_p_value_i[x] = p
                    corr_result[pix] = corr_result_i
                    corr_result_p_value[pix] = corr_result_p_value_i
                df_p_corr = T.dic_to_df(corr_result, 'pix')
                df_p_corr_p_value = T.dic_to_df(corr_result_p_value, 'pix')
                T.save_df(df_p_corr, outf_p_corr)
                T.save_df(df_p_corr_p_value, outf_p_corr_p_value)
                T.df_to_excel(df_p_corr, join(outdir_i, f'{window}_corr.xlsx'))
                T.df_to_excel(df_p_corr_p_value, join(outdir_i, f'{window}_corr_p_value.xlsx'))

    def y_variable(self):
        return 'late_LAI3g'

    def x_variables(self,period_in):
        period_list = ['early', 'peak', 'late']
        y_fdir = join(Dataframe_moving_window().datadir, 'LAI3g')
        x_variables_dict = {}
        for period in period_list:
            x_fdir = join(Dataframe_moving_window().datadir, 'X', period)
            x_list = []
            for f in T.listdir(x_fdir):
                var_name = f.split('.')[0].replace('during_', '')
                fpath = join(x_fdir, f)
                x_list.append(var_name)
            x_variables_dict[period] = x_list
        return x_variables_dict[period_in]

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
        outdir = join(self.this_class_png,'corr')
        # outdir = join(self.this_class_png,'p_corr_detrend')
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        period_list = ['early', 'peak', 'late']
        fdir = join(self.this_class_arr,'corr')
        # fdir = join(self.this_class_arr,'p_corr_detrend')
        # ltd_list = ('Non-sig', 'energy-limited', 'water-limited')
        ltd_list = ('energy-limited', 'water-limited')
        for ltd in ltd_list:
            for period in period_list:
                x_list = self.x_variables(period)
                fdir_i = join(fdir,period)
                outdir_i = join(outdir,period)
                T.mkdir(outdir_i,force=True)
                for x in x_list:
                    window_number = len(T.listdir(fdir_i))
                    color_list = T.gen_colors(window_number)
                    plt.figure(figsize=(4, 4))
                    for window in T.listdir(fdir_i):
                        fpath = join(fdir_i,window,f'{window}_corr.df')
                        df = T.load_df(fpath)
                        df = Main_flow_2.Dataframe_func(df).df
                        df = df[df['limited_area']==ltd]
                        # limited_area_list = T.get_df_unique_val_list(df, 'limited_area')
                        # print(limited_area_list)
                        # exit()
                        # T.print_head_n(df)
                        # exit()
                        xvals = df[x].values
                        xx,yy = Plot().plot_hist_smooth(xvals,bins=80,alpha=0)
                        plt.plot(xx,yy,label=window,color=color_list[int(window)])
                    plt.title(f'{x}')
                    outf = join(outdir_i,f'{ltd}_{x}.pdf')
                    plt.tight_layout()
                    plt.savefig(outf)
                    plt.close()

    def moving_window_trend(self):
        outdir = join(self.this_class_tif, 'moving_window_trend')
        T.mkdir(outdir, force=True)
        T.open_path_and_file(outdir)
        period_list = ['early', 'peak', 'late']
        fdir = join(self.this_class_arr, 'corr')
        for period in period_list:
            x_list = self.x_variables(period)
            fdir_i = join(fdir, period)
            window_number = len(T.listdir(fdir_i))
            outdir_i = join(outdir, period)
            T.mkdir(outdir_i, force=True)
            for x in tqdm(x_list,desc=f'{period}'):
                arrs = []
                for window in T.listdir(fdir_i):
                    fpath = join(fdir_i, window, f'{window}_corr.df')
                    df = T.load_df(fpath)
                    spatial_dict = T.df_to_spatial_dic(df, x)
                    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
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
                outf_trend = join(outdir_i, f'{x}.tif')
                outf_p = join(outdir_i, f'{x}_p.tif')
                DIC_and_TIF().pix_dic_to_tif(trend_spatial_dict, outf_trend)
                DIC_and_TIF().pix_dic_to_tif(trend_p_spatial_dict, outf_p)

    def over_all_corr(self):
        data_dir = '/Volumes/NVME2T/greening_project_redo/data/original_20221107/'
        outdir = join(self.this_class_tif, 'over_all_corr')
        T.mkdir(outdir, force=True)
        period_list = ['early', 'peak', 'late']
        y_var = self.y_variable()
        for period in period_list:
            outdir_i = join(outdir, period)
            T.mkdir(outdir_i, force=True)
            spatial_dict_y_f = join(data_dir,'LAI3g',f'during_{period}_LAI3g.npy')
            spatial_dict_y = T.load_npy(spatial_dict_y_f)
            x_var_list = self.x_variables(period)
            for x in x_var_list:
                outf = join(outdir_i, f'{period}_{x}.tif')
                spatial_dict_f = join(data_dir, f'X/{period}',f'during_{x}.npy')
                spatial_dict = T.load_npy(spatial_dict_f)
                corr_dict = {}
                for pix in tqdm(spatial_dict,desc=f'{period} {x}'):
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


    def bivariate_plot(self):

        pass


class Bivariate_plot:


    def __init__(self):
        pass

    def run(self):
        fdir1 = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/moving_window_trend'
        fdir2 = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/over_all_corr'
        outdir = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/bivariate_plot'
        T.mkdir(outdir, force=True)
        # period_list = ['early', 'peak', 'late']
        period_list = ['late']
        for period in period_list:
            fdir1_i = join(fdir1, period)
            fdir2_i = join(fdir2, period)
            outdir_i = join(outdir, period)
            T.mkdir(outdir_i, force=True)
            for x in T.listdir(fdir1_i):
                if not x.endswith('.tif'):
                    continue
                if x.endswith('_p.tif'):
                    continue
                title = x.split('.')[0]
                tif1 = join(fdir1_i, x)
                tif2 = join(fdir2_i, f'{period}_{x}')
                # print(tif2)
                # print(isfile(tif1), isfile(tif2))
                outf = join(outdir_i, x)
                x_label = 'moving window trend'
                y_label = 'long term correlation'
                min1 = -0.03
                max1 = 0.03
                min2 = -0.3
                max2 = 0.3
                self.plot_bivariate_map(tif1,tif2,x_label,y_label,min1,max1,min2,max2,outf,title)
        pass

    def plot_bivariate_map(self,tif1,tif2,x_label,y_label,min1,max1,min2,max2,outf,title=''):
        # x_label = 'moving_window_trend'
        # y_label = 'correlation'
        n = (5, 5)
        zcmap = self.gen_zcmap(n)
        arr1,originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tif1)
        arr2 = ToRaster().raster2array(tif2)[0]
        arr1[arr1<-999] = np.nan
        arr2[arr2<-999] = np.nan

        bin1 = np.linspace(min1,max1,n[0]+1)
        bin2 = np.linspace(min2,max2,n[1]+1)

        spatial_dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        spatial_dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_all = {'arr1':spatial_dict1,'arr2':spatial_dict2}
        df = T.spatial_dics_to_df(dict_all)

        blend_arr = []

        for r in range(len(arr1)):
            temp = []
            for c in range(len(arr1[0])):
                val1 = arr1[r][c]
                val2 = arr2[r][c]
                if val1 > max1:
                    val1 = max1
                if val1 < min1:
                    val1 = min1
                if val2 > max2:
                    val2 = max2
                if val2 < min2:
                    val2 = min2
                if np.isnan(val1) or np.isnan(val2):
                    temp.append(np.array([1,1,1,1]))
                    continue
                for i in range(len(bin1)-1):
                    if val1 >= bin1[i] and val1 <= bin1[i+1]:
                        for j in range(len(bin2)-1):
                            if val2 >= bin2[j] and val2 <= bin2[j+1]:
                                # print(zcmap[i][j])
                                color = zcmap[j][i] * 255
                                # print(color)
                                temp.append(color)
            temp = np.array(temp)
            blend_arr.append(temp)
        blend_arr = np.array(blend_arr)
        print(np.shape(blend_arr))
        # exit()
        # newRasterfn = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/test.tif'
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(outf)
        # define a projection and extent
        raster = gdal.Open(outf)
        geotransform = raster.GetGeoTransform()
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        raster.SetProjection(outRasterSRS.ExportToWkt())
        n_plot = (101,101)
        zcmap = self.gen_zcmap(n_plot)
        x_ticks = []
        y_ticks = []
        bin1 = np.linspace(min1, max1, n_plot[0] + 1)
        bin2 = np.linspace(min2, max2, n_plot[1] + 1)
        for i in range(len(bin1) - 1):
            for j in range(len(bin2) - 1):
                x_ticks.append((bin1[i] + bin1[i + 1]) / 2)
                y_ticks.append((bin2[j] + bin2[j + 1]) / 2)
        x_ticks = list(set(x_ticks))
        y_ticks = list(set(y_ticks))
        x_ticks.sort()
        y_ticks.sort()
        x_ticks = [round(x, 2) for x in x_ticks]
        y_ticks = [round(y, 2) for y in y_ticks]
        # x_ticks = x_ticks[::-1]
        # y_ticks = y_ticks[::-1]
        plt.imshow(zcmap)
        plt.xticks(list(range(len(x_ticks)))[::10],x_ticks[::10],rotation=90)
        plt.yticks(list(range(len(y_ticks)))[::10],y_ticks[::10])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outf.replace('.tif','.pdf'))

    def gen_zcmap(self,n):
        # corner_colors = ("#0058B7", '#FFFFFF', "#8F00FF", "#FF0000")
        corner_colors = ("#1D2D69", '#A1FF64', "#E9FF64", "#BD2ECC")
        # corner_colors = ("#DC9EC5", '#788AA9', "#F3F3F3", "#8CCBA4")
        # corner_colors = ("#FF0000", '#000000', "#FFFFFF", "#0058B7")
        zcmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
        # xcmap = sns.diverging_palette(180, 360,s=100,l=75, as_cmap=True)
        # ycmap = sns.diverging_palette(90, 270,s=100,l=75, as_cmap=True)
        # zcmap = xycmap.mean_xycmap(xcmap=xcmap, ycmap=ycmap, n=n)
        # plt.imshow(zcmap)
        # plt.show()
        # plt.close()
        return zcmap
        pass


class All_variables_single_corr:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'All_variables_single_corr',
            result_root_this_script)
        self.datadir = join(self.this_class_arr,'data')
        pass

    def run(self):
        # self.phenology_df_to_spatial_dict()
        self.phenology_df_to_spatial_dict_late_end()
        # self.current_year_corr()
        # self.pre_year_corr()
        # self.current_year_corr_tif()
        # self.pre_year_corr_tif()
        # self.corr_statistic()
        # self.check_old_corr()
        # self.MAT_MAP_matrix()
        pass

    def phenology_df_to_spatial_dict(self):
        dff = join(self.datadir,'Phenology_predictors/MODIS_LAI/phenology_dataframe.df')
        outf = join(self.datadir,'X/early_start.npy')
        df = T.load_df(dff)
        spatial_dict = T.df_to_spatial_dic(df,'early_start')
        spatial_dict_new = {}
        for pix in spatial_dict:
            dict_i = spatial_dict[pix]
            year_list = []
            for year in dict_i:
                if year > 2018:
                    continue
                year_list.append(year)
            year_list.sort()
            early_start_list = []
            for year in year_list:
                val = dict_i[year]
                early_start_list.append(val)
            early_start_list = np.array(early_start_list)
            spatial_dict_new[pix] = early_start_list
        T.save_npy(spatial_dict_new,outf)

    def phenology_df_to_spatial_dict_late_end(self):
        dff = join(self.datadir,'Phenology_predictors/MODIS_LAI/phenology_dataframe.df')
        outf = join(self.datadir,'X/late_end.npy')
        df = T.load_df(dff)
        spatial_dict = T.df_to_spatial_dic(df,'late_end')
        spatial_dict_new = {}
        for pix in spatial_dict:
            dict_i = spatial_dict[pix]
            year_list = []
            for year in dict_i:
                if year > 2018:
                    continue
                year_list.append(year)
            year_list.sort()
            early_start_list = []
            for year in year_list:
                val = dict_i[year]
                early_start_list.append(val)
            early_start_list = np.array(early_start_list)
            spatial_dict_new[pix] = early_start_list
        T.save_npy(spatial_dict_new,outf)

    def y_variable(self):
        y_var = 'during_late_MODIS_LAI'
        y_f = join(self.datadir, f'Y/{y_var}.npy')
        y_dict = T.load_npy(y_f)
        return y_dict

    def x_variables(self):

        x_list_current_year = [
            'early_start',
            'during_early_peak_late_SPEI3',
            'during_late_Temp',
            'during_late_VPD',
            'during_peak_SPI3',
            'during_late_PAR',
            'during_late_CO2',
            'during_late_SPI3',
            'during_early_peak_late_SPI3',
                  ]
        x_list_current_year_fpath = {}
        for x in x_list_current_year:
            x_list_current_year_fpath[x] = join(self.datadir, f'X/{x}.npy')
        x_list_current_year_fpath['SWE'] = join(data_root, 'GlobSnow_v3/SWE/per_pix_annual_1999-2018')
        x_list_current_year_fpath['during_early_peak_MODIS_LAI'] = join(self.datadir, f'Y/during_early_peak_MODIS_LAI.npy')

        # -------
        x_list_previous_year = [
            'early_start',
            'during_early_peak_late_SPEI3',
                      ]
        x_list_previous_year_fpath = {}
        for x in x_list_previous_year:
            x_list_previous_year_fpath[x] = join(self.datadir, f'X/{x}.npy')
        x_list_previous_year_fpath['during_early_peak_late_MODIS_LAI'] = join(self.datadir, f'Y/during_early_peak_late_MODIS_LAI.npy')
        return x_list_current_year_fpath, x_list_previous_year_fpath


    def all_x_variables(self):
        x_list_current_year_fpath, x_list_previous_year_fpath = self.x_variables()
        all_x_variables_fpath = {}
        for x in x_list_current_year_fpath:
            all_x_variables_fpath[f'current_{x}'] = x_list_current_year_fpath[x]
        for x in x_list_previous_year_fpath:
            all_x_variables_fpath[f'pre_{x}'] = x_list_previous_year_fpath[x]
        return all_x_variables_fpath

    def current_year_corr(self):
        outdir = join(self.this_class_arr,'current_year_corr')
        T.mkdir(outdir)
        outf = join(outdir,'corr.df')
        x_list_current_year_fpath, _ =self.x_variables()
        y_spatial_dict = self.y_variable()
        all_corr = {}
        for x_var in x_list_current_year_fpath:
            if x_var == 'SWE':
                spatial_dict = T.load_npy_dir(x_list_current_year_fpath[x_var])
                spatial_dict_new = {}
                for pix in spatial_dict:
                    vals = spatial_dict[pix]
                    vals[vals < -999] = np.nan
                    spatial_dict_new[pix] = vals
                spatial_dict = spatial_dict_new
            else:
                spatial_dict = T.load_npy(x_list_current_year_fpath[x_var])
            corr_dict = {}
            p_dict = {}
            for pix in tqdm(spatial_dict,desc=x_var):
                x_val = spatial_dict[pix]
                if not pix in y_spatial_dict:
                    continue
                if not x_var in ['early_start','during_early_peak_MODIS_LAI']:
                    x_val = x_val[1:]
                y_val = y_spatial_dict[pix]
                try:
                    x_val = T.detrend_vals(x_val)
                    y_val = T.detrend_vals(y_val)
                    r,p = T.nan_correlation(x_val,y_val)
                except:
                    print(x_var, len(x_val), len(y_val))
                    continue
                corr_dict[pix] = r
                p_dict[pix] = p
            key_corr = f'{x_var}_corr'
            key_p = f'{x_var}_p'
            all_corr[key_corr] = corr_dict
            all_corr[key_p] = p_dict
        df = T.spatial_dics_to_df(all_corr)
        T.print_head_n(df,10)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


    def pre_year_corr(self):
        outdir = join(self.this_class_arr,'pre_year_corr')
        T.mkdir(outdir)
        outf = join(outdir,'pre_corr.df')
        _, x_list_previous_year_fpath =self.x_variables()
        y_spatial_dict = self.y_variable()
        all_corr = {}
        for x_var in x_list_previous_year_fpath:
            spatial_dict = T.load_npy(x_list_previous_year_fpath[x_var])
            corr_dict = {}
            p_dict = {}
            for pix in tqdm(spatial_dict,desc=x_var):
                x_val = spatial_dict[pix]
                if not pix in y_spatial_dict:
                    continue
                y_val = y_spatial_dict[pix]
                if not x_var in ['early_start','during_early_peak_late_MODIS_LAI']:
                    x_val = x_val[:-1]
                else:
                    x_val = x_val[:-1]
                    y_val = y_val[1:]
                # print(x_var)
                # print(x_var)
                # print(len(x_val),len(y_val))
                # exit()
                try:
                    x_val = T.detrend_vals(x_val)
                    y_val = T.detrend_vals(y_val)
                    r,p = T.nan_correlation(x_val,y_val) # todo: bug
                except:
                    print(x_var,len(x_val),len(y_val))
                    continue
                corr_dict[pix] = r
                p_dict[pix] = p
            key_corr = f'{x_var}_corr'
            key_p = f'{x_var}_p'
            all_corr[key_corr] = corr_dict
            all_corr[key_p] = p_dict
        df = T.spatial_dics_to_df(all_corr)
        T.print_head_n(df,10)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass

    def current_year_corr_tif(self):
        outdir = join(self.this_class_tif,'current_year_corr')
        T.mkdir(outdir)
        dff = join(self.this_class_arr,'current_year_corr','corr.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        for col in cols:
            spatial_dict = T.df_to_spatial_dic(df,col)
            outf = join(outdir,f'{col}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def pre_year_corr_tif(self):
        outdir = join(self.this_class_tif,'pre_year_corr')
        T.mkdir(outdir)
        dff = join(self.this_class_arr,'pre_year_corr','pre_corr.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        for col in cols:
            spatial_dict = T.df_to_spatial_dic(df,col)
            outf = join(outdir,f'{col}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def corr_statistic(self):
        spatial_dict_all = {}
        fdir1 = join(self.this_class_tif,'current_year_corr')
        fdir2 = join(self.this_class_tif,'pre_year_corr')
        for f in T.listdir(fdir1):
            if not f.endswith('.tif'):
                continue
            if '_p.tif' in f:
                continue
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(join(fdir1,f))
            var_name = 'current_'+f.split('.')[0]
            spatial_dict_all[var_name] = spatial_dict
        for f in T.listdir(fdir2):
            if not f.endswith('.tif'):
                continue
            if '_p.tif' in f:
                continue
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(join(fdir2,f))
            var_name = 'pre_'+f.split('.')[0]
            spatial_dict_all[var_name] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        df = Main_flow_2.Dataframe_func(df).df
        limited_area_list = T.get_df_unique_val_list(df,'limited_area')
        variables_list = []
        for var_i in spatial_dict_all:
            variables_list.append(var_i)
        variables_list.sort()
        for ltd in limited_area_list:
            df_ltd = df[df['limited_area']==ltd]
            plt.figure(figsize=(4, 4))
            mean_list = []
            for var_i in variables_list:
                vals = df_ltd[var_i].tolist()
                vals_mean = np.nanmean(vals)
                mean_list.append(vals_mean)
            plt.barh(variables_list,mean_list)
            plt.title(f'{ltd}')
            plt.tight_layout()
        plt.show()

    def check_old_corr(self):
        f1 = '/Volumes/NVME2T/greening_project_redo/data/1982_2018_final_wen/during_late_LAI3g_zscore.npy'
        f2 = '/Volumes/NVME2T/greening_project_redo/data/lai3g_pheno/early_start.npy'
        dict1 = T.load_npy(f1)
        dict2 = T.load_npy(f2)
        spatial_dict = {}
        for pix in tqdm(dict1):
            if not pix in dict2:
                continue
            vals1 = dict1[pix]
            vals2 = dict2[pix]
            vals1 = T.detrend_vals(vals1)
            vals2 = T.detrend_vals(vals2)
            if len(vals2) == 0:
                continue
            # print(len(vals1),len(vals2))
            # exit()
            try:
                r,p = T.nan_correlation(dict1[pix],dict2[pix])
            except:
                continue
            spatial_dict[pix] = r
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='RdBu_r')
        plt.colorbar()
        plt.show()

    def add_MAT_MAP_to_df(self,df):
        MAT_f = join(data_root,'MAT_MAP','MAT.tif')
        MAP_f = join(data_root,'MAT_MAP','MAP.tif')

        dict_mat = DIC_and_TIF().spatial_tif_to_dic(MAT_f)
        dict_map = DIC_and_TIF().spatial_tif_to_dic(MAP_f)
        df = T.add_spatial_dic_to_df(df,dict_mat,'MAT')
        df = T.add_spatial_dic_to_df(df,dict_map,'MAP')
        return df


    def MAT_MAP_matrix(self):
        # tif = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/All_variables_single_corr/current_year_corr/during_early_peak_late_SPEI3_corr.tif'
        fdir = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/All_variables_single_corr/current_year_corr'
        outdir = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/All_variables_single_corr/current_year_corr_MAT_MAP'
        T.mkdir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if '_p.tif' in f:
                continue
            tif = join(fdir,f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(tif)
            df = T.spatial_dics_to_df({'corr':spatial_dict})
            df = self.add_MAT_MAP_to_df(df)
            df = Main_flow_2.Dataframe_func(df).df
            mat_bins = np.arange(-20,21,2)
            map_bins = np.arange(0,2100,100)
            mat_df_group,bins_list_str = T.df_bin(df,'MAT',mat_bins)
            matrix = []
            x_label = []
            y_label = []
            for name_mat,mat_df_group_i in mat_df_group:
                y_label.append(name_mat.left)
                map_df_group,bins_list_str = T.df_bin(mat_df_group_i,'MAP',map_bins)
                temp = []
                for name_map,map_df_group_i in map_df_group:
                    if not name_map.left in x_label:
                        x_label.append(name_map.left)
                    vals = map_df_group_i['corr'].tolist()
                    if len(vals) < 10:
                        temp.append(np.nan)
                        continue
                    mean = np.nanmean(vals)
                    temp.append(mean)
                matrix.append(temp)
            plt.figure(figsize=(8, 8))
            plt.imshow(matrix,vmin=-0.5,vmax=0.5,cmap='RdBu_r')
            plt.xticks(range(len(x_label)),x_label,rotation=90)
            plt.yticks(range(len(y_label)),y_label)
            plt.colorbar()
            plt.title(f)
            plt.tight_layout()
            outf = join(outdir,f+'.pdf')
            plt.savefig(outf)
            plt.close()


class All_variables_trend:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'All_variables_trend',
            result_root_this_script)
        self.datadir = join(All_variables_single_corr().this_class_arr,'data')
        pass

    def run(self):
        # self.phenology_df_to_spatial_dict()
        # self.current_year_trend()
        # self.pre_year_trend()
        # self.y_trend()
        # self.current_year_trend_tif()
        # self.pre_year_trend_tif()
        pass

    def phenology_df_to_spatial_dict(self):
        dff = join(self.datadir,'Phenology_predictors/MODIS_LAI/phenology_dataframe.df')
        outf = join(self.datadir,'X/early_start.npy')
        df = T.load_df(dff)
        spatial_dict = T.df_to_spatial_dic(df,'early_start')
        spatial_dict_new = {}
        for pix in spatial_dict:
            dict_i = spatial_dict[pix]
            year_list = []
            for year in dict_i:
                if year > 2018:
                    continue
                year_list.append(year)
            year_list.sort()
            early_start_list = []
            for year in year_list:
                val = dict_i[year]
                early_start_list.append(val)
            early_start_list = np.array(early_start_list)
            spatial_dict_new[pix] = early_start_list
        T.save_npy(spatial_dict_new,outf)

    def y_variable(self):
        y_var = 'during_late_MODIS_LAI'
        y_f = join(self.datadir, f'Y/{y_var}.npy')
        y_dict = T.load_npy(y_f)
        return y_dict

    def x_variables(self):

        x_list_current_year = [
            'early_start',
            'during_early_peak_late_SPEI3',
            'during_late_Temp',
            'during_late_VPD',
            'during_peak_SPI3',
            'during_late_PAR',
            'during_late_CO2',
            'during_late_SPI3',
                  ]
        x_list_current_year_fpath = {}
        for x in x_list_current_year:
            x_list_current_year_fpath[x] = join(self.datadir, f'X/{x}.npy')
        x_list_current_year_fpath['SWE'] = join(data_root, 'GlobSnow_v3/SWE/per_pix_annual_1999-2018')
        x_list_current_year_fpath['during_early_peak_MODIS_LAI'] = join(self.datadir, f'Y/during_early_peak_MODIS_LAI.npy')

        # -------
        x_list_previous_year = [
            'early_start',
            'during_early_peak_late_SPEI3',
                      ]
        x_list_previous_year_fpath = {}
        for x in x_list_previous_year:
            x_list_previous_year_fpath[x] = join(self.datadir, f'X/{x}.npy')
        x_list_previous_year_fpath['during_early_peak_late_MODIS_LAI'] = join(self.datadir, f'Y/during_early_peak_late_MODIS_LAI.npy')
        return x_list_current_year_fpath, x_list_previous_year_fpath


    def current_year_trend(self):
        outdir = join(self.this_class_arr,'current_year_trend')
        T.mkdir(outdir)
        outf = join(outdir,'trend.df')
        x_list_current_year_fpath, _ =self.x_variables()
        y_spatial_dict = self.y_variable()
        all_corr = {}
        for x_var in x_list_current_year_fpath:
            if x_var == 'SWE':
                spatial_dict = T.load_npy_dir(x_list_current_year_fpath[x_var])
                spatial_dict_new = {}
                for pix in spatial_dict:
                    vals = spatial_dict[pix]
                    vals[vals < -999] = np.nan
                    spatial_dict_new[pix] = vals
                spatial_dict = spatial_dict_new
            else:
                spatial_dict = T.load_npy(x_list_current_year_fpath[x_var])
            corr_dict = {}
            p_dict = {}
            for pix in tqdm(spatial_dict,desc=x_var):
                x_val = spatial_dict[pix]
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(x_val))),x_val)
                except:
                    a,p = np.nan,np.nan
                corr_dict[pix] = a
                p_dict[pix] = p
            key_corr = f'{x_var}_corr'
            key_p = f'{x_var}_p'
            all_corr[key_corr] = corr_dict
            all_corr[key_p] = p_dict
        df = T.spatial_dics_to_df(all_corr)
        T.print_head_n(df,10)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


    def pre_year_trend(self):
        outdir = join(self.this_class_arr,'pre_year_trend')
        T.mkdir(outdir)
        outf = join(outdir,'pre_trend.df')
        _, x_list_previous_year_fpath =self.x_variables()
        y_spatial_dict = self.y_variable()
        all_corr = {}
        for x_var in x_list_previous_year_fpath:
            spatial_dict = T.load_npy(x_list_previous_year_fpath[x_var])
            trend_dict = {}
            p_dict = {}
            for pix in tqdm(spatial_dict,desc=x_var):
                x_val = spatial_dict[pix]
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(x_val))),x_val)
                except:
                    a,p = np.nan,np.nan
                trend_dict[pix] = a
                p_dict[pix] = p
            key_corr = f'{x_var}_corr'
            key_p = f'{x_var}_p'
            all_corr[key_corr] = trend_dict
            all_corr[key_p] = p_dict
        df = T.spatial_dics_to_df(all_corr)
        T.print_head_n(df,10)
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def y_trend(self):
        y_spatial_dict = self.y_variable()
        outdir = join(self.this_class_tif,'y_trend')
        T.mkdir(outdir)
        trend_dict = {}
        p_dict = {}
        for pix in tqdm(y_spatial_dict):
            x_val = y_spatial_dict[pix]
            try:
                a, b, r, p = T.nan_line_fit(list(range(len(x_val))), x_val)
            except:
                a, p = np.nan, np.nan
            trend_dict[pix] = a
            p_dict[pix] = p
        outf_a = join(outdir, 'MODIS_trend.tif')
        outf_p = join(outdir, 'MODIS_p.tif')
        DIC_and_TIF().pix_dic_to_tif(trend_dict, outf_a)
        DIC_and_TIF().pix_dic_to_tif(p_dict, outf_p)

    def current_year_trend_tif(self):
        outdir = join(self.this_class_tif,'current_year_trend')
        T.mkdir(outdir)
        dff = join(self.this_class_arr,'current_year_trend','trend.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        for col in cols:
            spatial_dict = T.df_to_spatial_dic(df,col)
            outf = join(outdir,f'{col}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

    def pre_year_trend_tif(self):
        outdir = join(self.this_class_tif,'pre_year_trend')
        T.mkdir(outdir)
        dff = join(self.this_class_arr,'pre_year_trend','pre_trend.df')
        df = T.load_df(dff)
        cols = df.columns.tolist()
        cols.remove('pix')
        for col in cols:
            spatial_dict = T.df_to_spatial_dic(df,col)
            outf = join(outdir,f'{col}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict,outf)

def gen_ocean():
    arr = np.ones((360, 720))
    DIC_and_TIF().arr_to_tif(arr, join('/Volumes/NVME2T/greening_project_redo/conf', 'ocean.tif'))
    pass

def main():
    # Dataframe().run()
    # Dataframe_per_value().run()
    # RF_per_value().run()
    # RF().run()
    # Partial_corr().run()
    # Partial_corr_per_pix().run()
    # RF_per_pix().run()
    # Correlation_per_pix().run()
    # Dataframe_moving_window().run()
    # Moving_window_p_correlation().run()
    # Moving_window_single_correlation().run()
    Bivariate_plot().run()
    # Test_wen_data().run()
    # Tipping_point().run()
    # All_variables_single_corr().run()
    # All_variables_trend().run()
    # gen_ocean()
    pass


if __name__ == '__main__':
    main()