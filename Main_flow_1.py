# coding=utf-8
import psutil

from preprocess import *
results_root_main_flow = join(results_root,'Main_flow_1')


class Global_vars:
    def __init__(self):
        self.land_tif = join(this_root,'conf/land.tif')
        pass

    def season_list(self):
        season_l = ['early','peak','late']
        return season_l

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


class Get_Monthly_Early_Peak_Late:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Get_Monthly_Early_Peak_Late',results_root_main_flow)

    def run(self):
        self.Monthly_Early_Peak_Late()
        pass


    def Monthly_Early_Peak_Late(self):
        outf = join(self.this_class_arr,'Monthly_Early_Peak_Late.df')
        vege_dir = vars_info_dic['LAI_3g']['path']
        vege_dic = T.load_npy_dir(vege_dir)
        result_dic = {}
        for pix in tqdm(vege_dic):
            vals = vege_dic[pix]
            if np.isnan(np.nanmean(vals)):
                continue
            vals = np.array(vals)
            val_reshape = vals.reshape((-1,12))
            val_reshape_T = val_reshape.T
            month_mean_list = []
            for month in val_reshape_T:
                month_mean = np.nanmean(month)
                month_mean_list.append(month_mean)
            isnan_list = np.isnan(month_mean_list)
            if True in isnan_list:
                continue
            max_n_index,max_n_val = T.pick_max_n_index(month_mean_list,n=2)
            peak_months_distance = abs(max_n_index[0]-max_n_index[1])
            if not peak_months_distance == 1:
                continue
            max_n_index = list(max_n_index)
            max_n_index.sort()
            if max_n_index[0]<=3:
                continue
            if max_n_index[1]>=9:
                continue
            peak_mon = np.array(max_n_index) + 1
            early_mon = list(range(4,peak_mon[0]))
            late_mon = list(range(peak_mon[1]+1,11))

            early_mon = np.array(early_mon)
            peak_mon = np.array(peak_mon)
            late_mon = np.array(late_mon)

            result_dic_i = {
                'early':early_mon,
                'peak':peak_mon,
                'late':late_mon,
            }
            result_dic[pix] = result_dic_i
        df = T.dic_to_df(result_dic,'pix')
        df = df.dropna()
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


class Pick_Early_Peak_Late_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Pick_Early_Peak_Late_value',results_root_main_flow)
        pass


    def run(self):
        self.Pick_variables()
        self.Pick_variables_accumulative()
        self.Pick_variables_min()
        pass


    def Pick_variables(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mk_dir(outdir)
        var_list = [
            'LAI_3g',
            'SPEI',
            'Temperature',
            'Soil moisture',
        ]
        EPL_dff = join(Get_Monthly_Early_Peak_Late().this_class_arr,'Monthly_Early_Peak_Late.df')
        EPL_df = T.load_df(EPL_dff)
        EPL_dic=  T.df_to_dic(EPL_df,'pix')
        for variable in var_list:
            outf = join(outdir,variable+'.df')
            fdir = vars_info_dic[variable]['path']
            start_year = vars_info_dic[variable]['start_year']
            dic = T.load_npy_dir(fdir)
            result_dic = {}
            for pix in tqdm(dic,desc=variable):
                if not pix in EPL_dic:
                    continue
                vals = dic[pix]
                EPL_dic_i = EPL_dic[pix]
                season_vals_dic = {}
                for season in EPL_dic_i:
                    if season == 'pix':
                        continue
                    grow_season = EPL_dic_i[season]
                    grow_season = list(grow_season)
                    annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=grow_season)
                    season_vals_dic[season] = annual_vals
                result_dic[pix] = season_vals_dic
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)
    def Pick_variables_min(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mk_dir(outdir)
        var_list = [
            'SPEI',
            'Soil moisture',
        ]
        EPL_dff = join(Get_Monthly_Early_Peak_Late().this_class_arr,'Monthly_Early_Peak_Late.df')
        EPL_df = T.load_df(EPL_dff)
        EPL_dic=  T.df_to_dic(EPL_df,'pix')
        for variable in var_list:
            outf = join(outdir,variable+'_min.df')
            fdir = vars_info_dic[variable]['path']
            start_year = vars_info_dic[variable]['start_year']
            dic = T.load_npy_dir(fdir)
            result_dic = {}
            for pix in tqdm(dic,desc=variable):
                if not pix in EPL_dic:
                    continue
                vals = dic[pix]
                EPL_dic_i = EPL_dic[pix]
                season_vals_dic = {}
                for season in EPL_dic_i:
                    if season == 'pix':
                        continue
                    grow_season = EPL_dic_i[season]
                    grow_season = list(grow_season)
                    annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=grow_season,method='sum')
                    season_vals_dic[season] = annual_vals
                result_dic[pix] = season_vals_dic
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)

    def Pick_variables_accumulative(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mk_dir(outdir)
        var_list = [
            'SPEI',
            'Soil moisture',
        ]
        EPL_dff = join(Get_Monthly_Early_Peak_Late().this_class_arr,'Monthly_Early_Peak_Late.df')
        EPL_df = T.load_df(EPL_dff)
        EPL_dic=  T.df_to_dic(EPL_df,'pix')
        for variable in var_list:
            outf = join(outdir,variable+'_accu.df')
            fdir = vars_info_dic[variable]['path']
            start_year = vars_info_dic[variable]['start_year']
            dic = T.load_npy_dir(fdir)
            result_dic = {}
            for pix in tqdm(dic,desc=variable):
                if not pix in EPL_dic:
                    continue
                vals = dic[pix]
                EPL_dic_i = EPL_dic[pix]
                season_vals_dic = {}
                for season in EPL_dic_i:
                    if season == 'pix':
                        continue
                    grow_season = EPL_dic_i[season]
                    grow_season = list(grow_season)
                    annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=grow_season,method='sum')
                    season_vals_dic[season] = annual_vals
                result_dic[pix] = season_vals_dic
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)


class RF:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('RF',results_root_main_flow)
        self.dff = join(self.this_class_arr,'Dataframe.df')
        self.__var_list()
        self.__var_list_train()
        pass

    def run(self):
        # self.build_df()
        self.cal_importance()
        pass

    def __var_list_train(self):
        # 'early_peak_SPEI_min_min'
        # 'early_peak_late_LAI_3g_mean'
        self.x_var_list_train = ['SOS', 'late_Temperature', 'late_Soil moisture', 'early_peak_SPEI_accu_sum', 'early_peak_LAI_3g_mean']
        self.y_var_train = 'late_LAI_3g'
        self.all_var_list_train = copy.copy(self.x_var_list_train)
        self.all_var_list_train.append(self.y_var_train)

        pass

    def __var_list(self):
        self.x_var_list = [
            'LAI_3g',
            'Soil moisture',
            'Soil moisture_accu',
            'Soil moisture_min',
            'SPEI',
            'SPEI_accu',
            'SPEI_min',
            'Temperature',
                    ]
        # self.y_var = 'LAI_3g'
        # self.all_var_list = copy.copy(self.x_var_list)
        # self.all_var_list.append(self.y_var)

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


    def cal_importance(self):
        df,dff = self.__load_df()
        y_var = self.y_var_train
        x_var_list = self.x_var_list_train
        df_i = pd.DataFrame()
        df_i[y_var] = df[y_var]
        df_i[x_var_list] = df[x_var_list]
        df_i = df_i.dropna()
        print(len(df_i))
        Y = df_i[y_var]
        X = df_i[x_var_list]
        # clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = self.permutation(X, Y, x_var_list)
        clf, importances_dic, r_X_dic, mse, r_model, score, Y_test, y_pred = self.RF_importance(X, Y, x_var_list)
        importances_dic['r'] = r_model
        importances_dic['score'] = score
        importances_dic['mse'] = mse
        print(importances_dic)
        pass
    def RF_importance(self, X, Y, variable_list):
        # from sklearn import XGboost
        # from sklearn.ensemble import GradientBoostingRegressor
        X = X[variable_list]
        r_X_dic = {}
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        clf = RandomForestRegressor(n_estimators=100, n_jobs=4)
        clf.fit(X_train, Y_train)
        importances = clf.feature_importances_
        # result = permutation_importance(clf, X_test, Y_test, scoring='r2',
        #                                 n_repeats=10, random_state=42,
        #                                 n_jobs=4)

        # importances = result.importances_mean
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

    def build_df(self):
        df = self.__gen_df_init()

        # season_list = Global_vars().season_list()
        # var_list = self.x_var_list
        # for var_ in var_list:
        #     for season in season_list:
        #         print(var_,season)
        #         df = self.add_each_season_to_df(df,var_,season)
        #
        # combine_season_list = ['early','peak']
        # var_ = 'LAI_3g'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)

        combine_season_list = ['early','peak','late']
        var_ = 'LAI_3g'
        method = 'mean'
        df = self.add_combine(df,combine_season_list,var_,method)

        #
        # combine_season_list = ['early', 'peak']
        # var_ = 'SPEI_accu'
        # method = 'sum'
        # df = self.add_combine(df, combine_season_list, var_, method)
        #
        # combine_season_list = ['early', 'peak']
        # var_ = 'Soil moisture_accu'
        # method = 'sum'
        #
        #
        # df = self.add_combine(df, combine_season_list, var_, method)
        # combine_season_list = ['early', 'peak']
        # var_ = 'SPEI_min'
        # method = 'min'
        # df = self.add_combine(df, combine_season_list, var_, method)
        #
        # combine_season_list = ['early', 'peak']
        # var_ = 'Soil moisture_min'
        # method = 'min'
        # df = self.add_combine(df, combine_season_list, var_, method)
        # df = self.add_sos_to_df(df)
        # df = self.add_CO2(df)
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

        pass

    def add_sos_to_df(self,df):
        sos_f = join(data_root,'Phenology/transform_early_peak_late_dormant_period_annual/early_start.npy')
        sos_dic = T.load_npy(sos_f)
        year_list = list(range(1982,2021))
        sos_dic_all = {}
        for pix in sos_dic:
            vals = sos_dic[pix]
            if len(vals) == 0:
                continue
            sos_dic_i = dict(zip(year_list,vals))
            sos_dic_all[pix] = sos_dic_i

        sos_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            year = row.year
            if not pix in sos_dic_all:
                sos_list.append(np.nan)
                continue
            sos_i = sos_dic_all[pix]
            if not year in sos_i:
                sos_list.append(np.nan)
                continue
            sos = sos_i[year]
            sos_list.append(sos)
        df['SOS'] = sos_list

        return df


    def add_each_season_to_df(self,df,var_,season):
        # var_ = 'LAI_3g'
        # season = 'late'
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df_var = T.load_df(dff)
        dic = T.df_to_spatial_dic(df_var,season)
        pix_list_ = DIC_and_TIF().void_spatial_dic()
        year_list_ = list(range(1982,2021))
        val_list = []
        year_list = []
        pix_list = []
        for pix in tqdm(pix_list_):
            if not pix in dic:
                continue
            vals = dic[pix]
            dic_i = dict(zip(year_list_,vals))
            for y in year_list_:
                if not y in dic_i:
                    continue
                val = dic_i[y]
                val_list.append(val)
                year_list.append(y)
                pix_list.append(pix)
        df['pix'] = pix_list
        df['year'] = year_list
        df[f'{season}_{var_}'] = val_list
        return df

    def add_combine(self,df,combine_season_list,var_,method):
        # combine_season_list = ['early','peak']
        # var_ = 'LAI_3g'
        print(f'combining {combine_season_list} {var_} with {method}')
        matrix = []
        for season in combine_season_list:
            col_name = f'{season}_{var_}'
            vals = df[col_name].tolist()
            vals = np.array(vals)
            matrix.append(vals)
        matrix = np.array(matrix)
        matrix_T = matrix.T
        result_list = None
        if method == 'sum':
            result_list = []
            for i in matrix_T:
                if T.is_all_nan(i):
                    result_list.append(np.nan)
                    continue
                result = np.nansum(i)
                result_list.append(result)
        elif method == 'mean':
            result_list = []
            for i in matrix_T:
                if T.is_all_nan(i):
                    result_list.append(np.nan)
                    continue
                result = np.nansum(i)
                result_list.append(result)
        elif method == 'min':
            result_list = []
            for i in matrix_T:
                if T.is_all_nan(i):
                    result_list.append(np.nan)
                    continue
                result = np.nanmin(i)
                result_list.append(result)
        else:
            raise
        new_col_name = f'{"_".join(combine_season_list)}_{var_}_{method}'
        df[new_col_name] = result_list
        return df


    def add_CO2(self,df):
        co2_dir = vars_info_dic['CO2']['path']
        co2_dic = T.load_npy_dir(co2_dir)
        year_list = list(range(1982,2021))
        pix_list = T.get_df_unique_val_list(df,'pix')
        co2_annual_dic = {}
        for pix in tqdm(pix_list):
            co2_vals = co2_dic[pix]
            co2_annual_vals = T.monthly_vals_to_annual_val(co2_vals)
            co2_dic_i = dict(zip(year_list, co2_annual_vals))
            co2_annual_dic[pix] = co2_dic_i

        val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            year = row.year
            co2_dic_i = co2_annual_dic[pix]
            co2 = co2_dic_i[year]
            val_list.append(co2)
        df['CO2'] = val_list
        return df



def main():
    # Get_Monthly_Early_Peak_Late().run()
    # Pick_Early_Peak_Late_value().run()
    RF().run()


if __name__ == '__main__':
    main()