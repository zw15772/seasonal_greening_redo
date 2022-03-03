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

class Phenology:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Phenology',
                                                                                       results_root_main_flow)


    def run(self):
        # 1 把多年的NDVI分成单年，分南北半球，分文件夹存储
        # fdir = data_root+'NDVI_phenology/tif_05deg_bi_weekly/'
        # outdir = data_root+'NDVI_phenology/tif_05deg_bi_weekly_separate/'
        # self.split_north_south_hemi(fdir,outdir)

        # 2 把单年的NDVI tif 转换成 perpix
        # for folder in ['north','south_modified']:
        #     fdir = data_root+'NDVI_phenology/tif_05deg_bi_weekly_separate/{}/'.format(folder)
        #     outdir = data_root+'NDVI_phenology/per_pix_separate/{}/'.format(folder)
        #     self.data_transform_split_files(fdir,outdir)
        # 3 hants smooth
        # self.hants()
        # self.check_hants()

        # 4 计算 top left right
        # self.SOS_EOS()
        # self.check_SOS_EOS()

        # 5 合成南北半球
        # self.compose_SOS_EOS()

        # 6 check sos
        # self.check_sos()
        pass



    def compose_SOS_EOS(self):
        outdir = self.this_class_arr + 'compose_SOS_EOS/'
        T.mk_dir(outdir)
        threshold_i = 0.5
        SOS_EOS_dic = DIC_and_TIF().void_spatial_dic()
        for hemi in ['north','south_modified']:
            fdir = self.this_class_arr + 'SOS_EOS/threshold_{}/{}/'.format(threshold_i, hemi)
            for f in T.listdir(fdir):
                # year = f.split('.')[0]
                # year = int(year)
                # print fdir + f
                dic = T.load_npy(fdir + f)
                for pix in dic:
                    vals = dic[pix]
                    vals = np.array(vals)
                    a,b = pix.split('.')
                    a = int(a)
                    b = int(b)
                    pix = (a,b)
                    SOS_EOS_dic[pix].append(vals)
        SOS_EOS_dic_np = {}
        for pix in SOS_EOS_dic:
            val = SOS_EOS_dic[pix]
            val = np.array(val)
            SOS_EOS_dic_np[pix] = val
        np.save(outdir + 'compose_SOS_EOS',SOS_EOS_dic_np)
        # spatial_dic = {}
        # for pix in SOS_EOS_dic:
        #     vals = SOS_EOS_dic[pix]
        #     length = len(vals)
        #     spatial_dic[pix] = length
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
            # exit()
        pass

    def split_north_south_hemi(self,fdir,outdir):
        # 1 north
        north_dir = outdir + 'north/'
        south_dir = outdir + 'south/'
        T.mk_dir(south_dir,force=True)
        T.mk_dir(north_dir,force=True)
        years = np.array(range(1982, 2016))
        for y in tqdm(years):
            for f in T.listdir(fdir):
                if not f.endswith('tif'):
                    continue
                year = int(f[:4])
                if y == year:
                    date = f[4:8]
                    # print date
                    # exit()
                    array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir+f)
                    array_south = copy.copy(array)
                    array_north = copy.copy(array)
                    array_south[:180] = -999999.
                    array_north[180:] = -999999.
                    north_outdir_y = north_dir + '{}/'.format(y)
                    south_outdir_y = south_dir + '{}/'.format(y)
                    Tools().mk_dir(north_outdir_y)
                    Tools().mk_dir(south_outdir_y)
                    south_outf = south_outdir_y + date + '.tif'
                    north_outf = north_outdir_y + date + '.tif'
                    DIC_and_TIF().arr_to_tif(array_south,south_outf)
                    DIC_and_TIF().arr_to_tif(array_north,north_outf)
        # 2 modify south
        south_modified_dir = outdir + 'south_modified/'
        T.mk_dir(south_modified_dir)
        for year in T.listdir(south_dir):
            outdir_y = south_modified_dir + '{}/'.format(year)
            T.mk_dir(outdir_y)
            for f in T.listdir(south_dir + year):
                mon = f.split('.')[0][:2]
                day = f.split('.')[0][2:]
                year = int(year)
                mon = int(mon)
                # print 'original date:',year,mon
                old_fname = south_dir + str(year) + '/' + f
                mon = mon - 6
                if mon <= 0:
                    year_new = year - 1
                    mon_new = mon + 12
                    new_fname = south_modified_dir + '{}/{:02d}{}.tif'.format(year_new,mon_new,day)
                else:
                    new_fname = south_modified_dir + '{}/{:02d}{}.tif'.format(year,mon,day)

                # print old_fname
                # print new_fname
                try:
                    shutil.copy(old_fname,new_fname)
                except Exception as e:
                    # print e
                    continue



    def split_files(self,fdir,outdir):
        Tools().mk_dir(outdir)
        years = np.array(range(1982, 2016))
        for y in tqdm(years):
            for f in T.listdir(fdir):
                year = int(f[:4])
                if y == year:
                    outdir_y = outdir + '{}/'.format(y)
                    Tools().mk_dir(outdir_y)
                    shutil.copy(fdir + f, outdir_y + f)


    def data_transform(self, fdir, outdir):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
        # 将空间图转换为数组
        # per_pix_data
        flist = T.listdir(fdir)
        date_list = []
        for f in flist:
            if f.endswith('.tif'):
                date = f.split('.')[0]
                date_list.append(date)
        date_list.sort()
        all_array = []
        for d in tqdm(date_list, 'loading...'):
            # for d in date_list:
            for f in flist:
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        # print(d)
                        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                        array = np.array(array, dtype=float)
                        # print np.min(array)
                        # print type(array)
                        # plt.imshow(array)
                        # plt.show()
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic['%03d.%03d' % (r, c)] = []
                void_dic_list.append('%03d.%03d' % (r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(range(row)):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' % (r, c)].append(val) # TODO: need to be transformed into tuple

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def data_transform_split_files(self,fdir,outdir):
        for year in T.listdir(fdir):
            # print year
            fdir_i = fdir + year + '/'
            outdir_i = outdir + year + '/'
            Tools().mk_dir(outdir_i, force=1)
            self.data_transform(fdir_i, outdir_i)

        pass

    def kernel_hants(self, params):
        outdir, y, fdir = params
        outdir_y = outdir + y + '/'
        Tools().mk_dir(outdir_y, force=1)
        for f in T.listdir(fdir + y):
            dic = dict(np.load(fdir + y + '/' + f).item())
            hants_dic = {}
            for pix in dic:
                vals = dic[pix]
                vals = np.array(vals)
                std = np.std(vals)
                if std == 0:
                    continue
                xnew, ynew = self.__interp__(vals)
                ynew = np.array([ynew])
                # print np.std(ynew)
                results = HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
                                fit_error_tolerance=std)
                result = results[0]

                # plt.plot(result)
                # plt.plot(range(len(ynew[0])),ynew[0])
                # plt.show()
                hants_dic[pix] = result
            np.save(outdir_y + f, hants_dic)

    def hants(self):
        for hemi in ['north','south_modified']:
            outdir = self.this_class_arr + 'hants_smooth/{}/'.format(hemi)
            fdir = data_root + 'NDVI_phenology/per_pix_separate/{}/'.format(hemi)
            params = []
            for y in T.listdir(fdir):
                params.append([outdir, y, fdir])
                # self.kernel_hants([outdir, y, fdir])
            MULTIPROCESS(self.kernel_hants, params).run(process=5)

    def check_hants(self):
        hemi = 'south_modified'
        fdir = self.this_class_arr + 'hants_smooth/{}/'.format(hemi)
        tropical_mask_dic = NDVI().tropical_mask_dic

        for year in T.listdir(fdir):
            perpix_dir = fdir + '{}/'.format(year)
            for f in T.listdir(perpix_dir):
                if not '021' in f:
                    continue
                dic = T.load_npy(perpix_dir + f)
                for pix in dic:
                    if pix in tropical_mask_dic:
                        continue
                    vals = dic[pix]
                    if len(vals) > 0:
                        # print pix,vals
                        plt.plot(vals)
                        plt.show()
                        sleep()
            exit()

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def SOS_EOS(self, threshold_i=0.5):
        for hemi in ['north','south_modified']:
            out_dir = self.this_class_arr + 'SOS_EOS/threshold_{}/{}/'.format(threshold_i,hemi)
            Tools().mk_dir(out_dir, force=1)
            # fdir = data_root + 'NDVI_phenology/HANTS/'
            fdir = self.this_class_arr + 'hants_smooth/{}/'.format(hemi)
            for y in tqdm(T.listdir(fdir)):
                year_dir = fdir + y + '/'
                result_dic = {}
                for f in T.listdir(year_dir):
                    dic = dict(np.load(year_dir + f).item())
                    for pix in dic:
                        try:
                            vals = dic[pix]
                            maxind = np.argmax(vals)
                            start = self.__search_left(vals, maxind, threshold_i)
                            end = self.__search_right(vals, maxind, threshold_i)
                            result = [start,maxind, end]
                            result_dic[pix] = result
                            # print result
                            # sleep()
                        except:
                            pass
                            # plt.plot(vals)
                            # plt.show()
                            # exit()
                        # plt.plot(vals)
                        # plt.plot(range(start,end),vals[start:end],linewidth=4,zorder=99,color='r')
                        # plt.title('start:{} \nend:{} \nduration:{}'.format(start,end,end-start))
                        # plt.show()
                np.save(out_dir + y, result_dic)

    def check_SOS_EOS(self,threshold_i=0.5):
        fdir = self.this_class_arr + 'SOS_EOS/threshold_{}/north/'.format(threshold_i)
        for f in T.listdir(fdir):
            dic = T.load_npy(fdir+f)
            spatial_dic = {}
            for pix in dic:
                SOS = dic[pix][0]
                pix = (int(pix.split('.')[0]),int(pix.split('.')[1]))
                spatial_dic[pix] = SOS
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr)
            plt.show()
        pass

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


class Analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Analysis',results_root_main_flow)

        pass

    def run(self):
        # self.Greeing_trend()
        self.Greeing_trend_two_period()
        pass


    def nan_linear_fit(self, val1_list, val2_list):
        # pearson correlation of val1 and val2, which contain Nan
        K = KDE_plot()
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        if len(val1_list_new) <= 3:
            a, b, r, p = np.nan, np.nan, np.nan, np.nan
        else:
            a, b, r, p = K.linefit(val1_list_new, val2_list_new)
        return a, b, r, p
    def Greeing_trend(self):
        var_ = 'LAI_3g'
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mk_dir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        for season in global_season_dic:
            print(season)
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            trend_dic = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x,y)
                trend_dic[pix] = a
            out_tif = join(outdir,f'{season}.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dic,out_tif)
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
            # plt.figure()
            # plt.imshow(arr,vmax=0.02,vmin=-0.02,cmap='RdBu')
            # plt.colorbar()
            # DIC_and_TIF().plot_back_ground_arr(Global_vars().land_tif)
            # plt.title(season)
        # plt.show()
        pass

    def Greeing_trend_two_period(self):
        var_ = 'LAI_3g'
        period_1 = list(range(1982,2002))
        period_2 = list(range(2002,2019))
        year_list = list(range(1982,2021))
        period_list = [period_1,period_2]
        outdir = join(self.this_class_tif,f'Greeing_trend_two_period/{var_}')
        T.mk_dir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        flag = 0
        for period in period_list:
            flag += 1
            for season in global_season_dic:
                print(season,flag)
                col_name = join(f'{season}')
                spatial_dic = T.df_to_spatial_dic(df,col_name)
                trend_dic = {}
                for pix in spatial_dic:
                    vals = spatial_dic[pix]
                    period = np.array(period)
                    period_index = period - global_start_year
                    vals_period = T.pick_vals_from_1darray(vals,period_index)
                    x = list(range(len(vals_period)))
                    y = vals_period
                    a, b, r, p = self.nan_linear_fit(x,y)
                    trend_dic[pix] = a
                out_tif = join(outdir,f'{flag}_{season}.tif')
                DIC_and_TIF().pix_dic_to_tif(trend_dic,out_tif)
                # arr = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
                # plt.figure()
                # plt.imshow(arr,vmax=0.02,vmin=-0.02,cmap='RdBu')
                # plt.colorbar()
                # DIC_and_TIF().plot_back_ground_arr(Global_vars().land_tif)
                # plt.title(season)
            # plt.show()
        pass


def main():
    Phenology().run()
    # Get_Monthly_Early_Peak_Late().run()
    # Pick_Early_Peak_Late_value().run()
    # RF().run()
    # Analysis().run()


if __name__ == '__main__':
    main()