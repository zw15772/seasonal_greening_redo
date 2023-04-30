# coding=utf-8
import matplotlib.pyplot as plt
from lytools import *
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
centimeter_factor = 1 / 2.54

T = Tools()

this_script_root = '/Users/liyang/Desktop/earlier_late/'
data_root = this_script_root + 'data/'

class Dataframe_Func:

    def __init__(self,df):
        self.df = self.add_Humid_nonhumid(df)

    def run(self):

        pass


    def P_PET_ratio(self):
        fdir = join(data_root,'aridity_P_PET_dic')
        # fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            vals[vals == 0] = np.nan
            if len(vals) == 0:
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            vals = T.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def P_PET_class(self,):
        dic = self.P_PET_ratio()
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
        P_PET_dic_reclass = self.P_PET_class()
        df = T.add_spatial_dic_to_df(df,P_PET_dic_reclass,'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Dryland'
        return df



class Amplifying_Stablilizing:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Amplifying_Stablilizing', this_script_root, mode=2)
        self.datadir = join(this_script_root, 'data')
        self.models_list = self.__get_model_list()
        self.period_list = ['during_early_peak', 'during_late']

        pass

    def run(self):

        # self.class_tif()
        # self.class_tif_old()
        # self.plot_tif()
        # self.ratio_statistic()
        # self.ratio_statistic_old_intersect()
        # self.ratio_statistic_old()
        # self.individual_model_ratio()
        # self.ratio_statistic_all_area()
        self.plot_ratio()
        # self.plot_ratio_all_area()

        pass

    def class_tif(self):
        p_level = 0.1
        outdir = join(self.this_class_tif,'class_tif')
        fdir = join(self.datadir,'trend')
        T.mk_dir(outdir)
        # print(self.models_list)
        # exit()
        class_label_dict = {
            'strong stabilizing': -1,
            'weak stabilizing': 0,
            'amplifying': 1,
            'no effect': -2,
            'other': -3,
        }
        for model in tqdm(self.models_list):
            # if not 'MODIS' in model:
            #     continue
            early_peak_f = join(fdir, f'during_early_peak_{model}_trend.tif')
            early_peak_p_f = join(fdir, f'during_early_peak_{model}_p_value.tif')
            late_f = join(fdir, f'during_late_{model}_trend.tif')
            late_p_f = join(fdir, f'during_late_{model}_p_value.tif')
            early_peak_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_f)
            early_peak_p_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_p_f)
            late_dict = DIC_and_TIF().spatial_tif_to_dic(late_f)
            late_p_dict = DIC_and_TIF().spatial_tif_to_dic(late_p_f)

            class_dict = {}
            class_dict_num = {}
            for pix in early_peak_dict:
                early_peak = early_peak_dict[pix]
                early_peak_p = early_peak_p_dict[pix]
                late = late_dict[pix]
                late_p = late_p_dict[pix]
                if np.isnan(early_peak):
                    continue
                if np.isnan(early_peak_p):
                    continue
                if not early_peak > 0:
                    class_i = 'other'
                else:
                    if late_p < p_level:
                        if late > early_peak:
                            class_i = 'amplifying'
                        elif late < 0:
                            class_i = 'strong stabilizing'
                        else:
                            if early_peak_p < p_level and late < early_peak and late > 0:
                                class_i = 'weak stabilizing'
                            # else:
                            #     class_i = 'no effect'
                            elif early_peak_p >= p_level and late < early_peak and late > 0:
                                class_i = 'no effect'
                            else:
                                print('early_peak',early_peak)
                                print('late',late)
                                print('early_peak_p',early_peak_p)
                                print('late_p',late_p)
                                raise IOError('check')
                    else:
                        if early_peak_p > p_level:
                            class_i = 'no effect'
                        else:
                            if early_peak > late and early_peak_p < p_level and late > 0:
                                class_i = 'weak stabilizing'
                            elif early_peak > late and early_peak_p < p_level and late < 0:
                                class_i = 'weak stabilizing'
                            elif early_peak < late and early_peak_p < p_level:
                                class_i = 'no effect'
                            else:
                                print('early_peak',early_peak)
                                print('late',late)
                                print('early_peak_p',early_peak_p)
                                print('late_p',late_p)
                                raise IOError('check')


                class_dict[pix] = class_i
                class_dict_num[pix] = class_label_dict[class_i]
            # exit()
            # color_list = ['#EDEDED','k', 'purple', '#FFFFCC', 'g']
            # cmap = T.cmap_blend(color_list)
            # plt.register_cmap(name='mycmap', cmap=cmap)
            # mpl.colormaps.register(name='mycmap', cmap=cmap)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(class_dict_num)
            # plt.imshow(arr,cmap='mycmap',interpolation='nearest')
            # plt.colorbar()
            # plt.title(model)
            # plt.show()
            outf = join(outdir, f'{model}.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def class_tif_old(self):
        outdir = join(self.this_class_tif,'class_tif_old')
        fdir = join(self.datadir,'trend')
        T.mk_dir(outdir)
        class_label_dict = {'stablilizing': -1, 'weak amplifying': 0, 'amplifying': 1}
        for model in tqdm(self.models_list):
            early_peak_f = join(fdir, f'during_early_peak_{model}_trend.tif')
            late_f = join(fdir, f'during_late_{model}_trend.tif')
            early_peak_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_f)
            late_dict = DIC_and_TIF().spatial_tif_to_dic(late_f)
            class_dict = {}
            class_dict_num = {}
            for pix in early_peak_dict:
                early_peak = early_peak_dict[pix]
                late = late_dict[pix]
                if not early_peak > 0:
                    continue
                if late <= 0:
                    class_i = 'stablilizing'
                else:
                    if early_peak >= late:
                        class_i = 'weak amplifying'
                    else:
                        class_i = 'amplifying'
                class_dict[pix] = class_i
                class_dict_num[pix] = class_label_dict[class_i]
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(class_dict_num)
            outf = join(outdir, f'{model}.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)


    def plot_tif(self):
        fdir = join(self.this_class_tif,'class_tif')
        outdir = join(self.this_class_png, 'plot_tif')
        T.mk_dir(outdir)
        color_list = ['#EDEDED','k', 'purple', '#FFFFCC', 'g']
        cmap = T.cmap_blend(color_list)
        # plt.register_cmap(name='mycmap', cmap=cmap)
        mpl.colormaps.register(name='mycmap',cmap=cmap)
        product = ['MODIS', 'LAI3g', 'Trendy_ensemble']
        for product_i in product:
            for f in T.listdir(fdir):
                if not product_i in f:
                    continue
                fpath = join(fdir, f)
                Plot().plot_ortho(fpath,cmap='mycmap',vmin=-3,vmax=1)
                outf = join(outdir, f.replace('.tif', '.png'))
                plt.savefig(outf, dpi=300)
                plt.close()
        T.open_path_and_file(outdir)

    def ratio_statistic(self):
        # water and energy limited
        outdir = join(self.this_class_arr,'ratio_statistic')
        T.mkdir(outdir)
        fdir = join(self.this_class_tif,'class_tif')
        # fdir = join(self.this_class_tif,'class_tif_old')
        all_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = dict_i
        df = T.spatial_dics_to_df(all_dict)
        # print(df<-1)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        df = Dataframe_Func(df).df
        HI_list = ['Humid','Dryland']
        class_label_dict = {
            'strong stabilizing': -1,
            'weak stabilizing': 0,
            'amplifying': 1,
            'no effect': -2,
            'other': -3,
        }
        class_label_dict = T.reverse_dic(class_label_dict)
        T.print_head_n(df)
        # cols = df.columns.tolist()
        cols = ['MODIS_LAI','LAI3g','Trendy_ensemble']
        all_results_dict = {}
        for HI in HI_list:
            df_HI = df[df['HI_reclass'] == HI]
            for col in cols:
                spatial_dict = T.df_to_spatial_dic(df_HI,col)
                count_dict = {
                    'strong stabilizing':0,
                    'weak stabilizing':0,
                    'amplifying':0,
                    'other':0,
                    'no effect':0,
                }
                total = 0
                for pix in spatial_dict:
                    val_i = spatial_dict[pix]
                    # if val_i == -2 or val_i == -3:
                    #     continue
                    class_i = class_label_dict[val_i][0]
                    # print(class_i)
                    count_dict[class_i] += 1
                    total += 1
                for class_i in count_dict:
                    count_dict[class_i] = count_dict[class_i]/total * 100
                key_i = f'{HI}_{col}'
                all_results_dict[key_i] = count_dict
        df = pd.DataFrame(all_results_dict)
        outf = join(outdir, 'ratio_statistic.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.open_path_and_file(outdir)
        # T.print_head_n(df)
        # exit()

    def ratio_statistic_old(self):
        # water and energy limited
        outdir = join(self.this_class_arr,'ratio_statistic_old')
        T.mkdir(outdir)
        # fdir = join(self.this_class_tif,'class_tif')
        fdir = join(self.this_class_tif,'class_tif_old')
        all_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = dict_i
        df = T.spatial_dics_to_df(all_dict)
        # print(df<-1)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        df = Dataframe_Func(df).df
        HI_list = ['Humid','Dryland']
        class_label_dict = {
            'strong stabilizing': -1,
            'weak stabilizing': 0,
            'amplifying': 1,
            # 'no effect': -2,
            # 'other': -3,
        }
        class_label_dict = T.reverse_dic(class_label_dict)
        T.print_head_n(df)
        # cols = df.columns.tolist()
        cols = ['MODIS_LAI','LAI3g','Trendy_ensemble']
        all_results_dict = {}
        for HI in HI_list:
            df_HI = df[df['HI_reclass'] == HI]
            for col in cols:
                spatial_dict = T.df_to_spatial_dic(df_HI,col)
                count_dict = {
                    'strong stabilizing':0,
                    'weak stabilizing':0,
                    'amplifying':0,
                    # 'other':0,
                    # 'no effect':0,
                }
                total = 0
                for pix in spatial_dict:
                    val_i = spatial_dict[pix]
                    # if val_i == -2 or val_i == -3:
                    #     continue
                    class_i = class_label_dict[val_i][0]
                    # print(class_i)
                    count_dict[class_i] += 1
                    total += 1
                for class_i in count_dict:
                    count_dict[class_i] = count_dict[class_i]/total * 100
                key_i = f'{HI}_{col}'
                all_results_dict[key_i] = count_dict
        df = pd.DataFrame(all_results_dict)
        outf = join(outdir, 'ratio_statistic.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.open_path_and_file(outdir)
        # T.print_head_n(df)
        # exit()

    def ratio_statistic_old_intersect(self):
        # water and energy limited
        outdir = join(self.this_class_arr,'ratio_statistic_old_intersect')
        T.mkdir(outdir)
        # fdir = join(self.this_class_tif,'class_tif')
        fdir = join(self.this_class_tif,'class_tif_old')
        selected_cols = ['pix','MODIS_LAI','LAI3g','Trendy_ensemble']
        all_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = dict_i
        df = T.spatial_dics_to_df(all_dict)
        df = df[selected_cols]

        # print(df<-1)
        df = df.dropna()
        # print(df)
        # exit()
        T.print_head_n(df)
        # exit()
        df = Dataframe_Func(df).df
        HI_list = ['Humid','Dryland']
        class_label_dict = {
            'strong stabilizing': -1,
            'weak stabilizing': 0,
            'amplifying': 1,
            # 'no effect': -2,
            # 'other': -3,
        }
        class_label_dict = T.reverse_dic(class_label_dict)
        T.print_head_n(df)
        # cols = df.columns.tolist()
        cols = ['MODIS_LAI','LAI3g','Trendy_ensemble']
        all_results_dict = {}
        for HI in HI_list:
            df_HI = df[df['HI_reclass'] == HI]
            for col in cols:
                spatial_dict = T.df_to_spatial_dic(df_HI,col)
                count_dict = {
                    'strong stabilizing':0,
                    'weak stabilizing':0,
                    'amplifying':0,
                    # 'other':0,
                    # 'no effect':0,
                }
                total = 0
                for pix in spatial_dict:
                    val_i = spatial_dict[pix]
                    # if val_i == -2 or val_i == -3:
                    #     continue
                    class_i = class_label_dict[val_i][0]
                    # print(class_i)
                    count_dict[class_i] += 1
                    total += 1
                for class_i in count_dict:
                    count_dict[class_i] = count_dict[class_i]/total * 100
                key_i = f'{HI}_{col}'
                all_results_dict[key_i] = count_dict
        df = pd.DataFrame(all_results_dict)
        outf = join(outdir, 'ratio_statistic.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.open_path_and_file(outdir)
        # T.print_head_n(df)
        # exit()

        pass

    def individual_model_ratio(self):
        outdir = join(self.this_class_arr, 'individual_models')
        T.mkdir(outdir)
        fdir = join(self.this_class_tif, 'individual_models')
        all_dict = {}
        key_list = []
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            key_list.append(key)
            all_dict[key] = dict_i
        # print(key_list)
        # exit()
        df = T.spatial_dics_to_df(all_dict)
        # df = df.dropna()
        df = Dataframe_Func(df).df
        HI_list = ['Humid', 'Dryland']
        class_label_dict = {
            'strong stabilizing': -1,
            'weak stabilizing': 0,
            'amplifying': 1,
            'no effect': -2,
            'other': -3,
        }
        class_label_dict = T.reverse_dic(class_label_dict)
        T.print_head_n(df)
        # exit()
        # cols = df.columns.tolist()
        cols = key_list
        all_results_dict = {}
        for HI in HI_list:
            df_HI = df[df['HI_reclass'] == HI]
            for col in cols:
                spatial_dict = T.df_to_spatial_dic(df_HI, col)
                count_dict = {
                    'strong stabilizing': 0,
                    'weak stabilizing': 0,
                    'amplifying': 0,
                    'other': 0,
                    'no effect':0
                }
                total = 0
                for pix in spatial_dict:
                    val_i = spatial_dict[pix]
                    if np.isnan(val_i):
                        continue
                    # print(val_i)
                    class_i = class_label_dict[val_i][0]
                    count_dict[class_i] += 1
                    total += 1
                for class_i in count_dict:
                    count_dict[class_i] = count_dict[class_i] / total * 100
                key_i = f'{HI}_{col}'
                all_results_dict[key_i] = count_dict
        df = pd.DataFrame(all_results_dict)
        outf = join(outdir, 'ratio_statistic.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.open_path_and_file(outdir)
        pass

    def ratio_statistic_all_area(self):
        outdir = join(self.this_class_arr,'ratio_statistic_all_area')
        T.mkdir(outdir)
        fdir = join(self.this_class_tif,'class_tif')
        all_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = dict_i
        df = T.spatial_dics_to_df(all_dict)
        # print(df<-1)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        # df = Dataframe_Func(df).df
        amp_stab_dict = {
            -1:'strong stablilizing',
            0:'weak stablilizing',
            1:'amplifying',
            # -2:'other'
        }
        T.print_head_n(df)
        # cols = df.columns.tolist()
        # cols = ['LAI3g','MODIS_LAI','Trendy_ensemble']
        cols = ['MODIS_LAI','LAI3g','Trendy_ensemble']
        all_results_dict = {}
        for col in cols:
            spatial_dict = T.df_to_spatial_dic(df,col)
            count_dict = {
                'strong stablilizing':0,
                'weak stablilizing':0,
                'amplifying':0,
                # 'other':0,
            }
            total = 0
            for pix in spatial_dict:
                val_i = spatial_dict[pix]
                if val_i == -2 or val_i == -3:
                    continue
                class_i = amp_stab_dict[val_i]
                count_dict[class_i] += 1
                total += 1
            for class_i in count_dict:
                count_dict[class_i] = count_dict[class_i]/total * 100
            key_i = f'{col}'
            all_results_dict[key_i] = count_dict
        df = pd.DataFrame(all_results_dict)
        outf = join(outdir, 'ratio_statistic.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.open_path_and_file(outdir)
        # T.print_head_n(df)
        # exit()

        pass


    def plot_ratio(self):
        outdir = join(self.this_class_png,'ratio_statistic')
        T.mkdir(outdir)
        # fpath1 = join(self.this_class_arr,'ratio_statistic','ratio_statistic.df')
        fpath1 = join(self.this_class_arr,'ratio_statistic_old_intersect','ratio_statistic.df')
        # fpath2 = join(self.this_class_arr, 'individual_models/ratio_statistic.df')
        df1 = T.load_df(fpath1)
        df = df1
        # df2 = T.load_df(fpath2)
        # df = pd.concat([df1,df2],axis=1)
        # T.print_head_n(df)
        # exit()
        # T.print_head_n(df)
        HI_list = ['Humid', 'Dryland']
        HI_list_rename = {
            'Humid':'Energy-limited',
            'Dryland':'Water-limited'
        }
        # T.print_head_n(df)
        # cols = df.columns.tolist()
        # cols1 = ['LAI3g', 'MODIS_LAI', 'Trendy_ensemble']
        cols1 = ['MODIS_LAI','LAI3g',  'Trendy_ensemble']
        cols2 = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5',
                'IBIS_S2_lai', 'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai',
                'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai',
                'YIBs_S2_Monthly_lai']
        # color_dict = {
        #     'LAI3g':'#1f77b4',
        #     'MODIS_LAI':'#ff7f0e',
        #     'Trendy_ensemble':'#2ca02c',
        # }
        color_dict = {
            'LAI3g':'g',
            'MODIS_LAI':'r',
            'Trendy_ensemble':'b',
        }

        # cols1.extend(cols2)
        cols = cols1
        smp_stab_list = ['strong stabilizing','weak stabilizing','amplifying']
        smp_stab_list_rename = {
            'strong stabilizing':'Strong stabilization',
            'weak stabilizing':'Weak stabilization',
            'amplifying':'Amplification',
            }
        for HI in HI_list:
            plt.figure(figsize=(18*centimeter_factor, 4*centimeter_factor))
            flag = 1
            for smp_stab in smp_stab_list:
                plt.subplot(1,3,flag)
                flag += 1
                x = []
                y = []
                color_list = []
                for col in cols:
                    key_i = f'{HI}_{col}'
                    color = color_dict[col] if col in color_dict else 'grey'
                    val_i = df.loc[smp_stab,key_i]
                    x.append(col)
                    y.append(val_i)
                    color_list.append(color)
                plt.bar(x,y,color=color_list,)
                colors = color_dict
                labels = list(colors.keys())
                handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
                plt.title(f'{HI_list_rename[HI]}\n{smp_stab_list_rename[smp_stab]}')
                plt.xticks([])
                plt.ylim(0,85)
            plt.legend(handles, labels)
            outf = join(outdir,f'{HI}.pdf')
        plt.show()
            # plt.savefig(outf)
            # plt.close()
        # T.open_path_and_file(outdir)


    def plot_ratio_all_area(self):
        outdir = join(self.this_class_png,'ratio_statistic_all_area')
        T.mkdir(outdir)
        fpath1 = join(self.this_class_arr,'ratio_statistic_all_area','ratio_statistic.df')
        df1 = T.load_df(fpath1)
        # T.print_head_n(df1)
        # exit()
        # T.print_head_n(df)
        amp_stab_dict = {
            -1: 'strong stablilizing',
            0: 'weak stablilizing',
            1: 'amplifying'
        }
        # T.print_head_n(df)
        # cols = df.columns.tolist()
        cols1 = ['MODIS_LAI', 'LAI3g', 'Trendy_ensemble']
        # cols2 = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5',
        #         'IBIS_S2_lai', 'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai',
        #         'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
        #         'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai',
        #         'YIBs_S2_Monthly_lai']
        color_dict = {
            'LAI3g':'#C5E2DF',
            'MODIS_LAI':'#C59EC4',
            'Trendy_ensemble':'gray',
        }
        # cols1.extend(cols2)
        cols = cols1
        smp_stab_list = ['strong stablilizing','weak stablilizing','amplifying']

        plt.figure(figsize=(18*centimeter_factor, 4*centimeter_factor))
        flag = 1
        for smp_stab in smp_stab_list:
            plt.subplot(1,3,flag)
            flag += 1
            x = []
            y = []
            color_list = []
            for col in cols:
                print(col)
                key_i = f'{col}'
                color = color_dict[col] if col in color_dict else 'grey'
                val_i = df1.loc[smp_stab,key_i]
                x.append(col)
                y.append(val_i)
                color_list.append(color)
            plt.bar(x,y,color=color_list)
            plt.title(f'{smp_stab}')
            plt.xticks([])
            plt.ylim(0,85)
        # outf = join(outdir,f'{HI}.pdf')
        plt.show()
            # plt.savefig(outf)
            # plt.close()
        # T.open_path_and_file(outdir)

    def __get_model_list(self):
        models_list = []
        fdir = join(self.datadir,'trend')
        for f in T.listdir(fdir):
            if not f.endswith('_trend.tif'):
                continue
            if 'during_early_peak' in f:
                model = f.replace('during_early_peak_', '')
                model = model.replace('_trend.tif', '')
                models_list.append(model)
            if 'during_late' in f:
                model = f.replace('during_late_', '')
                model = model.replace('_trend.tif', '')
                models_list.append(model)
        models_list = list(set(models_list))
        models_list.sort()

        return models_list


def main():
    Amplifying_Stablilizing().run()

    pass

if __name__ == '__main__':
    main()
