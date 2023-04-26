# coding=utf-8
import matplotlib.pyplot as plt
from lytools import *

T = Tools()

this_script_root = '/Users/liyang/Downloads/earlier_late/'

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
        # self.plot_tif()
        self.ratio_statistic()

    def class_tif(self):
        outdir = join(self.this_class_tif,'class_tif')
        T.mk_dir(outdir)
        class_label_dict = {'stablilizing': -1, 'weak amplifying': 0, 'amplifying': 1}
        for model in tqdm(self.models_list):
            early_peak_f = join(self.datadir, f'during_early_peak_{model}_trend.tif')
            late_f = join(self.datadir, f'during_late_{model}_trend.tif')
            early_peak_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_f)
            late_dict = DIC_and_TIF().spatial_tif_to_dic(late_f)
            class_dict = {}
            class_dict_num = {}
            for pix in early_peak_dict:
                early_peak = early_peak_dict[pix]
                late = late_dict[pix]
                if not early_peak > 0:
                    continue
                if late <=0:
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
        color_list = ['purple', '#FFFFCC', 'g']
        cmap = T.cmap_blend(color_list)
        plt.register_cmap(name='mycmap', cmap=cmap)
        for f in T.listdir(fdir):
            if not 'MODIS' in f:
                continue
            # if not 'LAI3g' in f:
            #     continue
            # if not 'Trendy_ensemble' in f:
            #     continue
            fpath = join(fdir, f)
            Plot().plot_ortho(fpath,cmap='mycmap',vmin=-1,vmax=1)
            outf = join(outdir, f.replace('.tif', '.png'))
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def ratio_statistic(self):
        fdir = join(self.this_class_tif,'class_tif')
        all_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            key = f.replace('.tif', '')
            all_dict[key] = dict_i
        df = T.spatial_dics_to_df(all_dict)
        df = df.dropna()
        spatial_dict = {}
        for i,row in df.iterrows():
            pix = row['pix']
            spatial_dict[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,interpolation='nearest',cmap='gray')
        plt.show()

        pass

    def __get_model_list(self):
        models_list = []
        for f in T.listdir(self.datadir):
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
