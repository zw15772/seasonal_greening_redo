# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from __init__ import *
centimeter_factor = 1 / 2.54
T = Tools()
result_root_this_script = '/Users/liyang/Desktop/process_MODIS_LAI_monthly'

class Plot_line:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Plot_line', result_root_this_script, mode=2)
        self.data_dir = '/Users/liyang/Desktop/process_MODIS_LAI_monthly/df'
        pass

    def run(self):
        self.plot()
        pass

    def plot(self):
        outdir = join(self.this_class_png, 'plot')
        T.mk_dir(outdir)
        mode = 'greening_browning'
        dff = join(self.data_dir, 'Build_process_early_greening_late_browning_05.df')

        # mode = 'greening_greening'
        # dff = join(self.data_dir, 'Build_process_early_greening_late_greening_05.df')

        df = T.load_df(dff)
        print(df.columns.tolist())
        anomaly_list = ['Precip', 'Temp', 'CCI_SM', 'GLEAM_SMroot', 'GLEAM_ET','Trendy_ensemble', 'LAI3g','MODIS_LAI'][::-1]
        climate_list = ['Rainfall', 'seasonal_Temperature_mean']
        VI_list = ['MODIS_LAI', 'LAI3g', 'Trendy_ensemble']
        # T.color_map_choice()
        # plt.show()
        color_list = T.gen_colors(len(anomaly_list),'turbo')

        Humid_Arid_list = T.get_df_unique_val_list(df, 'Humid_Arid')
        for zone in Humid_Arid_list:
            df_zone = df[df['Humid_Arid'] == zone]
            plt.figure(figsize=(8.8*centimeter_factor, 20*centimeter_factor))
            flag = 0
            y_ticks_list = []
            for anomaly in anomaly_list:
                vals = df_zone[anomaly].tolist()
                vals = np.array(vals)
                vals_mean = np.nanmean(vals, axis=0)
                vals_std = np.nanstd(vals, axis=0) / 4.
                # vals_mean = vals_mean + flag
                y_ticks = anomaly_list[flag]
                y_ticks_list.append(y_ticks)
                color = color_list[flag]
                plt.subplot(9, 1, flag + 1)
                plt.plot(vals_mean, label=anomaly, color=color, linewidth=1.5)
                plt.scatter(range(len(vals_mean)), vals_mean, s=20,linewidths=0,zorder=10, color=color)
                plt.fill_between(range(len(vals_mean)), vals_mean - vals_std, vals_mean + vals_std, alpha=0.2, color=color,linewidth=0)
                plt.hlines(0, 0, 11, colors='k', linestyles='dashed')
                plt.ylim(-0.8, 0.8)
                flag += 1
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)

                plt.title(f'{anomaly}')
                month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


            outf = join(outdir, f'{mode}_{zone}1.pdf')
            # plt.tight_layout()
            # plt.show()
            plt.savefig(outf)
            plt.close()

        pass


class Frequency:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Frequency', result_root_this_script, mode=2)
        self.data_dir = '/Users/liyang/Desktop/process_MODIS_LAI_monthly/npy'
        self.threshold_list = [0.00, 0.25, 0.5, 0.75, 1.00]
        self.product = 'MODIS'
        pass

    def run(self):
        # self.amplifying_tif()
        self.gen_df()
        pass


    def amplifying_tif(self):
        outdir = join(self.this_class_tif, 'amplifying_ratio')
        T.mk_dir(outdir)
        product = self.product
        threshold_list = self.threshold_list
        early_peak_f = join(self.data_dir,product, 'detrend_early_peak_MODIS_LAI_zscore.npy')
        late_f = join(self.data_dir,product, 'detrend_late_MODIS_LAI_zscore.npy')
        early_peak_dict = T.load_npy(early_peak_f)
        late_dict = T.load_npy(late_f)
        for threshold in threshold_list:
            threshold_str = f'{threshold:.2f}'
            amplifying_ratio_dict = {}
            for pix in tqdm(early_peak_dict,desc=f'{product}_{threshold_str}'):
                early_peak = early_peak_dict[pix]
                late = late_dict[pix]
                early_peak_condition = early_peak > threshold
                late_condition = late > threshold
                True_False = np.logical_and(early_peak_condition,late_condition)
                father = np.sum(early_peak_condition)
                son = np.sum(True_False)
                amplifying_ratio = son / father * 100
                amplifying_ratio_dict[pix] = amplifying_ratio
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(amplifying_ratio_dict)
            outf = join(outdir, f'{product}_{threshold_str}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        T.open_path_and_file(outdir)
        pass

    def gen_df(self):
        dff_template = '/Users/liyang/Desktop/process_MODIS_LAI_monthly/df/Build_process_early_greening_late_browning_05.df'
        fdir = join(self.this_class_tif, 'amplifying_ratio')
        threshold_list = self.threshold_list
        product = self.product
        df_template = T.load_df(dff_template)
        df_group_dict = T.df_groupby(df_template,'pix')
        Humid_Arid_spatial_dict = {}
        for pix in df_group_dict:
            df_i = df_group_dict[pix]
            Humid_Arid = df_i['Humid_Arid'].tolist()[0]
            Humid_Arid_spatial_dict[pix] = Humid_Arid

        all_spatial_dict = {}
        for threshold in threshold_list:
            threshold_str = f'{threshold:.2f}'
            f = join(fdir, f'{product}_{threshold_str}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
            all_spatial_dict[threshold_str] = spatial_dict
        df = T.spatial_dics_to_df(all_spatial_dict)
        df = T.add_spatial_dic_to_df(df,Humid_Arid_spatial_dict,'Humid_Arid')
        Humid_Arid_list = T.get_df_unique_val_list(df,'Humid_Arid')
        for Humid_Arid in Humid_Arid_list:
            print(Humid_Arid)
            df_Humid_Arid = df[df['Humid_Arid'] == Humid_Arid]
            x = []
            y = []
            for threshold in threshold_list:
                threshold_str = f'{threshold:.2f}'
                ratio_list = df_Humid_Arid[threshold_str].tolist()
                mean = np.nanmean(ratio_list)
                x.append(threshold)
                y.append(mean)
            # plt.figure()
            plt.plot(x,y,alpha=0.5,label=Humid_Arid)
            plt.scatter(x,y)
        plt.legend()
        plt.show()

def main():
    # Plot_line().run()
    Frequency().run()

if __name__ == '__main__':
    main()