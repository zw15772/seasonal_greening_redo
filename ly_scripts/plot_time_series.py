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

def main():
    Plot_line().run()

if __name__ == '__main__':
    main()