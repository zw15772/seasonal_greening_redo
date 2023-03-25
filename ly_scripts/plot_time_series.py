# coding=utf-8

from __init__ import *
T = Tools()
class Foo:

    def __init__(self):
        self.data_dir = '/Users/liyang/Desktop/process_MODIS_LAI_monthly'
        pass

    def run(self):
        self.plot()
        pass

    def plot(self):
        dff = join(self.data_dir, 'Build_process_early_greening_late_browning_05.df')
        df = T.load_df(dff)
        print(df.columns)
        # exit()
        Humid_Arid_list = T.get_df_unique_val_list(df, 'Humid_Arid')
        for zone in Humid_Arid_list:
            df_zone = df[df['Humid_Arid'] == zone]
            T.print_head_n(df_zone, 10)
            exit()

        pass

def main():
    Foo().run()

if __name__ == '__main__':
    main()