# coding=utf-8
import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import Main_flow
from preprocess import *
from __init__ import *
results_root_main_flow = join(results_root, 'Main_flow')
global_n = 15

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
        # df = df[df['lat'] < 60]
        # df = df[df['lat'] > 30]
        # df = df[df['HI_reclass']=='Non Humid']  # focus on dryland
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


    def P_PET_ratio(self):
        fdir = '/Volumes/SSD_sumsang/project_greening_redo/data/aridity_P_PET_dic'
        # fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            vals[vals == 0] = np.nan
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
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Non Humid'
        return df

    def add_lc(self,df):
        lc_f = join(GLC2000().datadir,'lc_dic_reclass.npy')
        lc_dic = T.load_npy(lc_f)
        df = T.add_spatial_dic_to_df(df,lc_dic,'GLC2000')
        return df

    def add_NDVI_mask(self, df):
        # ndvi_mask_tif = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
        ndvi_mask_tif = '/Volumes/NVME2T/greening_project_redo/conf/NDVI_mask.tif'
        # ndvi_mask_tif = join(NDVI().datadir,'NDVI_mask.tif')
        arr = ToRaster().raster2array(ndvi_mask_tif)[0]
        arr = T.mask_999999_arr(arr, warning=False)
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        df = T.add_spatial_dic_to_df(df, dic, 'ndvi_mask_mark')
        df = pd.DataFrame(df)
        df = df.dropna(subset=['ndvi_mask_mark'])
        return df

    def get_valid_pix_list(self):
        outdir = join(this_root,'temporary/get_valid_pix_list')
        T.mkdir(outdir,force=True)
        outf = join(outdir,'pix_list.npy')
        if isfile(outf):
            return np.load(outf)
        df = pd.DataFrame()
        void_dic = DIC_and_TIF().void_spatial_dic()
        pix_list_void = []
        for pix in void_dic:
            pix_list_void.append(pix)
        df['pix'] = pix_list_void
        df = self.add_NDVI_mask(df)
        df = self.add_Humid_nonhumid(df)
        df = df.dropna()
        pix_list = T.get_df_unique_val_list(df,'pix')
        np.save(outf,pix_list)
        return pix_list
        pass

    def add_lon_lat_to_df(self,df):
        lon_list = []
        lat_list = []
        D = DIC_and_TIF()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            lon,lat = D.pix_to_lon_lat(pix)
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def get_valid_pix_df(self):
        outdir = join(this_root,'temporary/get_valid_pix_df')
        T.mkdir(outdir,force=True)
        outf = join(outdir,'dataframe.df')
        if isfile(outf):
            return T.load_df(outf)
        df = pd.DataFrame()
        void_dic = DIC_and_TIF().void_spatial_dic()
        pix_list_void = []
        for pix in void_dic:
            pix_list_void.append(pix)
        df['pix'] = pix_list_void
        df = self.add_NDVI_mask(df)
        df = self.add_Humid_nonhumid(df)
        df = self.add_lon_lat_to_df(df)
        df = df[df['lat']>30]
        df = df.dropna()
        T.save_df(df,outf)
        return df



class HANTS:

    def __init__(self):

        pass

    # Computing diagonal for each row of a 2d array. See: http://stackoverflow.com/q/27214027/2459096
    def makediag3d(self,M):
        b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
        b[:, ::M.shape[1] + 1] = M
        return b.reshape((M.shape[0], M.shape[1], M.shape[1]))

    def get_starter_matrix(self,base_period_len, sample_count, frequencies_considered_count):
        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images
        mat = np.zeros(shape=(nr, sample_count))
        mat[0, :] = 1
        ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
        cs = np.cos(ang)
        sn = np.sin(ang)
        # create some standard sinus and cosinus functions and put in matrix
        i = np.arange(1, frequencies_considered_count + 1)
        ts = np.arange(sample_count)
        for column in range(sample_count):
            index = np.mod(i * ts[column], base_period_len)
            # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
            mat[2 * i - 1, column] = cs.take(index)
            mat[2 * i, column] = sn.take(index)

        return mat

    # import profilehooks
    # @profilehooks.profile(sort='time')
    def HANTS(self,sample_count, inputs,
              frequencies_considered_count=3,
              outliers_to_reject='Hi',
              low=0., high=255,
              fit_error_tolerance=5.,
              delta=0.1):
        """
        Function to apply the Harmonic analysis of time series applied to arrays
        sample_count    = nr. of images (total number of actual samples of the time series)
        base_period_len    = length of the base period, measured in virtual samples
                (days, dekads, months, etc.)
        frequencies_considered_count    = number of frequencies to be considered above the zero frequency
        inputs     = array of input sample values (e.g. NDVI values)
        ts    = array of size sample_count of time sample indicators
                (indicates virtual sample number relative to the base period);
                numbers in array ts maybe greater than base_period_len
                If no aux file is used (no time samples), we assume ts(i)= i,
                where i=1, ..., sample_count
        outliers_to_reject  = 2-character string indicating rejection of high or low outliers
                select from 'Hi', 'Lo' or 'None'
        low   = valid range minimum
        high  = valid range maximum (values outside the valid range are rejeced
                right away)
        fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
                fit are rejected)
        dod   = degree of overdeterminedness (iteration stops if number of
                points reaches the minimum required for curve fitting, plus
                dod). This is a safety measure
        delta = small positive number (e.g. 0.1) to suppress high amplitudes
        """

        # define some parameters
        base_period_len = sample_count  #

        # check which setting to set for outlier filtering
        if outliers_to_reject == 'Hi':
            sHiLo = -1
        elif outliers_to_reject == 'Lo':
            sHiLo = 1
        else:
            sHiLo = 0

        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images

        # create empty arrays to fill
        outputs = np.zeros(shape=(inputs.shape[0], sample_count))

        mat = self.get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

        # repeat the mat array over the number of arrays in inputs
        # and create arrays with ones with shape inputs where high and low values are set to 0
        mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
        p = np.ones_like(inputs)
        p[(low >= inputs) | (inputs > high)] = 0
        nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

        # prepare for while loop
        ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

        dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
        noutmax = sample_count - nr - dod
        for _ in range(sample_count):
            if ready.all():
                break
            # print '--------*-*-*-*',it.value, '*-*-*-*--------'
            # multiply outliers with timeseries
            za = np.einsum('ijk,ik->ij', mat, p * inputs)

            # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
            diag = self.makediag3d(p)
            A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
            # add delta to suppress high amplitudes but not for [0,0]
            A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
            A[:, 0, 0] = A[:, 0, 0] - delta

            # solve linear matrix equation and define reconstructed timeseries
            zr = np.linalg.solve(A, za)
            outputs = np.einsum('ijk,kj->ki', mat.T, zr)

            # calculate error and sort err by index
            err = p * (sHiLo * (outputs - inputs))
            rankVec = np.argsort(err, axis=1, )

            # select maximum error and compute new ready status
            maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
            ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)

            # if ready is still false
            if not ready.all():
                j = rankVec.take(sample_count - 1, axis=-1)

                p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                    int)  # *check
                nout += 1

        return outputs

    # Compute semi-random time series array with numb standing for number of timeseries
    def array_in(self,numb):
        y = np.array([5.0, 2.0, 10.0, 12.0, 18.0, 23.0, 27.0, 40.0, 60.0, 70.0, 90.0, 160.0, 190.0,
                      210.0, 104.0, 90.0, 170.0, 50.0, 120.0, 60.0, 40.0, 30.0, 28.0, 24.0, 15.0,
                      10.0])
        y = np.tile(y[None].T, (1, numb)).T
        kl = (np.random.randint(2, size=(numb, 26)) *
              np.random.randint(2, size=(numb, 26)) + 1)
        kl[kl == 2] = 0
        y = y * kl
        return y

class Phenology:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Phenology',
                                                                                       results_root_main_flow)

        self.product = 'LAI3g'
        # self.product = 'MODIS_LAI'


        self.datadir=join(data_root,self.product)
        self.resultdir = join(results_root, self.product)
        # Tools().open_path_and_file(self.this_class_arr)

    def run(self):

        # fdir = join(data_root,'tif_05')
        # outdir = join(self.datadir,'per_pix_annual')
        # self.data_transform_annual(fdir,outdir)
        # self.modify_first_year()
        # 3 hants smooth
        # self.hants()
        # self.check_hants()

        # self.annual_phenology(self.product)
        # self.compose_annual_phenology(self.product)
        # self.pick_daily_phenology()
        # self.pick_month_phenology()
        # self.data_clean(self.product)
        # self.average_phenology(self.product)
        # self.check_SOS_EOS(self.product)
        # self.check_compose_hants()
        self.check_pixel_phenology(self.product)
        # self.all_year_hants_annual()
        # self.all_year_hants_annual_mean()
        # self.all_year_hants()
        # self.longterm_mean_phenology()
        # self.check_lonterm_phenology()

        pass



    def compose_SOS_EOS(self):
        outdir = self.this_class_arr + 'compose_SOS_EOS/'
        T.mkdir(outdir)
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


    def kernel_hants(self, params):
        outdir, y, fdir = params
        outf = join(outdir,y)
        dic = T.load_npy_dir(join(fdir,y))
        hants_dic = {}
        spatial_dic = {}
        for pix in tqdm(dic,desc=y):
            r,c = pix
            if r > 180:
                continue
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            if T.is_all_nan(vals):
                continue
            vals = T.interp_nan(vals)
            if len(vals) == 1:
                continue
            spatial_dic[pix] = 1
            xnew, ynew = self.__interp__(vals)
            std = np.nanstd(ynew)
            std = float(std)
            ynew = np.array([ynew])
            results = HANTS().HANTS(sample_count=365, inputs=ynew, low=0, high=10,
                            fit_error_tolerance=std)
            result = results[0]
            if T.is_all_nan(result):
                continue
            hants_dic[pix] = result
        T.save_npy(hants_dic,outf)

    def modify_first_year(self):
        fdir = join(data_root,self.product,'per_pix_annual','2000')
        for f in tqdm(T.listdir(fdir)):
            dic = T.load_npy(join(fdir,f))
            for pix in dic:
                vals = dic[pix]
                vals = np.array(vals)
                if len(vals) == 20:
                    for i in range(3):
                        vals = np.insert(vals,0,-999999)
                dic[pix] = vals
            T.save_npy(dic,join(fdir,f))
        pass


    def hants(self):
        outdir = join(self.this_class_arr,f'hants_{self.product}')
        T.mkdir(outdir)
        fdir = join(self.datadir,'per_pix_annual')
        params = []
        for y in T.listdir(fdir):
            params.append([outdir, y, fdir])
            # self.kernel_hants([outdir, y, fdir])
        MULTIPROCESS(self.kernel_hants, params).run(process=4)

    def check_hants(self):
        fdir = join(self.this_class_arr,'hants')
        for year in T.listdir(fdir):
            f = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Phenology/1983.npy'
            dic = T.load_npy(f)
            # dic = T.load_npy(join(fdir,year))
            spatial_dic = {}
            for pix in dic:
                vals = dic[pix]
                spatial_dic[pix] = len(vals)
                # print(len(vals))
                # exit()
                # if len(vals) > 0:
                #     # print pix,vals
                #     plt.plot(vals)
                #     plt.show()
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF().plot_back_ground_arr(global_land_tif)
            plt.imshow(arr)
            plt.show()
            exit()
    def check_compose_hants(self):
        fdir = join(self.this_class_arr,'hants')
        f = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Phenology/compose_Early_Peak_Late_pick/phenology_dataframe.df'
        df = T.load_df(f)
        dic = T.df_to_dic(df,'pix')
        # dic = T.load_npy(join(fdir,year))
        spatial_dic = {}
        for pix in dic:
            vals = dic[pix]
            for key in vals:
                val = vals[key]
                print(key)
                print(val)
            exit()
            spatial_dic[pix] = len(vals)
            # print(len(vals))
            # exit()
            # if len(vals) > 0:
            #     # print pix,vals
            #     plt.plot(vals)
            #     plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.imshow(arr)
        plt.show()
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

    def __median_early_late(self,vals,sos,eos,peak):
        # 2 使用sos-peak peak-eos中位数作为sos和eos的结束和开始

        median_left = int((peak-sos)/2.)
        median_right = int((eos - peak)/2)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind

    def __day_to_month(self,doy):
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    # def SOS_EOS(self, threshold_i=0.5):
    #     out_dir = join(self.this_class_arr,'SOS_EOS')
    #     T.mkdir(out_dir)
    #     fdir = join(self.this_class_arr,'hants')
    #     for y in tqdm(T.listdir(fdir)):
    #         dic = T.load_npy(join(fdir,y))
    #         result_dic = {}
    #         for pix in dic:
    #             try:
    #                 vals = dic[pix]
    #                 maxind = np.argmax(vals)
    #                 start = self.__search_left(vals, maxind, threshold_i)
    #                 end = self.__search_right(vals, maxind, threshold_i)
    #                 # result = [start,maxind, end]
    #                 result = {
    #                     'SOS':start,
    #                     'Peak':maxind,
    #                     'EOS':end,
    #                 }
    #                 result_dic[pix] = result
    #             except:
    #                 pass
    #         df = T.dic_to_df(result_dic,'pix')

    def pick_phenology(self,vals,threshold_i):
        peak = np.argmax(vals)
        if peak == 0 or peak == (len(vals) - 1):
            return {}
        try:
            early_start = self.__search_left(vals, peak, threshold_i)
            late_end = self.__search_right(vals, peak, threshold_i)
        except:
            early_start = 60
            late_end = 130
            print(vals)
            plt.plot(vals)
            plt.show()
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
        # method 2
        early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

        early_period = early_end - early_start
        peak_period = late_start - early_end
        late_period = late_end - late_start
        dormant_period = 365 - (late_end - early_start)

        result = {
            'early_length': early_period,
            'mid_length': peak_period,
            'late_length': late_period,
            'dormant_length': dormant_period,
            'early_start': early_start,
            'early_start_mon': self.__day_to_month(early_start),

            'early_end': early_end,
            'early_end_mon': self.__day_to_month(early_end),

            'peak': peak,
            'peak_mon': self.__day_to_month(peak),

            'late_start': late_start,
            'late_start_mon': self.__day_to_month(late_start),

            'late_end': late_end,
            'late_end_mon': self.__day_to_month(late_end),
        }
        return result
        pass


    def annual_phenology(self,product,threshold_i=0.2):
        out_dir = join(self.this_class_arr, 'annual_phenology',product)
        T.mkdir(out_dir,force=True)
        hants_smooth_dir = join(self.this_class_arr, 'hants',product)
        for f in T.listdir(hants_smooth_dir):
            year = int(f.split('.')[0])
            outf_i = join(out_dir,f'{year}.df')
            hants_smooth_f = join(hants_smooth_dir,f)
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}
            for pix in tqdm(hants_dic,desc=str(year)):
                vals = hants_dic[pix]
                result = self.pick_phenology(vals,threshold_i)
                result_dic[pix] = result
            df = T.dic_to_df(result_dic,'pix')
            T.save_df(df,outf_i)
            T.df_to_excel(df,outf_i)
            np.save(outf_i,result_dic)

    def compose_annual_phenology(self,product):
        f_dir = join(self.this_class_arr, 'annual_phenology',product)
        outdir = join(self.this_class_arr,'compose_annual_phenology',product)
        T.mkdir(outdir,force=True)
        outf = join(outdir,'phenology_dataframe.df')
        all_result_dic = {}
        pix_list_all = []
        col_list = None
        for f in T.listdir(f_dir):
            if not f.endswith('.df'):
                continue
            df = T.load_df(join(f_dir,f))
            pix_list = T.get_df_unique_val_list(df,'pix')
            pix_list_all.append(pix_list)
            col_list = df.columns
        all_pix = []
        for pix_list in pix_list_all:
            for pix in pix_list:
                all_pix.append(pix)
        pix_list = T.drop_repeat_val_from_list(all_pix)

        col_list = col_list.to_list()
        col_list.remove('pix')
        for pix in pix_list:
            dic_i = {}
            for col in col_list:
                dic_i[col] = {}
            all_result_dic[pix] = dic_i
        # print(len(T.listdir(f_dir)))
        # exit()
        for f in tqdm(T.listdir(f_dir)):
            if not f.endswith('.df'):
                continue
            year = int(f.split('.')[0])
            df = T.load_df(join(f_dir,f))
            dic = T.df_to_dic(df,'pix')
            for pix in dic:
                for col in dic[pix]:
                    if col == 'pix':
                        continue
                    all_result_dic[pix][col][year] = dic[pix][col]
        df_all = T.dic_to_df(all_result_dic,'pix')
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf)
        np.save(outf,all_result_dic)

    def data_clean(self,product):  # 盖帽法

        f_dir = join(self.this_class_arr, 'compose_annual_phenology', product)
        outdir = join(self.this_class_arr, 'compose_annual_phenology_clean', product)
        T.mkdir(outdir, force=True)
        outf = join(outdir, 'phenology_dataframe.df')
        all_result_dic = {}
        pix_list_all = []

        for f in T.listdir(f_dir):
            if not f.endswith('.df'):
                continue
            df = T.load_df(join(f_dir, f))
            columns=df.columns
            column_list=[]
            for col in columns:
                if col=='pix':
                    continue
                column_list.append(col)
            for i, row in df.iterrows():
                pix = row['pix']
                lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                if lat>50:
                    continue
                address=Tools().lonlat_to_address(lon,lat)
                print(address)
                for col in column_list:
                    dic_i=row[col]
                    print(dic_i)
                    # values_list=dic_i.values()
                    # values_list=list(values_list)
                    series=pd.Series(dic_i)
                    cap_series=self.cap(series)
                    print(series)


                    series.hist(bins=50)


                    plt.figure()

                    print(cap_series)
                    plt.plot(series)
                    plt.title(address)
                    plt.plot(cap_series)
                    plt.show()

    def average_phenology(self,product):

        f_dir = join(self.this_class_arr, 'compose_annual_phenology', product)
        outdir = join(self.this_class_arr, 'average_phenology', product)
        T.mkdir(outdir, force=True)
        outf = join(outdir, 'phenology_dataframe.df')
        all_result_dic = {}


        for f in T.listdir(f_dir):
            if not f.endswith('.df'):
                continue
            df = T.load_df(join(f_dir, f))
            columns=df.columns
            column_list=[]
            for col in columns:
                if col=='pix':
                    continue
                column_list.append(col)

            pix_list = T.get_df_unique_val_list(df, 'pix')

########################################build dic##############################################################
            for pix in pix_list:
                dic_i = {}
                for col in column_list:
                    dic_i[col] = {}
                all_result_dic[pix] = dic_i

            for i, row in tqdm(df.iterrows(),total=len(df)):
                pix = row['pix']
                lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                # address=Tools().lonlat_to_address(lon,lat)
                # print(address)
                for col in column_list:
                    dic_i=row[col]
                    # print(dic_i)
                    values=dic_i.values()
                    values=list(values)
                    value_mean=np.mean(values)
                    value_=round(value_mean,0)
                    value_std=np.std(values)
                    all_result_dic[pix][col] = value_

        df_all = T.dic_to_df(all_result_dic, 'pix')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all, outf)
        np.save(outf, all_result_dic)



    def cap(self,x, quantile=(0.05, 0.95)):

        """盖帽法处理异常值
        Args：
            x：pd.Series列，连续变量
            quantile：指定盖帽法的上下分位数范围
        """

        # 生成分位数
        Q01, Q99 = x.quantile(quantile).values.tolist()

        # 替换异常值为指定的分位数
        if Q01 > x.min():
            x = x.copy()
            x.loc[x < Q01] = Q01

        if Q99 < x.max():
            x = x.copy()
            x.loc[x > Q99] = Q99

        return (x)

    def check_pixel_phenology(self,product):
        fdir = join(self.this_class_arr, 'Get_Monthly_Early_Peak_Late', product+'/')
        # fdir='/Volumes/SSD_sumsang/project_greening/Result/new_result/Main_flow/arr/Phenology/average_phenology/MODIS_LAI/'
        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = T.load_npy(fdir+f)
            spatial_dic = {}
            for pix in dic:

                SOS = dic[pix]['late']
                if len(SOS)==0:
                    continue
                SOS=SOS[0]
                spatial_dic[pix] = SOS
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr,vmin=180,vmax=350,cmap='jet')
            plt.imshow(arr, vmin=6, vmax=11, cmap='jet')
            plt.colorbar()
            plt.show()
        pass

    def check_SOS_EOS(self,product):
        fdir = join(self.this_class_arr, 'Get_Monthly_Early_Peak_Late', product+'/')
        # fdir='/Volumes/SSD_sumsang/project_greening/Result/new_result/Main_flow/arr/Phenology/average_phenology/MODIS_LAI/'
        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = T.load_npy(fdir+f)
            spatial_dic = {}
            for pix in dic:

                SOS = dic[pix]['late']
                if len(SOS)==0:
                    continue
                SOS=SOS[0]
                spatial_dic[pix] = SOS
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr,vmin=180,vmax=350,cmap='jet')
            plt.imshow(arr, vmin=6, vmax=11, cmap='jet')
            plt.colorbar()
            plt.show()
        pass

    def pick_daily_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]

        # product='MODIS_LAI' #  240 20yr
        # variable_list = ['LAI4g'] #长度39 468
        product = 'LAI3g'  # 长度37 444

        phenology_df = T.load_df(
            results_root + f'Main_flow/arr/Phenology/compose_annual_phenology_average/{product}/phenology_dataframe_{product}.df')

        outdir = results_root + f'Main_flow/arr/Phenology/pick_daily_phenology/{product}/'
        T.mkdir(outdir, force=True)
        outf= results_root + f'Main_flow/arr/Phenology/pick_daily_phenology/{product}/pick_daily.df'
        early_dic = {}
        peak_dic = {}
        late_dic = {}
        all_result_dic={}

        for i, row in tqdm(phenology_df.iterrows(),total=len(phenology_df)):
            pix = row['pix']
            all_result_dic[pix] = {}
            early_start=row['early_start']
            early_end = row['early_end']
            peak_start = row['early_end']
            peak_end = row['late_start']
            late_start = row['late_start']
            late_end = row['late_end']
            early_period= np.arange(int(early_start),int(early_end),1)
            # print(early_period)
            peak_period = np.arange(int(early_end),int(late_start),1)
            # print(peak_period)
            late_period = np.arange(int(late_start),int(late_end),1)
            # print(late_period)
            all_result_dic[pix]['early']=early_period
            all_result_dic [pix]['peak'] = peak_period
            all_result_dic [pix]['late'] = late_period

        df_all = T.dic_to_df(all_result_dic, 'pix')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all, outf)
        np.save(outf, all_result_dic)
        
    def pick_month_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]

        # product='MODIS_LAI' #  240 20yr
        # variable_list = ['LAI4g'] #长度39 468
        product = 'LAI3g'  # 长度37 444

        phenology_df = T.load_df(
            results_root + f'Main_flow/arr/Phenology/compose_annual_phenology_average/{product}/phenology_dataframe_{product}.df')

        outdir = results_root + f'Main_flow/arr/Phenology/pick_monthly_phenology/{product}/'
        T.mkdir(outdir, force=True)
        outf= results_root + f'Main_flow/arr/Phenology/pick_monthly_phenology/{product}/pick_daily.df'
        early_dic = {}
        peak_dic = {}
        late_dic = {}
        all_result_dic={}

        for i, row in tqdm(phenology_df.iterrows(),total=len(phenology_df)):
            pix = row['pix']
            all_result_dic[pix] = {}
            early_start=row['early_start_mon']
            early_end = row['early_end_mon']
            peak_start = row['early_end_mon']
            peak_end = row['late_start_mon']
            late_start = row['late_start_mon']
            late_end = row['late_end_mon']
            early_period= np.arange(int(early_start),int(early_end),1)
            # print(early_period)
            peak_period = np.arange(int(early_end),int(late_start),1)
            # print(peak_period)
            late_period = np.arange(int(late_start),int(late_end),1)
            # print(late_period)
            all_result_dic[pix]['early']=early_period
            all_result_dic [pix]['peak'] = peak_period
            all_result_dic [pix]['late'] = late_period

        df_all = T.dic_to_df(all_result_dic, 'pix')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all, outf)
        np.save(outf, all_result_dic)



    def data_transform_annual(self,fdir,outdir,date_fmt='yyyymmdd'):
        T.mkdir(outdir)
        year_list = []
        for f in T.listdir(fdir):
            y, m, d = Pre_Process().get_year_month_day(f,date_fmt=date_fmt)
            year_list.append(y)
        year_list = T.drop_repeat_val_from_list(year_list)
        # print(year_list)
        # exit()
        for year in year_list:
            outdir_i = join(outdir,f'{year}')
            T.mkdir(outdir_i)
            annual_f_list = []
            for f in T.listdir(fdir):
                y, m, d = Pre_Process().get_year_month_day(f,date_fmt=date_fmt)
                if y == year:
                    annual_f_list.append(f)
            Pre_Process().data_transform_with_date_list(fdir,outdir_i,annual_f_list)
            print(annual_f_list)
            print(year)
            # exit()
        # Pre_Process().monthly_compose()
        # exit()

    def all_year_hants(self):
        fdir = join(self.this_class_arr,'hants')
        outdir = join(self.this_class_arr,'all_year_hants')
        T.mkdir(outdir)
        pix_list = None
        for f in T.listdir(fdir):
            dic = T.load_npy(join(fdir,f))
            pix_list = []
            for pix in dic:
                pix_list.append(pix)
            break
        hants_vals_all_year_dic = {}
        for pix in pix_list:
            hants_vals_all_year_dic[pix] = []
        for f in T.listdir(fdir):
            print(f)
            dic = T.load_npy(join(fdir,f))
            for pix in dic:
                vals = dic[pix]
                if not pix in hants_vals_all_year_dic:
                    continue
                for val in vals:
                    hants_vals_all_year_dic[pix].append(val)
        T.save_distributed_perpix_dic(hants_vals_all_year_dic,outdir,1000)

    def all_year_hants_annual(self):
        fdir = join(self.this_class_arr,'hants')
        outdir = join(self.this_class_arr,'all_year_hants_annual')
        year_list = list(range(1982,2999))
        T.mkdir(outdir)
        pix_list = None
        for f in T.listdir(fdir):
            dic = T.load_npy(join(fdir,f))
            pix_list = []
            for pix in dic:
                pix_list.append(pix)
            break
        hants_vals_all_year_dic = {}
        for pix in pix_list:
            hants_vals_all_year_dic[pix] = []
        for f in T.listdir(fdir):
            print(f)
            dic = T.load_npy(join(fdir,f))
            for pix in dic:
                vals = dic[pix]
                if not pix in hants_vals_all_year_dic:
                    continue
                for val in vals:
                    hants_vals_all_year_dic[pix].append(val)
        hants_vals_all_year_dic_annual_dic = {}
        for pix in tqdm(hants_vals_all_year_dic):
            vals = hants_vals_all_year_dic[pix]
            vals = np.array(vals)
            vals_reshape = np.reshape(vals,(-1,365))
            dic_i = dict(zip(year_list,vals_reshape))
            dic_i = dic_i.items()
            dic_i = list(dic_i)
            dic_i = np.array(dic_i,dtype=object)
            hants_vals_all_year_dic_annual_dic[pix] = dic_i
        T.save_distributed_perpix_dic(hants_vals_all_year_dic_annual_dic,outdir,1000)

    def all_year_hants_annual_mean(self):
        fdir = join(self.this_class_arr,'hants')
        outdir = join(self.this_class_arr,'all_year_hants_annual_mean')
        year_list = list(range(1982,2999))
        T.mkdir(outdir)
        pix_list = None
        for f in T.listdir(fdir):
            dic = T.load_npy(join(fdir,f))
            pix_list = []
            for pix in dic:
                pix_list.append(pix)
            break
        hants_vals_all_year_dic = {}
        for pix in pix_list:
            hants_vals_all_year_dic[pix] = []
        for f in T.listdir(fdir):
            print(f)
            dic = T.load_npy(join(fdir,f))
            for pix in dic:
                vals = dic[pix]
                if not pix in hants_vals_all_year_dic:
                    continue
                for val in vals:
                    hants_vals_all_year_dic[pix].append(val)
        hants_vals_all_year_dic_annual_mean_dic = {}
        for pix in tqdm(hants_vals_all_year_dic):
            vals = hants_vals_all_year_dic[pix]
            vals = np.array(vals)
            vals_reshape = np.reshape(vals,(-1,365))
            vals_reshape_mean = np.mean(vals_reshape,axis=0)
            hants_vals_all_year_dic_annual_mean_dic[pix] = vals_reshape_mean
        T.save_distributed_perpix_dic(hants_vals_all_year_dic_annual_mean_dic,outdir,10000)

    def longterm_mean_phenology(self,threshold=0.2):
        fdir = join(self.this_class_arr,'all_year_hants')
        outdir = join(self.this_class_arr,'longterm_mean_phenology')
        T.mkdir(outdir)
        outf = join(outdir,'longterm_mean_phenology.df')
        result_dic = {}
        for f in tqdm(T.listdir(fdir)):
            dic = T.load_npy(join(fdir,f))
            for pix in dic:
                vals = dic[pix]
                vals_reshape = np.reshape(vals,(-1,365))
                annual_mean_vals = np.mean(vals_reshape,axis=0)
                result = self.pick_phenology(annual_mean_vals,threshold)
                result_dic[pix] =result
        df = T.dic_to_df(result_dic,'pix')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        pass

    def check_lonterm_phenology(self):
        f = join(self.this_class_arr,'longterm_mean_phenology/longterm_mean_phenology.df')
        df = T.load_df(f)
        cols = df.columns
        print(cols)
        # exit()
        spatial_dic = T.df_to_spatial_dic(df,'late_end_mon')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
        pass


class Get_Monthly_Early_Peak_Late:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('phenology',results_root_main_flow)

    def run(self):
        # self.Monthly_Early_Peak_Late()
        self.Monthly_Early_Peak_Late_via_DOY()
        # self.check_pix()
        pass


    def Monthly_Early_Peak_Late(self):

        outf = join(self.this_class_arr,'Monthly_Early_Peak_Late.df')

        product='MODIS_LAI'
        phenology_df = T.load_df(
            results_root + f'Main_flow/arr/Phenology/compose_annual_phenology_average/{product}/phenology_dataframe_{product}.df')
        phenology_df=phenology_df.dropna()


        sos_dict = T.df_to_spatial_dic(phenology_df,'early_start_mon')
        eos_dict = T.df_to_spatial_dic(phenology_df,'late_end_mon')
        vege_dir = f'/Volumes/SSD_sumsang/project_greening/Data/original_dataset/{product}_dic/'  # monthly
        vege_dic = T.load_npy_dir(vege_dir)
        result_dic = {}
        for pix in tqdm(vege_dic):
            if pix not in sos_dict:
                continue
            vals = vege_dic[pix]
            if T.is_all_nan(vals):
                continue
            sos = int(sos_dict[pix])
            eos = int(eos_dict[pix])

            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            val_reshape = vals.reshape((-1,12))
            val_reshape_T = val_reshape.T
            month_mean_list = []
            for month in val_reshape_T:
                month_mean = np.nanmean(month)
                month_mean_list.append(month_mean)
            isnan_list = np.isnan(month_mean_list)

            max_n_index,max_n_val = T.pick_max_n_index(month_mean_list,n=2)
            peak_months_distance = abs(max_n_index[0]-max_n_index[1])


            max_n_index = list(max_n_index)
            max_n_index.sort()
            if max_n_index[0]<3:
                continue
            if max_n_index[1]>=10:
                continue
            peak_mon = np.array(max_n_index) + 1
            if sos > peak_mon[0]:
                continue
            if peak_mon[-1]+1 > eos+1:
                continue
            early_mon = list(range(sos,peak_mon[0]))
            late_mon = list(range(peak_mon[-1]+1,eos+1))
            # print(early_mon)

            if peak_months_distance >= 2:
                early_mon = list(range(sos, peak_mon[0]))
                peak_mon = list(range(peak_mon[0],peak_mon[1]+1))
                late_mon = list(range(peak_mon[-1] + 1, eos+1))
                # print(peak_months_distance)
                # print(early_mon)
                # print(peak_mon)
                # print(late_mon)
                # print('--')
                # plt.plot(month_mean_list)
                # plt.show()
            early_mon = np.array(early_mon)
            peak_mon = np.array(peak_mon)
            late_mon = np.array(late_mon)
            # print(month_mean_list)
            # print(early_mon)
            # print(peak_mon)
            # print(late_mon)
            # plt.plot(month_mean_list)
            # plt.show()
            # exit()
            result_dic_i = {
                'early':early_mon,
                'peak':peak_mon,
                'late':late_mon,
            }
            result_dic[pix] = result_dic_i
        df = T.dic_to_df(result_dic,'pix')
        # df = df.dropna()
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        T.save_npy(result_dic,outf)

    def Monthly_Early_Peak_Late_via_DOY(self):

        outf = join(self.this_class_arr, 'Monthly_Early_Peak_Late_via_DOY.df')
        product = 'LAI3g'
        phenology_df = T.load_df(
            results_root + f'Main_flow/arr/Phenology/average_phenology/{product}/phenology_dataframe_{product}.df')
        phenology_df = phenology_df.dropna()
        # T.print_head_n(phenology_df,5)

        early_start_dict = T.df_to_spatial_dic(phenology_df, 'early_start')
        early_end_dict = T.df_to_spatial_dic(phenology_df, 'early_end')
        late_start_dict = T.df_to_spatial_dic(phenology_df, 'late_start')
        late_end_dict = T.df_to_spatial_dic(phenology_df, 'late_end')

        DOY_to_mon_dict = {}
        base_time = datetime.datetime(2000, 1, 1)
        for i in range(1,366):
            time_i = base_time + datetime.timedelta(days=i-1)
            mon = time_i.month
            DOY_to_mon_dict[i] = mon
        # DOY_to_mon_dict_reverse = T.reverse_dic(DOY_to_mon_dict)
        # print(DOY_to_mon_dict_reverse)

        vege_dir = f'/Volumes/SSD_sumsang/project_greening/Data/original_dataset/{product}_dic/'  # monthly
        # vege_dir = f'/Volumes/NVME2T/greening_project_redo/data/BU_MCD_LAI_CMG/resample_bill_05_monthly_max_compose_per_pix/'  # monthly
        vege_dic = T.load_npy_dir(vege_dir)
        # exit()
        result_dic = {}
        for pix in tqdm(vege_dic):
            r,c=pix
            # if r>150:
            #     continue
            # if r<120:
            #     continue
            if pix not in early_start_dict:
                continue
            vals = vege_dic[pix]
            if T.is_all_nan(vals):
                continue
            early_start = int(early_start_dict[pix])
            early_end = int(early_end_dict[pix])
            late_start = int(late_start_dict[pix])
            late_end = int(late_end_dict[pix])
            early_range = list(range(early_start,early_end+1))
            peak_range = list(range(early_end+1,late_start))
            late_range = list(range(late_start,late_end+1))
            early_range_mon = [DOY_to_mon_dict[i] for i in early_range]
            peak_range_mon = [DOY_to_mon_dict[i] for i in peak_range]
            late_range_mon = [DOY_to_mon_dict[i] for i in late_range]

            early_mon = sorted(list(set(early_range_mon)))
            peak_mon = sorted(list(set(peak_range_mon)))[1:]
            late_mon = sorted(list(set(late_range_mon)))[1:]

            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            val_reshape = vals.reshape((-1, 12))
            val_reshape_T = val_reshape.T
            month_mean_list = []
            for month in val_reshape_T:
                month_mean = np.nanmean(month)
                month_mean_list.append(month_mean)
            isnan_list = np.isnan(month_mean_list)

            max_n_index, max_n_val = T.pick_max_n_index(month_mean_list, n=2)
            peak_months_distance = abs(max_n_index[0] - max_n_index[1])

            max_n_index = list(max_n_index)
            max_n_index.sort()
            if max_n_index[0] < 3:
                continue
            if max_n_index[1] >= 10:
                continue

            early_mon = np.array(early_mon)
            peak_mon = np.array(peak_mon)
            late_mon = np.array(late_mon)
            # print(month_mean_list)
            # print(early_mon)
            # print(peak_mon)
            # print(late_mon)
            # plt.plot(month_mean_list)
            # plt.show()
            # exit()
            result_dic_i = {
                'early': early_mon,
                'peak': peak_mon,
                'late': late_mon,
            }
            result_dic[pix] = result_dic_i

            # plt.plot(month_mean_list)
            # early_mon=early_mon-1
            # peak_mon=peak_mon-1
            # late_mon=late_mon-1
            #
            # plt.scatter(early_mon,[month_mean_list[i] for i in early_mon],c='g',s=70,zorder=40)
            # plt.scatter(peak_mon, [month_mean_list[i] for i in peak_mon],c='r',s=70,zorder=40)
            # plt.scatter(late_mon, [month_mean_list[i] for i in late_mon],c='b',s=70,zorder=40)
            # plt.title(str(pix)+'\n'+str(early_mon)+'\n'+str(peak_mon)+'\n'+str(late_mon)+'\n'+str(early_start)+'\n'+str(late_end))
            # plt.grid()
            # plt.tight_layout()
            # plt.show()
        df = T.dic_to_df(result_dic, 'pix')
        # df = df.dropna()
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.save_npy(result_dic, outf)


    def check_pix(self):

        outf = join(self.this_class_arr, 'Monthly_Early_Peak_Late.df')
        product = 'MODIS_LAI'
        phenology_df = T.load_df(
            results_root + f'Main_flow/arr/Phenology/average_phenology/{product}/phenology_dataframe_{product}.df')
        phenology_df = phenology_df.dropna()

        sos_dict = T.df_to_spatial_dic(phenology_df, 'early_start_mon')
        eos_dict = T.df_to_spatial_dic(phenology_df, 'late_end_mon')
        vege_dir = f'/Volumes/SSD_sumsang/project_greening/Data/original_dataset/{product}_dic/'  # monthly
        vege_dic = T.load_npy_dir(vege_dir)
        result_dic = {}
        for pix in tqdm(vege_dic):
            r,c=pix
            if r>150:
                continue
            if r<120:
                continue
            if pix not in sos_dict:
                continue
            vals = vege_dic[pix]
            if T.is_all_nan(vals):
                continue
            sos = int(sos_dict[pix])
            eos = int(eos_dict[pix])

            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            val_reshape = vals.reshape((-1, 12))
            val_reshape_T = val_reshape.T
            month_mean_list = []
            for month in val_reshape_T:
                month_mean = np.nanmean(month)
                month_mean_list.append(month_mean)
            isnan_list = np.isnan(month_mean_list)

            max_n_index, max_n_val = T.pick_max_n_index(month_mean_list, n=2)
            peak_months_distance = abs(max_n_index[0] - max_n_index[1])

            max_n_index = list(max_n_index)
            max_n_index.sort()
            if max_n_index[0] < 3:
                continue
            if max_n_index[1] >= 10:
                continue
            peak_mon = np.array(max_n_index) + 1
            if sos > peak_mon[0]:
                continue
            if peak_mon[-1] + 1 > eos + 1:
                continue
            early_mon = list(range(sos, peak_mon[0]))
            late_mon = list(range(peak_mon[-1] + 1, eos + 1))
            # print(early_mon)

            if peak_months_distance >= 2:
                early_mon = list(range(sos, peak_mon[0]))
                peak_mon = list(range(peak_mon[0], peak_mon[1] + 1))
                late_mon = list(range(peak_mon[-1] + 1, eos + 1))
                # print(peak_months_distance)
                # print(early_mon)
                # print(peak_mon)
                # print(late_mon)
                # print('--')
                # plt.plot(month_mean_list)
                # plt.show()
            early_mon = np.array(early_mon)
            peak_mon = np.array(peak_mon)
            late_mon = np.array(late_mon)
            # print(month_mean_list)
            # print(early_mon)
            # print(peak_mon)
            # print(late_mon)
            # plt.plot(month_mean_list)
            # plt.show()
            # exit()
            result_dic_i = {
                'early': early_mon,
                'peak': peak_mon,
                'late': late_mon,
            }
            result_dic[pix] = result_dic_i
            plt.plot(month_mean_list)
            early_mon=early_mon-1
            peak_mon=peak_mon-1
            late_mon=late_mon-1

            plt.scatter(early_mon,[month_mean_list[i] for i in early_mon],c='g',s=20,zorder=10)
            plt.scatter(peak_mon, [month_mean_list[i] for i in peak_mon],c='r',s=20,zorder=10)
            plt.scatter(late_mon, [month_mean_list[i] for i in late_mon],c='b',s=20,zorder=10)
            plt.title(str(pix)+'\n'+str(early_mon)+'\n'+str(peak_mon)+'\n'+str(late_mon)+'\n'+str(sos)+'\n'+str(eos))
            plt.tight_layout()
            plt.show()
        df = T.dic_to_df(result_dic, 'pix')
        # df = df.dropna()
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        T.save_npy(result_dic, outf)

class Pick_Early_Peak_Late_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Pick_Early_Peak_Late_value',results_root_main_flow)
        pass


    def run(self):
        # self.Pick_variables()
        # self.Pick_variables_all_gs()
        # self.Pick_variables_accumulative()
        # self.Pick_variables_min()
        # self.df_to_dict()
        self.check_nan_num()
        pass



    def check_nan_num(self):
        x_var = 'Soil moisture'
        # dff = join(self.this_class_arr,f'Pick_variables/{x_var}.df')
        dff = join(self.this_class_arr,f'Pick_variables_all_gs/{x_var}.df')
        df = T.load_df(dff)
        df = Dataframe().add_NDVI_mask(df)
        # period_list = ['early','peak','late']
        period_list = ['all_gs']
        for period in period_list:
            dic = T.df_to_spatial_dic(df,period)
            spatial_dic = {}
            for pix in dic:
                vals = dic[pix]
                if type(vals)==float:
                    continue
                is_nan_list = np.isnan(vals)
                nan_num = T.count_num(is_nan_list,False)
                # spatial_dic[pix] = 1
                spatial_dic[pix] = nan_num/len(vals)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            arr = arr[:180]
            plt.figure()
            DIC_and_TIF().plot_back_ground_arr(global_land_tif,aspect='auto')
            plt.imshow(arr,aspect='auto')
            plt.title(period)
            plt.colorbar()
        plt.show()

        pass

    def df_to_dict(self):
        outdir = join(self.this_class_arr,'df_to_dict')
        T.mkdir(outdir)
        # var_ = 'VOD'
        var_ = 'LAI4g_101'
        outdir_var = join(outdir,var_)
        T.mkdir(outdir_var)
        dff = join(self.this_class_arr,'Pick_variables',var_+'.df')
        df = T.load_df(dff)
        columns = df.columns
        for col in columns:
            if 'pix' in col:
                continue
            print(col)
            dict_ = T.df_to_spatial_dic(df,col)
            outf = join(outdir_var,col)
            T.save_npy(dict_,outf)
        pass

    def Pick_variables_all_gs(self):
        outdir = join(self.this_class_arr,'Pick_variables_all_gs')
        T.mkdir(outdir)
        var_list = [
            # 'LAI_3g',
            # 'SPEI',
            # 'Temperature',
            'Soil moisture',
            # 'CO2',
            # 'Aridity',
            # 'VOD',
            # 'LAI4g_101',
            # 'MODIS_LAI_CMG',
        ]
        for variable in var_list:
            outf = join(outdir,variable+'.df')
            fdir = vars_info_dic[variable]['path']
            start_year = vars_info_dic[variable]['start_year']
            dic = T.load_npy_dir(fdir)
            result_dic = {}
            for pix in tqdm(dic,desc=variable):
                vals = dic[pix]
                vals = np.array(vals)
                vals[vals<-999] = np.nan
                annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=list(range(4,11)))
                season_dict = {'all_gs':annual_vals}
                result_dic[pix] = season_dict
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)

    def Pick_variables(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mkdir(outdir)
        var_list = [
            # 'LAI_3g',
            # 'SPEI',
            # 'Temperature',
            'Soil moisture',
            # 'CO2',
            # 'Aridity',
            # 'VOD',
            # 'LAI4g_101',
            # 'MODIS_LAI_CMG',
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
                vals = np.array(vals)
                vals[vals<-999] = np.nan
                EPL_dic_i = EPL_dic[pix]
                season_vals_dic = {}
                for season in EPL_dic_i:
                    if season == 'pix':
                        continue
                    grow_season = EPL_dic_i[season]
                    grow_season = list(grow_season)
                    if len(grow_season) == 0:
                        continue
                    annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=grow_season)
                    season_vals_dic[season] = annual_vals
                result_dic[pix] = season_vals_dic
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)


    def Pick_variables_min(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mkdir(outdir)
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
                    if len(grow_season) == 0:
                        continue
                    annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=grow_season,method='sum')
                    season_vals_dic[season] = annual_vals
                result_dic[pix] = season_vals_dic
            df_i = T.dic_to_df(result_dic,'pix')
            T.save_df(df_i,outf)
            T.df_to_excel(df_i,outf)

    def Pick_variables_accumulative(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mkdir(outdir)
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
                    if len(grow_season) == 0:
                        continue
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
        self.build_df()
        # self.cal_importance()
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
        # col_list = []
        # for var_ in var_list:
        #     for season in season_list:
        #         print(var_,season)
        #         col_name = f'{season}_{var_}'
        #         col_list.append(col_name)
        #         df = self.add_each_season_to_df(df,var_,season)
        # df = df.dropna(how='all',subset=col_list)

        # combine_season_list = ['early','peak']
        # var_ = 'LAI_3g'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)

        combine_season_list = ['early', 'peak']
        var_ = 'Soil moisture'
        method = 'mean'
        df = self.add_combine(df, combine_season_list, var_, method)

        # combine_season_list = ['early','peak','late']
        # var_ = 'LAI_3g'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)
        #
        # combine_season_list = ['early','peak','late']
        # var_ = 'Soil moisture'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)
        #
        # combine_season_list = ['early', 'peak', 'late']
        # var_ = 'SPEI_accu'
        # method = 'sum'
        # df = self.add_combine(df, combine_season_list, var_, method)
        #
        # #
        # combine_season_list = ['early', 'peak']
        # var_ = 'SPEI_accu'
        # method = 'sum'
        # df = self.add_combine(df, combine_season_list, var_, method)
        # # #
        # combine_season_list = ['early', 'peak']
        # var_ = 'Soil moisture_accu'
        # method = 'sum'
        # # #
        # # #
        # df = self.add_combine(df, combine_season_list, var_, method)
        # combine_season_list = ['early', 'peak']
        # var_ = 'SPEI_min'
        # method = 'min'
        # df = self.add_combine(df, combine_season_list, var_, method)
        # # #
        # combine_season_list = ['early', 'peak']
        # var_ = 'Soil moisture_min'
        # method = 'min'
        # df = self.add_combine(df, combine_season_list, var_, method)
        # df = self.add_sos_to_df(df)
        # df = self.add_CO2(df)
        # df = Dataframe().add_Humid_nonhumid(df)
        # df = Dataframe().add_lon_lat_to_df(df)
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
            vals = Pre_Process().cal_anomaly_juping(vals)
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
        P = Pre_Process()
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
                for y in year_list_:
                    val_list.append(np.nan)
                    year_list.append(np.nan)
                    pix_list.append(np.nan)
                continue
            vals = dic[pix]
            if type(vals) == float:
                for y in year_list_:
                    val_list.append(np.nan)
                    year_list.append(np.nan)
                    pix_list.append(np.nan)
                continue
            # vals = P.cal_anomaly_juping(vals)
            vals = P.cal_relative_change(vals)
            dic_i = dict(zip(year_list_,vals))
            for y in year_list_:
                if not y in dic_i:
                    val_list.append(np.nan)
                    year_list.append(np.nan)
                    pix_list.append(np.nan)
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
                result = np.nanmean(i)
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
            co2_annual_vals = Pre_Process().cal_anomaly_juping(co2_annual_vals)
            co2_dic_i = dict(zip(year_list, co2_annual_vals))
            co2_annual_dic[pix] = co2_dic_i

        val_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            year = row.year
            try:
                co2_dic_i = co2_annual_dic[pix]
            except:
                val_list.append(np.nan)
                continue
            co2 = co2_dic_i[year]
            val_list.append(co2)
        df['CO2'] = val_list
        return df



class Moving_window_RF:
    def __init__(self):
        pass

    def run(self):

        pass

    def build_df(self):


        pass

class Analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Analysis',results_root_main_flow)

        pass

    def run(self):
        # self.Greeing_trend()
        # self.Greeing_trend_relative_change()
        # self.Greeing_trend_combine()
        # self.Greeing_trend_lc_ratio()
        # self.Greeing_trend_lc_tif()
        # self.Greeing_trend_sos_conditon()
        # self.plot_pdf_Greeing_trend_sos_conditon()
        # self.SOS_trend()
        # self.carry_over_effect_bar_plot()
        # self.Greeing_trend_3_season_line()
        # self.Greeing_trend_two_period()
        # self.Jan_to_Dec_timeseries()
        # self.Jan_to_Dec_timeseries_sos_condition()
        # self.yvar_pdf_SOS_condition()
        # self.LAI_pdf_SOS_condition_climate_background()
        # self.LAI_spatial_SOS_condition()
        # self.Jan_to_Dec_timeseries_hants()
        # self.early_peak_greening_speedup_advanced_sos()
        # self.check()
        # self.carry_over_effect_with_sm_matrix()
        # self.inter_annual_carryover_effect()
        # self.correlation_advanced_sos()
        # self.correlation_sos_vs_vegetation()
        # self.correlation_pdf_sos_vs_vegetation()
        # self.correlation_positive_sos_trend()
        # self.carryover_effect_sm()
        self.product_comparison_statistics()
        pass

    def product_comparison_statistics(self):
        fdir = '/Volumes/NVME2T/greening_project_redo/data/product_comparison_2000-2018'
        outdir = join(self.this_class_png,'product_comparison_2000-2018')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        dic_all = {}
        for f in T.listdir(fdir):
            dic_i = DIC_and_TIF().spatial_tif_to_dic(join(fdir,f))
            key = f.split('.')[0]
            dic_all[key] = dic_i
        df = T.spatial_dics_to_df(dic_all)
        df = Dataframe().add_lon_lat_to_df(df)
        df = df[df['lat']>30]
        period_list = ['early','peak','late']
        df = df.dropna()
        for period in period_list:
            unique_value = T.get_df_unique_val_list(df,period)
            total = 0
            plt.figure()
            for uv in unique_value:
                if uv == 1:
                    continue
                df_i = df[df[period]==uv]
                ratio = len(df_i)/len(df)
                total += ratio
                # print(uv,ratio*100)
                plt.bar(uv,ratio*100)
            plt.ylim(0,60)
            plt.title(f'{period}')
            plt.savefig(join(outdir,f'{period}.pdf'))
            plt.close()
            # print(unique_value)
        # plt.show()


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

    def SOS_trend(self):
        outdir = join(self.this_class_tif,'SOS_trend')
        T.mkdir(outdir)
        outf = join(outdir,'SOS_trend.tif')
        phenology_dff = join(Phenology().this_class_arr,'compose_annual_phenology/phenology_dataframe.df')
        phenology_df = T.load_df(phenology_dff)
        print(phenology_df.columns)
        early_start_dic = T.df_to_spatial_dic(phenology_df,'early_start')
        spatial_sos_trend_dic = {}
        for pix in tqdm(early_start_dic):
            dic_i = early_start_dic[pix]
            # dic_i = dict(dic_i)
            year_list = []
            for year in dic_i:
                year_list.append(year)
            year_list.sort()
            sos_list = []
            for year in year_list:
                sos = dic_i[year]
                sos_list.append(sos)
            try:
                a,_,_,_ = T.nan_line_fit(year_list,sos_list)
                spatial_sos_trend_dic[pix] = a
            except:
                continue
        DIC_and_TIF().pix_dic_to_tif(spatial_sos_trend_dic,outf)
        pass

    def carry_over_effect_spatial(self):
        # df =
        pass


    def carry_over_effect_bar_plot(self):
        # carry over trend < 0 humid region, see sos trend
        humid_region_var = 'HI_reclass'
        sos_trend_var = 'sos_trend'  # original values trend
        carryover_trend_var = 'carryover_trend'  # moving window
        carryover_trend_p_var = 'carryover_trend_p'  # moving window
        sm_early_trend_var = 'sm_early_trend'  # original values trend
        sm_peak_trend_var = 'sm_peak_trend'  # original values trend
        sm_late_trend_var = 'sm_late_trend'  # original values trend
        lai_combine_trend_var = 'lai_combine_trend'  # original values trend

        mode = 'early-peak_late'  # carryover effect mode
        carryover_tif = join(Moving_window().this_class_tif,f'array_carry_over_effect/trend/{mode}.tif')  # moving window
        carryover_p_tif = join(Moving_window().this_class_tif,f'array_carry_over_effect/trend/{mode}_p.tif')  # moving window
        sos_tif = join(self.this_class_tif,'SOS_trend/SOS_trend.tif')
        sm_early_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/early.tif')
        sm_peak_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/peak.tif')
        sm_late_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/late.tif')
        lai_combine_tif = join(self.this_class_tif,'Greeing_trend_combine/LAI_3g/early-peak_LAI_3g.tif')

        carryover_dic = DIC_and_TIF().spatial_tif_to_dic(carryover_tif)
        carryover_p_dic = DIC_and_TIF().spatial_tif_to_dic(carryover_p_tif)
        sos_dic = DIC_and_TIF().spatial_tif_to_dic(sos_tif)
        sm_early_dic = DIC_and_TIF().spatial_tif_to_dic(sm_early_tif)
        sm_peak_dic = DIC_and_TIF().spatial_tif_to_dic(sm_peak_tif)
        sm_late_dic = DIC_and_TIF().spatial_tif_to_dic(sm_late_tif)
        lai_combine_dic = DIC_and_TIF().spatial_tif_to_dic(lai_combine_tif)

        spatial_dic_all = {
            'carryover_trend':carryover_dic,
            'carryover_trend_p':carryover_p_dic,
            'sos_trend':sos_dic,
            'sm_early_trend':sm_early_dic,
            'sm_peak_trend':sm_peak_dic,
            'sm_late_trend':sm_late_dic,
            'lai_combine_trend':lai_combine_dic,
        }
        df = T.spatial_dics_to_df(spatial_dic_all)
        df = Dataframe().add_Humid_nonhumid(df)
        df = df.dropna()
        df = df[df[carryover_trend_p_var]<0.05]
        T.print_head_n(df)
        # exit()
        region_list = T.get_df_unique_val_list(df,humid_region_var)
        carryover_trend_list = ['Carryover(+)','Carryover(-)',]
        for region in region_list:
            df_region = df[df[humid_region_var]==region]
            for carryover_trend in carryover_trend_list:
                if carryover_trend == 'Carryover(+)':
                    df_carryover_trend = df_region[df_region[carryover_trend_var]>0]
                elif carryover_trend == 'Carryover(-)':
                    df_carryover_trend = df_region[df_region[carryover_trend_var]<0]
                else:
                    raise
                sos_trend = df_carryover_trend[sos_trend_var].tolist()
                sm_early_trend = df_carryover_trend[sm_early_trend_var].tolist()
                sm_peak_trend = df_carryover_trend[sm_peak_trend_var].tolist()
                sm_late_trend = df_carryover_trend[sm_late_trend_var].tolist()
                lai_combine_trend = df_carryover_trend[lai_combine_trend_var].tolist()
                plt.figure()
                plt.bar('sos_trend',np.nanmean(sos_trend)/50.)
                plt.bar('sm_early_trend',np.nanmean(sm_early_trend)*10)
                plt.bar('sm_peak_trend',np.nanmean(sm_peak_trend)*10)
                plt.bar('sm_late_trend',np.nanmean(sm_late_trend)*10)
                plt.bar('lai_combine_trend',np.nanmean(lai_combine_trend))
                plt.title(f'{region}_{carryover_trend}')
        plt.show()

        pass

    def Greeing_trend(self):
        # var_ = 'LAI_3g'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        var_ = 'VOD'
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mkdir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        for season in global_season_dic:
            print(season)
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            trend_dic = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x,y)
                trend_dic[pix] = a
            out_tif = join(outdir,f'{season}.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dic,out_tif)
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
        #     plt.figure()
        #     plt.imshow(arr,vmax=0.02,vmin=-0.02,cmap='RdBu')
        #     plt.colorbar()
        #     DIC_and_TIF().plot_back_ground_arr(Global_vars().land_tif)
        #     plt.title(season)
        # plt.show()
        pass
    def Greeing_trend_relative_change(self):
        # var_ = 'LAI_3g'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        var_ = 'VOD'
        outdir = join(self.this_class_tif,f'Greeing_trend_relative_change/{var_}')
        T.mkdir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        for season in global_season_dic:
            print(season)
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            trend_dic = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                vals = Global_vars().cal_relative_change(vals)
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x,y)
                trend_dic[pix] = a
            out_tif = join(outdir,f'{season}.tif')
            DIC_and_TIF().pix_dic_to_tif(trend_dic,out_tif)
        #     arr = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
        #     plt.figure()
        #     plt.imshow(arr,vmax=0.02,vmin=-0.02,cmap='RdBu')
        #     plt.colorbar()
        #     DIC_and_TIF().plot_back_ground_arr(Global_vars().land_tif)
        #     plt.title(season)
        # plt.show()
        pass

    def Greeing_trend_combine(self):
        var_ = 'LAI_3g'
        mode = 'early-peak'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        outdir = join(self.this_class_tif,f'Greeing_trend_combine/{var_}')
        T.mkdir(outdir,force=True)
        out_tif = join(outdir,f'{mode}_{var_}.tif')
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        df = df.dropna()
        combine_season_list = mode.split('-')
        new_combined_colname = f'{mode}_{var_}_mean'
        df = T.combine_df_columns(df,combine_season_list,new_combined_colname)
        spatial_vals_dic = T.df_to_spatial_dic(df,new_combined_colname)
        trend_arr = DIC_and_TIF().pix_dic_to_spatial_arr_trend(spatial_vals_dic)
        DIC_and_TIF().arr_to_tif(trend_arr,out_tif)
        pass
    def Greeing_trend_lc_ratio(self):
        var_ = 'LAI_3g'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        outdir = join(self.this_class_tif,f'Greeing_trend_lc_ratio/{var_}')
        T.mkdir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        # lc_list = T.get_df_unique_val_list(df,'GLC2000')
        for season in global_season_dic:
            print(season)
            plt.figure()
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            result_dic = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                # print(len(vals))
                # exit()
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x,y)
                result = {'k':a,'p':p}
                result_dic[pix] = result
            df_temp = T.dic_to_df(result_dic,key_col_str='pix')
            df_temp = Dataframe().add_lc(df_temp)
            df_temp = Dataframe().add_Humid_nonhumid(df_temp)
            df_temp = df_temp.dropna()

            lc_list = T.get_df_unique_val_list(df_temp,'GLC2000')
            category_list = []
            for i,row in df_temp.iterrows():
                k = row['k']
                p = row['p']
                if p >= 0.1:
                    category = 'non-significant'
                elif 0.05 < p < 0.1:
                    if k > 0:
                        category = 'positive'
                    else:
                        category = 'negative'
                elif p <= 0.05:
                    if k > 0:
                        category = 'significant positive'
                    else:
                        category = 'significant negative'
                else:
                    category = 'non-significant'
                category_list.append(category)
            df_temp['category'] = category_list
            category_list_unique = ['significant positive','positive','non-significant','negative','significant negative']
            category_color_dic = {'non-significant':'grey','positive':'green','negative':'orange','significant positive':'blue','significant negative':'red'}
            for lc in lc_list:
                df_temp_lc = df_temp[df_temp['GLC2000'] == lc]
                ratio_dic = {}
                for category in category_list_unique:
                    df_category = df_temp_lc[df_temp_lc['category'] == category]
                    ratio = len(df_category)/len(df_temp_lc)
                    ratio_dic[category] = ratio
                print(lc)
                print(ratio_dic)
                bottom = 0
                for category in category_list_unique:
                    plt.bar(lc,ratio_dic[category],bottom=bottom,color=category_color_dic[category])
                    bottom += ratio_dic[category]
            plt.title(f'{season}')
            # plt.legend()
        plt.show()


        pass

    def Greeing_trend_lc_tif(self):
        var_ = 'LAI_3g'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        outdir = join(self.this_class_tif, f'Greeing_trend_lc_tif/{var_}')
        T.open_path_and_file(outdir)
        T.mkdir(outdir, force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr, f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        df = df.dropna()
        # lc_list = T.get_df_unique_val_list(df,'GLC2000')
        for season in global_season_dic:
            print(season)
            plt.figure()
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df, col_name)
            result_dic = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                # print(len(vals))
                # exit()
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x, y)
                result = {'k': a, 'p': p}
                result_dic[pix] = result
            df_temp = T.dic_to_df(result_dic, key_col_str='pix')
            # df_temp = Dataframe().add_lc(df_temp)
            # df_temp = Dataframe().add_Humid_nonhumid(df_temp)
            df_temp = df_temp.dropna()
            spatial_dic_k = T.df_to_spatial_dic(df_temp, 'k')
            spatial_dic_p = T.df_to_spatial_dic(df_temp, 'p')
            DIC_and_TIF().pix_dic_to_tif(spatial_dic_k, join(outdir, f'{season}_k.tif'))
            DIC_and_TIF().pix_dic_to_tif(spatial_dic_p, join(outdir, f'{season}_p.tif'))


        pass

    def Greeing_trend_sos_conditon(self):
        outdir = join(self.this_class_arr,'Greeing_trend_sos_conditon')
        T.mkdir(outdir)
        outf = join(outdir,'Greening_trend.df')
        var_ = 'LAI_3g'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        sos_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mkdir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        df_list = []
        for season in global_season_dic:
            print(season)
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            result_dic = {}
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                if not pix in sos_dic:
                    continue
                sos_dic_i = sos_dic[pix]
                year_list = []
                for year in sos_dic_i:
                    year_list.append(year)
                year_list.sort()
                sos_list = []
                for year in year_list:
                    sos = sos_dic_i[year]
                    sos_list.append(sos)
                a_sos, b_sos, r_sos, p_sos = self.nan_linear_fit(year_list, sos_list)
                x = list(range(len(vals)))
                y = vals
                a, b, r, p = self.nan_linear_fit(x,y)
                result_dic_i = {
                    f'{season}_sos_k':a_sos,
                    f'{season}_sos_p':p_sos,
                    f'{season}_{var_}_k':a,
                    f'{season}_{var_}_p':p,
                }
                result_dic[pix] = result_dic_i
            df_i = T.dic_to_df(result_dic,'pix')
            df_list.append(df_i)
        df_0 = pd.DataFrame()
        df_0 = T.join_df_list(df_0,df_list,'pix')
        T.save_df(df_0,outf)
        T.df_to_excel(df_0,outf)

    def plot_pdf_Greeing_trend_sos_conditon(self):
        y_var = 'LAI_3g'
        region_var = 'HI_reclass'
        p_threshold = 0.05
        fdir = join(self.this_class_arr, 'Greeing_trend_sos_conditon')
        dff = join(fdir, 'Greening_trend.df')
        df = T.load_df(dff)
        df = Dataframe().add_Humid_nonhumid(df)
        region_list = T.get_df_unique_val_list(df,region_var)[::-1]
        # region_list = T.get_df_unique_val_list(df,region_var)
        for region in region_list:
            df_region = df[df[region_var]==region]
            for season in global_season_dic:
                k_key = f'{season}_{y_var}_k'
                p_key = f'{season}_{y_var}_p'
                k_sos_key = f'{season}_sos_k'
                p_sos_key = f'{season}_sos_p'
                df_i = pd.DataFrame()
                df_i['pix'] = df_region['pix']
                df_i['k'] = df_region[k_key]
                df_i['p'] = df_region[p_key]
                df_i['k_sos'] = df_region[k_sos_key]
                df_i['p_sos'] = df_region[p_sos_key]
                df_i = df_i.dropna()
                # df_i = df_i[df_i['p']<p_threshold]
                # df_i = df_i[df_i['p_sos']<p_threshold]
                df_i = df_i[df_i['k_sos']<0]
                spatial_dic = T.df_to_spatial_dic(df_i,'k')
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                up,down = T.get_vals_std_up_down(arr)
                plt.figure()
                plt.imshow(arr,vmin=down,vmax=up)
                plt.title(season)
                plt.colorbar()
                DIC_and_TIF().plot_back_ground_arr(global_land_tif)
                # plt.figure()
                # arr_flatten = arr.flatten()
                # arr_flatten = T.remove_np_nan(arr_flatten)
                # x,y = Plot().plot_hist_smooth(arr_flatten,bins=80,alpha=0)
                # plt.plot(x,y,label=f'{region}_{season}')
                # plt.legend()
            plt.show()




    def Greeing_trend_3_season_line(self):
        var_ = 'LAI_3g'
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mkdir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        df = Dataframe().add_Humid_nonhumid(df)
        df = Main_flow.Dataframe().add_lon_lat_to_df(df)
        df = df[df['lat']>30]
        # df = df[df['HI_reclass']=='Non Humid']
        df = df[df['HI_reclass']=='Humid']
        pix_list = T.get_df_unique_val_list(df,'pix')
        spatial_dic = {}
        for pix in pix_list:
            spatial_dic[pix] = 1
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
        for season in global_season_dic:
            print(season)
            col_name = join(f'{season}')
            spatial_dic = T.df_to_spatial_dic(df,col_name)
            trend_dic = {}
            matrix = []
            spatial_dic_1 = {}
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                if type(vals) == float:
                    continue
                if len(vals) == 0:
                    continue
                matrix.append(vals)
                spatial_dic_1[pix] = 1
            # print(matrix)
            # print(np.shape(matrix))
            # exit()
            annual_mean = np.nanmean(matrix,axis=0)
            x = list(range(len(annual_mean)))
            y = annual_mean
            a, b, r, p = self.nan_linear_fit(x, y)
            plt.figure()
            KDE_plot().plot_fit_line(a, b, r, p,x)
            plt.scatter(x,y,label=season)
            plt.legend()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_1)
        plt.figure()
        plt.imshow(arr)
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.show()
        pass

    def Greeing_trend_two_period(self):
        var_ = 'LAI_3g'
        start_year = vars_info_dic[var_]['start_year']
        period_1 = list(range(1982,2002))
        period_2 = list(range(2002,2019))
        year_list = list(range(1982,2021))
        period_list = [period_1,period_2]
        outdir = join(self.this_class_tif,f'Greeing_trend_two_period/{var_}')
        T.mkdir(outdir,force=True)
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
                    if type(vals) == float:
                        continue
                    period = np.array(period)
                    period_index = period - start_year
                    vals_period = T.pick_vals_from_1darray(vals,period_index)
                    x = list(range(len(vals_period)))
                    y = vals_period
                    try:
                        a, b, r, p = self.nan_linear_fit(x,y)
                        trend_dic[pix] = a
                    except:
                        continue
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


    def Jan_to_Dec_timeseries(self):
        var_ = 'LAI_3g'
        fdir = vars_info_dic[var_]['path']
        start_year = vars_info_dic[var_]['start_year']
        void_df = Global_vars().get_valid_pix_df()
        # HI_region = 'Non Humid'
        HI_region = 'Humid'
        void_df = void_df[void_df['HI_reclass']==HI_region]
        selected_pix_list = T.get_df_unique_val_list(void_df,'pix')
        selected_pix_list = set(selected_pix_list)
        spatial_dic = T.df_to_spatial_dic(void_df,'lat')
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.imshow(arr)
        plt.figure()
        dic = T.load_npy_dir(fdir)
        # period_list_dic = {
        #     '1980s':list(range(1982,1991)),
        #     '1990s':list(range(1991,2001)),
        #     '2000s':list(range(2001,2011)),
        #     '2010s':list(range(2011,2019)),
        # }
        period_list_dic = {
            '1982-2001': list(range(1982, 2002)),
            '2002-2018': list(range(2002, 2019)),
            # '2000s': list(range(2001, 2011)),
            # '2010s': list(range(2011, 2019)),
        }
        for period in period_list_dic:
            print(period)
            matrix = []
            for pix in dic:
                if not pix in selected_pix_list:
                    continue
                vals = dic[pix]
                if T.is_all_nan(vals):
                    continue
                isnan = np.isnan(vals)
                if True in isnan:
                    continue
                period_range = period_list_dic[period]
                period_range = np.array(period_range)
                period_index = period_range - start_year
                annual_vals = T.monthly_vals_to_annual_val(vals,method='array')
                picked_annual_vals = T.pick_vals_from_1darray(annual_vals,period_index)
                monthly_mean = np.nanmean(picked_annual_vals,axis=0)
                matrix.append(monthly_mean)
            # print(np.shape(matrix))
            # exit()
            monthly_mean_all = np.nanmean(matrix,axis=0)
            monthly_std_all = np.nanstd(matrix,axis=0)
            x = list(range(len(monthly_mean_all)))
            y = monthly_mean_all
            yerr = monthly_std_all
            title = f'{period}-{HI_region}'
            plt.plot(x,y,label=title)
            # Plot().plot_line_with_error_bar(x,y,yerr)
        plt.legend()
        plt.show()
        # outdir = join(self.this_class_tif,f'Jan_to_Dec_timeseries/{var_}')
        # T.mkdir(outdir,force=True)
        pass


    def Jan_to_Dec_timeseries_hants(self):
        annual_hants_fdir = join(Phenology().this_class_arr, 'all_year_hants_annual')
        annual_vals_dic = T.load_npy_dir(annual_hants_fdir)
        void_df = Global_vars().get_valid_pix_df()
        HI_region = 'Non Humid'
        # HI_region = 'Humid'
        region_list = self.get_region_pix_list(HI_region)

        # void_df = void_df[void_df['HI_reclass']==HI_region]
        # selected_pix_list = T.get_df_unique_val_list(void_df,'pix')
        # selected_pix_list = set(selected_pix_list)
        # spatial_dic = T.df_to_spatial_dic(void_df,'lat')
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        # plt.imshow(arr)
        # plt.figure()
        # dic = T.load_npy_dir(fdir)
        period_list_dic = {
            '1980s':list(range(1982,1991)),
            '1990s':list(range(1991,2001)),
            '2000s':list(range(2001,2011)),
            '2010s':list(range(2011,2019)),
        }
        # period_list_dic = {
        #     '1982-2001': list(range(1982, 2002)),
        #     '2002-2018': list(range(2002, 2019)),
        # }
        for period in period_list_dic:
            print(period)
            selected_year = period_list_dic[period]
            # print(selected_year)
            matrix = []
            for pix in annual_vals_dic:
                if not pix in region_list:
                    continue
                if not pix in annual_vals_dic:
                    continue
                annual_vals_dic_i = annual_vals_dic[pix]
                annual_vals_dic_i = dict(annual_vals_dic_i)
                for year in selected_year:
                    if not year in annual_vals_dic_i:
                        continue
                    vals = annual_vals_dic_i[year]
                    matrix.append(vals)
            # print(np.shape(matrix))
            # exit()
            monthly_mean_all = np.nanmean(matrix,axis=0)
            monthly_std_all = np.nanstd(matrix,axis=0)
            x = list(range(len(monthly_mean_all)))
            y = monthly_mean_all
            yerr = monthly_std_all
            title = f'{period}-{HI_region}'
            # title = f'{period}'
            plt.plot(x,y,label=title)
            # Plot().plot_line_with_error_bar(x,y,yerr)
        plt.legend()
        plt.show()
        # outdir = join(self.this_class_tif,f'Jan_to_Dec_timeseries/{var_}')
        # T.mkdir(outdir,force=True)
        pass


    def get_phenology_spatial_dic(self,var_name,isanomaly=False):
        sos_dff = join(Phenology().this_class_arr, 'compose_annual_phenology/phenology_dataframe.df')
        sos_df = T.load_df(sos_dff)
        early_start_dic = T.df_to_spatial_dic(sos_df, var_name)
        phenology_dic = {}
        for pix in early_start_dic:
            year_list = []
            dic_i = early_start_dic[pix]
            if isanomaly:
                for year in dic_i:
                    year_list.append(year)
                year_list.sort()
                val_list = []
                for year in year_list:
                    val = dic_i[year]
                    val_list.append(val)
                val_list_anomaly = Pre_Process().cal_anomaly_juping(val_list)
                dic_i_anomaly = dict(zip(year_list, val_list_anomaly))
                phenology_dic[pix] = dic_i_anomaly
            else:
                phenology_dic[pix] = dic_i
        return phenology_dic



    def Jan_to_Dec_timeseries_sos_condition(self):
        y_var = 'LAI_3g'

        lai_dir = vars_info_dic[y_var]['path']
        start_year = vars_info_dic[y_var]['start_year']
        lai_dic = T.load_npy_dir(lai_dir)
        phenology_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        all_spatial_dic = {
            'lai':lai_dic,
            'sos':phenology_dic
        }
        region = 'Humid'
        # region = 'Non Humid'
        df = T.spatial_dics_to_df(all_spatial_dic)
        df = Dataframe().add_Humid_nonhumid(df)
        df = df[df['HI_reclass']==region]
        df = df.dropna()
        lai_dic = T.df_to_spatial_dic(df,'lai')
        early_start_dic_anomaly = T.df_to_spatial_dic(df,'sos')

        year_list = list(range(start_year, 9999))
        matrix = []
        for pix in tqdm(lai_dic):
            if not pix in early_start_dic_anomaly:
                continue
            vals = lai_dic[pix]
            vals = T.detrend_vals(vals)
            vals = Pre_Process().z_score_climatology(vals)
            # vals = Pre_Process().cal_anomaly_juping(vals)
            vals_reshape = np.reshape(vals, (-1, 12))
            dic_i = dict(zip(year_list, vals_reshape))
            early_start_dic_anomaly_i = early_start_dic_anomaly[pix]
            # print(early_start_dic_anomaly_i)
            if type(early_start_dic_anomaly_i) == float:
                continue
            for year in dic_i:
                if not year in early_start_dic_anomaly_i:
                    continue
                sos_anomaly = early_start_dic_anomaly_i[year]
                if sos_anomaly < 0:
                # if sos_anomaly > 0:
                    matrix.append(dic_i[year])
            # vals_2017 = dic_i[2017]
            # if np.nanmean(vals_2018[:7]) < 0.1:
            # all_vals_2017.append(vals_2017)
        monthly_val = np.nanmean(matrix, axis=0)
        # plt.plot(monthly_val_2017,label='2017')
        plt.plot(monthly_val)
        # plt.legend()
        plt.title(region)
        plt.show()
        pass

    def LAI_spatial_SOS_condition(self):
        outdir = join(self.this_class_png,'LAI_spatial_SOS_condition')
        T.mkdir(outdir)
        y_var = 'LAI_3g'
        region_var = 'HI_reclass'
        condition_list = ['Advanced SOS','Delayed SOS']
        # lai_dir = vars_info_dic[y_var]['path']
        start_year = vars_info_dic[y_var]['start_year']
        lai_dff = join(Pick_Early_Peak_Late_value().this_class_arr,'Pick_variables/LAI_3g.df')
        lai_df = T.load_df(lai_dff)
        lai_df = T.combine_df_columns(lai_df,['early','peak','late'],'all_gs')
        # season = 'all_gs'
        # season = 'peak'
        # season = 'late'
        seasonlist = global_season_dic
        seasonlist.append('all_gs')
        color_list = KDE_plot().makeColours(list(range(8)),cmap='jet')
        for season in seasonlist:
            print(season)
            flag = 0
            lai_dic = T.df_to_spatial_dic(lai_df,season)
            phenology_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
            all_spatial_dic = {
                'lai':lai_dic,
                'sos':phenology_dic
            }
            # region = 'Humid'
            # region = 'Non Humid'
            df = T.spatial_dics_to_df(all_spatial_dic)
            df = Dataframe().add_Humid_nonhumid(df)
            region_list = T.get_df_unique_val_list(df,region_var)
            # region_list = ['Humid']
            # region_list = ['Non Humid']
            # df = df[df['HI_reclass']==region]
            df = df.dropna()
            for region in region_list:
                df_region = df[df[region_var]==region]
                # plt.figure()
                for condition in condition_list:
                    title = f'{season}_{region}_{condition}'
                    lai_dic = T.df_to_spatial_dic(df_region,'lai')
                    early_start_dic_anomaly = T.df_to_spatial_dic(df_region,'sos')
                    year_list = list(range(start_year, 9999))
                    spatial_dic_result = {}
                    for pix in tqdm(lai_dic):
                        if not pix in early_start_dic_anomaly:
                            continue
                        vals = lai_dic[pix]
                        vals = T.detrend_vals(vals)
                        # vals = Pre_Process().z_score_climatology(vals)
                        vals = Pre_Process().cal_anomaly_juping(vals)
                        # vals_reshape = np.reshape(vals, (-1, 12))
                        dic_i = dict(zip(year_list, vals))
                        early_start_dic_anomaly_i = early_start_dic_anomaly[pix]
                        # print(early_start_dic_anomaly_i)
                        if type(early_start_dic_anomaly_i) == float:
                            continue
                        picked_vals = []
                        for year in dic_i:
                            if not year in early_start_dic_anomaly_i:
                                continue
                            sos_anomaly = early_start_dic_anomaly_i[year]
                            if condition == 'Advanced SOS':
                                if sos_anomaly < 0:
                                    picked_vals.append(dic_i[year])
                            elif condition == 'Delayed SOS':
                                if sos_anomaly > 0:
                                    picked_vals.append(dic_i[year])
                            else:
                                raise
                        if T.is_all_nan(picked_vals):
                            continue
                        longterm_mean = np.nanmean(picked_vals)
                        spatial_dic_result[pix] = longterm_mean

                    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_result)
                    DIC_and_TIF().plot_back_ground_arr(global_land_tif)
                    up, down = T.get_vals_std_up_down(arr)
                    plt.imshow(arr, vmax=up, vmin=down)
                    plt.colorbar()
                    # arr_flatten = arr.flatten()
                    # arr_flatten = T.remove_np_nan(arr_flatten)
                    # x,y = Plot().plot_hist_smooth(arr_flatten,bins=80,range=(-0.4,0.4),alpha=0)
                    # plt.figure(figsize=(6,3))
                    # plt.plot(x,y,label=title,color=color_list[flag])
                    # flag += 1

                    plt.title(title)
                    # plt.vlines(x=0,ymin=0,ymax=.2,linestyles='dashed')
                    # plt.ylim(0,0.2)
                    # print(title)
                    # plt.legend()
                    plt.savefig(join(outdir,title+'.pdf'))
                    plt.close()
                    # plt.show()

    def yvar_pdf_SOS_condition(self):
        # y_var = 'Soil moisture'
        # bin_range = (-0.02,0.02)
        # y_var = 'LAI_3g'
        # bin_range = (-0.4,0.4)
        y_var = 'SPEI'
        bin_range = (-1, 1)

        outdir = join(self.this_class_png,f'yvar_pdf_SOS_condition/{y_var}')
        T.mkdir(outdir,force=True)
        region_var = 'HI_reclass'
        condition_list = ['Advanced SOS','Delayed SOS']
        # lai_dir = vars_info_dic[y_var]['path']
        start_year = vars_info_dic[y_var]['start_year']
        lai_dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{y_var}.df')
        lai_df = T.load_df(lai_dff)
        lai_df = T.combine_df_columns(lai_df,['early','peak','late'],'all_gs')
        # season = 'all_gs'
        # season = 'peak'
        # season = 'late'
        seasonlist = global_season_dic
        seasonlist.append('all_gs')
        color_list = KDE_plot().makeColours(list(range(8)),cmap='jet')
        for season in seasonlist:
            print(season)
            flag = 0
            lai_dic = T.df_to_spatial_dic(lai_df,season)
            phenology_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
            all_spatial_dic = {
                'lai':lai_dic,
                'sos':phenology_dic
            }
            # region = 'Humid'
            # region = 'Non Humid'
            df = T.spatial_dics_to_df(all_spatial_dic)
            df = Dataframe().add_Humid_nonhumid(df)
            region_list = T.get_df_unique_val_list(df,region_var)
            # region_list = ['Humid']
            # region_list = ['Non Humid']
            # df = df[df['HI_reclass']==region]
            df = df.dropna()
            for region in region_list:
                df_region = df[df[region_var]==region]
                # plt.figure()
                for condition in condition_list:
                    title = f'{season}_{region}_{condition}'
                    lai_dic = T.df_to_spatial_dic(df_region,'lai')
                    early_start_dic_anomaly = T.df_to_spatial_dic(df_region,'sos')
                    year_list = list(range(start_year, 9999))
                    spatial_dic_result = {}
                    for pix in tqdm(lai_dic):
                        if not pix in early_start_dic_anomaly:
                            continue
                        vals = lai_dic[pix]
                        vals = T.detrend_vals(vals)
                        # vals = Pre_Process().z_score_climatology(vals)
                        vals = Pre_Process().cal_anomaly_juping(vals)
                        # vals_reshape = np.reshape(vals, (-1, 12))
                        dic_i = dict(zip(year_list, vals))
                        early_start_dic_anomaly_i = early_start_dic_anomaly[pix]
                        # print(early_start_dic_anomaly_i)
                        if type(early_start_dic_anomaly_i) == float:
                            continue
                        picked_vals = []
                        for year in dic_i:
                            if not year in early_start_dic_anomaly_i:
                                continue
                            sos_anomaly = early_start_dic_anomaly_i[year]
                            if condition == 'Advanced SOS':
                                if sos_anomaly < 0:
                                    picked_vals.append(dic_i[year])
                            elif condition == 'Delayed SOS':
                                if sos_anomaly > 0:
                                    picked_vals.append(dic_i[year])
                            else:
                                raise
                        if T.is_all_nan(picked_vals):
                            continue
                        longterm_mean = np.nanmean(picked_vals)
                        spatial_dic_result[pix] = longterm_mean

                    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_result)
                    arr_flatten = arr.flatten()
                    arr_flatten = T.remove_np_nan(arr_flatten)
                    plt.figure(figsize=(6,3))
                    x,y = Plot().plot_hist_smooth(arr_flatten,bins=80,range=bin_range,alpha=0)
                    # x,y = Plot().plot_hist_smooth(arr_flatten,bins=80,alpha=0)
                    plt.plot(x,y,label=title,color=color_list[flag])
                    flag += 1
                    # up,down = T.get_vals_std_up_down(arr)
                    # plt.imshow(arr,vmax=up,vmin=down)
                    # plt.colorbar()
                    # plt.title(title)
                    plt.vlines(x=0,ymin=0,ymax=.1,linestyles='dashed')
                    plt.ylim(0,0.1)
                    print(title)
                    plt.legend()
                    plt.savefig(join(outdir,title+'.pdf'))
                    plt.close()
                    # plt.show()


    def __get_MA_dic(self,climate_var):
        outdir = join(temporary_root,'MA_dic')
        T.mkdir(outdir)
        outf = join(outdir,climate_var+'.npy')
        climate_dir = vars_info_dic[climate_var]['path']
        climate_dic = T.load_npy_dir(climate_dir)
        if isfile(outf):
            return T.load_npy(outf),climate_dic
        MA_climate_arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(climate_dic)
        MA_climate_dic = DIC_and_TIF().spatial_arr_to_dic(MA_climate_arr)
        T.save_npy(MA_climate_dic,outf)
        return MA_climate_dic,climate_dic


    def LAI_pdf_SOS_condition_climate_background(self):
        outdir = join(self.this_class_png,'LAI_pdf_SOS_condition_climate_background')
        climate_var = 'Precipitation'
        MA_climate_dic, climate_dic = self.__get_MA_dic(climate_var)
        T.mkdir(outdir)
        y_var = 'LAI_3g'
        region_var = 'HI_reclass'
        # condition_list = ['Advanced SOS','Delayed SOS']
        condition_list = ['Advanced SOS']
        # lai_dir = vars_info_dic[y_var]['path']
        start_year = vars_info_dic[y_var]['start_year']
        lai_dff = join(Pick_Early_Peak_Late_value().this_class_arr,'Pick_variables/LAI_3g.df')
        lai_df = T.load_df(lai_dff)
        lai_df = T.combine_df_columns(lai_df,['early','peak','late'],'all_gs')
        # season = 'all_gs'
        # season = 'peak'
        # season = 'late'
        seasonlist = global_season_dic
        seasonlist.append('all_gs')
        seasonlist = ['late']
        color_list = KDE_plot().makeColours(list(range(8)),cmap='jet')
        for season in seasonlist:
            print(season)
            flag = 0
            lai_dic = T.df_to_spatial_dic(lai_df,season)
            phenology_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
            all_spatial_dic = {
                'lai':lai_dic,
                'sos':phenology_dic
            }
            # region = 'Humid'
            # region = 'Non Humid'
            df = T.spatial_dics_to_df(all_spatial_dic)
            df = Dataframe().add_Humid_nonhumid(df)
            # region_list = T.get_df_unique_val_list(df,region_var)
            # region_list = ['Humid']
            region_list = ['Non Humid']
            # df = df[df['HI_reclass']==region]
            df = df.dropna()
            for region in region_list:
                df_region = df[df[region_var]==region]
                # plt.figure()
                for condition in condition_list:
                    title = f'{season}_{region}_{condition}'
                    lai_dic = T.df_to_spatial_dic(df_region,'lai')
                    early_start_dic_anomaly = T.df_to_spatial_dic(df_region,'sos')
                    year_list = list(range(start_year, 9999))
                    spatial_dic_result = {}

                    for pix in tqdm(lai_dic):
                        if not pix in early_start_dic_anomaly:
                            continue
                        vals = lai_dic[pix]
                        vals = T.detrend_vals(vals)
                        # vals = Pre_Process().z_score_climatology(vals)
                        vals = Pre_Process().cal_anomaly_juping(vals)
                        # vals_reshape = np.reshape(vals, (-1, 12))
                        dic_i = dict(zip(year_list, vals))
                        early_start_dic_anomaly_i = early_start_dic_anomaly[pix]
                        # print(early_start_dic_anomaly_i)
                        if type(early_start_dic_anomaly_i) == float:
                            continue
                        picked_vals = []
                        for year in dic_i:
                            if not year in early_start_dic_anomaly_i:
                                continue
                            sos_anomaly = early_start_dic_anomaly_i[year]
                            if condition == 'Advanced SOS':
                                if sos_anomaly < 0:
                                    picked_vals.append(dic_i[year])
                            elif condition == 'Delayed SOS':
                                if sos_anomaly > 0:
                                    picked_vals.append(dic_i[year])
                            else:
                                raise
                        if T.is_all_nan(picked_vals):
                            continue
                        longterm_mean = np.nanmean(picked_vals)
                        if longterm_mean > 0:
                        # if longterm_mean < 0:
                            spatial_dic_result[pix] = longterm_mean
                    for pix in tqdm(spatial_dic_result):
                        climate_ts = climate_dic[pix]
                        climate_ts_seasonal = T.monthly_vals_to_annual_val(climate_ts,method='array')
                        climate_ts_seasonal_mean = np.nanmean(climate_ts_seasonal,axis=0)
                        plt.plot(climate_ts_seasonal_mean,color='gray',alpha=0.002)
                    plt.ylim(0,150)
                    plt.title(title)
                    plt.ylabel(climate_var)
                    plt.xlabel('Month')
                    plt.xticks(T.mon)
                    plt.show()
                    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_result)
                    # plt.imshow(arr)
                    # plt.colorbar()
                    # print(title)
                    # # plt.savefig(join(outdir,title+'.pdf'))
                    # # plt.close()
                    # plt.show()


    def P_PET_dic_reclass(self,P_PET_dic_class):
        reclass_dic = {}
        for pix in P_PET_dic_class:
            val = P_PET_dic_class[pix]
            if not val == 'Humid':
                val_new = 'Non Humid'
            else:
                val_new = 'Humid'
            reclass_dic[pix] = val_new
        return reclass_dic


    def get_region_pix_list(self,region='Non Humid'):
        P_PET_dic_class = Main_flow.Dataframe().P_PET_class()
        P_PET_dic_reclass = self.P_PET_dic_reclass(P_PET_dic_class)
        P_PET_dic_reclass_reverse = T.reverse_dic(P_PET_dic_reclass)
        non_humid_pix_list = P_PET_dic_reclass_reverse[region]
        non_humid_pix_list = set(non_humid_pix_list)
        return non_humid_pix_list

    def early_peak_greening_speedup_advanced_sos(self):
        outdir = join(self.this_class_arr,'early_peak_greening_speedup_advanced_sos')
        T.mkdir(outdir)
        outf = join(outdir,'result.npy')
        # region filter


        # load phenology files
        annual_hants_fdir = join(Phenology().this_class_arr,'all_year_hants_annual')
        long_term_hants_fdir = join(Phenology().this_class_arr,'all_year_hants_annual_mean')
        phenology_longterm_dff = join(Phenology().this_class_arr,'longterm_mean_phenology/longterm_mean_phenology.df')
        phenology_annual_dff = join(Phenology().this_class_arr,'compose_annual_phenology/phenology_dataframe.df')
        annual_vals_dic = T.load_npy_dir(annual_hants_fdir)
        long_term_vals_dic = T.load_npy_dir(long_term_hants_fdir)
        phenology_longterm_df = T.load_df(phenology_longterm_dff)
        phenology_annual_df = T.load_df(phenology_annual_dff)
        phenology_longterm_dic = T.df_to_dic(phenology_longterm_df,'pix')
        phenology_annual_dic = T.df_to_dic(phenology_annual_df,'pix')
        region_pix_list = self.get_region_pix_list()
        # analysis

        # phenology columns:
        ###############################
        # dormant_length
        # early_end	early_end_mon	early_length	early_start	early_start_mon
        # late_end	late_end_mon	late_length	late_start	late_start_mon
        # mid_length	peak	peak_mon
        ###############################
        result_dic = {}
        for pix in tqdm(long_term_vals_dic):
            # if not pix in non_humid_pix_list:
            #     continue
            long_term_vals = long_term_vals_dic[pix]
            annual_vals_dic_i = annual_vals_dic[pix]
            annual_vals_dic_i = dict(annual_vals_dic_i)
            if not pix in phenology_longterm_dic:
                continue
            phenology_longterm_dic_i = phenology_longterm_dic[pix]
            phenology_annual_dic_i = phenology_annual_dic[pix]
            early_start_longterm = phenology_longterm_dic_i['early_start']
            early_end_longterm = phenology_longterm_dic_i['early_end']
            late_start_longterm = phenology_longterm_dic_i['late_start']
            peak_period_longterm = list(range(early_end_longterm, late_start_longterm))
            early_period_longterm = list(range(early_start_longterm, early_end_longterm))

            early_vals_long_term = T.pick_vals_from_1darray(long_term_vals,early_period_longterm)
            peak_vals_long_term = T.pick_vals_from_1darray(long_term_vals,peak_period_longterm)
            early_vals_long_term_mean = np.mean(early_vals_long_term)
            peak_vals_long_term_mean = np.mean(peak_vals_long_term)
            picked_year = []
            for year in annual_vals_dic_i:
                annual_vals = annual_vals_dic_i[year]
                if not year in phenology_annual_dic_i['early_start']:
                    continue
                early_start = phenology_annual_dic_i['early_start'][year]
                early_end = phenology_annual_dic_i['early_end'][year]
                late_start = phenology_annual_dic_i['late_start'][year]
                # condition 1: advanced SOS
                if early_start > early_start_longterm:
                    continue
                # condition 2: early mean > long term early mean
                early_period_annual = list(range(early_start,early_end))
                early_vals_annual = T.pick_vals_from_1darray(annual_vals,early_period_annual)
                early_vals_annual_mean = np.mean(early_vals_annual)
                if early_vals_annual_mean < early_vals_long_term_mean:
                    continue
                # condition 3: peak mean > long term peak mean
                peak_period_annual = list(range(early_end, late_start))
                peak_vals_annual = T.pick_vals_from_1darray(annual_vals, peak_period_annual)
                peak_vals_annual_mean = np.mean(peak_vals_annual)
                if peak_vals_annual_mean < peak_vals_long_term_mean:
                    continue
                picked_year.append(year)
            if len(picked_year) == 0:
                continue
            result_dic[pix] = picked_year
            #     plt.plot(annual_vals,label=str(year))
            # plt.plot(long_term_vals,color='k',lw=4)
            # plt.scatter(early_start_longterm,long_term_vals[early_start_longterm],s=80,zorder=99,color='r')
            # plt.scatter(early_end_longterm,long_term_vals[early_end_longterm],s=80,zorder=99,color='r')
            # plt.scatter(late_start_longterm,long_term_vals[late_start_longterm],s=80,zorder=99,color='r')
            # plt.legend()
            # plt.show()
            # exit()
        T.save_npy(result_dic,outf)
    def check(self):
        f = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Analysis/early_peak_greening_speedup_advanced_sos/result.npy'
        dic = T.load_npy(f)
        spatial_dic = {}
        for pix in dic:
            vals = dic[pix]
            spatial_dic[pix] = len(vals)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        std = np.nanstd(arr)
        mean = np.nanmean(arr)
        up = mean + std
        down = mean - std
        plt.imshow(arr,vmin=down,vmax=up,aspect='auto')
        plt.colorbar()
        plt.show()

    def get_vals_std_up_down(self,vals):
        vals = np.array(vals)
        std = np.nanstd(vals)
        mean = np.nanmean(vals)
        up = mean + std
        down = mean - std
        return up,down

    def inter_annual_carryover_effect(self):
        df_dir = join(Pick_Early_Peak_Late_value().this_class_arr,'Pick_variables')
        early_lai_var = 'early_LAI_3g'
        peak_lai_var = 'peak_LAI_3g'
        late_lai_var = 'late_LAI_3g'
        df_list = []
        for f in T.listdir(df_dir):
            if not 'LAI' in f:
                continue
            if not f.endswith('.df'):
                continue
            var_name = f.split('.')[0]
            print(var_name)
            df = T.load_df(join(df_dir,f))
            for season in global_season_dic:
                df = T.rename_dataframe_columns(df,season,f'{season}_{var_name}')
            df_list.append(df)
        df = pd.DataFrame()
        df_all = T.join_df_list(df,df_list,'pix')
        print(df_all.columns)
        spatial_dic = {}
        for i,row in tqdm(df_all.iterrows(),total=len(df_all)):
            pix = row.pix
            early_vals = row[early_lai_var]
            # peak_lai_val = row[peak_lai_var]
            late_vals = row[late_lai_var]
            try:
                r,p = T.nan_correlation(early_vals,late_vals)
                # r,p = T.nan_correlation(peak_lai_val,late_vals)
                spatial_dic[pix] = r
            except:
                continue
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        up,down = self.get_vals_std_up_down(arr)
        plt.imshow(arr,vmin=down,vmax=up,aspect='auto')
        plt.colorbar()
        plt.show()
        exit()
        # dff = join(RF().this_class_arr, 'Dataframe.df')
        # df = T.load_df(dff)
        # early_peak_LAI_3g_mean_var = 'early_peak_LAI_3g_mean'
        # # early_peak_LAI_3g_mean_var = 'early_LAI_3g'
        # # early_peak_LAI_3g_mean_var = 'peak_LAI_3g'
        # early_peak_late_sm_var = 'early_peak_late_Soil moisture_mean'
        # # early_peak_late_sm_var = 'early_peak_Soil moisture_accu_sum'
        # # early_peak_late_sm_var = 'early_peak_SPEI_accu_sum'
        # early_peak_late_sm_var = 'early_peak_late_SPEI_accu_sum'
        # late_lai_var = 'late_LAI_3g'
        # sos_var = 'SOS'
        # df = df[df[early_peak_LAI_3g_mean_var] > 0]
        # df = df[df[sos_var] < 0]
        pass



    def carry_over_effect_with_sm_matrix(self):
        # matrix
        # bin_n = 21
        bin_n = 11
        dff = join(RF().this_class_arr,'Dataframe.df')
        df = T.load_df(dff)

        # early_peak_LAI_3g_mean_var = 'early_peak_LAI_3g_mean'
        early_peak_LAI_3g_mean_var = 'early_LAI_3g'
        # early_peak_LAI_3g_mean_var = 'peak_LAI_3g'
        # early_peak_late_sm_var = 'early_peak_late_Soil moisture_mean'
        early_peak_late_sm_var = 'early_peak_Soil moisture_mean'
        # early_peak_late_sm_var = 'early_peak_Soil moisture_accu_sum'
        # early_peak_late_sm_var = 'early_peak_SPEI_accu_sum'
        # early_peak_late_sm_var = 'early_peak_late_SPEI_accu_sum'
        late_lai_var = 'late_LAI_3g'
        # late_lai_var = 'peak_LAI_3g'
        sos_var = 'SOS'
        # df = df[df[early_peak_LAI_3g_mean_var]>0]
        df = df[df[sos_var]<0]
        humid_var = 'Humid'
        # humid_var = 'Non Humid'
        df = df[df['HI_reclass'] == humid_var]
        print(df.columns)
        #
        # T.print_head_n(df)
        # print(df)
        # exit()
        sm_vals = df[early_peak_late_sm_var].tolist() # min: -.15 max: .15
        early_peak_LAI_vals = df[early_peak_LAI_3g_mean_var].tolist() # min: 0 max: 2
        # plt.hist(early_peak_LAI_vals,bins=80)
        # plt.show()
        sm_vals = T.remove_np_nan(sm_vals)
        # print(sm_vals)
        # exit()
        early_peak_LAI_vals = T.remove_np_nan(early_peak_LAI_vals)
        sm_bins = []
        for n in range(bin_n):
            sm_bins.append(np.quantile(sm_vals,n/bin_n))
        early_peak_LAI_bins = []
        for n in range(bin_n):
            early_peak_LAI_bins.append(np.quantile(early_peak_LAI_vals,n/bin_n))

        # sm_bins = np.linspace(-0.15,0.15,bin_n)
        # early_peak_LAI_bins = np.linspace(0,2,bin_n)
        matrix = []
        for sm_i in range(len(sm_bins)):
            if sm_i+1 >= len(sm_bins):
                continue
            df_sm = df[df[early_peak_late_sm_var]>sm_bins[sm_i]]
            df_sm = df_sm[df_sm[early_peak_late_sm_var]<=sm_bins[sm_i+1]]
            # print(len(df_sm))
            temp = []
            for lai_i in range(len(early_peak_LAI_bins)):
                if lai_i+1 >= len(early_peak_LAI_bins):
                    continue
                # print(early_peak_LAI_bins[lai_i])
                df_lai = df_sm[df_sm[early_peak_LAI_3g_mean_var]>early_peak_LAI_bins[lai_i]]
                df_lai = df_lai[df_lai[early_peak_LAI_3g_mean_var]<=early_peak_LAI_bins[lai_i+1]]
                # print(len(df_lai))
                # exit()
                late_lai_vals = df_lai[late_lai_var].tolist()
                # early_lai_vals = df_lai[early_peak_LAI_3g_mean_var].tolist()
                # r,p = T.nan_correlation(late_lai_vals,early_lai_vals)
                late_lai_mean = np.nanmean(late_lai_vals)
                temp.append(late_lai_mean)
                # temp.append(p)
            matrix.append(temp)
        mean = np.nanmean(matrix)
        std = np.nanstd(matrix)
        up = mean + std
        down = mean - std
        early_peak_LAI_bins_ticks = [f'{i:0.2f}' for i in early_peak_LAI_bins]
        sm_bins_ticks = [f'{i:0.2f}' for i in sm_bins]
        plt.xticks(range(bin_n),early_peak_LAI_bins_ticks,rotation=90)
        plt.yticks(range(bin_n),sm_bins_ticks)
        plt.imshow(matrix,vmin=-0.1,vmax=0.1,cmap='RdBu')
        # plt.imshow(matrix,vmin=down,vmax=up,cmap='RdBu')
        # plt.imshow(matrix,cmap='RdBu')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(late_lai_var, rotation=270)
        plt.xlabel(early_peak_LAI_3g_mean_var)
        plt.ylabel(early_peak_late_sm_var)
        plt.title(humid_var)
        plt.tight_layout()
        plt.show()
        pass


    def correlation_advanced_sos(self):
        outdir = join(self.this_class_png,'correlation_advanced_sos')
        T.open_path_and_file(outdir)
        # var1 = 'LAI_3g'
        var1 = 'Soil moisture'
        var2 = 'LAI_3g'
        var1_start_year = vars_info_dic[var1]['start_year']
        var2_start_year = vars_info_dic[var2]['start_year']
        # var1_season = f'early-peak-late_{var1}'
        var1_season = f'late_{var1}'
        var2_season = f'early_{var2}'
        outf_pdf = join(outdir,f'{var1_season}-{var2_season}_advanced.pdf')
        outf_tif = join(outdir,f'{var1_season}-{var2_season}_advanced_spatial.pdf')
        # outf_p = join(outdir,f'{var1_season}-{var2_season}_p_delayed.pdf')
        # outf_sos = join(outdir,f'advanced_sos_year_number.tif')
        T.mkdir(outdir)
        early_start_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        df = Global_vars().load_df()
        region_list = T.get_df_unique_val_list(df,'HI_reclass')
        region_dic = T.df_to_spatial_dic(df,'HI_reclass')
        region_dic_reverse = T.reverse_dic(region_dic)
        # print(df.columns)
        # exit()
        season_list = var1_season.split('_')[0]
        combine_list = []
        for season in season_list.split('-'):
            combine_list.append(f'{season}_{var1}')
        # print(combine_list)
        # exit()
        df = T.combine_df_columns(df,combine_list,f'{season_list}_{var1}',)

        var1_dic = T.df_to_spatial_dic(df,var1_season)
        var2_dic = T.df_to_spatial_dic(df,var2_season)
        spatial_dic = {}
        for pix in tqdm(var1_dic):
            if not pix in early_start_dic:
                continue
            sos = early_start_dic[pix]
            sos_reverse = T.reverse_dic(sos)
            advanced_sos_year = []
            for sos_i in sos_reverse:
                if sos_i < 0:
                    years = sos_reverse[sos_i]
                    for year in years:
                        advanced_sos_year.append(year)
            advanced_sos_year.sort()
            advanced_sos_year = np.array(advanced_sos_year)
            val1 = var1_dic[pix]
            val2 = var2_dic[pix]
            try:
                if T.is_all_nan(val1):
                    continue
                if T.is_all_nan(val2):
                    continue
            except:
                continue
            val1_pick = T.pick_vals_from_1darray(val1, advanced_sos_year - var1_start_year)
            val2_pick = T.pick_vals_from_1darray(val2, advanced_sos_year - var2_start_year)
            r,p = T.nan_correlation(val1_pick,val2_pick)
            spatial_dic[pix] = r
            # spatial_dic_p[pix] = p
        for region in region_list:
            region_pix = region_dic_reverse[region]
            selected_val = []
            for pix in spatial_dic:
                if pix in region_pix:
                    selected_val.append(spatial_dic[pix])
            selected_val = T.remove_np_nan(selected_val)
            x,y = Plot().plot_hist_smooth(selected_val,bins=80,alpha=0)
            plt.plot(x,y,label=region)
        plt.legend()
        plt.title(f'{var1_season}-{var2_season}_advanced_sos')
        plt.savefig(outf_pdf)

        plt.figure(figsize=(10,6))
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,vmin=-0.8,vmax=0.8,aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var1_season}-{var2_season}_advanced_sos')
        plt.savefig(outf_tif)


    def correlation_positive_sos_trend(self):
        outdir = join(self.this_class_tif,'correlation_positive_sos_trend')

        var1 = 'LAI_3g'
        # var2 = 'Soil moisture'
        var2 = 'LAI_3g'
        var1_start_year = vars_info_dic[var1]['start_year']
        var2_start_year = vars_info_dic[var2]['start_year']
        var1_season = f'early-peak_{var1}'
        var2_season = f'late_{var2}'
        outf = join(outdir,f'{var1_season}-{var2_season}_.tif')
        outf_p = join(outdir,f'{var1_season}-{var2_season}_p.tif')
        T.mkdir(outdir)
        early_start_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        df = Global_vars().load_df()
        # print(df.columns)
        # exit()
        season_list = var1_season.split('_')[0]
        combine_list = []
        for season in season_list.split('-'):
            combine_list.append(f'{season}_{var1}')
        df = T.combine_df_columns(df,combine_list,f'{season_list}_{var1}',)

        var1_dic = T.df_to_spatial_dic(df,var1_season)
        var2_dic = T.df_to_spatial_dic(df,var2_season)
        spatial_dic = {}
        spatial_dic_p = {}
        for pix in tqdm(var1_dic):
            if not pix in early_start_dic:
                continue
            sos_dic = early_start_dic[pix]
            year_list = []
            for year in sos_dic:
                year_list.append(year)
            year_list.sort()
            sos_list = []
            for year in year_list:
                sos = sos_dic[year]
                sos_list.append(sos)
            a, _, _, _ = T.nan_line_fit(year_list,sos_list)
            # if not a < 0:
            #     continue
            val1 = var1_dic[pix]
            val2 = var2_dic[pix]
            if T.is_all_nan(val1):
                continue
            if T.is_all_nan(val2):
                continue
            r,p = T.nan_correlation(val1,val2)
            spatial_dic[pix] = r
            spatial_dic_p[pix] = p
        DIC_and_TIF().pix_dic_to_tif(spatial_dic,outf)
        DIC_and_TIF().pix_dic_to_tif(spatial_dic_p,outf_p)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        pass


    def sort_dic_key(self,dic:dict):
        key_list = []
        for key in dic:
            key_list.append(key)
        key_list.sort()
        return key_list

    def correlation_sos_vs_vegetation(self):
        outdir = join(self.this_class_png,'correlation_sos_vs_vegetation')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        y_var = 'LAI_3g'
        condition_list = ['Advanced SOS','Delayed SOS','All SOS']
        start_year = vars_info_dic[y_var]['start_year']
        early_start_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        df = Global_vars().load_df()
        df = Dataframe().combine_season(df,y_var)
        global_season_dic.append('all_gs')
        year_list_all = list(range(start_year,9999))
        for condition in condition_list:
            outdir_i = join(outdir,condition)
            T.mkdir(outdir_i)
            for season in global_season_dic:
                col_name = f'{season}_{y_var}'
                spatial_dic = T.df_to_spatial_dic(df,col_name)
                corr_dic = {}
                for pix in tqdm(spatial_dic,desc=season):
                    vege_vals = spatial_dic[pix]
                    if T.is_all_nan(vege_vals):
                        continue
                    vege_dic = dict(zip(year_list_all,vege_vals))
                    if not pix in early_start_dic:
                        continue
                    sos_dic_i = early_start_dic[pix]
                    year_list = self.sort_dic_key(sos_dic_i)
                    vege_vals_pick = []
                    sos_pick = []
                    for year in year_list:
                        sos = sos_dic_i[year]
                        if condition == 'Advanced SOS':
                            if sos < 0:
                                vege_vals_pick.append(vege_dic[year])
                                sos_pick.append(sos)
                        elif condition == 'Delayed SOS':
                            if sos > 0:
                                vege_vals_pick.append(vege_dic[year])
                                sos_pick.append(sos)
                        elif condition == 'All SOS':
                            vege_vals_pick.append(vege_dic[year])
                            sos_pick.append(sos)
                        else:
                            raise
                    r,p = T.nan_correlation(vege_vals_pick,sos_pick)
                    corr_dic[pix] = r
                outf = join(outdir_i,f'{y_var}_{season}.pdf')
                # DIC_and_TIF().pix_dic_to_tif(corr_dic,outf)
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(corr_dic)
                plt.figure(figsize=(8,4))
                plt.imshow(arr,vmin=-0.8,vmax=0.8,aspect='auto')
                plt.colorbar()
                DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
                # plt.show()
                plt.title(f'{condition}_{y_var}_{season}')
                plt.savefig(outf)
                plt.close()


    def correlation_pdf_sos_vs_vegetation(self):
        outdir = join(self.this_class_png,'correlation_pdf_sos_vs_vegetation')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        y_var = 'LAI_3g'
        region_var = 'HI_reclass'
        condition_list = ['Advanced SOS','Delayed SOS','All SOS']
        start_year = vars_info_dic[y_var]['start_year']
        early_start_dic = self.get_phenology_spatial_dic('early_start',isanomaly=True)
        df = Global_vars().load_df()
        df = Dataframe().combine_season(df,y_var)
        region_list = T.get_df_unique_val_list(df,region_var)
        region_dic = T.df_to_spatial_dic(df,region_var)
        region_dic_reverse = T.reverse_dic(region_dic)
        global_season_dic.append('all_gs')
        year_list_all = list(range(start_year,9999))
        for condition in condition_list:
            outdir_i = join(outdir,condition)
            T.mkdir(outdir_i)
            for season in global_season_dic:
                col_name = f'{season}_{y_var}'
                spatial_dic = T.df_to_spatial_dic(df,col_name)
                corr_dic = {}
                for pix in tqdm(spatial_dic,desc=season):
                    vege_vals = spatial_dic[pix]
                    if T.is_all_nan(vege_vals):
                        continue
                    vege_dic = dict(zip(year_list_all,vege_vals))
                    if not pix in early_start_dic:
                        continue
                    sos_dic_i = early_start_dic[pix]
                    year_list = self.sort_dic_key(sos_dic_i)
                    vege_vals_pick = []
                    sos_pick = []
                    for year in year_list:
                        sos = sos_dic_i[year]
                        if condition == 'Advanced SOS':
                            if sos < 0:
                                vege_vals_pick.append(vege_dic[year])
                                sos_pick.append(sos)
                        elif condition == 'Delayed SOS':
                            if sos > 0:
                                vege_vals_pick.append(vege_dic[year])
                                sos_pick.append(sos)
                        elif condition == 'All SOS':
                            vege_vals_pick.append(vege_dic[year])
                            sos_pick.append(sos)
                        else:
                            raise
                    r,p = T.nan_correlation(vege_vals_pick,sos_pick)
                    corr_dic[pix] = r
                outf = join(outdir_i,f'{y_var}_{season}.pdf')
                # DIC_and_TIF().pix_dic_to_tif(corr_dic,outf)
                for region in region_list:
                    region_pix = region_dic_reverse[region]
                    picked_corr = []
                    for pix in corr_dic:
                        if pix in region_pix:
                            picked_corr.append(corr_dic[pix])
                    picked_corr = T.remove_np_nan(picked_corr)
                    x,y = Plot().plot_hist_smooth(picked_corr,bins=80,alpha=0)
                    plt.plot(x,y,label=region)
                plt.legend()
                plt.title(f'{condition}_{y_var}_{season}')
                # plt.show()
                plt.savefig(outf)
                plt.close()


    def carryover_effect_sm(self):
        tif = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/tif/Analysis/correlation_positive_sos_trend/early-peak_LAI_3g-late_LAI_3g_.tif'
        dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        sm_dir = vars_info_dic['Soil moisture']['path']
        sm_dic = T.load_npy_dir(sm_dir)
        # sm_mean = DIC_and_TIF().pix_dic_to_spatial_arr_mean(sm_dic)
        # plt.imshow(sm_mean)
        # plt.colorbar()
        # plt.show()
        matrix = []
        spatial_dic = {}
        for pix in tqdm(dic):
            carryover_val = dic[pix]
            if np.isnan(carryover_val):
                continue
            if carryover_val > 0:
                continue
            sm_vals = sm_dic[pix]
            if T.is_all_nan(sm_vals):
                continue
            sm_vals = Pre_Process().z_score_climatology(sm_vals)
            sm_vals_reshape = np.reshape(sm_vals,(-1,12))
            sm_vals_reshape_monthly_mean = np.nanmean(sm_vals_reshape,axis=0)
            # if True in np.isnan(sm_vals_reshape_monthly_mean):
            #     continue
            # plt.plot(sm_vals_reshape_monthly_mean)
            # plt.show()
            # plt.imshow(sm_vals_reshape)
            matrix.append(sm_vals_reshape_monthly_mean)
            spatial_dic[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.figure()
        matrix_mean = np.nanmean(matrix,axis=0)
        plt.plot(matrix_mean)
        plt.show()
        pass

    def plot_product_accordance(self):


        pass

class Dataframe:

    def __init__(self):
        self.this_class_arr = join(results_root_main_flow,'arr/Dataframe/')
        self.dff = self.this_class_arr + 'dataframe_1982-2018.df'
        self.P_PET_fdir = data_root+ 'original_dataset/aridity_P_PET_dic/'
        T.mkdir(self.this_class_arr,force=True)


    def run(self):
        df = self.__gen_df_init()
        # df = self.add_data(df)
        # df = self.add_lon_lat_to_df(df)
        df = self.add_Humid_nonhumid(df)
        # df = self.combine_season(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff, random=False)


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


    def __add_pix_to_df(self,df):
        pix_dic = DIC_and_TIF().void_spatial_dic()
        pix_list = []
        for pix in pix_dic:
            pix_list.append(pix)
        pix_list.sort()
        df['pix'] = pix_list
        return df


    def combine_season(self,df,yvar):
        combine_col_list = [f'early_{yvar}', f'late_{yvar}', f'peak_{yvar}',]
        new_col_name = f'all_gs_{yvar}'
        df = T.combine_df_columns(df,combine_col_list,new_col_name)
        return df
        pass

    def add_data(self,df):
        fdir = join(Pick_Early_Peak_Late_value().this_class_arr,'Pick_variables')
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            # print(f)
            var_name = f.split('.')[0]
            df_i = T.load_df(join(fdir,f))
            dic_i = T.df_to_dic(df_i,'pix')
            result_dic = {}
            for pix in dic_i:
                dic_i_i = dic_i[pix]
                dic_i_i_new = {}
                for col_i in dic_i_i:
                    if col_i == 'pix':
                        continue
                    vals = dic_i_i[col_i]
                    new_col_i = f'{col_i}_{var_name}'
                    dic_i_i_new[new_col_i] = vals
                result_dic[pix] = dic_i_i_new
            df_new = T.dic_to_df(result_dic,'pix')
            df_list.append(df_new)
        df_all = T.join_df_list(df,df_list,'pix')
        return df_all


    def add_ly_NDVI(self,df):
        fdir = join(data_root,'NDVI_ly/per_pix_seasonal')
        for season in global_season_dic:
            f = join(fdir,f'{season}.npy')
            dic = T.load_npy(f)
            df = T.add_spatial_dic_to_df(df,dic,f'{season}_ly_NDVI')
        return df

    def add_GEE_LAI(self,df):
        fdir = join(data_root, 'GEE_AVHRR_LAI/per_pix_seasonal')
        for season in global_season_dic:
            f = join(fdir, f'{season}.npy')
            dic = T.load_npy(f)
            df = T.add_spatial_dic_to_df(df, dic, f'{season}_GEE_AVHRR_LAI')
        return df
        pass

    def add_lon_lat_to_df(self,df):
        lon_list = []
        lat_list = []
        D = DIC_and_TIF()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            lon,lat = D.pix_to_lon_lat(pix)
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df

    def drop_n_std(self,vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_class(self):
        outdir = join(self.this_class_arr,'P_PET_class')
        T.mkdir(outdir)
        outf = join(outdir,'P_PET_class.npy')
        if isfile(outf):
            return T.load_npy(outf)
        dic = self.P_PET_ratio(self.P_PET_fdir)
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
        T.save_npy(dic_reclass,outf)
        return dic_reclass

    def P_PET_ratio(self,P_PET_fdir):
        # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals,warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term
    def add_Humid_nonhumid(self,df):
        P_PET_dic_reclass = self.P_PET_class()
        df = T.add_spatial_dic_to_df(df,P_PET_dic_reclass,'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Dryland'
        return df

    def add_lc(self,df):
        lc_f = join(GLC2000().datadir,'lc_dic_reclass.npy')
        lc_dic = T.load_npy(lc_f)
        df = T.add_spatial_dic_to_df(df,lc_dic,'GLC2000')
        return df

    def add_NDVI_mask(self, df):
        # ndvi_mask_tif = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
        ndvi_mask_tif = '/Volumes/NVME2T/greening_project_redo/conf/NDVI_mask.tif'
        # ndvi_mask_tif = join(NDVI().datadir,'NDVI_mask.tif')
        arr = ToRaster().raster2array(ndvi_mask_tif)[0]
        arr = T.mask_999999_arr(arr, warning=False)
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        df = T.add_spatial_dic_to_df(df, dic, 'ndvi_mask_mark')
        df = pd.DataFrame(df)
        df = df.dropna(subset=['ndvi_mask_mark'])
        return df

class Moving_window:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Moving_window_1982-2018',results_root_main_flow)
        self.__var_list()
        self.end_year = 2018
        # self.end_year = 2015
        pass

    def __load_df(self):
        df = Global_vars().load_df()
        return df

    def run(self):
        # self.print_var_list()
        # exit()
        # self.single_correlation()
        # self.single_correlation_matrix_plot()
        self.single_correlation_pdf_plot()
        # self.single_correlation_time_series_plot()

        # self.trend()
        # self.trend_with_advanced_sos()
        # self.check_moving_window_sos_trend()
        # self.trend_time_series_plot()
        # self.trend_matrix_plot()
        # self.trend_area_ratio_timeseries()
        # self.trend_area_ratio_timeseries_advanced_sos()

        # self.array()
        # self.array_carry_over_effect()
        # self.array_carry_over_effect_spatial_trend()

        # self.mean()
        # self.mean_time_series_plot()
        # self.mean_matrix_plot()

        # self.carryover_partial_correlation()
        # self.pdf_plot('partial_correlation_orign_nodetrend')
        # self.pdf_plot('partial_correlation_orign_detrend')
        # self.pdf_plot('partial_correlation_anomaly_nodetrend')
        # self.pdf_plot('partial_correlation_anomaly_detrend')
        # self.pdf_plot('partial_correlation')
        # self.multi_regression()

        # self.LAI_trend_vs_vpd_LAI_correlation()
        # self.LAI_trend_vs_vpd_LAI_correlation_all_year()

        pass

    def __var_list(self):
        # self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'PAR', 'Precip', 'SPEI3', 'VPD', 'temperature']
        # self.x_var_list = ['Aridity', 'CO2', 'SPEI', 'SPEI_accu', 'SPEI_min', 'Soil moisture', 'Soil moisture_accu', 'Soil moisture_min', 'Temperature']
        self.x_var_list = ['Aridity', 'CO2', 'SPEI', 'SPEI_accu', 'SPEI_min',
                           'Soil moisture', 'Soil moisture_accu', 'Soil moisture_min', 'Temperature']
        # self.y_var = 'LAI_GIMMS'
        self.y_var = 'LAI_3g'
        self.all_var_list = copy.copy(self.x_var_list)
        self.all_var_list.append(self.y_var)

    def __partial_corr_var_list(self):
        # y_var = f'LAI_GIMMS'
        y_var = f'LAI_3g'
        x_var_list = [f'CO2',
                           f'VPD',
                           f'PAR',
                           f'temperature',
                           f'CCI_SM',
                      ]
        all_vars_list = copy.copy(x_var_list)
        all_vars_list.append(y_var)

        return x_var_list,y_var,all_vars_list

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
        x_var_list_valid = []
        for x in x_var_list:
            if x in df:
                x_var_list_valid.append(x)
        for x in x_var_list_valid:
            x_var_list_valid_new_cov=copy.copy(x_var_list_valid)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def __cal_multi_regression(self,df,x_var_list,y_var):
        model = LinearRegression()
        x_var_new = []
        for x in x_var_list:
            if x in df:
                x_var_new.append(x)
        X = df[x_var_new]
        Y = df[y_var]
        model.fit(X,Y)
        coef = model.coef_
        result_dic = dict(zip(x_var_new,coef))
        return result_dic

    def print_var_list(self):
        df = self.__load_df()
        var_list = []
        for col in df:
            col = str(col)
            for season in global_season_dic:
                if season in col:
                    col_new = col.replace(f'{season}_','')
                    var_list.append(col_new)
        var_list = T.drop_repeat_val_from_list(var_list)
        print(var_list)


    def __gen_moving_window_index(self,start_year,end_year):
        # start_year = global_start_year
        # end_year = global_end_year
        n = global_n
        moving_window_index_list = []
        for y in range(start_year,end_year + 1):
            moving_window_index = list(range(y,y+n))
            moving_window_index_list.append(moving_window_index)
            if y+n > end_year:
                break
        return moving_window_index_list

    def add_constant_value_to_df(self,df):
        df = Dataframe().add_lon_lat_to_df(df)
        df = Dataframe().add_Humid_nonhumid(df)
        df = Dataframe().add_lc(df)
        df = Dataframe().add_NDVI_mask(df)
        return df

    def single_correlation(self):
        outdir = join(self.this_class_arr,'single_correlation')
        T.mkdir(outdir)
        outf = join(outdir,f'single_correlation_{global_n}.df')
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}

        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.x_var_list:
                    x_var = f'{season}_{x}'
                    y_var = f'{season}_{self.y_var}'
                    xval_all = df[x_var]
                    yval_all = df[y_var]
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list,xval_all))
                    y_spatial_dic = dict(zip(df_pix_list,yval_all))
                    for pix in y_spatial_dic:
                        xval = x_spatial_dic[pix]
                        yval = y_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        if type(yval) == float:
                            continue
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        yval_pick = T.pick_vals_from_1darray(yval,window_index)
                        r,p = T.nan_correlation(xval_pick,yval_pick)
                        key_r = f'{n}_{x_var}_r'
                        key_p = f'{n}_{x_var}_p'
                        results_dic[pix][key_r] = r
                        results_dic[pix][key_p] = p

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def carryover_partial_correlation(self):
        outdir = join(self.this_class_arr,'carryover_partial_correlation')
        T.mkdir(outdir)
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        x_var_list,y_var,all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df,'pix')
        print(x_var_list)
        print(y_var)
        exit()
        for n in range(len(moving_window_index_list)):
            outf = join(outdir, f'{n:02d}-{len(moving_window_index_list)}_partial_correlation.df')
            window_index = np.array(moving_window_index_list[n], dtype=int)
            window_index = window_index - global_start_year
            results_dic = {}
            for pix in pix_list:
                results_dic[pix] = {}
            for season in global_season_dic:
                key = f'{n:02d}_{season}'
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{n+1}/{len(moving_window_index_list)}'):
                    pix = row.pix
                    df_for_partial_corr = pd.DataFrame()
                    for x_var in x_var_list:
                        x_col_name = f'{season}_{x_var}'
                        xval = row[x_col_name]
                        if type(xval) == float:
                            continue
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        xval_pick = Pre_Process().cal_anomaly_juping(xval_pick)
                        xval_pick = T.detrend_vals(xval_pick)
                        df_for_partial_corr[x_var] = xval_pick
                    y_col_name = f'{season}_{self.y_var}'
                    yval = row[y_col_name]
                    if type(yval) == float:
                        continue
                    yval_pick = T.pick_vals_from_1darray(yval, window_index)
                    yval_pick = Pre_Process().cal_anomaly_juping(yval_pick)
                    yval_pick = T.detrend_vals(yval_pick)

                    df_for_partial_corr[self.y_var] = yval_pick
                    df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                    partial_correlation,partial_correlation_p_value = self.__cal_partial_correlation(df_for_partial_corr,x_var_list,y_var)
                    results_dic[pix][key] = partial_correlation

            df_result = T.dic_to_df(results_dic,'pix')
            T.print_head_n(df_result)
            # print(df_result)
            df_result = self.add_constant_value_to_df(df_result)
            T.save_df(df_result,outf)
            T.df_to_excel(df_result,outf)


    def multi_regression(self):
        outdir = join(self.this_class_arr,'multi_regression')
        T.mkdir(outdir)
        start_year = global_start_year
        moving_window_index_list = self.__gen_moving_window_index(start_year,self.end_year)
        df = self.__load_df()
        x_var_list,y_var,all_vars_list = self.__partial_corr_var_list()
        pix_list = T.get_df_unique_val_list(df,'pix')

        for n in range(len(moving_window_index_list)):
            outf = join(outdir, f'{n:02d}-{len(moving_window_index_list)}_partial_correlation.df')
            window_index = np.array(moving_window_index_list[n], dtype=int)
            window_index = window_index - global_start_year
            results_dic = {}
            for pix in pix_list:
                results_dic[pix] = {}
            for season in global_season_dic:
                key = f'{n:02d}_{season}'
                # if not season == 'early':
                #     continue
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{n+1}/{len(moving_window_index_list)}'):
                    pix = row.pix
                    df_for_partial_corr = pd.DataFrame()
                    for x_var in x_var_list:
                        x_col_name = f'{season}_{x_var}'
                        xval = row[x_col_name]
                        xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        df_for_partial_corr[x_var] = xval_pick
                    y_col_name = f'{season}_{self.y_var}'
                    yval = row[y_col_name]
                    if type(yval) == float:
                        continue
                    # print(yval)
                    yval_pick = T.pick_vals_from_1darray(yval, window_index)
                    df_for_partial_corr[self.y_var] = yval_pick
                    df_for_partial_corr = df_for_partial_corr.dropna(axis=1)
                    multi_reg = self.__cal_multi_regression(df_for_partial_corr,x_var_list,y_var)
                    results_dic[pix][key] = multi_reg
            # print(results_dic)
            # exit()
            df_result = T.dic_to_df(results_dic,'pix')
            T.print_head_n(df_result)
            # print(df_result)
            df_result = self.add_constant_value_to_df(df_result)
            T.save_df(df_result,outf)
            T.df_to_excel(df_result,outf)

    def __get_window_list(self,df,x_var_1):
        # x_var_1 = self.x_var_list[0]
        # x_var_1 = self.y_var
        window_list = []
        for col in df:
            col = str(col)
            if x_var_1 in col:
                window = col.split('_')[0]
                window_list.append(int(window))
        window_list = T.drop_repeat_val_from_list(window_list)
        return window_list

    def single_correlation_matrix_plot(self):
        outdir = join(self.this_class_png,'plot_single_correlation')
        T.mkdir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list = self.x_var_list
        window_list = self.__get_window_list(df,var_list[0])
        # variable = var_list[0]
        for variable in var_list:
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}_r'].tolist()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap='RdBu_r',vmin=-0.3,vmax=0.3)
                        # plt.scatter(x_list, y, c=z, s=120, marker='s', cmap='RdBu_r')
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                # plt.show()
                plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                plt.close()

    def trend_with_advanced_sos(self):
        early_start_dic = Analysis().get_phenology_spatial_dic('early_start',isanomaly=True)
        outdir = join(self.this_class_arr,'trend_with_advanced_sos')
        T.mkdir(outdir)
        outf = join(outdir,'trend.df')
        outf_sos = join(outdir,'trend_sos.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year,self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        results_dic_sos_trend = {}
        for pix in pix_list:
            results_dic[pix] = {}
            results_dic_sos_trend[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            global_season_dic.insert(0,'gs')
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                # print(self.all_var_list[0])
                # exit()
                # for x in [self.y_var]:
                for x in self.all_var_list:
                    x_var = f'{season}_{x}'
                    print(x_var)
                    if not x_var in df.columns:
                        continue
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        if not pix in early_start_dic:
                            continue
                        sos_dic_i = early_start_dic[pix]
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        year_list = []
                        sos_list = []
                        for year in sos_dic_i:
                            year_list.append(year)
                        year_list.sort()
                        for year in year_list:
                            sos = sos_dic_i[year]
                            sos_list.append(sos)
                        try:
                            sos_trend = T.nan_line_fit(year_list,sos_list)[0]
                            key_sos_r = f'{n}_sos_r'
                            results_dic_sos_trend[pix][key_sos_r] = sos_trend
                        except:
                            sos_trend = np.nan
                        if np.isnan(sos_trend):
                            continue
                        if sos_trend > 0:
                            continue
                        # print(sos_trend)
                        # exit()
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        # print(window_index)
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        except:
                            continue
                        # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                        try:
                            a, b, r, p = K.linefit(list(range(len(xval_pick))),xval_pick)
                            key_r = f'{n}_{x_var}_r'
                            key_p = f'{n}_{x_var}_p'
                            results_dic[pix][key_r] = a
                            results_dic[pix][key_p] = p
                        except:
                            continue
        df_result_sos = T.dic_to_df(results_dic_sos_trend, 'pix')
        df_result_sos = self.add_constant_value_to_df(df_result_sos)
        T.save_df(df_result_sos, outf_sos)
        T.df_to_excel(df_result_sos, outf_sos)
        df_result = T.dic_to_df(results_dic,'pix')
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def check_moving_window_sos_trend(self):
        dff = join(self.this_class_arr,'trend_with_advanced_sos/trend_sos.df')
        outdir = join(self.this_class_png,'check_moving_window_sos_trend')
        T.mkdir(outdir)
        df = T.load_df(dff)
        for col in df.columns:
            if not 'sos' in col:
                continue
            spatial_dic = T.df_to_spatial_dic(df,col)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            up,down = T.get_vals_std_up_down(arr)
            plt.imshow(arr,vmin=down,vmax=up)
            plt.colorbar()
            plt.savefig(join(outdir,col))
            plt.close()
        print(df.columns)
        exit()

        pass

    def trend(self):
        outdir = join(self.this_class_arr,'trend')
        T.mkdir(outdir)
        outf = join(outdir,'trend.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year,self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.all_var_list:
                    x_var = f'{season}_{x}'
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        # print(window_index)
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval,window_index)
                        except:
                            continue
                        # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                        try:
                            a, b, r, p = K.linefit(list(range(len(xval_pick))),xval_pick)
                            key_r = f'{n}_{x_var}_r'
                            key_p = f'{n}_{x_var}_p'
                            results_dic[pix][key_r] = a
                            results_dic[pix][key_p] = p
                        except:
                            continue

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)

    def trend_matrix_plot(self):
        outdir = join(self.this_class_png,'trend_matrix_plot')
        T.mkdir(outdir)
        dff = join(self.this_class_arr, 'trend/trend.df')
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list = self.all_var_list
        window_list = self.__get_window_list(df,'VPD')
        # variable = var_list[0]
        for variable in var_list:
            # if not 'VPD' in variable:
            #     continue
            print(variable)
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            # print(variable)
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        # print(df_HI)
                        # print(window_list)
                        # exit()
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}_r'].tolist()
                            # print(df_w)
                            # exit()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        if 'CCI' in variable:
                            cmap = 'RdBu'
                        else:
                            cmap = 'RdBu_r'
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap=cmap)
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                # plt.show()
                plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                plt.close()

    def trend_time_series_plot(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        fdir = join(self.this_class_arr, 'trend')
        dff = join(fdir, 'trend.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df,HI_reclass_var)
        # df = df[df[HI_reclass_var]=='Non Humid']
        # title = 'Non Humid'

        df = df[df[HI_reclass_var]=='Humid']
        title = 'Humid'

        # print(HI_reclass_list)
        # exit()
        lc_var = 'GLC2000'
        lc_list = T.get_df_unique_val_list(df,lc_var)
        window_list = self.__get_window_list(df,self.all_var_list[0])
        # print(window_list)
        # exit()
        for xvar in self.all_var_list:
            if not 'LAI' in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}_r'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list,color=color_dic[season],label=season)
            plt.legend()
            plt.title(xvar)
        plt.title(title)
        plt.show()
        exit()
        # df_all = df[df['HI_reclass']==humid]
        # # df
        # K = KDE_plot()
        # val_length = 37
        # y_var = f'{season}_{y_variable}'
        # window_list = []
        # for col in df_all:
        #     if f'{y_variable}' in str(col):
        #         window = col.split('_')[0]
        #         window_list.append(window)
        # window_list=T.drop_repeat_val_from_list(window_list)
        #
        # mean_list = []
        # std_list = []
        # for j in window_list:
        #     y_val = df_all[f'{j}_{season}_{y_variable}_trend'].to_list()
        #     y_val_mean = np.nanmean(y_val)
        #     y_val_std = np.nanstd(y_val)
        #     mean_list.append(y_val_mean)
        #     std_list.append(y_val_std)
        #     # print(df_co2)
        #     # matrix.append(y_list)
        # # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        # y_list = mean_list
        # x_list = range(len(y_list))
        # plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # # plt.imshow(matrix)
        # # plt.colorbar()
        # plt.xlabel('year')
        # plt.ylabel(y_variable)
        # plt.title(humid)
        # # plt.show()

    def trend_area_ratio_timeseries(self):
        y_variable = self.y_var
        # y_variable = 'VOD'
        # y_variable = 'GIMMS_NDVI'
        dff = join(self.this_class_arr, 'trend', 'trend.df')
        # print(dff)
        # exit()
        df_all = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        df_all = Dataframe().add_Humid_nonhumid(df_all)
        df_all = df_all[df_all[HI_reclass_var]=='Non Humid']
        title = 'Non Humid'

        # df_all = df_all[df_all[HI_reclass_var] == 'Humid']
        # title = 'Humid'
        for season in global_season_dic:
            plt.figure()
            color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
            K = KDE_plot()
            y_var = f'{season}_{y_variable}'
            window_list = self.__get_window_list(df_all,y_var)
            for w in window_list:
                df_new = pd.DataFrame()
                df_new['r'] = df_all[f'{w}_{y_var}_r']
                df_new['p'] = df_all[f'{w}_{y_var}_p']
                df_new = df_new.dropna()
                df_p_non_sig = df_new[df_new['p'] > 0.1]
                df_p_05 = df_new[df_new['p'] <= 0.1]
                df_p_05 = df_p_05[df_p_05['p'] >= 0.05]
                df_p_sig = df_new[df_new['p'] < 0.05]
                pos_05 = df_p_05[df_p_05['r'] >= 0]
                neg_05 = df_p_05[df_p_05['r'] < 0]
                pos_sig = df_p_sig[df_p_sig['r'] >= 0]
                neg_sig = df_p_sig[df_p_sig['r'] < 0]
                non_sig_ratio = len(df_p_non_sig) / len(df_new)
                pos_05_ratio = len(pos_05) / len(df_new)
                neg_05_ratio = len(neg_05) / len(df_new)
                pos_sig_ratio = len(pos_sig) / len(df_new)
                neg_sig_ratio = len(neg_sig) / len(df_new)
                bars = [pos_sig_ratio, pos_05_ratio, non_sig_ratio, neg_05_ratio, neg_sig_ratio]

                bottom = 0
                color_flag = 0
                for b in bars:
                    plt.bar(f'{w}', b, bottom=bottom, color=color_list[color_flag])
                    bottom += b
                    color_flag += 1
            plt.title(f'{title}-{y_var}')
            plt.tight_layout()
        plt.show()


    def trend_area_ratio_timeseries_advanced_sos(self):
        y_variable = self.y_var
        # y_variable = 'VOD'
        # y_variable = 'GIMMS_NDVI'
        dff = join(self.this_class_arr, 'trend_with_advanced_sos', 'trend.df')
        # print(dff)
        # exit()
        df_all = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        df_all = Dataframe().add_Humid_nonhumid(df_all)
        region = 'Humid'
        df_all = df_all[df_all[HI_reclass_var]==region]
        title = region

        # df_all = df_all[df_all[HI_reclass_var] == 'Humid']
        # title = 'Humid'
        for season in global_season_dic:
            plt.figure()
            color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
            K = KDE_plot()
            y_var = f'{season}_{y_variable}'
            window_list = self.__get_window_list(df_all,y_var)
            for w in window_list:
                df_new = pd.DataFrame()
                df_new['pix'] = df_all['pix']
                df_new['r'] = df_all[f'{w}_{y_var}_r']
                df_new['p'] = df_all[f'{w}_{y_var}_p']
                df_new = df_new.dropna()
                pix_list = T.get_df_unique_val_list(df_new,'pix')
                spatial_dic = {}
                for pix in pix_list:
                    spatial_dic[pix] = 1
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)

                # print(df_new)
                # exit()
                df_p_non_sig = df_new[df_new['p'] > 0.1]
                df_p_05 = df_new[df_new['p'] <= 0.1]
                df_p_05 = df_p_05[df_p_05['p'] >= 0.05]
                df_p_sig = df_new[df_new['p'] < 0.05]
                pos_05 = df_p_05[df_p_05['r'] >= 0]
                neg_05 = df_p_05[df_p_05['r'] < 0]
                pos_sig = df_p_sig[df_p_sig['r'] >= 0]
                neg_sig = df_p_sig[df_p_sig['r'] < 0]
                non_sig_ratio = len(df_p_non_sig) / len(df_new)
                pos_05_ratio = len(pos_05) / len(df_new)
                neg_05_ratio = len(neg_05) / len(df_new)
                pos_sig_ratio = len(pos_sig) / len(df_new)
                neg_sig_ratio = len(neg_sig) / len(df_new)
                bars = [pos_sig_ratio, pos_05_ratio, non_sig_ratio, neg_05_ratio, neg_sig_ratio]

                bottom = 0
                color_flag = 0
                for b in bars:
                    plt.bar(f'{w}', b, bottom=bottom, color=color_list[color_flag])
                    bottom += b
                    color_flag += 1
            plt.title(f'{title}-{y_var}')
            plt.tight_layout()
            plt.figure()
            plt.imshow(arr)
        plt.show()

    def mean(self):
        outdir = join(self.this_class_arr, 'mean')
        T.mkdir(outdir)
        outf = join(outdir, 'mean_relative_change.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year, self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df, 'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                for x in self.all_var_list:
                    x_var = f'{season}_{x}'
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix'].tolist()
                    if not len(xval_all) == len(df_pix_list):
                        raise UserWarning
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        xval = Global_vars().cal_relative_change(xval)
                        window_index = np.array(moving_window_index_list[n], dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval, window_index)
                            mean_val = np.nanmean(xval_pick)
                            # r,p = T.nan_correlation(list(range(len(xval_pick))),xval_pick)
                            key = f'{n}_{x_var}'
                            results_dic[pix][key] = mean_val
                        except:
                            continue
        df_result = T.dic_to_df(results_dic, 'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

        pass

    def mean_time_series_plot(self):
        # y_variable = 'GIMMS_NDVI'
        # season = 'peak'
        # humid = 'Non Humid'
        color_dic = {
            'early':'g',
            'peak':'r',
            'late':'b',
        }
        fdir = join(self.this_class_arr, 'mean')
        dff = join(fdir, 'mean.df')
        df = T.load_df(dff)
        HI_reclass_var = 'HI_reclass'
        HI_reclass_list = T.get_df_unique_val_list(df,HI_reclass_var)
        df = df[df[HI_reclass_var]=='Non Humid']
        title = 'Non Humid'

        # df = df[df[HI_reclass_var]=='Humid']
        # title = 'Humid'

        # print(HI_reclass_list)
        # exit()
        lc_var = 'GLC2000'
        lc_list = T.get_df_unique_val_list(df,lc_var)
        window_list = self.__get_window_list(df,'LAI_GIMMS')
        # print(window_list)
        # exit()
        for xvar in self.all_var_list:
            if not 'LAI' in xvar:
                continue
            plt.figure()
            for season in global_season_dic:
                mean_list = []
                for w in window_list:
                    col_name = f'{w}_{season}_{xvar}'
                    print(col_name)
                    vals = df[col_name].tolist()
                    mean = np.nanmean(vals)
                    mean_list.append(mean)
                plt.plot(mean_list,color=color_dic[season],label=season)
            plt.legend()
            plt.title(xvar)
        plt.title(title)
        plt.show()
        exit()
        # df_all = df[df['HI_reclass']==humid]
        # # df
        # K = KDE_plot()
        # val_length = 37
        # y_var = f'{season}_{y_variable}'
        # window_list = []
        # for col in df_all:
        #     if f'{y_variable}' in str(col):
        #         window = col.split('_')[0]
        #         window_list.append(window)
        # window_list=T.drop_repeat_val_from_list(window_list)
        #
        # mean_list = []
        # std_list = []
        # for j in window_list:
        #     y_val = df_all[f'{j}_{season}_{y_variable}_trend'].to_list()
        #     y_val_mean = np.nanmean(y_val)
        #     y_val_std = np.nanstd(y_val)
        #     mean_list.append(y_val_mean)
        #     std_list.append(y_val_std)
        #     # print(df_co2)
        #     # matrix.append(y_list)
        # # y_list = SMOOTH().smooth_convolve(mean_list,window_len=7)
        # y_list = mean_list
        # x_list = range(len(y_list))
        # plt.plot(x_list, y_list,lw=4,alpha=0.8,label=season,color=color_dic[season])
        # # plt.imshow(matrix)
        # # plt.colorbar()
        # plt.xlabel('year')
        # plt.ylabel(y_variable)
        # plt.title(humid)
        # # plt.show()


    def mean_matrix_plot(self):
        outdir = join(self.this_class_png,'mean_matrix_plot')
        T.mkdir(outdir)
        dff = join(self.this_class_arr, 'mean/mean.df')
        df = T.load_df(dff)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list = self.all_var_list
        window_list = self.__get_window_list(df,'VPD')
        # variable = var_list[0]
        for variable in var_list:
            # if not 'VPD' in variable:
            #     continue
            print(variable)
            lc_list = ['Evergreen', 'Deciduous', 'Shrubs', 'Grass']
            HI_class_list = ['Humid', 'Non Humid']
            HI_class_var = 'HI_reclass'
            matrix = []
            # print(variable)
            for HI_class in HI_class_list:
                plt.figure(figsize=(10,7))
                for season in global_season_dic:
                    for lc in lc_list:
                        # val_length = 34
                        K = KDE_plot()
                        df_lc = df[df['GLC2000'] == lc]
                        df_HI = df_lc[df_lc[HI_class_var] == HI_class]
                        # print(df_HI)
                        # print(window_list)
                        # exit()
                        mean_list = []
                        x_list = []
                        for w in window_list:
                            df_w = df_HI[f'{w}_{season}_{variable}'].tolist()
                            # print(df_w)
                            # exit()
                            mean = np.nanmean(df_w)
                            mean_list.append(mean)
                            x_list.append(w)

                        # matrix.append(mean_list)
                        y = [f'{season}_{variable}_{lc}'] * len(mean_list)
                        if 'CCI' in variable:
                            cmap = 'RdBu'
                        else:
                            cmap = 'RdBu_r'
                        z = mean_list
                        plt.scatter(x_list, y, c=z, s=120, marker='s', cmap=cmap,vmin=0,vmax=20)
                        plt.subplots_adjust(
                            top=0.926,
                            bottom=0.08,
                            left=0.404,
                            right=0.937,
                            hspace=0.2,
                            wspace=0.2
                        )
                plt.colorbar()
                title = f'{variable}_{HI_class}'
                plt.title(title)
                # plt.tight_layout()
                plt.axis('equal')
                plt.show()
                # plt.savefig(join(outdir, f'{variable}_{HI_class}.pdf'))
                # plt.close()

    def single_correlation_pdf_plot(self):
        outdir = join(self.this_class_png, 'single_correlation_pdf_plot')
        T.mkdir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list = self.x_var_list
        window_list = self.__get_window_list(df, var_list[0])
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for variable in var_list:
            print(variable)
            matrix = []
            for season in global_season_dic:
                K = KDE_plot()
                plt.figure(figsize=(5,3))
                for w in window_list:
                    vals = df[f'{w}_{season}_{variable}_r'].tolist()
                    vals = T.remove_np_nan(vals)
                    x,y = Plot().plot_hist_smooth(vals,alpha=0,bins=80)
                    plt.plot(x,y,color=gradient_color[w],label=str(w))
                plt.title(variable)
                plt.tight_layout()
                # plt.show()
                plt.savefig(join(outdir,f'{variable}-{season}.pdf'))
                # plt.legend()
                # plt.savefig(join(outdir,f'legend.pdf'))
                plt.close()
                # exit()
        pass


    def single_correlation_time_series_plot(self):
        # outdir = join(self.this_class_png, 'single_correlation_pdf_plot')
        # T.mkdir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        ### check spatial ###
        # spatial_dic = T.df_to_spatial_dic(df,'ndvi_mask_mark')
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list,_,_ = self.__partial_corr_var_list()
        window_list = self.__get_window_list(df, var_list[0])
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for season in global_season_dic:
            K = KDE_plot()
            plt.figure(figsize=(5, 3))
            for variable in var_list:
                print(variable)
                x = []
                y = []
                for w in window_list:
                    vals = df[f'{w}_{season}_{variable}_r'].tolist()
                    mean = np.nanmean(vals)
                    x.append(w)
                    y.append(mean)
                plt.plot(x,y,label=variable)
            plt.title('Moving window single correlation\n'+season)
            plt.tight_layout()
            plt.legend()
        plt.show()
        pass



    def pdf_plot(self,func_name):
        outdir = join(self.this_class_png, func_name+'_pdf')
        fdir = join(self.this_class_arr,func_name)
        T.mkdir(outdir)
        # dff = join(self.this_class_arr, f'{func_name}/{func_name}.df')
        # df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mkdir(outdir)
        var_list,_,_ = self.__partial_corr_var_list()
        window_list = []
        for f in T.listdir(fdir):
            window = f.split('-')[0]
            window_list.append(window)
        window_list = T.drop_repeat_val_from_list(window_list)
        window_list = [int(i) for i in window_list]
        # variable = var_list[0]
        gradient_color = KDE_plot().makeColours(window_list,'Spectral')
        for variable in var_list:
            print(variable)
            matrix = []
            for season in global_season_dic:
                K = KDE_plot()
                plt.figure(figsize=(5,3))
                for w in window_list:
                    dff = f'{w:02d}-23_partial_correlation.df'
                    df = T.load_df(join(fdir,dff))
                    dic_all = df[f'{w:02d}_{season}'].tolist()
                    vals = []
                    for dic_i in dic_all:
                        # print(pix)
                        # dic_i = dic_all[pix]
                        # print(dic_i)
                        if type(dic_i) == float:
                            continue
                        if not variable in dic_i:
                            continue
                        val = dic_i[variable]
                        vals.append(val)
                    vals = T.remove_np_nan(vals)
                    x,y = Plot().plot_hist_smooth(vals,alpha=0,bins=80)
                    plt.plot(x,y,color=gradient_color[w],label=str(w))
                # plt.show()
                plt.title(variable)
                plt.tight_layout()
                plt.savefig(join(outdir,f'{variable}-{season}.pdf'))

                # plt.legend()
                # plt.savefig(join(outdir,f'legend.pdf'))
                plt.close()
                # exit()
        pass



    def LAI_trend_vs_vpd_LAI_correlation(self):
        correlation_fdir = join(self.this_class_arr,'single_correlation')
        dff_corr = join(correlation_fdir,f'single_correlation_{global_n}.df')
        df_corr = T.load_df(dff_corr)
        df_corr = Global_vars().clean_df(df_corr)
        trend_corr_dff= join(self.this_class_arr,f'trend/trend_{global_n}.df')
        df_trend = T.load_df(trend_corr_dff)
        df_trend = Global_vars().clean_df(df_trend)

        x_var_list,_,_ = self.__partial_corr_var_list()
        window_list = self.__get_window_list(df_trend,'Aridity')
        color_list = KDE_plot().makeColours(window_list, 'Spectral')
        print(window_list)
        print(df_trend)
        print(trend_corr_dff)
        for x_var in x_var_list:
            for season in global_season_dic:
                plt.figure()
                for w in window_list:
                    dic_trend = T.df_to_spatial_dic(df_trend,f'{w}_{season}_{self.y_var}_r')
                    dic_corr = T.df_to_spatial_dic(df_corr,f'{w}_{season}_{x_var}_r')
                    x = []
                    y = []
                    for pix in dic_trend:
                        if not pix in dic_corr:
                            continue
                        x.append(dic_trend[pix])
                        y.append(dic_corr[pix])
                    # KDE_plot().plot_scatter(x,y)
                    # plt.show()
                    plt.scatter(x,y,s=1,alpha=0.2,color=color_list[w])
                plt.title(f'{x_var}_{season}')
                plt.xlabel(f'{self.y_var} trend')
                plt.ylabel(f'{x_var} vs LAI Correlation')
            plt.show()

        pass

    def array(self):
        outdir = join(self.this_class_arr,'array')
        T.mkdir(outdir)
        outf = join(outdir,'array_y.df')
        moving_window_index_list = self.__gen_moving_window_index(global_start_year,self.end_year)
        # print(moving_window_index_list)
        # exit()
        df = self.__load_df()
        pix_list = T.get_df_unique_val_list(df,'pix')
        results_dic = {}
        for pix in pix_list:
            results_dic[pix] = {}
        K = KDE_plot()
        for n in tqdm(range(len(moving_window_index_list))):
            # print(n)
            # if not n == 0:
            #     continue
            for season in global_season_dic:
                # if not season == 'early':
                #     continue
                # for x in self.all_var_list:
                for x in [self.y_var]:
                    x_var = f'{season}_{x}'
                    xval_all = df[x_var].tolist()
                    df_pix_list = df['pix']
                    x_spatial_dic = dict(zip(df_pix_list, xval_all))
                    for pix in x_spatial_dic:
                        xval = x_spatial_dic[pix]
                        if type(xval) == float:
                            continue
                        # xval = Pre_Process().cal_anomaly_juping(xval)
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        # print(window_index)
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval,window_index)
                            xval_pick = Pre_Process().cal_anomaly_juping(xval_pick)
                            results_dic[pix][f'{n:02d}_{season}_{x}'] = xval_pick
                        except:
                            continue

        df_result = T.dic_to_df(results_dic,'pix')
        T.print_head_n(df_result)
        print(df_result)
        df_result = self.add_constant_value_to_df(df_result)
        T.save_df(df_result,outf)
        T.df_to_excel(df_result,outf)


    def array_carry_over_effect(self):
        # corr_mode = 'early_late'
        # corr_mode = 'peak_late'
        corr_mode = 'early-peak_late'

        mode_list = corr_mode.split('_')
        outdir = join(self.this_class_tif,f'array_carry_over_effect/{corr_mode}')
        T.mkdir(outdir,force=True)
        dff = join(self.this_class_arr,'array/array_y.df')
        df = T.load_df(dff)
        var_name = 'LAI_3g'
        window_list = self.__get_window_list(df,var_name)
        pix_list = T.get_df_unique_val_list(df,'pix')
        for w in tqdm(window_list):
            window_dic = {}
            for season in global_season_dic:
                col_name = f'{w:02d}_{season}_{var_name}'
                spatial_dic = T.df_to_spatial_dic(df,col_name)
                window_dic[season] = spatial_dic
            spatial_dic_corr = {}
            for pix in pix_list:
                if '-' in mode_list[0]:
                    mode_list_1,mode_list_2 = mode_list[0].split('-')
                    # print(mode_list_1)
                    # print(mode_list_2)
                    vals1_1 = window_dic[mode_list_1][pix]
                    vals1_2 = window_dic[mode_list_2][pix]
                    vals1 = (vals1_1+vals1_2) / 2.
                    vals2 = window_dic[mode_list[1]][pix]
                else:
                    vals1 = window_dic[mode_list[0]][pix]
                    vals2 = window_dic[mode_list[1]][pix]
                try:
                    r,p = T.nan_correlation(vals1,vals2)
                except:
                    r = np.nan
                spatial_dic_corr[pix] = r
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_corr)
            outf = join(outdir,f'{w:02d}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dic_corr,outf)
        pass




    def array_carry_over_effect_spatial_trend(self):
        # corr_mode = 'early_late'
        # corr_mode = 'peak_late'
        corr_mode = 'early-peak_late'
        outdir = join(self.this_class_tif,f'array_carry_over_effect/trend/')
        outf = join(outdir,f'{corr_mode}.tif')
        outf_p = join(outdir,f'{corr_mode}_p.tif')
        T.mkdir(outdir)
        fdir = join(self.this_class_tif,f'array_carry_over_effect/{corr_mode}')
        all_dic = {}
        window_list = []
        dic_i = None
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            window = f.split('.')[0]
            dic_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            all_dic[window] = dic_i
            window_list.append(window)
        pix_list = []
        for pix in dic_i:
            pix_list.append(pix)

        spatial_dic_trend = {}
        spatial_dic_trend_p = {}
        for pix in tqdm(pix_list):
            vals_list = []
            for w in window_list:
                val = all_dic[w][pix]
                vals_list.append(val)
            a,_,_,p = T.nan_line_fit(list(range(len(vals_list))),vals_list)
            spatial_dic_trend[pix] = a
            spatial_dic_trend_p[pix] = p
        DIC_and_TIF().pix_dic_to_tif(spatial_dic_trend,outf)
        DIC_and_TIF().pix_dic_to_tif(spatial_dic_trend_p,outf_p)
        pass


    def LAI_trend_vs_vpd_LAI_correlation_all_year(self):
        df = Global_vars().load_df()
        x_var_list,_,_ = self.__partial_corr_var_list()
        for x_var in x_var_list:
            if not 'CCI' in x_var:
                continue
            for season in global_season_dic:
                var_name = f'{season}_{x_var}'
                x_spatial_dic = T.df_to_spatial_dic(df,var_name)
                y_spatial_dic = T.df_to_spatial_dic(df,f'{season}_{self.y_var}')
                x = []
                y = []
                for pix in tqdm(x_spatial_dic):
                    xvals = x_spatial_dic[pix]
                    yvals = y_spatial_dic[pix]
                    if type(yvals) == float:
                        continue
                    if type(xvals) == float:
                        continue
                    y_k, b, r, p = KDE_plot().linefit(list(range(len(yvals))),yvals)
                    corr,_ = T.nan_correlation(xvals,yvals)
                    x.append(corr)
                    y.append(y_k)
                plt.figure()
                plt.scatter(x,y,s=1)
                plt.xlabel(f'{x_var}-{self.y_var} correlation')
                plt.ylabel(f'{self.y_var} trend')
                plt.title(season)
            plt.show()

class Partial_corr:

    def __init__(self):
        self.__var_list()
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Partial_corr',results_root_main_flow)

        # self.dff = join(self.this_class_arr,f'Dataframe_{season}.df')


    def run(self):
        self.build_df()
        self.cal_partial_corr()
        # self.tif_partial_corr_spatial()
        pass

    def __var_list(self):
        # self.x_var_list = ['Aridity', 'CCI_SM', 'CO2', 'PAR', 'Precip', 'SPEI3', 'VPD', 'temperature']
        # self.x_var_list = [
        #     'early_peak_late_Soil moisture_mean',
        #     'early_peak_LAI_3g_mean',
        #     'early',
        #     'early',
        # ]
        # self.y_var = 'LAI_GIMMS'
        self.y_var = 'late_lai'
        # self.all_var_list = copy.copy(self.x_var_list)
        # self.all_var_list.append(self.y_var)


    def __partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p

    def __cal_partial_correlation(self,df,x_var_list):

        partial_correlation={}
        partial_correlation_p_value = {}
        for x in x_var_list:
            # print(x)
            x_var_list_valid_new_cov=copy.copy(x_var_list)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,self.y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def __cal_anomaly(self,vals):
        mean = np.nanmean(vals)
        anomaly_list = []
        for val in vals:
            anomaly = val - mean
            anomaly_list.append(anomaly)
        anomaly_list = np.array(anomaly_list)
        return anomaly_list

    def __cal_relative_change(self,vals):
        base = vals[:3]
        base = np.nanmean(base)
        relative_change_list = []
        for val in vals:
            change_rate = (val - base) / base
            relative_change_list.append(change_rate)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list


    def build_df(self):
        outdir = join(self.this_class_arr,'Dataframe')
        T.mkdir(outdir)
        outf = join(outdir,'partial_corr_df.df')
        early_start_dic = Analysis().get_phenology_spatial_dic('early_start',isanomaly=True)
        df_partial_correlation = pd.DataFrame()
        df = Global_vars().load_df()
        print(df.columns)

        all_pix_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            if not pix in early_start_dic:
                continue
            sos_dic_i = early_start_dic[pix]
            early_lai = row['early_LAI_3g']
            peak_lai = row['peak_LAI_3g']
            late_lai = row['late_LAI_3g']
            sm_early = row['early_Soil moisture']
            sm_peak = row['peak_Soil moisture']
            sm_late = row['late_Soil moisture']
            sos_list = []
            year_list = []
            for year in sos_dic_i:
                year_list.append(year)
            year_list.sort()
            for year in year_list:
                sos = sos_dic_i[year]
                sos_list.append(sos)
            # try:
            #     sos_trend, b, r, p  = T.nan_line_fit(year_list,sos_list)
            # except:
            #     sos_trend = np.nan
            late_temperature = row['late_Temperature']
            try:
                all_gs_sm = np.nanmean([sm_early,sm_peak,sm_late],axis=0)
            except:
                print(sm_early,sm_peak,sm_late)
                exit()
                all_gs_sm = np.nan
            early_peak_lai = np.nanmean([early_lai,peak_lai],axis=0)
            dic_i = {
                    'late_lai':late_lai,
                    'early_peak_lai':early_peak_lai,
                    'all_gs_sm':all_gs_sm,
                    'sos':sos_list,
                    'late_temperature':late_temperature,
                 }
            all_pix_dic[pix] = dic_i
        df_all = T.dic_to_df(all_pix_dic,'pix')
        print(df_all.columns)
        T.print_head_n(df_all)
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf)

    def cal_partial_corr(self):
        dff = join(self.this_class_arr, 'Dataframe/partial_corr_df.df')
        outdir = join(self.this_class_arr,f'{self.y_var}')
        T.mkdir(outdir,force=True)
        df = T.load_df(dff)
        print(df.columns)
        partial_corr_r_spatial_dic = {}
        partial_corr_p_spatial_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            # if r > 120:
            #     continue
            # df_i = pd.DataFrame(row)
            val_dic = dict(row)
            y_vals = val_dic[self.y_var]
            if type(y_vals) == float:
                continue
            vals_len = len(y_vals)
            df_i = pd.DataFrame()
            success = 1
            for col in df.columns:
                if col == 'pix':
                    continue
                vals_list = []
                vals = val_dic[col]
                for val in vals:
                    vals_list.append(val)
                if vals_len != len(vals_list):
                    success = 0
                    continue
                df_i[col] = vals_list
            df_i = df_i.dropna(axis=1)
            if success == 0:
                continue
            if not self.y_var in df_i.columns:
                continue
            # print(df_i)
            # exit()
            x_var_valid = []
            for col in df_i.columns:
                if col == self.y_var:
                    continue
                else:
                    x_var_valid.append(col)
            dic_partial_corr,dic_partial_corr_p = self.__cal_partial_correlation(df_i,x_var_valid)
            partial_corr_r_spatial_dic[pix] = dic_partial_corr
            partial_corr_p_spatial_dic[pix] = dic_partial_corr_p
        df_partial_corr_r = T.dic_to_df(partial_corr_r_spatial_dic,'pix')
        df_partial_corr_p = T.dic_to_df(partial_corr_p_spatial_dic,'pix')
        T.save_df(df_partial_corr_r,join(outdir,'partial_corr_r.df'))
        T.df_to_excel(df_partial_corr_r,join(outdir,'partial_corr_r.df'))
        T.save_df(df_partial_corr_p,join(outdir,'partial_corr_p.df'))
        T.df_to_excel(df_partial_corr_p,join(outdir,'partial_corr_p.df'))



    def tif_partial_corr_spatial(self):
        outdir = join(self.this_class_tif,'partial_corr_spatial')
        T.mkdir(outdir)
        dff = join(self.this_class_arr,self.y_var,'partial_corr_r.df')
        df = T.load_df(dff)
        for col in df.columns:
            if col == 'pix':
                continue
            print(col)
            spatial_dic = T.df_to_spatial_dic(df,col)
            DIC_and_TIF().pix_dic_to_tif(spatial_dic,join(outdir,f'{col}.tif'))

        pass
class Drought_event:

    def __init__(self):

        pass

    def run(self):
        # self.shp_to_raster()
        self.plot_seasonal_line()
        pass

    def shp_to_raster(self):
        in_shp = join(this_root,'conf/Europe.shp')
        output_raster = join(this_root,'conf/Europe.tif')
        pixel_size = 0.5
        in_raster_template = join(this_root,'conf/land.tif')
        ToRaster().shp_to_raster(in_shp, output_raster, pixel_size, in_raster_template=in_raster_template)

        pass

    def plot_seasonal_line(self):
        y_var = 'LAI_3g'
        europe_tif = join(this_root,'conf/Europe.tif')
        europe_dic = DIC_and_TIF().spatial_tif_to_dic(europe_tif)
        sos_dff = join(Phenology().this_class_arr,'compose_annual_phenology/phenology_dataframe.df')
        sos_df = T.load_df(sos_dff)
        early_start_dic = T.df_to_spatial_dic(sos_df,'early_start')
        early_start_dic_anomaly = {}
        for pix in early_start_dic:
            year_list = []
            dic_i = early_start_dic[pix]
            for year in dic_i:
                year_list.append(year)
            year_list.sort()
            val_list = []
            for year in year_list:
                val = dic_i[year]
                val_list.append(val)
            val_list_anomaly = Pre_Process().cal_anomaly_juping(val_list)
            dic_i_anomaly = dict(zip(year_list,val_list_anomaly))
            early_start_dic_anomaly[pix] = dic_i_anomaly

        europe_pix = []
        for pix in europe_dic:
            val = europe_dic[pix]
            if val == 1:
                europe_pix.append(pix)
        lai_dir = vars_info_dic[y_var]['path']
        start_year = vars_info_dic[y_var]['start_year']
        lai_dic = T.load_npy_dir(lai_dir)
        lai_df = T.spatial_dics_to_df({'lai':lai_dic})
        # Dataframe().add
        # print(lai_df)
        # exit()
        year_list = list(range(start_year,9999))
        all_vals_2018 = []
        all_vals_2017 = []
        pick_year = 2010
        for pix in tqdm(europe_pix):
            vals = lai_dic[pix]
            vals = T.detrend_vals(vals)
            vals = Pre_Process().z_score_climatology(vals)
            # vals = Pre_Process().cal_anomaly_juping(vals)
            vals_reshape = np.reshape(vals,(-1,12))
            dic_i = dict(zip(year_list,vals_reshape))
            vals_2018 = dic_i[pick_year]
            # vals_2017 = dic_i[2017]
            # if np.nanmean(vals_2018[:7]) < 0.1:
            all_vals_2018.append(vals_2018)
            # all_vals_2017.append(vals_2017)
        monthly_val_2018 = np.nanmean(all_vals_2018,axis=0)
        monthly_val_2017 = np.nanmean(all_vals_2017,axis=0)
        # plt.plot(monthly_val_2017,label='2017')
        plt.plot(monthly_val_2018,label=f'{pick_year}')
        plt.legend()
        plt.show()


class Time_series:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = T.mk_class_dir('Time_series',results_root_main_flow)

        pass

    def run(self):
        self.dataframe_all_year()
        # self.dataframe_2000_2016()
        # self.plot_ly_data_timeseries()
        # self.plot_dict()
        # self.mean_annual_spatial()
        pass

    def mean_annual_spatial(self):
        # fdir = '/Volumes/NVME2T/greening_project_redo/data/LAI4g_101/per_pix'
        # title = 'LAI4g'
        fdir = '/Volumes/NVME2T/greening_project_redo/data/BU_MCD_LAI_CMG/resample_ly_per_pix'
        title = 'BU_MCD_LAI_CMG'

        dic = T.load_npy_dir(fdir)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(dic)
        arr[arr<-99999] = np.nan
        plt.imshow(arr,vmin=0,vmax=4)
        plt.title(title)
        plt.show()
        # fdir = '/Volumes/NVME2T/greening_project_redo/data/BU_MCD_LAI_CMG/resample_ly'
        # outdir = '/Volumes/NVME2T/greening_project_redo/data/BU_MCD_LAI_CMG/resample_ly_per_pix'
        # T.mkdir(outdir)
        # Pre_Process().data_transform(fdir,outdir)
        # arr_sum = 0
        # for tif in T.listdir(fdir):
        #     arr = ToRaster().raster2array(join(fdir,tif))[0]



    def plot_all_season(self):

        pass

    def plot_dict(self):
        fdir = '/Volumes/NVME2T/greening_project_redo/data/test_data/0416'

        all_dict = {}
        key_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            dic = T.load_npy(fpath)
            all_dict[f[:-4]] = dic
            key_list.append(f[:-4])
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe().add_Humid_nonhumid(df)
        df = Dataframe().add_NDVI_mask(df)
        df = Dataframe().add_lon_lat_to_df(df)
        df = df[df['lat']>30]
        df = df[df['ndvi_mask_mark']==1]
        df = df[df['HI_reclass']=='Non Humid']
        df = df.dropna(subset=key_list)
        for key in key_list:
            if 'MODIS' in key:
                start_year = 2000
            else:
                start_year = 1982
            vals = df[key].tolist()
            vals = np.array(vals)
            vals_new = []
            for vals_i in vals:
                vals_len = len(vals_i)
                if vals_len < 12:
                    continue
                vals_new.append(vals_i)
            vals_new = np.array(vals_new)
            vals_mean = np.nanmean(vals_new,axis=0)
            year_list = list(range(start_year,start_year+len(vals_mean)))
            plt.plot(year_list,vals_mean,label=key)
        plt.legend()
        plt.show()

    def dataframe_all_year(self):
        # dff = '/Volumes/NVME2T/greening_project_redo/data/vege_dataframe/Data_frame_1982-2020.df'
        # dff = '/Volumes/NVME2T/greening_project_redo/data/vege_dataframe/Data_frame_1982-2020(1).df'
        dff = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Time_series/Data_frame_1982-2020.df'
        ndvi_mask_tif = '/Volumes/NVME2T/greening_project_redo/conf/NDVI_mask.tif'
        ndvi_mask_dic = DIC_and_TIF().spatial_tif_to_dic(ndvi_mask_tif)
        outdir = join(self.this_class_png,'all_year')
        T.mkdir(outdir)
        # T.open_path_and_file(outdir)
        df = T.load_df(dff)
        df = df[df['row']<120]
        df = T.add_spatial_dic_to_df(df,ndvi_mask_dic,'ndvi_mask')
        df = df[df['ndvi_mask']==1]
        columns_list = df.columns
        for col in columns_list:
            if col == 'pix':
                continue
            print(col)
        # exit()Data_frame_1982-2020(1).df
        vege_product_list = ['LAI3g','LAI4g','MODIS_LAI','VOD']
        vege_product_color_dict = {'LAI3g': 'red', 'LAI4g': 'blue', 'MODIS_LAI': 'green', 'VOD': 'black'}
        season_list = ['early','peak','late']
        # mode_list = ['relative_change','raw']
        mode_list = ['relative_change']
        # mode_list = ['raw']
        HI_class_list = T.get_df_unique_val_list(df,'HI_class')
        for humid in ['Humid','Non Humid']:
            if humid == 'Humid':
                df_HI = df[df['HI_class']=='Humid']
            else:
                df_HI = df[df['HI_class']!='Humid']
            for mode in mode_list:
                for season in season_list:
                    plt.figure()
                    plt.title(f'{season}-{humid}')
                    for vege_product in vege_product_list:
                        col_name_i = f'{vege_product}_{season}_{mode}'
                        for col_name in columns_list:
                            if col_name_i in col_name:
                                if '2000-2016' in col_name:
                                    continue
                                print(col_name_i)
                                df_vals = df_HI[col_name]
                                df_years = df_HI['year']
                                df_pix = df_HI['pix']
                                df_i = pd.DataFrame({'pix':df_pix,'year':df_years,'vals':df_vals})
                                years_list = T.get_df_unique_val_list(df_i,'year')
                                vals_list = []
                                std_list = []
                                for year in tqdm(years_list):
                                    df_year_i = df_i[df_i['year'] == year]
                                    if len(df_year_i) == 0:
                                        vals_list.append(np.nan)
                                    else:
                                        vals_year = df_year_i['vals']
                                        pix = df_year_i['pix']
                                        vals_year_mean = np.nanmean(vals_year)
                                        vals_year_std = np.nanstd(vals_year)
                                        vals_list.append(vals_year_mean)
                                        std_list.append(vals_year_std)
                                # plt.plot(years_list,vals_list,label=f'{col_name}')
                                plt.plot(years_list,vals_list,label=f'{col_name}',color=vege_product_color_dict[vege_product])
                                a, b, r, p = T.nan_line_fit(years_list,vals_list)
                                # plt.plot(years_list,a*np.array(years_list)+b,'--',color='black')
                                plt.plot(years_list,a*np.array(years_list)+b,'--',color=vege_product_color_dict[vege_product])
                                # Plot().plot_line_with_error_bar(years_list,vals_list,std_list,label=f'{col_name}')
                    plt.grid(1)
                    plt.legend()
                    plt.savefig(join(outdir,f'{season}_{humid}_{mode}.pdf'))
                    plt.close()
                    # plt.show()
        # plt.show()
    def dataframe_2000_2016(self):
        outdir = join(self.this_class_png,'2000_2016')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        dff = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/arr/Time_series/Data_frame_1982-2020.df'
        df = T.load_df(dff)
        df = df[df['row']<120]
        columns_list = df.columns
        for col in columns_list:
            if col == 'pix':
                continue
            print(col)
        # exit()
        vege_product_list = ['LAI3g','LAI4g','MODIS_LAI','VOD']
        vege_product_color_dict = {'LAI3g':'red','LAI4g':'blue','MODIS_LAI':'green','VOD':'black'}
        season_list = ['early','peak','late']
        # mode_list = ['relative_change','raw']
        mode_list = ['relative_change']
        # mode_list = ['raw']

        for humid in ['Humid','Non-Humid']:
            if humid == 'Humid':
                df_HI = df[df['HI_class']=='Humid']
            else:
                df_HI = df[df['HI_class']!='Humid']
            for season in season_list:
                plt.figure()
                plt.title(f'{season}-{humid}')
                for vege_product in vege_product_list:
                    for mode in mode_list:
                        col_name_i = f'{vege_product}_{season}_{mode}'
                        for col_name in columns_list:
                            if col_name_i in col_name:
                                if not '2000-2016' in col_name:
                                    continue
                                print(col_name_i)
                                df_vals = df_HI[col_name]
                                df_years = df_HI['year']
                                df_pix = df_HI['pix']
                                df_i = pd.DataFrame({'pix':df_pix,'year':df_years,'vals':df_vals})
                                years_list = T.get_df_unique_val_list(df_i,'year')
                                vals_list = []
                                for year in tqdm(years_list):
                                    df_year_i = df_i[df_i['year'] == year]
                                    if len(df_year_i) == 0:
                                        vals_list.append(np.nan)
                                    else:
                                        vals_year = df_year_i['vals']
                                        pix = df_year_i['pix']
                                        vals_year_mean = np.nanmean(vals_year)
                                        vals_list.append(vals_year_mean)
                                # plt.plot(years_list,vals_list,label=f'{col_name}')
                                plt.plot(years_list,vals_list,label=f'{vege_product}',color=vege_product_color_dict[vege_product])
                                a, b, r, p = T.nan_line_fit(years_list, vals_list)
                                years_list = np.arange(2000,2017)
                                years_list = list(years_list)
                                plt.plot(years_list, a * np.array(years_list) + b, '--', color=vege_product_color_dict[vege_product])
                plt.legend()
                plt.savefig(join(outdir,f'{season}_{humid}.pdf'))
                plt.close()
        # plt.show()

    def cal_relative_change(self,vals):
        base = np.nanmean(vals)
        # base = np.nanmean(base)
        relative_change_list = []
        for val in vals:
            change_rate = (val - base) / base
            relative_change_list.append(change_rate)
        relative_change_list = np.array(relative_change_list)
        return relative_change_list

    def ly_data(self,var_name,period,ishumid):
        start_year = vars_info_dic[var_name]['start_year']
        fdir = join(Pick_Early_Peak_Late_value().this_class_arr,'Pick_variables')
        fname = f'{var_name}.df'
        dff = join(fdir,fname)
        df = T.load_df(dff)
        df = Dataframe().add_NDVI_mask(df)
        df = Dataframe().add_lon_lat_to_df(df)
        df = Dataframe().add_Humid_nonhumid(df)
        df = df[df['lat']>30]
        df = df[df['HI_reclass']==ishumid]
        spatial_dict = T.df_to_spatial_dic(df,period)
        # print(spatial_dict)
        # exit()
        valid_arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dict)
        all_vals = []
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            if type(vals) == float:
                continue
            if T.is_all_nan(vals):
                continue
            # vals = self.cal_relative_change(vals)

            all_vals.append(vals)
        all_vals = np.array(all_vals)
        all_vals_mean = np.nanmean(all_vals,axis=0)
        x_list = np.arange(len(all_vals_mean))+start_year
        plt.plot(x_list,all_vals_mean,label=f'{var_name}')
        plt.title(f'{period}-{ishumid}')
        return valid_arr

    def plot_ly_data_timeseries(self):
        var_list = ['LAI4g_101','MODIS_LAI_CMG']
        period_list = ['early','peak','late']
        vmin_max_dict = {
            'early':{'vmin':0,'vmax':2},
            'peak':{'vmin':0,'vmax':3.5},
            'late':{'vmin':0,'vmax':2},
        }
        ishumid = 'Non Humid'
        # ishumid = 'Humid'
        for period in period_list:
            # if not period == 'late':
            #     continue
            plt.figure()
            arr_list = []
            for var_name in var_list:
                valid_arr = self.ly_data(var_name,period,ishumid)
                arr_list.append(valid_arr)
            plt.legend()
            plt.grid()
            for i in range(len(arr_list)):
                plt.figure()
                plt.imshow(arr_list[i],vmin=vmin_max_dict[period]['vmin'],vmax=vmin_max_dict[period]['vmax'],cmap='jet_r')
                plt.colorbar()
                DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif)
                plt.title(f'{var_list[i]}_{period}_{ishumid}')
        plt.show()
        pass


class Plot_Trend_Spatial:

    def __init__(self):
        self.this_class_arr,self.this_class_tif,self.this_class_png = \
            T.mk_class_dir('Plot_Trend_Spatial',results_root_main_flow)

    def run(self):
        self.plot_spatial()
        pass

    def trend_spatial_pvalue_point_shp(self):
        fdir = join(self.this_class_tif,'trend_spatial_tif')
        outdir = join(self.this_class_tif,'trend_spatial_pvalue_point_shp')
        T.mkdir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('_p.tif'):
                continue
            intif = join(fdir,f)
            outtif = join(outdir,f)
            ToRaster().resample_reproj(intif,outtif,res=2)
        for f in T.listdir(outdir):
            out_shp_f = join(outdir,f+'.shp')
            if not f.endswith('_p.tif'):
                continue
            arr = ToRaster().raster2array(join(outdir,f))[0]
            arr = T.mask_999999_arr(arr,warning=False)
            arr[arr>0.1] = np.nan
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            point_list = []
            for pix in dic:
                val = dic[pix]
                if np.isnan(val):
                    continue
                lon,lat = DIC_and_TIF(tif_template=join(outdir,f)).pix_to_lon_lat(pix)
                # print(lon,lat)
                list_i = [lon,lat,1]
                point_list.append(list_i)
            T.point_to_shp(point_list,out_shp_f)


    def tif_statistic(self,trend_tif,p_tif,vmin,vmax):
        trend_arr = ToRaster().raster2array(trend_tif)[0]
        trend_arr = trend_arr[:120]
        trend_arr = T.mask_999999_arr(trend_arr,warning=False)
        trend_arr_flatten = trend_arr.flatten()
        p_arr = ToRaster().raster2array(p_tif)[0]
        p_arr = p_arr[:120]
        p_arr = T.mask_999999_arr(p_arr,warning=False)
        # p_arr = T.remove_np_nan(p_arr)

        trend_dict = DIC_and_TIF().spatial_arr_to_dic(trend_arr)
        p_dict = DIC_and_TIF().spatial_arr_to_dic(p_arr)
        dict_all = {'trend':trend_dict,'p':p_dict}
        df = T.spatial_dics_to_df(dict_all)
        df = Dataframe().add_Humid_nonhumid(df)
        # T.print_head_n(df,n=10)
        ## re index datafram
        df = df.reset_index()
        humid_nonhumid_list = ['Humid','Non Humid']
        humid_nonhumid_result_dict = {}
        for humid in humid_nonhumid_list:
            df_humid = df[df['HI_reclass']==humid]
            trend_values = df_humid['trend'].tolist()
            trend_hist = np.histogram(trend_values, bins=100, density=True, range=(vmin, vmax))
            for i,row in tqdm(df_humid.iterrows(),total=len(df_humid)):
                p = row['p']
                trend = row['trend']
                if p < 0.05:
                    if trend>0:
                        df_humid.loc[i,'class'] = 'significant_positive\np<0.05'
                    else:
                        df_humid.loc[i,'class'] = 'significant_negative\np<0.05'
                elif 0.05 <= p < 0.1:
                    if trend>0:
                        df_humid.loc[i,'class'] = 'positive\n0.05<p<0.1'
                    else:
                        df_humid.loc[i,'class'] = 'negative\n0.05<p<0.1'
                else:
                    df_humid.loc[i,'class'] = 'not_significant\np>0.1'
            # sig_ratio_list = ['significant_positive\np<0.05','positive\n0.05<p<0.1','not_significant\np>0.1','negative\n0.05<p<0.1','significant_negative\np<0.05']

            df_sig_pos = df_humid[df_humid['class']=='significant_positive\np<0.05']
            df_sig_neg = df_humid[df_humid['class']=='significant_negative\np<0.05']
            df_not_sig = df_humid[df_humid['class']=='not_significant\np>0.1']
            df_pos = df_humid[df_humid['class']=='positive\n0.05<p<0.1']
            df_neg = df_humid[df_humid['class']=='negative\n0.05<p<0.1']
            ratio_sig_pos = len(df_sig_pos)/len(df_humid)
            ratio_sig_neg = len(df_sig_neg)/len(df_humid)
            ratio_not_sig = len(df_not_sig)/len(df_humid)
            ratio_pos = len(df_pos)/len(df_humid)
            ratio_neg = len(df_neg)/len(df_humid)
            sig_ratio_dict = {
                'significant_positive\np<0.05':ratio_sig_pos,
                'significant_negative\np<0.05':ratio_sig_neg,
                'not_significant\np>0.1':ratio_not_sig,
                'positive\n0.05<p<0.1':ratio_pos,
                'negative\n0.05<p<0.1':ratio_neg
            }
            humid_nonhumid_result_dict[humid] = (trend_hist,sig_ratio_dict,trend_arr)
        return humid_nonhumid_result_dict

    def plot_spatial(self):
        # fpath = 'MODIS_LAI_peak_relative_change_trend.tif'
        fdir = join(self.this_class_arr,'spatial_tif/2000-2018')
        outdir = join(self.this_class_png,'2000-2018_plot')
        T.mkdir(outdir)
        product_list = ['LAI4g','LAI3g','MODIS_LAI','VOD']
        period_list = ['early','peak','late']
        mode_list = ['relative_change']
        tail_list = ['_p_value.tif','_trend.tif']
        color_list = ['brown','white','green']
        cmap = T.cmap_blend(color_list)
        for product in product_list:
            for period in period_list:
                for mode in mode_list:
                    print(product,period,mode)
                    # 'MODIS_LAI_peak_relative_change_trend.tif'
                    f_pvalue = f'{product}_{period}_{mode}{tail_list[0]}'
                    f_trend = f'{product}_{period}_{mode}{tail_list[1]}'
                    intif_pvalue = join(fdir,f_pvalue)
                    intif_trend = join(fdir,f_trend)
                    vmin = -5
                    vmax = 5
                    # trend_hist,sig_ratio_dict,trend_arr = self.tif_statistic(intif_trend,intif_pvalue,vmin,vmax)
                    result_dict = self.tif_statistic(intif_trend,intif_pvalue,vmin,vmax)

                    sig_ratio_list = ['significant_positive\np<0.05','positive\n0.05<p<0.1','not_significant\np>0.1','negative\n0.05<p<0.1','significant_negative\np<0.05']
                    fig,axs = plt.subplots(2,2,figsize=(10,5))
                    for humid in result_dict:
                        trend_hist, sig_ratio_dict, trend_arr = result_dict[humid]
                        axs[0][0].plot(trend_hist[1][:-1],trend_hist[0],label=humid)
                        axs[0][0].set_title(f'Histogram')
                        axs[0][0].set_xlabel('trend')
                        axs[0][0].set_ylabel('density')
                        if humid == 'Humid':
                            axs[0][1].bar(sig_ratio_list,[sig_ratio_dict[i] for i in sig_ratio_list])
                            axs[0][1].set_ylim(0,0.8)
                            axs[0][1].set_title(f'{humid}-Significant ratio')
                            axs[0][1].set_ylabel('ratio')
                            axs[0][1].xaxis.set_ticks([])
                            # axs[0][1].set_xticklabels(sig_ratio_list,rotation=45)
                        else:
                            axs[1][1].bar(sig_ratio_list, [sig_ratio_dict[i] for i in sig_ratio_list])
                            axs[1][1].set_ylim(0, 0.8)
                            axs[1][1].set_title(f'{humid}-Significant ratio')
                            axs[1][1].set_ylabel('ratio')
                            axs[1][1].xaxis.set_ticks(sig_ratio_list)
                            axs[1][1].set_xticklabels(sig_ratio_list, rotation=35)
                        if humid == 'Humid':
                            pcm = axs[1][0].imshow(trend_arr,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
                            DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,ax=axs[1][0])
                            # axs[1][0].set_title(f'spatial trend')
                            axs[1][0].axis('off')
                            fig.colorbar(pcm, ax=axs[1][0], location='bottom', shrink=.3, label='spatial trend')
                        fig.suptitle(f'{product}_{period}_{mode}')
                        ## set subplot legends
                        axs[0][0].legend(loc='upper left')
                    outf = join(outdir,f'{product}_{period}_{mode}.pdf')
                    plt.tight_layout()
                    plt.savefig(outf)
                    plt.close()
                    # plt.show()

        # for f in T.listdir(fdir):
        #     if not f.endswith('.tif'):
        #         continue
        #     if not 'trend' in f:
        #         continue
        #     intif = join(fdir,f)
        #     self.tif_statistic(intif)


        pass


class Sankey_plot:

    def __init__(self):
        Y_name = 'MODIS_LAI'
        # Y_name = 'LAI4g'
        # self.fdir = f'/Volumes/SSD_sumsang/project_greening/Result/new_result/multiregression/daily/{Y_name}/'
        self.fdir=results_root+f'partial_correlation/partial_correlation_relative_change_trend/daily/{Y_name}'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot',results_root_main_flow)
        outdir = join(self.this_class_arr,f'{Y_name}')
        T.mkdir(outdir)
        self.dff = join(outdir,'dataframe.df')
        pass

    def run(self):
        # self.plot_p_value_spatial()
        # df,var_list = self.join_dataframe(self.fdir)
        # df = self.build_sankey_plot(df,var_list)
        #
        df = self.__gen_df_init()
        # df = Dataframe().add_Humid_nonhumid(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        self.plot_Sankey(df,True)
        # self.plot_Sankey(df,False)
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def join_dataframe(self,fdir):

        df_list = []
        var_list = []
        for f in T.listdir(fdir):
            print(f)
            # period = f.split('.')[0].split('_')[-2]
            period = f.split('.')[0].split('_')[-3]
            dic = T.load_npy(join(fdir,f))
            df_i = T.dic_to_df(dic,key_col_str='pix')
            old_col_list = []
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                old_col_list.append(col)
                var_list.append(col)
            if '_p_value_' in f:
                new_col_list = [f'{col}_p_value' for col in old_col_list]
            else:
                new_col_list = [f'{col}' for col in old_col_list]
            for i in range(len(old_col_list)):
                new_name = new_col_list[i]
                old_name = old_col_list[i]
                df_i = T.rename_dataframe_columns(df_i,old_name,new_name)
            df_list.append(df_i)
        df = pd.DataFrame()
        df = Tools().join_df_list(df,df_list,'pix')
        ## re-index dataframe
        df = df.reset_index(drop=True)
        return df,var_list

    def build_sankey_plot(self,df,var_list,p_threshold=0.1):

        # period_list = ['early','peak','late']
        for var_ in var_list:
            # for period in period_list:
            var_name_corr = f'{var_}'
            var_name_p_value = f'{var_}_p_value'
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{var_}'):
                corr = row[var_name_corr]


        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


    def __get_var_list(self,fdir):
        var_list = []
        for f in T.listdir(fdir):
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir, f))
            df_i = T.dic_to_df(dic, key_col_str='pix')
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                var_list.append(col)
            break
        return var_list

    def __add_alpha_to_color(self,hexcolor,alpha=0.4):
        rgb = T.hex_color_to_rgb(hexcolor)
        rgb = list(rgb)
        rgb[-1] = alpha
        rgb = tuple(rgb)
        rgb_str = 'rgba'+str(rgb)
        return rgb_str

    def plot_Sankey(self,df,ishumid):
        if ishumid:
            df = df[df['HI_reclass'] == 'Humid']
            outdir = join(self.this_class_png, 'Sankey_plot/Humid')
        else:
            df = df[df['HI_reclass'] == 'Non Humid']
            outdir = join(self.this_class_png, 'Sankey_plot/Non_Humid')
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        var_list = self.__get_var_list(self.fdir)

        period_list = ['early','peak','late']
        status_list = ['Positive','Non_significant','Negative']
        early_status_list = [f'early-{status}' for status in status_list]
        peak_status_list = [f'peak-{status}' for status in status_list]
        late_status_list = [f'late-{status}' for status in status_list]
        node_list = early_status_list + peak_status_list + late_status_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        color_dict = {'Non_significant': self.__add_alpha_to_color('#CCCCCC'),
                      'Positive': self.__add_alpha_to_color('#00ACAE'),
                      'Negative': self.__add_alpha_to_color('#FF8E42'),
                      }
        node_color_list = [color_dict[condition] for condition in status_list]
        node_color_list = node_color_list + node_color_list + node_color_list


        for var_ in var_list:
            early_class_var = f'{var_}_early_class'
            peak_class_var = f'{var_}_peak_class'
            late_class_var = f'{var_}_late_class'

            source = []
            target = []
            value = []
            # color_list = []
            # node_list_anomaly_value_mean = []
            # anomaly_value_list = []
            node_list_with_ratio = []
            for early_status in early_status_list:
                df_early = df[df[early_class_var] == early_status]
                ratio = len(df_early)/len(df)
                node_list_with_ratio.append(ratio)
            for peak_status in peak_status_list:
                df_peak = df[df[peak_class_var] == peak_status]
                ratio = len(df_peak)/len(df)
                node_list_with_ratio.append(ratio)
            for late_status in late_status_list:
                df_late = df[df[late_class_var] == late_status]
                ratio = len(df_late)/len(df)
                node_list_with_ratio.append(ratio)
            node_list_with_ratio = [round(i,3) for i in node_list_with_ratio]

            for early_status in early_status_list:
                df_early = df[df[early_class_var] == early_status]
                early_count = len(df_early)
                for peak_status in peak_status_list:
                    df_peak = df_early[df_early[peak_class_var] == peak_status]
                    peak_count = len(df_peak)
                    source.append(position_dict[early_status])
                    target.append(position_dict[peak_status])
                    value.append(peak_count)
                    for late_status in late_status_list:
                        df_late = df_peak[df_peak[late_class_var] == late_status]
                        late_count = len(df_late)
                        source.append(position_dict[peak_status])
                        target.append(position_dict[late_status])
                        value.append(late_count)
            link = dict(source=source, target=target, value=value,)
            node = dict(label=node_list_with_ratio, pad=100,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        # x=node_x,
                        # y=node_y,
                        color=node_color_list
                    )
            data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
            fig = go.Figure(data)
            fig.update_layout(title_text=f'{var_}')
            # fig.write_html(join(outdir, f'{var_}.html'))
            fig.write_image(join(outdir, f'{var_}.png'))
            # fig.show()
        pass


    def plot_p_value_spatial(self):
        fdir = self.fdir
        var_ = 'Temp'
        # var_ = 'CCI_SM'
        period = 'early'
        f = join(fdir, '2000-2018_partial_correlation_p_value_early_LAI3g.npy')
        dict_ = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(dict_):
            dict_i = dict_[pix]
            if not var_ in dict_i:
                continue
            value = dict_i[var_]
            spatial_dict[pix] = value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr = arr[:180]
        arr[arr>0.1] = 1
        plt.imshow(arr, cmap='jet',aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var_}_{period}')
        plt.show()

        pass


class Sankey_plot_max_contribution:
    # todo: 6-11
    # done

    def __init__(self):
        self.Y_name = 'LAI3g'
        # self.Y_name = 'LAI4g'
        # self.Y_name = 'MODIS-LAI'
        self.var_list = ['CCI_SM', 'PAR', 'Temp', 'VPD','CO2']
        self.period_list = ['early', 'peak', 'late']
        self.fdir=f'/Volumes/SSD_sumsang/project_greening/Result/new_result/multiregression/daily/{self.Y_name}'
        # self.fdir = f'/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_zscore/{self.Y_name}'
        # self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot_max_contribution',results_root_main_flow)
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot_max_contribution_detrend',
                                                                                       results_root_main_flow)
        outdir = join(self.this_class_arr,f'{self.Y_name}')
        T.mkdir(outdir)
        self.dff = join(outdir,'dataframe.df')
        # T.open_path_and_file(outdir)

        pass

    def run(self):
        # self.plot_p_value_spatial()
        df,var_list = self.join_dataframe(self.fdir)
        # df = self.build_sankey_plot(df,var_list)
        # # #
        df = self.__gen_df_init()
        # df = Dataframe().add_Humid_nonhumid(df)
        # # df = self.add_is_significant(df)
        #
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        # self.plot_Sankey(df,True)

        self.plot_Sankey(df,False)
        # self.plot_max_corr_spatial(df)
        # self.max_contribution_bar()
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def plot_max_corr_spatial(self,df):
        outdir = join(self.this_class_tif,'plot_max_corr_spatial')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        period_list = self.period_list
        var_list = self.var_list
        var_color_dict = {'CCI_SM':'#ff7f0e','CO2':'#1f77b4','PAR':'#2ca02c','Temp':'#9467bd','VPD':'#d62728'}
        var_value_dict = {'CCI_SM':0,'CO2':1,'PAR':2,'Temp':3,'VPD':4,'None':np.nan,'nan':np.nan}
        color_list = []
        for var_ in var_list:
            color_list.append(var_color_dict[var_])
        cmap = T.cmap_blend(color_list)
        for period in period_list:
            col_name = f'{period}_max_var'
            spatial_dict = T.df_to_spatial_dic(df,col_name)
            spatial_dict_value = {}
            for pix in spatial_dict:
                var_ = spatial_dict[pix]
                var_ = str(var_)
                var_ = var_.replace(period+'_','')
                # print(var_)
                value = var_value_dict[var_]
                spatial_dict_value[pix] = value
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_value)
            arr = arr[:180]
            # plt.figure()
            # DIC_and_TIF().plot_back_ground_arr(global_land_tif,aspect='auto')
            # plt.imshow(arr, cmap=cmap,aspect='auto')
            # plt.colorbar()
            # plt.title(f'{self.Y_name}_{period}')
            # T.plot_colors_palette(cmap)
            outf = join(outdir,f'{self.Y_name}_{period}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.show()

    def join_dataframe(self,fdir):

        df_list = []
        var_list = []
        for f in T.listdir(fdir):
            # print(f)
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir,f))
            df_i = T.dic_to_df(dic,key_col_str='pix')
            old_col_list = []
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                old_col_list.append(col)
                var_list.append(col)
            if '_p_value_' in f:
                new_col_list = [f'{period}_{col}_p_value' for col in old_col_list]
            else:
                new_col_list = [f'{period}_{col}' for col in old_col_list]
            for i in range(len(old_col_list)):
                new_name = new_col_list[i]
                old_name = old_col_list[i]
                df_i = T.rename_dataframe_columns(df_i,old_name,new_name)
            df_list.append(df_i)
        df = pd.DataFrame()
        df = Tools().join_df_list(df,df_list,'pix')
        ## re-index dataframe
        df = df.reset_index(drop=True)
        return df,var_list


    def build_sankey_plot(self,df,var_list,p_threshold=0.1):

        period_list = ['early','peak','late']
        for period in period_list:
            var_name_list = []
            for var_ in var_list:
                var_name_corr = f'{period}_{var_}'
                var_name_list.append(var_name_corr)
            for i,row in tqdm(df.iterrows(),total=len(df)):
                value_dict = {}
                for var_ in var_name_list:
                    value = row[var_]
                    value = abs(value)
                    value_dict[var_] = value
                max_key = T.get_max_key_from_dict(value_dict)
                df.loc[i,f'{period}_max_var'] = max_key
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


    def __get_var_list(self,fdir):
        var_list = []
        for f in T.listdir(fdir):
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir, f))
            df_i = T.dic_to_df(dic, key_col_str='pix')
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                var_list.append(col)
            break
        return var_list

    def __add_alpha_to_color(self,hexcolor,alpha=0.8):
        rgb = T.hex_color_to_rgb(hexcolor)
        rgb = list(rgb)
        rgb[-1] = alpha
        rgb = tuple(rgb)
        rgb_str = 'rgba'+str(rgb)
        return rgb_str

    def add_is_significant(self,df):
        # print(df)
        # exit()
        p_threshold = 0.1
        period_list = ['early','peak','late']
        for period in period_list:
            print(period)
            max_var_col = f'{period}_max_var'
            for i,row in tqdm(df.iterrows(),total=len(df)):
                max_var = row[max_var_col]
                max_var_p_var = f'{max_var}_p_value'
                try:
                    p_value = row[max_var_p_var]
                except:
                    p_value = 1
                if p_value > p_threshold:
                    df.loc[i,f'{period}_significant'] = False
                else:
                    df.loc[i,f'{period}_significant'] = True

        return df


    def plot_Sankey(self,df,ishumid):
        if ishumid:
            df = df[df['HI_reclass'] == 'Humid']
            outdir = join(self.this_class_png, f'{self.Y_name}/Humid')
            title = 'Humid'
        else:
            df = df[df['HI_reclass'] == 'Dryland']
            outdir = join(self.this_class_png, f'{self.Y_name}/Dryland')
            title = 'Dryland'
        T.mkdir(outdir,force=True)
        # T.open_path_and_file(outdir)
        var_list = self.__get_var_list(self.fdir)

        period_list = ['early','peak','late']
        # status_list = ['Positive','Non_significant','Negative']
        # early_status_list = [f'early-{status}' for status in status_list]
        # peak_status_list = [f'peak-{status}' for status in status_list]
        # late_status_list = [f'late-{status}' for status in status_list]
        early_max_var_list = [f'early_{var}' for var in var_list]
        peak_max_var_list = [f'peak_{var}' for var in var_list]
        late_max_var_list = [f'late_{var}' for var in var_list]

        node_list = early_max_var_list+peak_max_var_list+late_max_var_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        color_dict = {'CO2': self.__add_alpha_to_color('#1f77b4'),
                      'CCI_SM': self.__add_alpha_to_color('#00E7FF'),
                      'PAR': self.__add_alpha_to_color('#FFFF00'),
                      'Temp': self.__add_alpha_to_color('#FF0000'),
                      'VPD': self.__add_alpha_to_color('#B531AF'),
                      }

        node_color_list = [color_dict[var_] for var_ in var_list]
        node_color_list = node_color_list * 3
        # print(node_color_list)
        # exit()
        early_max_var_col = 'early_max_var'
        peak_max_var_col = 'peak_max_var'
        late_max_var_col = 'late_max_var'

        source = []
        target = []
        value = []
        # color_list = []
        # node_list_anomaly_value_mean = []
        # anomaly_value_list = []
        node_list_with_ratio = []
        node_name_list = []
        for early_status in early_max_var_list:
            # print(early_status)
            # exit()
            df_early = df[df[early_max_var_col] == early_status]
            vals = df_early[early_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_early)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{early_status} {vals_mean:.2f}')
        for peak_status in peak_max_var_list:
            df_peak = df[df[peak_max_var_col] == peak_status]
            vals = df_peak[peak_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_peak)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{peak_status} {vals_mean:.2f}')
        for late_status in late_max_var_list:
            df_late = df[df[late_max_var_col] == late_status]
            vals = df_late[late_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_late)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{late_status} {vals_mean:.2f}')
        node_list_with_ratio = [round(i,3) for i in node_list_with_ratio]

        for early_status in early_max_var_list:
            df_early = df[df[early_max_var_col] == early_status]
            df_early = df_early[df_early['early_significant'] == True]
            early_count = len(df_early)
            for peak_status in peak_max_var_list:
                df_peak = df_early[df_early[peak_max_var_col] == peak_status]
                df_peak = df_peak[df_peak['peak_significant'] == True]

                peak_count = len(df_peak)
                source.append(position_dict[early_status])
                target.append(position_dict[peak_status])
                value.append(peak_count)
                for late_status in late_max_var_list:
                    df_late = df_peak[df_peak[late_max_var_col] == late_status]
                    df_late = df_late[df_late['late_significant'] == True]

                    late_count = len(df_late)
                    source.append(position_dict[peak_status])
                    target.append(position_dict[late_status])
                    value.append(late_count)
        link = dict(source=source, target=target, value=value,)
        # node = dict(label=node_list_with_ratio, pad=100,
        node = dict(label=node_name_list, pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    # x=node_x,
                    # y=node_y,
                    color=node_color_list
                )
        data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
        fig = go.Figure(data)
        fig.update_layout(title_text=f'{title}')
        # fig.write_html(join(outdir, f'{title}.html'))
        # fig.write_image(join(outdir, f'{title}.png'))
        fig.show()


    def plot_p_value_spatial(self):
        fdir = self.fdir
        var_ = 'Temp'
        # var_ = 'CCI_SM'
        period = 'early'
        f = join(fdir, '2000-2018_partial_correlation_p_value_early_LAI3g.npy')
        dict_ = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(dict_):
            dict_i = dict_[pix]
            if not var_ in dict_i:
                continue
            value = dict_i[var_]
            spatial_dict[pix] = value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr = arr[:180]
        arr[arr>0.1] = 1
        plt.imshow(arr, cmap='jet',aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var_}_{period}')
        plt.show()

        pass

    def max_contribution_bar(self):
        fdir = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/tif/Sankey_plot_max_contribution/plot_max_corr_spatial'
        outdir = join(self.this_class_png, 'max_contribution_bar')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        var_value_dict = {'CCI_SM': 0, 'CO2': 1, 'PAR': 2, 'Temp': 3, 'VPD': 4}
        color_dict = {'CCI_SM': '#00E7FF', 'CO2': '#00FF00', 'PAR': '#FFFF00', 'Temp': '#FF0000', 'VPD': '#B531AF'}
        var_value_dict_reverse = T.reverse_dic(var_value_dict)
        all_dict = {}
        product_list = []
        period_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fname = f.split('.')[0]
            product,period = fname.split('_')
            product_list.append(product)
            period_list.append(period)
            arr = ToRaster().raster2array(join(fdir, f))[0]
            arr = T.mask_999999_arr(arr,warning=False)
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            key = f'{product}_{period}'
            all_dict[key] = dic
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe().add_Humid_nonhumid(df)
        humid_var = 'HI_reclass'
        humid_list = T.get_df_unique_val_list(df, humid_var)
        for humid in humid_list:
            df_humid = df[df[humid_var] == humid]
            fig,axs = plt.subplots(3,3,figsize=(10,10))
            product_list = list(set(product_list))
            period_list = ['early','peak','late']
            for m,product in enumerate(product_list):
                for n,period in enumerate(period_list):
                    col = f'{product}_{period}'
                    df_i = df_humid[col]
                    df_i = df_i.dropna()
                    unique_value = T.get_df_unique_val_list(df_humid,col)
                    x_list = []
                    y_list = []
                    color_list = []
                    for val in unique_value:
                        df_i_i = df_i[df_i == val]
                        ratio = len(df_i_i)/len(df_i) * 100
                        x = var_value_dict_reverse[int(val)][0]
                        x_list.append(x)
                        y_list.append(ratio)
                        color_list.append(color_dict[x])
                    # plt.figure()
                    # print(m,n)
                    axs[m][n].bar(x_list,y_list,color=color_list)
                    axs[m][n].set_title(f'{product}_{period}')
                    axs[m][n].set_ylim(0,47)
            plt.suptitle(f'{humid}')
            plt.tight_layout()
            outf = join(outdir, f'{humid}.pdf')
            plt.savefig(outf)
            plt.close()

        pass

class Sankey_plot_single_max_contribution:

    def __init__(self):
        # self.Y_name = 'LAI3g'
        self.Y_name = 'LAI4g'
        self.var_list = ['CCI_SM', 'CO2', 'PAR', 'Temp', 'VPD']
        self.period_list = ['early', 'peak', 'late']
        self.fdir = f'/Volumes/NVME2T/greening_project_redo/data/Sankey_plot_data/simple_corr/{self.Y_name}'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot_single_max_contribution',results_root_main_flow)
        outdir = join(self.this_class_arr,f'{self.Y_name}')
        T.mkdir(outdir)
        self.dff = join(outdir,'dataframe.df')
        T.open_path_and_file(outdir)
        pass

    def run(self):
        # self.get_max_corr_var()
        # #
        df = self.__gen_df_init()
        # df = Dataframe().add_Humid_nonhumid(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        # self.plot_Sankey(df,True)
        # self.plot_Sankey(df,False)
        self.plot_max_corr_spatial(df)
        pass


    def plot_max_corr_spatial(self,df):
        period_list = self.period_list
        var_list = self.var_list
        var_color_dict = {'CCI_SM':'#ff7f0e','CO2':'#1f77b4','PAR':'#2ca02c','Temp':'#9467bd','VPD':'#d62728'}
        var_value_dict = {'CCI_SM':0,'CO2':1,'PAR':2,'Temp':3,'VPD':4}
        color_list = []
        for var_ in var_list:
            color_list.append(var_color_dict[var_])
        cmap = T.cmap_blend(color_list)
        for period in period_list:
            col_name = f'{period}_max_var'
            spatial_dict = T.df_to_spatial_dic(df,col_name)
            spatial_dict_value = {}
            for pix in spatial_dict:
                var_ = spatial_dict[pix]
                var_ = str(var_)
                var_ = var_.replace(period+'_','')
                value = var_value_dict[var_]
                spatial_dict_value[pix] = value
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_value)
            arr = arr[:180]
            plt.figure()
            DIC_and_TIF().plot_back_ground_arr(global_land_tif,aspect='auto')
            plt.imshow(arr, cmap=cmap,aspect='auto')
            plt.colorbar()
            plt.title(f'{self.Y_name}_{period}')
            # T.plot_colors_palette(cmap)
        plt.show()


    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df


    def get_max_corr_var(self):
        fdir = self.fdir
        var_list = self.var_list
        period_list = self.period_list
        all_dict = {}
        col_list = []
        for period in period_list:
            for var in var_list:
                r_f = join(fdir, f'{var}_{period}_zscore_{self.Y_name}_r.npy')
                arr = np.load(r_f)
                arr[arr<-9999] = np.nan
                dic = DIC_and_TIF().spatial_arr_to_dic(arr)
                col_name = f'{period}_{var}'
                all_dict[col_name] = dic
                col_list.append(col_name)
        df = T.spatial_dics_to_df(all_dict)
        df = df.reset_index()
        df = df.dropna(subset=col_list,how='all')
        T.print_head_n(df,5)
        for period in period_list:
            col_name_list = []
            for var in var_list:
                col_name = f'{period}_{var}'
                col_name_list.append(col_name)
            for i,row in tqdm(df.iterrows(),total=len(df),desc=period):
                dict_val_i = {}
                for col in col_name_list:
                    val = row[col]
                    val = abs(val)
                    dict_val_i[col] = val
                max_var = T.get_max_key_from_dict(dict_val_i)
                df.loc[i, f'{period}_max_var'] = max_var
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)


    def join_dataframe(self,fdir):

        df_list = []
        var_list = []
        for f in T.listdir(fdir):
            # print(f)
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir,f))
            df_i = T.dic_to_df(dic,key_col_str='pix')
            old_col_list = []
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                old_col_list.append(col)
                var_list.append(col)
            if '_p_value_' in f:
                new_col_list = [f'{period}_{col}_p_value' for col in old_col_list]
            else:
                new_col_list = [f'{period}_{col}' for col in old_col_list]
            for i in range(len(old_col_list)):
                new_name = new_col_list[i]
                old_name = old_col_list[i]
                df_i = T.rename_dataframe_columns(df_i,old_name,new_name)
            df_list.append(df_i)
        df = pd.DataFrame()
        df = Tools().join_df_list(df,df_list,'pix')
        ## re-index dataframe
        df = df.reset_index(drop=True)
        return df,var_list

    def get_max_key_from_dict(self,input_dict):
        max_key = None
        max_value = -np.inf
        for key in input_dict:
            value = input_dict[key]
            if value > max_value:
                max_key = key
                max_value = value
        return max_key


    def build_sankey_plot(self,df,var_list,p_threshold=0.1):

        period_list = ['early','peak','late']
        for period in period_list:
            var_name_list = []
            for var_ in var_list:
                var_name_corr = f'{period}_{var_}'
                var_name_list.append(var_name_corr)
            for i,row in tqdm(df.iterrows(),total=len(df)):
                value_dict = {}
                for var_ in var_name_list:
                    value = row[var_]
                    value = abs(value)
                    value_dict[var_] = value
                max_key = self.get_max_key_from_dict(value_dict)
                df.loc[i,f'{period}_max_var'] = max_key
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)



    def __add_alpha_to_color(self,hexcolor,alpha=0.8):
        rgb = T.hex_color_to_rgb(hexcolor)
        rgb = list(rgb)
        rgb[-1] = alpha
        rgb = tuple(rgb)
        rgb_str = 'rgba'+str(rgb)
        return rgb_str

    def plot_Sankey(self,df,ishumid):
        if ishumid:
            df = df[df['HI_reclass'] == 'Humid']
            outdir = join(self.this_class_png, f'{self.Y_name}/Humid')
            title = 'Humid'
        else:
            df = df[df['HI_reclass'] == 'Non Humid']
            outdir = join(self.this_class_png, f'{self.Y_name}/Non_Humid')
            title = 'Non Humid'
        T.mkdir(outdir,force=True)
        T.open_path_and_file(outdir)
        var_list = self.var_list

        period_list = ['early','peak','late']
        # status_list = ['Positive','Non_significant','Negative']
        # early_status_list = [f'early-{status}' for status in status_list]
        # peak_status_list = [f'peak-{status}' for status in status_list]
        # late_status_list = [f'late-{status}' for status in status_list]
        early_max_var_list = [f'early_{var}' for var in var_list]
        peak_max_var_list = [f'peak_{var}' for var in var_list]
        late_max_var_list = [f'late_{var}' for var in var_list]

        node_list = early_max_var_list+peak_max_var_list+late_max_var_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        color_dict = {'CO2': self.__add_alpha_to_color('#00FF00'),
                      'CCI_SM': self.__add_alpha_to_color('#00E7FF'),
                      'PAR': self.__add_alpha_to_color('#FFFF00'),
                      'Temp': self.__add_alpha_to_color('#FF0000'),
                      'VPD': self.__add_alpha_to_color('#B531AF'),
                      }
        node_color_list = [color_dict[var_] for var_ in var_list]
        node_color_list = node_color_list * 3
        # print(node_color_list)
        # exit()
        early_max_var_col = 'early_max_var'
        peak_max_var_col = 'peak_max_var'
        late_max_var_col = 'late_max_var'

        source = []
        target = []
        value = []
        # color_list = []
        # node_list_anomaly_value_mean = []
        # anomaly_value_list = []
        node_list_with_ratio = []
        node_name_list = []
        for early_status in early_max_var_list:
            df_early = df[df[early_max_var_col] == early_status]
            ratio = len(df_early)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(early_status)
        for peak_status in peak_max_var_list:
            df_peak = df[df[peak_max_var_col] == peak_status]
            ratio = len(df_peak)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(peak_status)
        for late_status in late_max_var_list:
            df_late = df[df[late_max_var_col] == late_status]
            ratio = len(df_late)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(late_status)
        node_list_with_ratio = [round(i,3) for i in node_list_with_ratio]

        for early_status in early_max_var_list:
            df_early = df[df[early_max_var_col] == early_status]
            early_count = len(df_early)
            for peak_status in peak_max_var_list:
                df_peak = df_early[df_early[peak_max_var_col] == peak_status]
                peak_count = len(df_peak)
                source.append(position_dict[early_status])
                target.append(position_dict[peak_status])
                value.append(peak_count)
                for late_status in late_max_var_list:
                    df_late = df_peak[df_peak[late_max_var_col] == late_status]
                    late_count = len(df_late)
                    source.append(position_dict[peak_status])
                    target.append(position_dict[late_status])
                    value.append(late_count)
        link = dict(source=source, target=target, value=value,)
        # node = dict(label=node_list_with_ratio, pad=100,
        node = dict(label=node_name_list, pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    # x=node_x,
                    # y=node_y,
                    color=node_color_list
                )
        data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
        fig = go.Figure(data)
        fig.update_layout(title_text=f'{title}')
        fig.write_html(join(outdir, f'{title}.html'))
        # fig.write_image(join(outdir, f'{title}.png'))
        # fig.show()
        # exit()

    def plot_p_value_spatial(self):
        fdir = self.fdir
        var_ = 'Temp'
        # var_ = 'CCI_SM'
        period = 'early'
        f = join(fdir, '2000-2018_partial_correlation_p_value_early_LAI3g.npy')
        dict_ = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(dict_):
            dict_i = dict_[pix]
            if not var_ in dict_i:
                continue
            value = dict_i[var_]
            spatial_dict[pix] = value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr = arr[:180]
        arr[arr>0.1] = 1
        plt.imshow(arr, cmap='jet',aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var_}_{period}')
        plt.show()

        pass

class Sankey_plot_PLS:

    def __init__(self):
        # self.Y_name = 'LAI3g'
        self.Y_name = 'LAI4g'
        # self.Y_name = 'MODIS-LAI'
        self.var_list = ['CCI_SM', 'CO2', 'PAR', 'Temp', 'VPD']
        self.period_list = ['early', 'peak', 'late']
        self.fdir = f'/Volumes/NVME2T/greening_project_redo/data/Sankey_plot_data/PLS/{self.Y_name}'
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Sankey_plot_PLS',results_root_main_flow)
        outdir = join(self.this_class_arr,f'{self.Y_name}')
        T.mkdir(outdir)
        self.dff = join(outdir,'dataframe.df')
        # T.open_path_and_file(outdir)

        pass

    def run(self):
        # self.plot_p_value_spatial()
        # df,var_list = self.join_dataframe(self.fdir)
        df = self.build_sankey_plot(df,var_list)
        # #
        df = self.__gen_df_init()
        # df = Dataframe().add_Humid_nonhumid(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        self.plot_Sankey(df,True)
        # self.plot_Sankey(df,False)
        # self.plot_max_corr_spatial(df)
        # self.max_contribution_bar()
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def plot_max_corr_spatial(self,df):
        outdir = join(self.this_class_tif,'plot_max_corr_spatial')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        period_list = self.period_list
        var_list = self.var_list
        var_color_dict = {'CCI_SM':'#ff7f0e','CO2':'#1f77b4','PAR':'#2ca02c','Temp':'#9467bd','VPD':'#d62728'}
        var_value_dict = {'CCI_SM':0,'CO2':1,'PAR':2,'Temp':3,'VPD':4,'None':np.nan,'nan':np.nan}
        color_list = []
        for var_ in var_list:
            color_list.append(var_color_dict[var_])
        cmap = T.cmap_blend(color_list)
        for period in period_list:
            col_name = f'{period}_max_var'
            spatial_dict = T.df_to_spatial_dic(df,col_name)
            spatial_dict_value = {}
            for pix in spatial_dict:
                var_ = spatial_dict[pix]
                var_ = str(var_)
                var_ = var_.replace(period+'_','')
                # print(var_)
                value = var_value_dict[var_]
                spatial_dict_value[pix] = value
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_value)
            arr = arr[:180]
            # plt.figure()
            # DIC_and_TIF().plot_back_ground_arr(global_land_tif,aspect='auto')
            # plt.imshow(arr, cmap=cmap,aspect='auto')
            # plt.colorbar()
            # plt.title(f'{self.Y_name}_{period}')
            # T.plot_colors_palette(cmap)
            outf = join(outdir,f'{self.Y_name}_{period}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.show()

    def join_dataframe(self,fdir):

        df_list = []
        var_list = []
        for f in T.listdir(fdir):
            # print(f)
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir,f))
            df_i = T.dic_to_df(dic,key_col_str='pix')
            old_col_list = []
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                old_col_list.append(col)
                var_list.append(col)
            if '_p_value_' in f:
                new_col_list = [f'{period}_{col}_p_value' for col in old_col_list]
            else:
                new_col_list = [f'{period}_{col}' for col in old_col_list]
            for i in range(len(old_col_list)):
                new_name = new_col_list[i]
                old_name = old_col_list[i]
                df_i = T.rename_dataframe_columns(df_i,old_name,new_name)
            df_list.append(df_i)
        df = pd.DataFrame()
        df = Tools().join_df_list(df,df_list,'pix')
        ## re-index dataframe
        df = df.reset_index(drop=True)
        return df,var_list


    def build_sankey_plot(self,df,var_list,p_threshold=0.1):

        period_list = ['early','peak','late']
        for period in period_list:
            var_name_list = []
            for var_ in var_list:
                var_name_corr = f'{period}_{var_}'
                var_name_list.append(var_name_corr)
            for i,row in tqdm(df.iterrows(),total=len(df)):
                value_dict = {}
                for var_ in var_name_list:
                    value = row[var_]
                    value = abs(value)
                    value_dict[var_] = value
                max_key = T.get_max_key_from_dict(value_dict)
                df.loc[i,f'{period}_max_var'] = max_key
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


    def __get_var_list(self,fdir):
        var_list = []
        for f in T.listdir(fdir):
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir, f))
            df_i = T.dic_to_df(dic, key_col_str='pix')
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                var_list.append(col)
            break
        return var_list

    def __add_alpha_to_color(self,hexcolor,alpha=0.8):
        rgb = T.hex_color_to_rgb(hexcolor)
        rgb = list(rgb)
        rgb[-1] = alpha
        rgb = tuple(rgb)
        rgb_str = 'rgba'+str(rgb)
        return rgb_str

    def plot_Sankey(self,df,ishumid):
        if ishumid:
            df = df[df['HI_reclass'] == 'Humid']
            outdir = join(self.this_class_png, f'{self.Y_name}/Humid')
            title = 'Humid'
        else:
            df = df[df['HI_reclass'] == 'Dryland']
            outdir = join(self.this_class_png, f'{self.Y_name}/Dryland')
            title = 'Dryland'
        T.mkdir(outdir,force=True)
        # T.open_path_and_file(outdir)
        var_list = self.__get_var_list(self.fdir)

        period_list = ['early','peak','late']
        # period_list = ['late']
        early_max_var_list = [f'early_{var}' for var in var_list]
        peak_max_var_list = [f'peak_{var}' for var in var_list]
        late_max_var_list = [f'late_{var}' for var in var_list]

        node_list = early_max_var_list+peak_max_var_list+late_max_var_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        color_dict = {'CO2': self.__add_alpha_to_color('#00FF00'),
                      'CCI_SM': self.__add_alpha_to_color('#00E7FF'),
                      'PAR': self.__add_alpha_to_color('#FFFF00'),
                      'Temp': self.__add_alpha_to_color('#FF0000'),
                      'VPD': self.__add_alpha_to_color('#B531AF'),
                      }
        node_color_list = [color_dict[var_] for var_ in var_list]
        node_color_list = node_color_list * 3
        # print(node_color_list)
        # exit()
        early_max_var_col = 'early_max_var'
        peak_max_var_col = 'peak_max_var'
        late_max_var_col = 'late_max_var'

        source = []
        target = []
        value = []
        # color_list = []
        # node_list_anomaly_value_mean = []
        # anomaly_value_list = []
        node_list_with_ratio = []
        node_name_list = []
        for early_status in early_max_var_list:
            # print(early_status)
            # exit()
            df_early = df[df[early_max_var_col] == early_status]
            vals = df_early[early_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_early)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{early_status} {vals_mean:.2f}')
        for peak_status in peak_max_var_list:
            df_peak = df[df[peak_max_var_col] == peak_status]
            vals = df_peak[peak_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_peak)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{peak_status} {vals_mean:.2f}')
        for late_status in late_max_var_list:
            df_late = df[df[late_max_var_col] == late_status]
            vals = df_late[late_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_late)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{late_status} {vals_mean:.2f}')
        node_list_with_ratio = [round(i,3) for i in node_list_with_ratio]

        for early_status in early_max_var_list:
            df_early = df[df[early_max_var_col] == early_status]
            early_count = len(df_early)
            for peak_status in peak_max_var_list:
                df_peak = df_early[df_early[peak_max_var_col] == peak_status]
                peak_count = len(df_peak)
                source.append(position_dict[early_status])
                target.append(position_dict[peak_status])
                value.append(peak_count)
                for late_status in late_max_var_list:
                    df_late = df_peak[df_peak[late_max_var_col] == late_status]
                    late_count = len(df_late)
                    source.append(position_dict[peak_status])
                    target.append(position_dict[late_status])
                    value.append(late_count)
        link = dict(source=source, target=target, value=value,)
        # node = dict(label=node_list_with_ratio, pad=100,
        node = dict(label=node_name_list, pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    # x=node_x,
                    # y=node_y,
                    color=node_color_list
                )
        data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
        fig = go.Figure(data)
        fig.update_layout(title_text=f'{title}')
        fig.write_html(join(outdir, f'{title}.html'))
        # fig.write_image(join(outdir, f'{title}.png'))
        # fig.show()


    def plot_p_value_spatial(self):
        fdir = self.fdir
        var_ = 'Temp'
        # var_ = 'CCI_SM'
        period = 'early'
        f = join(fdir, '2000-2018_partial_correlation_p_value_early_LAI3g.npy')
        dict_ = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(dict_):
            dict_i = dict_[pix]
            if not var_ in dict_i:
                continue
            value = dict_i[var_]
            spatial_dict[pix] = value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr = arr[:180]
        arr[arr>0.1] = 1
        plt.imshow(arr, cmap='jet',aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var_}_{period}')
        plt.show()

        pass

    def max_contribution_bar(self):
        fdir = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/tif/Sankey_plot_max_contribution/plot_max_corr_spatial'
        outdir = join(self.this_class_png, 'max_contribution_bar')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        var_value_dict = {'CCI_SM': 0, 'CO2': 1, 'PAR': 2, 'Temp': 3, 'VPD': 4}
        color_dict = {'CCI_SM': '#00E7FF', 'CO2': '#00FF00', 'PAR': '#FFFF00', 'Temp': '#FF0000', 'VPD': '#B531AF'}
        var_value_dict_reverse = T.reverse_dic(var_value_dict)
        all_dict = {}
        product_list = []
        period_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fname = f.split('.')[0]
            product,period = fname.split('_')
            product_list.append(product)
            period_list.append(period)
            arr = ToRaster().raster2array(join(fdir, f))[0]
            arr = T.mask_999999_arr(arr,warning=False)
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            key = f'{product}_{period}'
            all_dict[key] = dic
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe().add_Humid_nonhumid(df)
        humid_var = 'HI_reclass'
        humid_list = T.get_df_unique_val_list(df, humid_var)
        for humid in humid_list:
            df_humid = df[df[humid_var] == humid]
            fig,axs = plt.subplots(3,3,figsize=(10,10))
            product_list = list(set(product_list))
            period_list = ['early','peak','late']
            for m,product in enumerate(product_list):
                for n,period in enumerate(period_list):
                    col = f'{product}_{period}'
                    df_i = df_humid[col]
                    df_i = df_i.dropna()
                    unique_value = T.get_df_unique_val_list(df_humid,col)
                    x_list = []
                    y_list = []
                    color_list = []
                    for val in unique_value:
                        df_i_i = df_i[df_i == val]
                        ratio = len(df_i_i)/len(df_i) * 100
                        x = var_value_dict_reverse[int(val)][0]
                        x_list.append(x)
                        y_list.append(ratio)
                        color_list.append(color_dict[x])
                    # plt.figure()
                    # print(m,n)
                    axs[m][n].bar(x_list,y_list,color=color_list)
                    axs[m][n].set_title(f'{product}_{period}')
                    axs[m][n].set_ylim(0,47)
            plt.suptitle(f'{humid}')
            plt.tight_layout()
            outf = join(outdir, f'{humid}.pdf')
            plt.savefig(outf)
            plt.close()

        pass


class multiregression_plot:
    #todo 6-20 实现nc 文章 df = df[df['row'] < 120]

    def __init__(self):
        # df = df[df['HI_class'] == 'Humid']
        # df = df[df['NDVI_MASK'] == 1]
        # df=df[df['max_trend']<10]
        # df = df[df['landcover'] !='cropland']
        pass

    def run(self):
        self.plot_matrix()

    def plot_matrix(self):
        # f = '/Volumes/NVME2T/greening_project_redo/data/220624/Dryland.df'
        f = '/Volumes/NVME2T/greening_project_redo/data/220624/Humid.df'
        df = T.load_df(f)
        var_list = ['CCI_SM', 'CO2', 'PAR', 'Temp', 'VPD']
        columns_list = list(df.columns)
        columns_list.remove('drivers')
        period_list = ['early','peak','late']

        print(columns_list)
        model_list = []
        for column in columns_list:
            period = column.split('_')[0]
            model = column.split('_')[1:]
            model = '_'.join(model)
            model_list.append(model)
        model_list = list(set(model_list))
        model_list.sort()
        for var_ in var_list:
            fig, ax = plt.subplots()
            matrix = []
            yticks_list = []
            for period in period_list:
                temp = []
                xticks_list = []
                for model in model_list:
                    df_i = df[df['drivers']==var_]
                    col = f'{period}_{model}'
                    value = df_i[col].tolist()[0]
                    temp.append(value)
                    xticks_list.append(model)
                matrix.append(temp)
                yticks_list.append(period)
            matrix = np.array(matrix)
            sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax)
            plt.xticks(range(len(xticks_list)), xticks_list, rotation=90)
            plt.yticks(range(len(yticks_list)), yticks_list)
            plt.title(f'{var_}')
            plt.tight_layout()
            plt.show()




class Moving_window_1:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Moving_window_1', results_root_main_flow)
        pass


    def run(self):
        fdir = join(self.this_class_arr,'1982-2020_during_late_window15_LAI4g')
        period = 'late'
        self.plot_pdf(fdir,period)
        pass

    def plot_pdf(self,fdir,period):

        outdir = join(self.this_class_png, period)
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        dic_all = {}
        variables_list = []
        window_list = []
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('correlation.npy'):
                continue
            dic = T.load_npy(join(fdir, f))
            dic_new = {}
            for pix in dic:
                dic_i = dic[pix]
                dic_i_new = {}
                for key_i in dic_i:
                    val_i = dic_i[key_i]
                    new_key_i = key_i.split(f'_{period}')[0]
                    if not new_key_i in variables_list:
                        variables_list.append(new_key_i)
                    dic_i_new[new_key_i] = val_i
                dic_new[pix] = dic_i_new
            window = f.split('.')[0].split('_')[-2]
            window = window.replace('window','')
            window = int(window)
            dic_all[window] = dic_new
            window_list.append(window)
        gradient_color = KDE_plot().makeColours(window_list, 'Spectral')

        df = T.spatial_dics_to_df(dic_all)
        df = Dataframe().add_Humid_nonhumid(df)
        humid_var = 'HI_reclass'
        humid_list = T.get_df_unique_val_list(df, humid_var)
        for humid in humid_list:
            df_humid = df[df[humid_var] == humid]
            for var_ in variables_list:
                for w in window_list:
                    spatial_dic = T.df_to_spatial_dic(df_humid,w)
                    vals_list = []
                    for pix in spatial_dic:
                        dic_i = spatial_dic[pix]
                        if not var_ in dic_i:
                            continue
                        val = dic_i[var_]
                        vals_list.append(val)
                    vals_list = T.remove_np_nan(vals_list)
                    x, y = Plot().plot_hist_smooth(vals_list, alpha=0, bins=80)
                    plt.plot(x, y, color=gradient_color[w], label=str(w))
                plt.title(f'{var_}_{period}_{humid}')
                plt.tight_layout()
                plt.savefig(join(outdir, f'{var_}_{period}_{humid}.pdf'))
                # plt.legend()
                # plt.savefig(join(outdir,f'legend.pdf'))
                plt.close()

def main():
    # Phenology().run()
    # Get_Monthly_Early_Peak_Late().run()
    # Pick_Early_Peak_Late_value().run()
    # Dataframe().run()
    # RF().run()
    # Moving_window_RF().run()
    # Analysis().run()
    # Moving_window().run()
    # Global_vars().get_valid_pix_df()
    # Drought_event().run()
    # Partial_corr().run()
    # Time_series().run()
    # Plot_Trend_Spatial().run()
    # Sankey_plot().run()
    # Sankey_plot_max_contribution().run()
    # Sankey_plot_single_max_contribution().run()
    # Sankey_plot_PLS().run()
    # Moving_window_1().run()
    multiregression_plot().run()

    pass


if __name__ == '__main__':
    main()