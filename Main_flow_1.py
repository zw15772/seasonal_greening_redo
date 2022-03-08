# coding=utf-8
import Main_flow
from preprocess import *
results_root_main_flow = join(results_root,'Main_flow_1')
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
        fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
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
        T.mk_dir(outdir,force=True)
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
        T.mk_dir(outdir,force=True)
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
        self.datadir=join(data_root,'LAI_3g')

    def run(self):

        # fdir = join(self.datadir,'LAI_TIFF_resample_0.5')
        # outdir = join(self.datadir,'per_pix_annual')
        # self.data_transform_annual(fdir,outdir)
        # 3 hants smooth
        # self.hants()
        # self.check_hants()

        # self.annual_phenology()
        # self.compose_annual_phenology()
        # self.check_compose_hants()
        # self.all_year_hants_annual()
        # self.all_year_hants_annual_mean()
        # self.all_year_hants()
        # self.longterm_mean_phenology()
        self.check_lonterm_phenology()

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
            if T.is_all_nan(vals):
                continue
            print(vals)
            vals = T.interp_nan(vals)
            print(vals)
            print('---')
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
        # T.save_npy(hants_dic,outf)
        exit()

    def hants(self):
        outdir = join(self.this_class_arr,'hants')
        T.mk_dir(outdir)
        fdir = join(self.datadir,'per_pix_annual')
        params = []
        for y in T.listdir(fdir):
            params.append([outdir, y, fdir])
            self.kernel_hants([outdir, y, fdir])
        # MULTIPROCESS(self.kernel_hants, params).run(process=4)

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
    #     T.mk_dir(out_dir)
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


    def annual_phenology(self,threshold_i=0.2):
        out_dir = join(self.this_class_arr, 'annual_phenology')
        T.mk_dir(out_dir)
        hants_smooth_dir = join(self.this_class_arr, 'hants')
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
            # np.save(outf_i,result_dic)

    def compose_annual_phenology(self):
        f_dir = join(self.this_class_arr, 'annual_phenology')
        outdir = join(self.this_class_arr,'compose_annual_phenology')
        T.mk_dir(outdir)
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


    def data_transform_annual(self,fdir,outdir):
        T.mk_dir(outdir)
        year_list = []
        for f in T.listdir(fdir):
            y, m, d = Pre_Process().get_year_month_day(f)
            year_list.append(y)
        year_list = T.drop_repeat_val_from_list(year_list)
        # print(year_list)
        # exit()
        for year in year_list:
            outdir_i = join(outdir,f'{year}')
            T.mk_dir(outdir_i)
            annual_f_list = []
            for f in T.listdir(fdir):
                y, m, d = Pre_Process().get_year_month_day(f)
                if y == year:
                    annual_f_list.append(f)
            Pre_Process().data_transform_with_date_list(fdir,outdir_i,annual_f_list)
            print(annual_f_list)
            print(year)
            # exit()
        # Pre_Process().monthly_compose()
        exit()

    def all_year_hants(self):
        fdir = join(self.this_class_arr,'hants')
        outdir = join(self.this_class_arr,'all_year_hants')
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
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
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Get_Monthly_Early_Peak_Late',results_root_main_flow)

    def run(self):
        self.Monthly_Early_Peak_Late()
        # self.check_pix()
        pass


    def Monthly_Early_Peak_Late(self):
        outf = join(self.this_class_arr,'Monthly_Early_Peak_Late.df')
        vege_dir = vars_info_dic['LAI_3g']['path']
        vege_dic = T.load_npy_dir(vege_dir)
        result_dic = {}
        for pix in tqdm(vege_dic):
            vals = vege_dic[pix]
            if T.is_all_nan(vals):
                continue
            vals = np.array(vals)
            val_reshape = vals.reshape((-1,12))
            val_reshape_T = val_reshape.T
            month_mean_list = []
            for month in val_reshape_T:
                month_mean = np.nanmean(month)
                month_mean_list.append(month_mean)
            isnan_list = np.isnan(month_mean_list)

            max_n_index,max_n_val = T.pick_max_n_index(month_mean_list,n=2)
            peak_months_distance = abs(max_n_index[0]-max_n_index[1])
            # if peak_months_distance >= 3:
            #     continue

            max_n_index = list(max_n_index)
            max_n_index.sort()
            if max_n_index[0]<=3:
                continue
            if max_n_index[1]>=10:
                continue
            peak_mon = np.array(max_n_index) + 1
            early_mon = list(range(4,peak_mon[0]))
            late_mon = list(range(peak_mon[-1]+1,11))
            # print(early_mon)

            if peak_months_distance >= 2:
                early_mon = list(range(4, peak_mon[0]))
                peak_mon = list(range(peak_mon[0],peak_mon[1]+1))
                late_mon = list(range(peak_mon[-1] + 1, 11))
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
            print(month_mean_list)
            print(early_mon)
            print(peak_mon)
            print(late_mon)
            plt.plot(month_mean_list)
            plt.show()
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


    def check_pix(self):
        dff = join(self.this_class_arr,'Monthly_Early_Peak_Late.df')
        df = T.load_df(dff)
        pix_list = T.get_df_unique_val_list(df,'pix')
        spatial_dic = {}
        for pix in pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr(global_land_tif)
        plt.imshow(arr)
        plt.show()
        pass

class Pick_Early_Peak_Late_value:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Pick_Early_Peak_Late_value',results_root_main_flow)
        pass


    def run(self):
        # self.Pick_variables()
        # self.Pick_variables_accumulative()
        # self.Pick_variables_min()
        pass


    def Pick_variables(self):
        outdir = join(self.this_class_arr,'Pick_variables')
        T.mk_dir(outdir)
        var_list = [
            'LAI_3g',
            'SPEI',
            'Temperature',
            'Soil moisture',
            'CO2',
            'Aridity',
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
        #
        # combine_season_list = ['early','peak','late']
        # var_ = 'LAI_3g'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)

        # combine_season_list = ['early','peak','late']
        # var_ = 'Soil moisture'
        # method = 'mean'
        # df = self.add_combine(df,combine_season_list,var_,method)

        combine_season_list = ['early', 'peak', 'late']
        var_ = 'SPEI_accu'
        method = 'sum'
        df = self.add_combine(df, combine_season_list, var_, method)
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
            vals = P.cal_anomaly_juping(vals)
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


class Analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Analysis',results_root_main_flow)

        pass

    def run(self):
        # self.Greeing_trend()
        # self.Greeing_trend_combine()
        # self.SOS_trend()
        self.carry_over_effect_bar_plot()
        # self.Greeing_trend_3_season_line()
        # self.Greeing_trend_two_period()
        # self.Jan_to_Dec_timeseries()
        # self.Jan_to_Dec_timeseries_hants()
        # self.early_peak_greening_speedup_advanced_sos()
        # self.check()
        # self.carry_over_effect_with_sm()
        # self.inter_annual_carryover_effect()
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
        T.mk_dir(outdir)
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

    def carry_over_effect_bar_plot(self):
        # carry over trend < 0 humid region, see sos trend
        humid_region_var = 'HI_reclass'
        region = 'Non Humid'
        mode = 'early-peak_late'
        carryover_tif = join(Moving_window().this_class_tif,f'array_carry_over_effect/trend/{mode}.tif')
        sos_tif = join(self.this_class_tif,'SOS_trend/SOS_trend.tif')
        sm_early_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/early.tif')
        sm_peak_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/peak.tif')
        sm_late_tif = join(self.this_class_tif,'Greeing_trend/Soil moisture/late.tif')
        carryover_dic = DIC_and_TIF().spatial_tif_to_dic(carryover_tif)
        sos_dic = DIC_and_TIF().spatial_tif_to_dic(sos_tif)
        sm_early_dic = DIC_and_TIF().spatial_tif_to_dic(sm_early_tif)
        sm_peak_dic = DIC_and_TIF().spatial_tif_to_dic(sm_peak_tif)
        sm_late_dic = DIC_and_TIF().spatial_tif_to_dic(sm_late_tif)
        spatial_dic_all = {
            'carryover_trend':carryover_dic,
            'sos_trend':sos_dic,
            'sm_early_trend':sm_early_dic,
            'sm_peak_trend':sm_peak_dic,
            'sm_late_trend':sm_late_dic,
        }
        df = T.spatial_dics_to_df(spatial_dic_all)
        df = Dataframe().add_Humid_nonhumid(df)
        T.print_head_n(df)
        exit()
        # df = df[df[humid_region_var]==region]
        # df = df[df['carryover_trend']<0]
        # sm_trend_dic = T.df_to_spatial_dic(df,'sm_trend')
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(sm_trend_dic)
        # sm_trend = np.nanmean(arr)
        # print(sm_trend)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        pass

    def Greeing_trend(self):
        # var_ = 'LAI_3g'
        # var_ = 'SPEI'
        var_ = 'Soil moisture'
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

    def Greeing_trend_combine(self):
        var_ = 'LAI_3g'
        mode = 'early-peak'
        # var_ = 'SPEI'
        # var_ = 'Soil moisture'
        outdir = join(self.this_class_tif,f'Greeing_trend_combine/{var_}')
        T.mk_dir(outdir,force=True)
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

    def Greeing_trend_3_season_line(self):
        var_ = 'LAI_3g'
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mk_dir(outdir,force=True)
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
        # T.mk_dir(outdir,force=True)
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
        # T.mk_dir(outdir,force=True)
        pass


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
        T.mk_dir(outdir)
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



    def carry_over_effect_with_sm(self):
        # matrix
        # bin_n = 21
        bin_n = 11
        dff = join(RF().this_class_arr,'Dataframe.df')
        df = T.load_df(dff)

        early_peak_LAI_3g_mean_var = 'early_peak_LAI_3g_mean'
        # early_peak_LAI_3g_mean_var = 'early_LAI_3g'
        # early_peak_LAI_3g_mean_var = 'peak_LAI_3g'
        early_peak_late_sm_var = 'early_peak_late_Soil moisture_mean'
        # early_peak_late_sm_var = 'early_peak_Soil moisture_accu_sum'
        # early_peak_late_sm_var = 'early_peak_SPEI_accu_sum'
        early_peak_late_sm_var = 'early_peak_late_SPEI_accu_sum'
        late_lai_var = 'late_LAI_3g'
        sos_var = 'SOS'
        df = df[df[early_peak_LAI_3g_mean_var]>0]
        df = df[df[sos_var]<0]
        # humid_var = 'Humid'
        # humid_var = 'Non Humid'
        # df = df[df['HI_reclass'] == humid_var]
        print(df.columns)
        #
        # print(len(df))
        # exit()
        sm_vals = df[early_peak_late_sm_var].tolist() # min: -.15 max: .15
        early_peak_LAI_vals = df[early_peak_LAI_3g_mean_var].tolist() # min: 0 max: 2
        # plt.hist(early_peak_LAI_vals,bins=80)
        # plt.show()
        sm_vals = T.remove_np_nan(sm_vals)
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
                early_lai_vals = df_lai[early_peak_LAI_3g_mean_var].tolist()
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
        # plt.imshow(matrix,vmin=-0.05,vmax=0.05,cmap='RdBu')
        plt.imshow(matrix,vmin=down,vmax=up,cmap='RdBu')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(late_lai_var, rotation=270)
        plt.xlabel(early_peak_LAI_3g_mean_var)
        plt.ylabel(early_peak_late_sm_var)
        # plt.title(humid_var)
        plt.tight_layout()
        plt.show()
        pass

class Dataframe:

    def __init__(self):
        self.this_class_arr = join(results_root_main_flow,'arr/Dataframe/')
        self.dff = self.this_class_arr + 'dataframe_1982-2018.df'
        self.P_PET_fdir = data_root+ 'aridity_P_PET_dic/'
        T.mk_dir(self.this_class_arr,force=True)


    def run(self):
        df = self.__gen_df_init()
        df = self.add_data(df)
        df = self.add_lon_lat_to_df(df)
        df = self.add_Humid_nonhumid(df)
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
        T.mk_dir(outdir)
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
        # self.single_correlation_pdf_plot()
        # self.single_correlation_time_series_plot()

        # self.trend()
        # self.trend_time_series_plot()
        # self.trend_matrix_plot()
        # self.trend_area_ratio_timeseries()

        # self.array()
        # self.array_carry_over_effect()
        self.array_carry_over_effect_spatial_trend()

        # self.mean()
        # self.mean_time_series_plot()
        # self.mean_matrix_plot()

        # self.partial_correlation()
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
        self.x_var_list = ['Aridity', 'CO2', 'SPEI', 'SPEI_accu', 'SPEI_min', 'Soil moisture', 'Soil moisture_accu', 'Soil moisture_min', 'Temperature']
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
        T.mk_dir(outdir)
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

    def partial_correlation(self):
        # outdir = join(self.this_class_arr,'partial_correlation_orign_nodetrend')
        # outdir = join(self.this_class_arr,'partial_correlation_orign_detrend')
        # outdir = join(self.this_class_arr,'partial_correlation_anomaly_nodetrend')
        outdir = join(self.this_class_arr,'partial_correlation_anomaly_detrend')
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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


    def trend(self):
        outdir = join(self.this_class_arr,'trend')
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'trend/trend.df')
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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

    def mean(self):
        outdir = join(self.this_class_arr, 'mean')
        T.mk_dir(outdir)
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
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, 'mean/mean.df')
        df = T.load_df(dff)
        df = df.dropna()
        # T.print_head_n(df)
        # exit()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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
        T.mk_dir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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
        # T.mk_dir(outdir)
        dff = join(self.this_class_arr, f'single_correlation/single_correlation_{global_n}.df')
        df = T.load_df(dff)
        df = Global_vars().clean_df(df)
        ### check spatial ###
        # spatial_dic = T.df_to_spatial_dic(df,'ndvi_mask_mark')
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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
        T.mk_dir(outdir)
        # dff = join(self.this_class_arr, f'{func_name}/{func_name}.df')
        # df = T.load_df(dff)
        # outdir = join(self.this_class_png, 'matrix_trend_moving_window')
        # T.mk_dir(outdir)
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
        T.mk_dir(outdir)
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
                        window_index = np.array(moving_window_index_list[n],dtype=int)
                        window_index = window_index - global_start_year
                        window_index = [int(i) for i in window_index]
                        # print(window_index)
                        try:
                            xval_pick = T.pick_vals_from_1darray(xval,window_index)
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
        T.mk_dir(outdir,force=True)
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
        corr_mode = 'early_late'
        # corr_mode = 'peak_late'
        # corr_mode = 'early-peak_late'
        outdir = join(self.this_class_tif,f'array_carry_over_effect/trend/')
        outf = join(outdir,f'{corr_mode}.tif')
        T.mk_dir(outdir)
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
        for pix in tqdm(pix_list):
            vals_list = []
            for w in window_list:
                val = all_dic[w][pix]
                vals_list.append(val)
            a,_,_,_ = T.nan_line_fit(list(range(len(vals_list))),vals_list)
            spatial_dic_trend[pix] = a
        DIC_and_TIF().pix_dic_to_tif(spatial_dic_trend,outf)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_trend)
        # plt.imshow(arr,cmap='RdBu',vmin=-0.05,vmax=0.05)
        # plt.colorbar()
        # plt.show()
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

def main():
    # Phenology().run()
    # Get_Monthly_Early_Peak_Late().run()
    # Pick_Early_Peak_Late_value().run()
    # Dataframe().run()
    # RF().run()
    Analysis().run()
    # Moving_window().run()
    # Global_vars().get_valid_pix_df()

    pass


if __name__ == '__main__':
    main()