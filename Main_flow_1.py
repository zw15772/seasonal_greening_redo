# coding=utf-8
import Main_flow
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
        self.hants()
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


    def kernel_hants(self, params):
        outdir, y, fdir = params
        outf = join(outdir,y)
        dic = T.load_npy_dir(join(fdir,y))
        hants_dic = {}
        for pix in tqdm(dic,desc=y):
            r,c = pix
            if r > 180:
                continue
            vals = dic[pix]
            vals = np.array(vals)
            if T.is_all_nan(vals):
                continue
            xnew, ynew = self.__interp__(vals)
            std = np.nanstd(ynew)
            std = float(std)
            ynew = np.array([ynew])
            # print np.std(ynew)
            results = HANTS().HANTS(sample_count=365, inputs=ynew, low=0, high=10,
                            fit_error_tolerance=std)
            result = results[0]
            if T.is_all_nan(result):
                continue
            hants_dic[pix] = result
        T.save_npy(hants_dic,outf)

    def hants(self):
        outdir = join(self.this_class_arr,'hants')
        T.mk_dir(outdir)
        fdir = join(self.datadir,'per_pix_annual')
        params = []
        for y in T.listdir(fdir):
            params.append([outdir, y, fdir])
            # self.kernel_hants([outdir, y, fdir])
        MULTIPROCESS(self.kernel_hants, params).run(process=4)

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
        self.Greeing_trend_3_season()
        # self.Greeing_trend_two_period()
        # self.Jan_to_Dec_timeseries()
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

    def Greeing_trend_3_season(self):
        var_ = 'LAI_3g'
        outdir = join(self.this_class_tif,f'Greeing_trend/{var_}')
        T.mk_dir(outdir,force=True)
        dff = join(Pick_Early_Peak_Late_value().this_class_arr,f'Pick_variables/{var_}.df')
        df = T.load_df(dff)
        df = Main_flow.Dataframe().add_lon_lat_to_df(df)
        df = df[df['lat']>30]
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
            for pix in spatial_dic:
                vals = spatial_dic[pix]
                matrix.append(vals)
            annual_mean = np.nanmean(matrix,axis=0)
            x = list(range(len(annual_mean)))
            y = annual_mean
            a, b, r, p = self.nan_linear_fit(x, y)
            # plt.figure()
            KDE_plot().plot_fit_line(a, b, r, p,x)
            plt.scatter(x,y,label=season)
            plt.legend()
        plt.show()
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


    def Jan_to_Dec_timeseries(self):
        var_ = 'LAI_3g'
        fdir = vars_info_dic[var_]['path']
        start_year = vars_info_dic[var_]['start_year']
        void_df = Global_vars().get_valid_pix_df()
        HI_region = 'Non Humid'
        # HI_region = 'Humid'
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




def main():
    Phenology().run()
    # Get_Monthly_Early_Peak_Late().run()
    # Pick_Early_Peak_Late_value().run()
    # RF().run()
    # Analysis().run()
    # Global_vars().get_valid_pix_df()

    pass


if __name__ == '__main__':
    main()