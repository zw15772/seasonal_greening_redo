# coding=utf-8

from __init__ import *

T = Tools()

class GOES_process:

    def __init__(self):
        self.data_dir = join(data_root,'GOES')
        pass


    def run(self):
        self.nc_to_tif()
        self.interpolate()

    def nc_to_tif(self):

        geo_f = join(self.data_dir,'nc/geo/Conus_Lat_Lon_GOES16.nc')
        data_dir = join(self.data_dir,'nc/data')
        outdir = join(self.data_dir,'tif')
        T.mk_dir(outdir)
        geo_nc = Dataset(geo_f)
        print(geo_nc.variables.keys())
        lon = geo_nc.variables['longitude'][:]
        lat = geo_nc.variables['latitude'][:]
        lon = np.array(lon)
        lat = np.array(lat)
        lon = lon.T
        lat = lat.T
        for f in T.listdir(data_dir):
            fpath = join(data_dir,f)
            ncin = Dataset(fpath)
            arrs = ncin.variables['lst'][:]
            flag = 0
            for arr in arrs:
                flag += 1
                outf = join(outdir,f'{flag:02d}.tif')
                print(arr.shape)
                arr = np.array(arr)
                arr[arr<1] = np.nan
                arr = arr.T
                # plt.imshow(arr)
                # plt.show()
                row = arr.shape[0]
                col = arr.shape[1]
                pixelWidth = (np.nanmax(lon) - np.nanmin(lon)) / col
                pixelHeight = (np.nanmax(lat) - np.nanmin(lat)) / row

                arr_flatten = arr.flatten()
                lon_flatten = lon.flatten()
                lat_flatten = lat.flatten()

                self.lon_lat_val_to_tif(lon_flatten,lat_flatten,arr_flatten,outf,pixelHeight, pixelWidth,row,col)


    def lon_lat_val_to_tif(self, lon_list, lat_list, val_list, outtif,pixelHeight, pixelWidth,row,col):
        originY = np.nanmax(lat_list)
        originX = np.nanmin(lon_list)

        spatial_dic = {}
        for i in tqdm(range(len(lon_list))):
            lon = lon_list[i]
            lat = lat_list[i]
            val = val_list[i]
            if np.isnan(lon) or np.isnan(lat):
                continue
            # print(lon,lat)
            r = abs(round((lat - originY) / pixelHeight))
            c = abs(round((lon - originX) / pixelWidth))
            r = int(r)
            c = int(c)
            spatial_dic[(r, c)] = val
        spatial = []
        for r in tqdm(range(row)):
            temp = []
            for c in range(col):
                key = (r, c)
                try:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                except:
                    temp.append(np.nan)
            spatial.append(temp)
        spatial = np.array(spatial, dtype=float)
        longitude_start = originX
        latitude_start = originY
        ToRaster().array2raster(outtif, longitude_start, latitude_start, pixelWidth, -pixelHeight, spatial)


    def interpolate(self):
        f = '/Volumes/NVME2T/greening_project_redo/data/GOES/tif/01.tif'
        outf = '/Volumes/NVME2T/greening_project_redo/data/GOES/tif/01_interpolate.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array_copy = array.copy()

        for i in tqdm(range(len(array))):
            for j in range(len(array[0])):
                val = array[i][j]
                if np.isnan(val):
                    if i-1<0 or j-1<0 or i+1>len(array)-1 or j+1>len(array[0])-1:
                        vals_mean = np.nan
                        continue
                    val1 = array[i-1][j-1]
                    val2 = array[i-1][j]
                    val3 = array[i-1][j+1]
                    val4 = array[i][j-1]
                    val6 = array[i][j+1]
                    val7 = array[i+1][j-1]
                    val8 = array[i+1][j]
                    val9 = array[i+1][j+1]
                    val_list = [val1,val2,val3,val4,val6,val7,val8,val9]
                    vals_mean = np.nanmean(val_list)
                    array_copy[i][j] = vals_mean
        ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array_copy)



def main():
    GOES_process().run()

    pass

if __name__ == '__main__':
    main()
    pass