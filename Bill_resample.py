################################
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from ToTiff import WriteToTif
# from input_raster import input_raster
import osgeo.gdal as gdal
import osgeo.osr as osr
import osgeo.gdalconst as gdalconst
import netCDF4 as pycdf


####Calculate the pixel area in km2 for a given raster and write to geotiff
def area(fid):
    # read data
    band = fid.GetRasterBand(1)
    data = band.ReadAsArray()
    geot = fid.GetGeoTransform()
    print(geot)
    # Get rows, cols, res
    rows = data.shape[0]
    cols = data.shape[1]
    xRes = geot[1]
    yRes = -geot[5]
    # Get extent
    minx = geot[0]
    miny = geot[3] + cols * geot[4] + rows * geot[5]
    maxx = geot[0] + cols * geot[1] + rows * geot[2]
    maxy = geot[3]

    # Get lat, lon, area (km2) and write to tiff
    radians = 0.0174532925
    radius = 6378.137  # km
    lat = np.linspace(maxy, miny, rows)
    lon = np.linspace(minx, maxx, cols)
    xGrid, yGrid = np.meshgrid(lon, lat)
    area = (np.sin(yGrid * radians + 0.5 * yRes * radians) - np.sin(yGrid * radians - 0.5 * yRes * radians)) * (
                xRes * radians) * radius * radius  # km2
    WriteToTif(area, geot, data_dir + "Area_WUS_4km.tiff", gdalconst.GDT_Float32)  # write out area raster

    return area


####Write an input dataset to geotiff
def WriteToTif(src, geot, filename, dtype):
    cols = src.shape[1]
    rows = src.shape[0]
    sr_wgs84 = osr.SpatialReference()
    sr_wgs84.ImportFromEPSG(4326)  # 4326 is the EPSG code for WGS84
    ds_out = gdal.GetDriverByName('GTiff').Create(filename, cols, rows, 1, dtype)
    band_out = ds_out.GetRasterBand(1)
    # band_out.Fill(np.nan) #no data value
    # band_out.SetNoDataValue(np.nan)
    ds_out.SetProjection(
        sr_wgs84.ExportToWkt())  # convert to well-known text format for setting projection on output raster
    ds_out.SetGeoTransform(geot)
    band_out.WriteArray(src)
    ds_out.FlushCache()


####Calculate totals for a given landcover mask
def totals(data, lc, mask, classes, classnames, t0, filename):
    '''
    all variables resolutons are in 0.05 * 0.05 degrees
    :param data:  Leaf area index data
    :param lc: Vegetation land cover ratio
    :param mask: vegetated and non-vegetated mask
    :param classes:
    :param classnames:
    :param t0:
    :param filename:
    :return:
    '''
    # variables
    years = data.shape[0]
    rows = data.shape[1]
    cols = data.shape[2]
    cols_res = 360.0 / cols  ## 360/7200=0.05
    rows_res = 180.0 / rows  ## 180/3600=0.05

    # Calculate area of each gridcell m2
    radians = 0.0174532925
    radius = 6378137  # m
    lats = np.linspace((-rows / 2) * rows_res - ((rows_res) / 2), (rows / 2) * rows_res + ((rows_res) / 2), rows)
    lons = np.linspace(0, (cols) * cols_res - cols_res, cols)
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    area = (np.sin(lats_grid * radians + 0.5 * rows_res * radians) - np.sin(
        lats_grid * radians - 0.5 * rows_res * radians)) * (cols_res * radians) * radius * radius

    print(lc * area).sum()  ## lc: ratio of landcover to area of gridcell. valid landcover pixels number/total gridcell number

    # variables
    n = len(classes)

    # Totals by land cover class
    time = np.zeros((years))
    totals = np.zeros((n + 1, years))
    for i in np.arange(n):  # landcover classes
        lc_mask = np.zeros((rows, cols))
        lc_mask[mask == i + 1] = 1
        for j in np.arange(years):  # years
            time[j] = t0 + j
            totals[0, j] = (data[j, :, :] * lc * area).sum() / (lc * area).sum()  # global
            totals[i + 1, j] = (data[j, :, :] * lc * area * lc_mask).sum() / (lc * area * lc_mask).sum()

    # Output
    ofile = open(filename, 'w')
    output = np.column_stack((time, np.transpose(totals)))
    np.savetxt(ofile, output, fmt=','.join(['%i'] + ['%.3f'] * (n + 1)),
               header=','.join(['Year'] + [classnames[0]] + [classnames[1]] + [classnames[2]] + [classnames[3]] +
                               [classnames[4]] + [classnames[5]] + [classnames[6]]),
               delimiter=',', comments='')

    return area


###Calculate distributions for a given landcover mask
def DistrByMask(data_raster, mask_raster, classes, filename):
    # storage array
    rows = data_raster.shape[0]
    cols = data_raster.shape[1]
    max_pixels = 1000000
    m = np.empty((classes, max_pixels))
    m[:, :] = np.nan
    for i in np.arange(classes):
        # create cz mask
        mask = np.zeros((rows, cols))
        mask[mask_raster == i + 1] = 1
        masked_data = data_raster * mask
        masked_data.flatten()
        masked_data = masked_data[masked_data != 0]
        m[i, 0:len(masked_data)] = masked_data
    # Output
    ofile = open(filename, 'w')
    output = np.transpose(m)
    np.savetxt(ofile, output, fmt='%.4f', delimiter=',')

##########################################

def main():
    a=np.array([[1,2,3,4,5,6,7,8,9,10],
                [11,12,13,14,15,16,17,18,19,20],
                [21,22,23,24,25,26,27,28,29,30],])
    print((a*a*a).sum())
    plt.imshow(a)
    plt.show()

if __name__ == '__main__':
    main()