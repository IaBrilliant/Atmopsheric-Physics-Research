#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:55:21 2022

@author: brilliant
"""
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import scipy as sp
from scipy import signal  

'''Importing Datasets & Setting up the Coordinate System'''

# Observational Windspeed variable in m/s: 
ds_ws = xr.open_dataset("/rds/general/project/circulates/live/data/20CR/ua850_v2c_185101-201412.nc")
# Observational Pressure variable in hPa:
ds_pressure = xr.open_dataset("/rds/general/project/circulates/live/data/20CR/ps_v2c_185101-201412.nc")

# Creating equally spaced longitude array (longit_real) that includes real coordinates
longit_real = np.empty([128*25])
for i in range(0, 128):
    if i == 127:
        longit_real[(i*25):(i+1)*25] = np.linspace(ds_ws.lon[i], 360.0, 26)[0:-1] # interpolating up to 360
    else:
        longit_real[(i*25):(i+1)*25] = np.linspace(ds_ws.lon[i], ds_ws.lon[i+1], 26)[0:-1]
old_lons = np.append(np.array(ds_ws.lon[:]), 360.0) # interpolating up to 360

# Creating equally spaced latitude array that includes real coordinates
latit_real = np.empty([31*25])
for i in range(0, 31):
    latit_real[(i*25):(i+1)*25] = np.linspace(ds_ws.lat[i], ds_ws.lat[i+1], 26)[0:-1]
latit_real = np.append(latit_real, ds_ws.lat[31])

# Coordinates needed for NAO calculation:
# latit 65.68921234 - iceland 24
# longit 341.8875 - iceland 39
# latit 37.78471054 - azores 24
# longit 334.35 - azores 22
print(np.where(latit_real == 65.68921234), np.where(longit_real == 341.8875), 'Iceland') 
print(np.where(latit_real == 37.78471054), np.where(longit_real == 334.35), 'Azores')

#%%

'''Creating a Windspeed file'''

path = "/rds/general/user/.../home/.../ws_interpolated2.nc"

ds_ua850 = nc.Dataset(path, 'w', format = 'NETCDF4')
time = ds_ua850.createDimension('time', 1968)
lat = ds_ua850.createDimension('lat', 31*25+1)
lon = ds_ua850.createDimension('lon', 128)

times = ds_ua850.createVariable('time', np.float32, ('time',))
lats = ds_ua850.createVariable('lat', np.float32, ('lat',))
lons = ds_ua850.createVariable('lon', np.float32, ('lon',))
value = ds_ua850.createVariable('value', np.float32, ('time', 'lat', 'lon',))
value.units = 'm/s'
times[:] = ds_ws.time[:]
lons[:] = ds_ws.lon[:]
lats[:] = latit_real

# Only meridional interpolation was performed; no need for zonal as the mean was taken. 

# Spatial Interpolation using Cubic Spline:  
for i in range(0, 1968):
    ds = ds_ws.isel(time=i).isel(level=0)
    values = ds.ua
    value[i,:,:] = sp.interpolate.griddata(ds_ws.lat[:], values, latit_real, method = 'cubic')

ds_ua850.close()

'''Creating a Pressure file'''

vals = np.empty([1968, 3, 3])

path_ic = "/rds/general/user/.../home/.../ps_interpolated.nc"
ps_ic = nc.Dataset(path_ic, 'w', format = 'NETCDF4')
time = ps_ic.createDimension('time', 1968)
lat = ps_ic.createDimension('lat', 776)
lon = ps_ic.createDimension('lon', 128)

times = ps_ic.createVariable('time', np.float32, ('time',))
lats = ps_ic.createVariable('lat', np.float32, ('lat',))
lons = ps_ic.createVariable('lon', np.float32, ('lon',))
value = ps_ic.createVariable('value', np.float32, ('time', 'lat', 'lon',))
value.units = 'm/s'
times[:] = ds_pressure.time[:]
lons[:] = ds_pressure.lon[:]
lats[:] = latit_real

# Spatial Interpolation using Cubic Spline:  
for i in range(0, 1968):
    ds = ds_pressure.isel(time=i)
    values = ds.ps
    value[i,:,:] = sp.interpolate.griddata(ds_ws.lat[:], values, latit_real, method = 'cubic')

ps_ic.close()

#%%

'''Winter mean NetCDF file'''

a = xr.open_dataset("/rds/general/user/.../home/.../ws_interpolated2.nc")
b = xr.open_dataset("/rds/general/user/.../home/.../ps_interpolated.nc")
ps_inter = b.to_array()[0, :, :, :]
ds_ua850 = a.to_array()[0, :, 1:, :] #the 0th latitude yields nan values

# Defining useful terms
lat_ws_D = np.empty([154, 775])  
lat_ws_J = np.empty([154, 775])  
lat_ws_F = np.empty([154, 775])
winter_speeds = np.empty([154, 775, 128])

jet_strength = np.empty([154])
jet_latitude = np.empty([154])
NAO_index = np.empty([154])
header_param = "Year\t\tLatitude\tStrength\tNAO_index"

path = "/rds/general/user/.../home/.../ws_winter.nc"
avg_ws = nc.Dataset(path, 'w', format = 'NETCDF4')
time = avg_ws.createDimension('time', 154)
lat = avg_ws.createDimension('lat', 775)
lon = avg_ws.createDimension('lon', 128)

times = avg_ws.createVariable('time', np.float32, ('time'))
lats = avg_ws.createVariable('lat', np.float32, ('lat'))
lons = avg_ws.createVariable('lon', np.float32, ('lon'))
value = avg_ws.createVariable('value', np.float32, ('time', 'lat', 'lon'))
value.units = 'm/s'
times[:] = np.arange(1861, 2015)
lats[:] = latit_real[1:]
lons[:] = ds_ws.lon[:]

#should take values from 300 (107) to 360 (128) deg i.e 0 - 60 W
lat_ws_D[:, :] = np.mean(ds_ua850[119:1967:12, :, 107:], axis = 2) #1967 because 1968 is December
lat_ws_J[:, :] = np.mean(ds_ua850[120:1968:12, :, 107:], axis = 2) #should be 154
lat_ws_F[:, :] = np.mean(ds_ua850[121:1968:12, :, 107:], axis = 2) #should be 154

for i in range(10, 164): # Number of years from 1861 to 2014 
    value[i-10, :, :] = np.mean(ds_ua850[(12*i-1):(12*i+2), :, :], axis = 0) # Zonal Mean over DJF; 
    azores_ps = ps_inter[(12*i-1):(12*i+2), 449, 119] # Azores DJF
    iceland_ps = ps_inter[(12*i-1):(12*i+2), 199, 121] # Iceland DJF
    NAO_index[i-10] = np.average(azores_ps - iceland_ps)/100 # Division by 100 to convert into hPa
    jet_strength[i-10] = np.average([max(lat_ws_D[i-10]), max(lat_ws_J[i-10]), max(lat_ws_F[i-10])])
    # 1 is added to the index because the first latitude in latit_real is not used in lat_ws_D
    jet_latitude[i-10] = np.average([latit_real[np.where(lat_ws_D[i-10] == max(lat_ws_D[i-10]))[0][0]+1], \
                                     latit_real[np.where(lat_ws_J[i-10] == max(lat_ws_J[i-10]))[0][0]+1], \
                                     latit_real[np.where(lat_ws_F[i-10] == max(lat_ws_F[i-10]))[0][0]+1]])
avg_ws.close()
    
params = np.column_stack([ds_ws.time[120:1957:12].dt.year, jet_latitude, jet_strength, NAO_index])
np.savetxt('parameters.txt', params, fmt = '%.5f', delimiter = '\t', header = header_param, comments = '')
print('Parameters are saved')

#%%

""" Regression Analysis """

c = xr.open_dataset("/rds/general/user/.../home/.../ws_winter.nc")
ds_avgws = c.to_array()[0, :, :, :]

years, jet_latitude, jet_strength, NAO_index = np.loadtxt("parameters.txt", skiprows = 1, unpack = True)
jet_strength_de = signal.detrend(jet_strength)
jet_latitude_de = signal.detrend(jet_latitude)
NAO_index_de = signal.detrend(NAO_index)

ws_transp = ds_avgws.transpose('lat', 'time', 'lon') #creating time series

# NAO Regression Slopes Dataset 
path = "/rds/general/user/.../home/.../nao_slope.nc"
nao_slope = nc.Dataset(path, 'w', format = 'NETCDF4')
lat = nao_slope.createDimension('lat', 775)
lon = nao_slope.createDimension('lon', 128)

lats = nao_slope.createVariable('lat', np.float32, ('lat'))
lons = nao_slope.createVariable('lon', np.float32, ('lon'))
value = nao_slope.createVariable('value', np.float32, ('lat', 'lon'))
value.units = 'm/s/hPa'
lats[:] = latit_real[1:]
lons[:] = ds_ws.lon[:]

for i in range(0, 775):
    value[i, :] = np.array(np.polyfit(NAO_index_de, ws_transp[i, :, :], 1)[0, :]) #slopes
nao_slope.close()

# Jet Latitude Regression Slopes Dataset 
path = "/rds/general/user/.../home/.../lat_slope.nc"
lat_slope = nc.Dataset(path, 'w', format = 'NETCDF4')
lat = lat_slope.createDimension('lat', 775)
lon = lat_slope.createDimension('lon', 128)

lats = lat_slope.createVariable('lat', np.float32, ('lat'))
lons = lat_slope.createVariable('lon', np.float32, ('lon'))
value = lat_slope.createVariable('value', np.float32, ('lat', 'lon'))
value.units = 'deg'
lats[:] = latit_real[1:]
lons[:] = ds_ws.lon[:]

for i in range(0, 775):
    value[i, :] = np.array(np.polyfit(jet_latitude_de, ws_transp[i, :, :], 1)[0, :])
lat_slope.close()

# Jet Strength Regression Slopes Dataset
path = "/rds/general/user/.../home/.../str_slope.nc"
str_slope = nc.Dataset(path, 'w', format = 'NETCDF4')
lat = str_slope.createDimension('lat', 775)
lon = str_slope.createDimension('lon', 128)

lats = str_slope.createVariable('lat', np.float32, ('lat'))
lons = str_slope.createVariable('lon', np.float32, ('lon'))
value = str_slope.createVariable('value', np.float32, ('lat', 'lon'))
value.units = 'm/s/deg'
lats[:] = latit_real[1:]
lons[:] = ds_ws.lon[:]

for i in range(0, 775):
    value[i, :] = np.array(np.polyfit(jet_strength_de, ws_transp[i, :, :], 1)[0, :])
str_slope.close()

print("Slopes Are Saved")

# Averaging ws_winter Time Series
path = "/rds/general/user/.../home/.../ua850_contour.nc"
cont = nc.Dataset(path, 'w', format = 'NETCDF4')
lat = cont.createDimension('lat', 775)
lon = cont.createDimension('lon', 128)

lats = cont.createVariable('lat', np.float32, ('lat'))
lons = cont.createVariable('lon', np.float32, ('lon'))
value = cont.createVariable('value', np.float32, ('lat', 'lon'))
value.units = 'm/s'
lats[:] = latit_real[1:]
lons[:] = ds_ws.lon[:]
value[:, :] = np.mean(ds_avgws[:, :, :], axis = 0)

cont.close()
print("Speeds are Time Averaged")

#%%

""" Plotting results """

# conda install basemap
os.environ["PROJ_LIB"] = "/rds/general/user/.../home/anaconda3/envs/test1/share/proj"

ds_cont = xr.open_dataset("/rds/general/user/.../home/.../ua850_contour.nc")
ds_nao = xr.open_dataset("/rds/general/user/.../home/.../nao_slope.nc")
ds_lat = xr.open_dataset("/rds/general/user/.../home/.../lat_slope.nc")
ds_str = xr.open_dataset("/rds/general/user/.../home/.../str_slope.nc")

# projection, lat/lon extents and resolution of polygons to draw
# resolutions: c - crude, l - low, i - intermediate, h - high, f - full
map = Basemap(projection = 'cyl', llcrnrlon = -100., llcrnrlat = 1.39, \
              urcrnrlon = 50.0, urcrnrlat = 90., resolution = 'l') 
map.drawcoastlines(linewidth = 0.4)
map.drawparallels(np.arange(10, 81, 10), labels = [1,0,0,0], linewidth = 0.3)
map.drawmeridians(np.arange(-90, 46, 45), labels = [0,0,0,1], linewidth = 0.3)

ua = np.array(ds_cont.value)
clevs = np.array([-9, -6, -3, 0, 3, 6, 9])
nao = np.array(ds_nao.value)
nao_levels = np.arange(-0.6, 0.7, 0.1)
jet_strength = np.array(ds_str.value)
strength_levels = np.arange(-2.4, 2.8, 0.4) 
jet_latitude = np.array(ds_lat.value)
latitude_levels = np.arange(-1.2, 1.4, 0.2)

lines = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', 'solid']
cols = ['black', 'black', 'black', 'black', 'black', 'black', 'black']
width = [0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]
x, y = np.meshgrid(ds_ws.lon[:], latit_real[1:])

cs1 = map.contour(x, y, ua[:, :], clevs, colors = cols, linestyles = lines, linewidths = width, latlon = 1)
plt.clabel(cs1, fontsize=10, inline=2, fmt = '%.f')
cs2 = map.contourf(x, y, nao[:, :], nao_levels, cmap = 'bwr', latlon = 1)
#cs3 = map.contourf(x, y, jet_strength[:, :], strength_levels, cmap = 'bwr', latlon = 1) # Jet Strength
#cs4 = map.contourf(x, y, jet_latitude[:, :], latitude_levels, cmap = 'bwr', latlon = 1) # Jet Latitude
cbar = map.colorbar(cs2, 'bottom')

plt.title('Regression slope of winter mean UA850 on NAO index')
cbar.set_label('m/s/hPa')
plt.savefig("ua850_nao.png", dpi = 250)

#plt.title('Regression slope of winter mean UA850 on jet latitude')
#cbar.set_label('m/s/$\deg$')
#plt.savefig("ua850_latitude0-60.png", dpi = 250)

#plt.title('Regression slope of winter mean UA850 on jet strength')
#plt.savefig("ua850_strength0-60.png", dpi = 250)
#colors = 'coolwarm'
