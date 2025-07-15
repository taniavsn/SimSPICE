from sunraster.instr.spice import read_spice_l2_fits # type: ignore
import os
import numpy as np
from astropy.io import fits
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings 
warnings.filterwarnings("ignore")


class Sprout_ML:
    ''' 
    The SPROUTS data consist of a file with 8 windows: 'O III 703 / Mg IX 706 (Merged)', 'O II 718 - Peak (spectral bin 2)', 'S IV 750/ Mg IX (spectral bin 2)', 'Ne VIII 770 (Merged)', 
    'S V 786 / O IV 787 (Merged)', 'Ly-gamma-CIII group (Merged)', 'N III 991 (Merged)' and 'O VI 1032 (Merged)'

    Each of those windows is a 'raster', i.e. a 3D data cube with 2 spatial dimensions and 1 spectral dimension (we disregard the time dimension). 
    In other words, each raster is a 2D image where each pixel is a spectrum.
    '''
    #min and max peak half width (sigma)
    MIN_WIDTH = 0.2 #Angstrom
    MAX_WIDTH = 0.8 #Angstrom # prev:2
    LINES = ['O III 702', 'O III 703', 'Mg IX 706', 'O II 718', 'Mg IX 749', 'S IV 750','Ne VIII 770', 'Mg VIII 772', 
        'S V 786', 'O IV 787', 'O IV 790', 'Ly-gamma 972', 'C III 977', 'O I / Na VI', 'N III 989', 'N III 991', 'O VI 1032']

    SIZE_CROPPED_MAP = 116160
    SHAPE_CROPPED_MAP = (605,192)


    def __init__(self, dir, filename, padding_size=10):
        self.filename = os.path.join(dir,filename)
        self.fitsfile = filename
        self.exposure = self.get_exposure(dir+filename)
        self.padding_size = padding_size
        self.common_wvl = np.array([ 698.6 ,  698.79,  698.98,  699.17,  699.36,  699.55,  699.74,
        699.93,  700.12,  700.31,  700.5 ,  700.69,  700.88,  701.07,
        701.26,  701.45,  701.64,  701.83,  702.02,  702.21,  702.4 ,
        702.59,  702.78,  702.97,  703.16,  703.35,  703.54,  703.73,
        703.92,  704.11,  704.3 ,  704.49,  704.68,  704.87,  705.06,
        705.25,  705.44,  705.63,  705.82,  706.01,  706.2 ,  706.39,
        706.58,  706.77,  706.96,  707.15,  707.34,  707.53,  707.72,
        707.91,  708.1 ,  708.29,  708.48,  708.67,  708.86,  709.05,
        709.24,  709.43,  709.62,  709.81,  714.3 ,  714.49,  714.68,
        714.87,  715.06,  715.25,  715.44,  715.63,  715.82,  716.01,
        716.2 ,  716.39,  716.58,  716.77,  716.96,  717.15,  717.34,
        717.53,  717.72,  717.91,  718.1 ,  718.29,  718.48,  718.67,
        718.86,  719.05,  719.24,  719.43,  719.62,  719.81,  720.  ,
        720.19,  720.38,  720.57,  720.76,  720.95,  721.14,  721.33,
        721.52,  721.71,  721.9 ,  722.09,  722.28,  722.47,  722.66,
        745.9 ,  746.09,  746.28,  746.47,  746.66,  746.85,  747.04,
        747.23,  747.42,  747.61,  747.8 ,  747.99,  748.18,  748.37,
        748.56,  748.75,  748.94,  749.13,  749.32,  749.51,  749.7 ,
        749.89,  750.08,  750.27,  750.46,  750.65,  750.84,  751.03,
        751.22,  751.41,  751.6 ,  751.79,  751.98,  752.17,  752.36,
        752.55,  752.74,  752.93,  753.12,  753.31,  753.5 ,  753.69,
        753.88,  754.07,  754.26,  764.7 ,  764.89,  765.08,  765.27,
        765.46,  765.65,  765.84,  766.03,  766.22,  766.41,  766.6 ,
        766.79,  766.98,  767.17,  767.36,  767.55,  767.74,  767.93,
        768.12,  768.31,  768.5 ,  768.69,  768.88,  769.07,  769.26,
        769.45,  769.64,  769.83,  770.02,  770.21,  770.4 ,  770.59,
        770.78,  770.97,  771.16,  771.35,  771.54,  771.73,  771.92,
        772.11,  772.3 ,  772.49,  772.68,  772.87,  773.06,  773.25,
        773.44,  773.63,  773.82,  774.01,  774.2 ,  774.39,  774.58,
        774.77,  774.96,  775.15,  775.34,  775.53,  775.72,  775.91,
        776.1 ,  781.6 ,  781.79,  781.98,  782.17,  782.36,  782.55,
        782.74,  782.93,  783.12,  783.31,  783.5 ,  783.69,  783.88,
        784.07,  784.26,  784.45,  784.64,  784.83,  785.02,  785.21,
        785.4 ,  785.59,  785.78,  785.97,  786.16,  786.35,  786.54,
        786.73,  786.92,  787.11,  787.3 ,  787.49,  787.68,  787.87,
        788.06,  788.25,  788.44,  788.63,  788.82,  789.01,  789.2 ,
        789.39,  789.58,  789.77,  789.96,  790.15,  790.34,  790.53,
        790.72,  790.91,  791.1 ,  791.29,  791.48,  791.67,  791.86,
        792.05,  792.24,  792.43,  792.62,  792.81,  966.9 ,  967.09,
        967.28,  967.47,  967.66,  967.85,  968.04,  968.23,  968.42,
        968.61,  968.8 ,  968.99,  969.18,  969.37,  969.56,  969.75,
        969.94,  970.13,  970.32,  970.51,  970.7 ,  970.89,  971.08,
        971.27,  971.46,  971.65,  971.84,  972.03,  972.22,  972.41,
        972.6 ,  972.79,  972.98,  973.17,  973.36,  973.55,  973.74,
        973.93,  974.12,  974.31,  974.5 ,  974.69,  974.88,  975.07,
        975.26,  975.45,  975.64,  975.83,  976.02,  976.21,  976.4 ,
        976.59,  976.78,  976.97,  977.16,  977.35,  977.54,  977.73,
        977.92,  978.11,  978.3 ,  978.49,  978.68,  978.87,  979.06,
        979.25,  979.44,  979.63,  979.82,  980.01,  980.2 ,  980.39,
        980.58,  980.77,  980.96,  981.15,  981.34,  981.53,  981.72,
        981.91,  985.8 ,  985.99,  986.18,  986.37,  986.56,  986.75,
        986.94,  987.13,  987.32,  987.51,  987.7 ,  987.89,  988.08,
        988.27,  988.46,  988.65,  988.84,  989.03,  989.22,  989.41,
        989.6 ,  989.79,  989.98,  990.17,  990.36,  990.55,  990.74,
        990.93,  991.12,  991.31,  991.5 ,  991.69,  991.88,  992.07,
        992.26,  992.45,  992.64,  992.83,  993.02,  993.21,  993.4 ,
        993.59,  993.78,  993.97,  994.16,  994.35,  994.54,  994.73,
        994.92,  995.11, 1027.2 , 1027.39, 1027.58, 1027.77, 1027.96,
       1028.15, 1028.34, 1028.53, 1028.72, 1028.91, 1029.1 , 1029.29,
       1029.48, 1029.67, 1029.86, 1030.05, 1030.24, 1030.43, 1030.62,
       1030.81, 1031.  , 1031.19, 1031.38, 1031.57, 1031.76, 1031.95,
       1032.14, 1032.33, 1032.52, 1032.71, 1032.9 , 1033.09, 1033.28,
       1033.47, 1033.66, 1033.85, 1034.04, 1034.23, 1034.42, 1034.61,
       1034.8 , 1034.99, 1035.18, 1035.37, 1035.56, 1035.75, 1035.94,
       1036.13, 1036.32, 1036.51])
    
    def get_exposure(self, filename):
        '''Open the file and get the data'''
        return read_spice_l2_fits(filename, memmap=False)
 
    def get_keys(self):
        '''Prints the different windows of the Sprout file'''
        return list(self.exposure.keys())
    
    def get_nbr_keys(self):
        '''Get the number of windows in the file'''
        return len(self.get_keys())
    
    def get_rasters(self):
        '''Returns a list of the rasters present in the file'''
        rasters = []
        for key in self.get_keys():
            rasters.append(self.exposure[key])
        return rasters
    
    def get_wcs(self, bindown=False):
        ''' returns a list of wcs corresponding to the rasters present in the file'''
        rasters = self.get_rasters()
        wcss = [copy.deepcopy(r.wcs) for r in rasters]  # Make a deep copy to avoid modifying original wcs
        if bindown:
            for x in range(len(wcss)):
                wcss[x].wcs.crpix[1] = (wcss[x].wcs.crpix[1]/2)
                wcss[x].wcs.cdelt[1] = (wcss[x].wcs.cdelt[1]*2)
        return wcss
    
    def get_cubes(self, croplatbottom=725, croplattop=120):
        '''Returns a list of the actual numerical data of the rasters present in the file in W/m**2/nm/str'''
        cubes = []
        for key in self.get_keys():
            # indexation starts at the top therefore top to bottom
            cubes.append(np.nan_to_num(self.exposure[key][0, :, croplattop:croplatbottom, :].data)) 
        return cubes
    
    def get_cubes_uncropped(self):
        '''Returns a list of the actual numerical data of the rasters present in the file'''
        cubes = []
        for key in self.get_keys():
            cubes.append(np.nan_to_num(self.exposure[key][0,:,:,:].data))
        return cubes
    
    def whatsgoingon(self, key='Ne VIII 770 (Merged)', slice=20, 
                     croplattop=None, croplatbottom=None):
        
        cube = self.exposure[key][0, :, croplattop:croplatbottom, :].data
        wcs = self.get_wcs()[6]
        ax = plt.subplot(projection=wcs, slices=('x', 'y', 20, 1))
        im = ax.imshow((cube[slice, :, :])**(1/2.2), aspect=1/4, 
                       cmap='gist_heat', vmin=0)
        plt.colorbar(im, ax=ax)
        
        # Remove axis labels and ticks for the "Time" axis
        ax.coords[3].set_ticks_visible(False)  # Hide ticks on the "time" axis
        ax.coords[3].set_ticklabel_visible(False)  # Hide tick labels on the "time" axis
        ax.coords[3].set_axislabel(' ')  # Remove the "time" axis label
        
        # Optionally, remove x and y axis labels if needed
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')

        datetime_str = (self.fitsfile).split('_')[3]
        date = datetime_str[:8]  
        time = datetime_str[9:] 
        plt.title(f"{date[:4]}-{date[4:6]}-{date[6:]} T {time[:2]}:{time[2:4]}:{time[4:]}")
        #plt.show()

    def get_file_info(self):
        '''Get info from the header of the file'''
        hdul = fits.open(self.filename)
        print(hdul.info(),'\n----------------------------------------')
        rast = self.get_rasters()[0]
        print(rast,'\n----------------------------------------')
        print(rast.wcs)
        hdul.close()

    def get_raster_dimensions(self):
        print(self.filename, ' : ', self.exposure[self.get_keys()[0]].data.shape)
    
    def get_file_time(self):
        print(self.get_rasters()[0])

    def get_wvl_arrays_A(self):
        wavelengths_arrays = []
        rasters = self.get_rasters()
        for raster in rasters:
            wavelengths_arrays.append(raster.spectral_axis.value*(10**10)) # turn to Angstrom
        return wavelengths_arrays

    def pad_wvl_arrays(self):
        ''' 
        Resolution of each spectral window : 0.19 A
        Pad each wavelength array by 20 values left and right, i.e. by about 3.8 Angstrom

        Returns a list (usually len 8) of arrays containing the padded wvl arrays
        '''
        wvlarrays_padded = []
        wavelength_arrays = self.get_wvl_arrays_A()
        for array in wavelength_arrays:
            step_size = array[11]-array[10]
            left_pad_wvl = array[0] - step_size * np.arange(self.padding_size, 0, -1) 
            right_pad_wvl = array[-1] + step_size * np.arange(1, self.padding_size+1) 
            padded_wavelengths = np.concatenate([left_pad_wvl, array, right_pad_wvl])
            wvlarrays_padded.append(padded_wavelengths)
            
        return wvlarrays_padded

    def pad_flux_array(self, method='zeros'):
        ''' 
        flux_arrays = list
                list of datacubes containing the spectra
        method: str or None
                Way to fill the flux values. 
                If median, the median value of the spectrum is used, NaNs if None.
        size = int
                number of added data points on each side of the spectrum

        Returns flux_arrays_padded, masks where:
            - flux_arrays_padded is a list of all cubes with each spectrum padded.
            - masks is a list of arrays with the same shape as flux_arrays_padded, 
            containing 1 for original values and 0 for padding.
        '''
        flux_arrays_padded = []
        masks = []
        flux_arrays = self.get_cubes()
        
        for flux in flux_arrays:  # flux shape should be (x, 610, 192)
            padded_spectra = np.zeros((flux.shape[0] + 2 * self.padding_size, flux.shape[1], flux.shape[2]))
            mask = np.zeros_like(padded_spectra, dtype=int)  # Mask array with 0 for padding
            
            for i in range(flux.shape[1]):
                for j in range(flux.shape[2]):
                    # Create padding based on the method
                    tab = np.linspace(0, 1, self.padding_size)
                    if method == 'median':
                        pad_flux = np.full_like(tab, np.nanmedian(flux[:, i, j]))
                    elif method == 'zeros':
                        pad_flux = np.zeros_like(tab)
                    else:
                        pad_flux = np.full_like(tab, np.nan)
                    
                    # Concatenate the padded values with the actual spectrum
                    padded_spectrum = np.concatenate([pad_flux, flux[:, i, j], pad_flux])
                    padded_spectra[:, i, j] = padded_spectrum
                    
                    # Set the mask to 1 for the actual data range
                    mask[self.padding_size:self.padding_size + flux.shape[0], i, j] = 1

            flux_arrays_padded.append(padded_spectra)
            masks.append(mask)
        
        return flux_arrays_padded, masks


def interpolate_arrays(common_wvl, wvl_old, flux_array, mask_array=None):
    '''
    Returns a new array of interpolated values of flux on a common wavelength array.
    '''
    if mask_array is not None:
        f = interp1d(wvl_old, mask_array)
        mask_new = f(common_wvl)
        f = interp1d(wvl_old, flux_array)
        flux_new = f(common_wvl)
        return flux_new, mask_new
    else:    
        f = interp1d(wvl_old, flux_array)
        flux_new = f(common_wvl)
        return flux_new
    
