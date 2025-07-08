## map item on SPROUTS map  
from sunraster.instr.spice import read_spice_l2_fits # type: ignore
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr # type: ignore
import colorcet as cc # type: ignore

WAVELENGTHS_ARRAY = [ 698.6 ,  698.79,  698.98,  699.17,  699.36,  699.55,  699.74,
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
       1036.13, 1036.32, 1036.51]

SIZE_CROPPED_MAP = 116160
SHAPE_CROPPED_MAP = (605,192)

def plot_n_random_spectra_cluster(labels, stacked_outputs, chosen_cluster, dataset, nbr_items=5, plot_on_map=False, log_scale=True,
                                  dataset_path="C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train.nc"):
    '''
    Plots nbr_item belonging to a given cluster.
    dataset: SproutDataset object, with augmentation_type set to None.
    '''
    labels_reshaped = labels.reshape(-1, 1)
    print('labels shape: ', labels_reshaped.shape)
    outputs_with_labels = np.hstack((stacked_outputs, labels_reshaped))
    idxes = np.where(outputs_with_labels[:,-1] == chosen_cluster)[0]
    random_spectra_idx = np.random.choice(idxes, size=nbr_items, replace=False)
    print('Items selected : ', random_spectra_idx)
    for i in random_spectra_idx:
        plt.figure(figsize=[12,4], tight_layout=True)
        item = dataset.__getitem__(i)
        plt.plot(WAVELENGTHS_ARRAY, item[0].squeeze())
        if log_scale:
            plt.yscale('log')
        plt.title(f'Item number #{i}')
        plt.show()
    if plot_on_map:
        for i in random_spectra_idx:
            map_item_map(i, dataset_path)


def plot_average_spectra_cluster(labels, stacked_outputs, chosen_cluster, dataset, log_scale=True,
                                  dataset_path="C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train.nc"):
    '''
    Plots the average spectra of a given cluster. 
    dataset: SproutDataset object, with augmentation_type set to None.
    '''
    labels_reshaped = labels.reshape(-1, 1)
    print('labels shape: ', labels_reshaped.shape)
    outputs_with_labels = np.hstack((stacked_outputs, labels_reshaped))
    av_spectra = []
    for i in range(len(np.where(labels==chosen_cluster)[0])):
        item = dataset.__getitem__(i)
        av_spectra.append(item[0].squeeze())
        

    plt.figure(figsize=[12,4], tight_layout=True)
    plt.plot(WAVELENGTHS_ARRAY, np.nanmean(av_spectra, axis=0), label = 'mean spectrum')
    plt.plot(WAVELENGTHS_ARRAY, np.nanmedian(av_spectra, axis=0), alpha=0.5, label = 'median spectrum')
    if log_scale:
        plt.yscale('log')
    plt.title(f'Cluster #{chosen_cluster}')
    plt.legend()
    plt.show()
    return av_spectra


def map_clusters(labels, dataset_path="C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train.nc", selected_clusters = None, max_ticks=10):
    '''
    Maps the fits file according to the clusters determined by HDBscan
    '''
    import xarray as xr
    dataset = xr.open_dataset(dataset_path)
    nbr_files = int(len(dataset['index'])/SIZE_CROPPED_MAP)
    for x in range(nbr_files):
        current_labels = labels[SIZE_CROPPED_MAP * x : SIZE_CROPPED_MAP * (x + 1)].reshapeSHAPE_CROPPED_MAP
        if selected_clusters is not None:
            masked_labels = np.where(np.isin(current_labels, selected_clusters), current_labels, np.nan)
        else:
            masked_labels = current_labels

        unique_clusters = np.unique(current_labels[~np.isnan(current_labels)])
        cmap = get_cmap("cet_glasbey_bw", len(unique_clusters))
        print("Unique Clusters:", unique_clusters)

        img = plt.imshow(masked_labels, cmap=cmap, vmin = int(np.nanmin(masked_labels)), vmax=int(np.nanmax(masked_labels)), aspect=1 / 4)

        datetime_str = str(dataset.isel(index=SIZE_CROPPED_MAP*x+10)['filename'].data).split('_')[3]
        date = datetime_str[:8]  
        time = datetime_str[9:] 
        plt.title(f"{date[:4]}-{date[4:6]}-{date[6:]} T {time[:2]}:{time[2:4]}:{time[4:]}")

        if len(unique_clusters) > max_ticks:
            tick_indices = np.linspace(0, len(unique_clusters) - 1, max_ticks, dtype=int)
            tick_labels = unique_clusters[tick_indices]
        else:
            tick_labels = unique_clusters
        cbar = plt.colorbar(img, ticks=tick_labels)
        cbar.ax.set_yticklabels(tick_labels.astype(int))

        # plt.show()


def map_item_map(item_nbr:int, dataset:str="C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train.nc",
                 plot:bool=False, title:str=' ', nanquant=0.99,
                 data_dir:str='C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\data_L2\\', 
                 key:str='Ne VIII 770 (Merged)', croplatbottom:int=725, croplattop:int=120):
    '''
    Maps an item from the dataset onto its corresponding solar map.
    item_nbr: the index of the wanted item.
    dataset: the path and filename of the dataset (.nc) desired.
    data_dir: the location of SPICE's .fits files. 
    '''
    dataset = xr.open_dataset(dataset)
    if isinstance(item_nbr, int):
        filename = str(dataset.isel(index=item_nbr)['filename'].data)
        print(filename)
        
        exposure = read_spice_l2_fits(data_dir+filename, memmap=False)
        cube = exposure[key][0,:,croplattop:croplatbottom,:].data
        plt.imshow(cube[20, :, :], aspect=1/4, cmap='gist_heat', vmax=np.nanquantile(cube[20, :, :], nanquant))
        plt.colorbar()
        print(cube.shape)
        if plot:
            i,j = (dataset.isel(index=item_nbr)['x-index'].data, dataset.isel(index=item_nbr)['y-index'].data)
            plt.plot(j,i, color='red', marker='+', label=str(item_nbr))

    else:
        filenames = str(dataset.isel(index=item_nbr)['filename'].data)
        for x in range(len(filenames)):    
            print(item_nbr, ' -> ', filenames[x])
            x=0 
            try:
                for item_nbr in item_nbr: 
                    filename=filenames[x]
                    i_s = [dataset.isel(index=item_nbr)['x-index'].data for item_nbr in item_nbr]
                    j_s = [dataset.isel(index=item_nbr)['y-index'].data for item_nbr in item_nbr]
                    exposure = read_spice_l2_fits(data_dir+filename, memmap=False)
                    cube = exposure[key][0,:,croplattop:croplatbottom,:].data
                    plt.imshow(cube[20, :, :], aspect=1/4, cmap='gist_heat', vmax=np.nanquantile(cube[20, :, :], nanquant))
                    plt.colorbar()
                    print(cube.shape)
                    if plot:
                        for i, j in zip(i_s, j_s):    
                            plt.plot(j,i, color='red', marker='+', label=str(item_nbr))
            except:
                x+=1


    if title:
        plt.title(title)
    plt.legend()

