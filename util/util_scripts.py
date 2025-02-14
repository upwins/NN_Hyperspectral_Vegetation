import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from collections import Counter
import json

# Set the fontsize and create a function to colot plots be a specific metadata category
def set_color(x, colormap_name = 'gray'):
    # This function returns a dictionary of colors based on the input numpy x of metadata values
    # The default colormap is tab10 (tableau 10), but any named colormap can be used
    
    #categories = np.unique(x)
    categories = x

    if colormap_name == 'custom':
        color_list = ['r','g','b','c','m','y','maroon','orange',
              'darkgreen','peru','navy','teal',
              'indigo','crimson','pink','slategrey',
              'darkgoldenrod','purple']
        clr = {}
        for i,category in enumerate(categories):
            clr[category] = color_list[i]
    else:
        clr = {}
        cmap = matplotlib.colormaps.get_cmap(colormap_name)
        for i,category in enumerate(categories):
            clr[category] = cmap(i/len(categories))
    return clr

def sort_dict_by_list(my_dict, key_order):
    """Sorts a dictionary based on a given order of keys."""

    sorted_dict = {}
    for key in key_order:
        if key in my_dict:
            sorted_dict[key] = my_dict[key]
    return sorted_dict

class SpectralCollection():
    def __init__(self, df): 
        self.df = df
        self.names = df['ASD UPWINS base_fname'].to_numpy()
        self.fnames = df['ASD base_fname'].to_numpy()
        #print(df['spectrum'][0])
        self.wl = np.array(list(df['spectrum'][0].keys()), dtype=float)
        self.spectra = np.array([list(spectrum.values()) for spectrum in df['spectrum']])

        # remove spectra with large derivate, which are the spectra collected using sunlight
        #derivitive = np.mean(np.abs(self.spectra[:,1:]-self.spectra[:,:-1]), axis=1)
        #select_indices = np.where(derivitive<0.0025)[0]
        select_indices = np.arange(len(self.spectra))

        # create variables for the library data 
        self.names = self.names[select_indices]
        self.fnames = self.fnames[select_indices]
        self.spectra = self.spectra[select_indices,:]
        
        self.nSpec = len(self.names)
        self.nBands = len(self.wl)

        # Extract the metadata as numpy arrays
        self.genus = []
        self.species = []
        self.principle_part = []
        self.health = []
        self.age = []
        self.lifecycle = []
        self.DateTimeId = []
        self.DateTimeKeys = []
        self.month = []
        self.plant_type = []
        self.name = []
        self.name_full_category = []
        self.name_genus_species = []
        self.fname = []

        select_indices = []

        for i in range(len(df)):
            
            try:

                row = df.loc[df['ASD UPWINS base_fname'] == self.names[i]]

                # if the spectrum has a corresponding row in the csv
                if (row['sub-category'].values[0] not in ['NA','backrounds']) & (row['category'].values[0] in ['vegetation','target_vegetation','soil','road']):
                    select_indices.append(i)
                    
                    self.plant_type.append(row['sub-category'].values[0])
                    self.genus.append(row['genus'].values[0])
                    self.species.append(row['species'].values[0])
                    self.principle_part.append(row['principal_part'].values[0])
                    self.health.append(row['health'].values[0])
                    self.age.append(row['age'].values[0])
                    self.lifecycle.append(row['lifecycle'].values[0])
                    self.name_genus_species.append(row['genus'].values[0]+'_'+row['species'].values[0])
                    self.DateTimeId.append(row['DateTimeUniqueIdentifier'].values[0])
                    self.month.append(row['DateTimeUniqueIdentifier'].values[0][4:6])
                    self.DateTimeKeys.append(row['DateTimeUniqueIdentifier'].values[0][:11])
                    self.fname.append(row['ASD base_fname'].values[0])
                    
                    if row['genus'].values[0]=='NA':
                        #not in our primary target vegetation library
                        self.name.append(row['sub-category'].values[0])
                        self.name_full_category.append(row['sub-category'].values[0]+'_'+row['principal_part'].values[0]+'_'+row['health'].values[0]+'_'+row['age'].values[0])
                    else:
                        self.name.append(row['genus'].values[0]+'_'+row['species'].values[0])
                        self.name_full_category.append(row['genus'].values[0]+'_'+row['species'].values[0]+'_'+row['principal_part'].values[0]+'_'+row['health'].values[0]+'_'+row['age'].values[0])
            except:
                continue

        #print(len(select_indices))

        self.names = self.names[select_indices]
        self.fnames = self.fnames[select_indices]
        self.spectra = self.spectra[select_indices,:]
        # create variables for the library metadata 
        self.nSpec = len(self.names)
        self.nBands = len(self.wl)

        #print(f'Number of spectra: {nSpec}')
        #print(f'Number of bands: {nBands}')
        self.genus = np.asarray(self.genus)
        self.species = np.asarray(self.species)
        self.principle_part = np.asarray(self.principle_part)
        self.health = np.asarray(self.health)
        self.age = np.asarray(self.age)
        self.lifecycle = np.asarray(self.lifecycle)
        self.DateTimeId = np.asarray(self.DateTimeId)
        self.DateTimeKeys = np.asarray(self.DateTimeKeys)
        self.month = np.asarray(self.month)
        self.name = np.asarray(self.name)
        self.name_genus_species = np.asarray(self.name_genus_species)
        self.name_full_category = np.asarray(self.name_full_category)
        self.plant_type = np.asarray(self.plant_type)
        self.fname = np.asarray(self.fname)

        self.code_dict = {
            'name': (self.name, []),
            'fname': (self.fname, []),
            'full_name': (self.names, []),
            'genus': (self.genus, []),
            'species': (self.species, []),
            'age': (self.age, ['PE', '1G', '2G', 'J', 'M', 'N']),
            'health': (self.health, []),
            'part': (self.principle_part, ['MX', 'L', 'ST', 'SP', 'LG', 'FL', 'FR', 'SE']),
            'type': (self.plant_type, []),
            'lifecycle': (self.lifecycle, ['D', 'RE', 'FLG', 'FFG', 'FRG', 'N']),
            'date': (self.DateTimeKeys, [])
        }

        self.colors = {}
        codes_with_colors = ['age', 'part']
        
        for code in codes_with_colors:

            predefined_categories = self.code_dict[code][1]
            categories = np.unique(self.code_dict[code][0])

            sorted_categories = sorted(categories, key=lambda x: predefined_categories.index(x) if x in predefined_categories else len(predefined_categories) + list(categories).index(x))
            
            colors = set_color(sorted_categories)
            self.colors.update(colors)


    def plot_totals_each_species(self):
        # Count the occurrences of each unique value
        name_counts = Counter(self.name)

        #print(name_counts.items())
        #print(*name_counts.items())

        # Extract the labels and values
        labels, values = zip(*name_counts.items())
        # Create the horizontal bar plot
        plt.figure(figsize=(8, 6))
        bars = plt.barh(labels, values, color='skyblue')
        # Add text labels on the bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(value), ha='left', va='center')
        # Set the labels and title
        plt.ylabel('Species')
        plt.xlabel('Number of Occurrences')
        plt.title('Number of Occurrences of Each Species')
        plt.tight_layout()
        plt.show()

    def plot_mean_spectrum_each_species(self):

        plt.figure(figsize=(14,10))
        plt.grid(True)

        for species in np.unique(self.name):
            species_indices = np.where(self.name==species)[0]
            mean_spectrum = np.mean(self.spectra[species_indices], axis=0)

            plt.plot(self.wl, mean_spectrum, label=f'{species} ({len(species_indices)} spectra)', lw=1)
        
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Mean Spectrum of Each Species')
        plt.legend(bbox_to_anchor=(0.0, -0.05), loc='upper left', ncols=3)

        plt.show()

        return

    def df_principle_part_counts_each_species(self) -> pd.DataFrame:
        
       
        names = np.unique(self.name)
        names_indices = [np.where(self.name==name)[0] for name in names]

        #names_indices_array = np.array(indices)
        
        #for name, indices in zip(names, list_selected_indices):
        #    parts, part_counts = np.unique(self.principle_part[indices], return_counts=True)
        #    print(name, indices.size)
            
        #    for x, y in zip(parts, part_counts):
        #        print(x, y)
            #print(self.principle_part[name_indicies].shape)
            #parts, counts = np.unique(self.principle_part[name_indicies], return_counts=True)
            #print(np.asarray(parts, counts).T)


        part_names = np.unique(self.principle_part).tolist()
        part_totals_for_each_name = []

        for i, val in enumerate(names):

            dict_of_part_counts = dict(zip(part_names, [0] * len(part_names)))

            part_counts = Counter(self.principle_part[names_indices[i]])
            
            for key, value in part_counts.items():
                dict_of_part_counts[str(key)] = value
            
            part_totals_for_each_name.append(dict_of_part_counts)

        df = pd.DataFrame(part_totals_for_each_name, index=names)
        #df["Totals"] = [len(val) for i, val in enumerate(names_indices)]
        df.insert(0, "Totals", [len(name_indices) for name_indices in names_indices])

        return df

    def df_with_codes(self, codes, list_by) -> pd.DataFrame:
        
        #title = ""
        
        #title = title[:-2]

        vals = np.unique(self.code_dict[list_by][0])
        vals_totals = []
        count_totals = []

        for val in vals:
            selected_indices = np.arange(len(self.code_dict[list_by][0]))
            #selected_indices = np.where(self.code_dict[list_by][0]==val)[0]
            selected_indices = selected_indices[self.code_dict[list_by][0][selected_indices]==val]

            full_dict_of_counts = {}
            for code in codes:
                code_vals = np.unique(self.code_dict[code][0]).tolist()

                dict_of_counts = dict(zip(code_vals, [0] * len(code_vals)))
                counts = Counter(self.code_dict[code][0][selected_indices])
                
                for k, v in counts.items():
                    dict_of_counts[str(k)] = v
                
                sorted_dict_of_counts = {key: dict_of_counts[key] for key in self.code_dict[code][1] if key in dict_of_counts}
                
                if not sorted_dict_of_counts:
                    sorted_dict_of_counts = dict_of_counts
                   

                full_dict_of_counts.update(sorted_dict_of_counts)

            count_totals.append(full_dict_of_counts)
            vals_totals.append(len(selected_indices))


        df = pd.DataFrame(count_totals, index=vals)
        df.drop('N', axis=1, inplace=True)
        df.insert(0, "Total", vals_totals)
        # create a title for the df using
        #df.style.set_caption("")

        return df


    # fix variable references
    def plot_spectra_for_name_full_category_by_date(self, name_full_cat: str):
        selected_indices = np.where(self.name_full_category==name_full_cat)[0]
        #print(selected_indices)
        #print(DateTimeId[selected_indices])

        # Group DateTimeId based on the first 11 characters
        grouped_dates = defaultdict(list)
        for i, date_id in enumerate(self.DateTimeId[selected_indices]):
            group_key = date_id[:11]
            # Store the original index from selected_indices
            grouped_dates[group_key].append(selected_indices[i])

        # Sort the keys (date strings) chronologically
        sorted_keys = sorted(grouped_dates.keys())

        # Create a new dictionary with sorted keys
        sorted_grouped_dates = {key: grouped_dates[key] for key in sorted_keys}

        # Print the sorted grouped indices
        #for date_key, group in sorted_grouped_dates.items():
        #    print(f"Original Indices for group {date_key}: {group}")
        
        
        plt.figure(figsize=(14,10))
        plt.grid(True)

        clr = set_color(range(len(sorted_grouped_dates)), 'custom')


        for i, (group_name, group_indices) in enumerate(sorted_grouped_dates.items()):
            mean_spectrum = np.mean(self.spectra[group_indices,:], axis=0)
            plt.plot(self.wl, mean_spectrum, label=group_name, lw=1, c=clr[i])

        #for idx in selected_indices:
        #    plt.plot(wl, spectra[idx], label=DateTimeId[idx], lw=1, c=clr[DateTimeId[idx]])

        plt.xlabel('Wavelength')
        plt.ylabel('Refletance')
        plt.title(f'{name_full_cat} ({len(selected_indices)} Spectra)')
        plt.legend(bbox_to_anchor=(0.0, -0.05), loc='upper left', ncols=3)
    
    def select_indicies_with_filter(self, filter):
        selected_indices = np.arange(len(self.name))

        for k,v in filter.items():
            selected_indices = selected_indices[self.code_dict[k][0][selected_indices]==v]

        return selected_indices

    def plot_with_filter(self, filter, plotby, colormap='gray'):
        selected_indices = np.arange(len(self.name))
        title = ""

        for k,v in filter.items():
            selected_indices = selected_indices[self.code_dict[k][0][selected_indices]==v]
            title = title + k + '=' + v + ', '

        title = title[:-2]

        plt.figure(figsize=(14,10))
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        plt.grid(which='major', linestyle='-', linewidth=1.0)
        
        cmap = matplotlib.colormaps.get_cmap(colormap)

        if (plotby=='date'):

            # Group DateTimeId based on the first 11 characters
            grouped_dates = defaultdict(list)
            for i, date_id in enumerate(self.DateTimeId[selected_indices]):
                group_key = date_id[:8]
                # Store the original index from selected_indices
                grouped_dates[group_key].append(selected_indices[i])

            # Sort the keys (date strings) chronologically
            sorted_keys = sorted(grouped_dates.keys())

            # Create a new dictionary with sorted keys
            sorted_grouped_dates = {key: grouped_dates[key] for key in sorted_keys}
            
            for i, (group_name, group_indices) in enumerate(sorted_grouped_dates.items()):
                mean_spectrum = np.mean(self.spectra[group_indices], axis=0)
                plt.plot(self.wl, mean_spectrum, label=f'{group_name} ({len(group_indices)} spectra)', lw=1.3, c=cmap(i/len(sorted_grouped_dates)))

        else:
        
            unique_vals = np.unique(self.code_dict[plotby][0][selected_indices])
            unique_vals = list(unique_vals)
            
            if (len(self.code_dict[plotby][1]) > 0):
                #order = ['PE', '1G', '2G', 'J', 'M', 'D', 'N']
                #order = self.code_dict[plotby][1]
                #unique_vals = sorted(unique_vals, key=lambda x: order.index(x))

                predefined_categories = self.code_dict[plotby][1]
                unique_vals = sorted(unique_vals, key=lambda x: predefined_categories.index(x) if x in predefined_categories else len(predefined_categories) + unique_vals.index(x))


            for i, unique_val in enumerate(unique_vals):
                #indices = np.where(self.code_dict[plot_by][0][selected_indices]==unique_val)[0] #don't use; this doesn't work; bad indexing
                indices = selected_indices[self.code_dict[plotby][0][selected_indices]==unique_val]
                mean_spectrum = np.mean(self.spectra[indices], axis=0)

                plt.plot(self.wl, mean_spectrum, label=f'{unique_val} ({len(indices)} spectra)', lw=1.3, c=(self.colors[unique_val] if unique_val in self.colors else cmap(i/len(unique_vals))))

        # Plot visible wavelengths
        plt.axvline(x=450, color='b', linestyle=(0, (7, 10)), alpha=0.5)
        plt.axvline(x=550, color='g', linestyle=(0, (7, 10)), alpha=0.5)
        plt.axvline(x=650, color='r', linestyle=(0, (7, 10)), alpha=0.5)

        plt.annotate('Visible Spectrum', xy=(0.13, 1.01), xytext=(0.13, 1.02), xycoords='axes fraction', 
            fontsize=10, ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-[, widthB=4.0, lengthB=0.2', lw=1.0, color='k'), annotation_clip=False)

        # Plot key wavelengths
        a_val = 1200
        b_val = 1500
        c_val = 1950

        plt.axvline(x=a_val, color='0.3', linestyle=(0, (7, 10)))
        plt.axvline(x=b_val, color='0.3', linestyle=(0, (7, 10)))
        plt.axvline(x=c_val, color='0.3', linestyle=(0, (7, 10)))

        plt.annotate('A', xy=(a_val, 1), xytext=(a_val - 10, 1.01), xycoords=('data', 'axes fraction'),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))

        plt.annotate('B', xy=(b_val, 1), xytext=(b_val - 11, 1.01), xycoords=('data', 'axes fraction'),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))

        plt.annotate('C', xy=(c_val, 1), xytext=(c_val - 12, 1.01), xycoords=('data', 'axes fraction'),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f'Mean spectra by {plotby} of {title}', loc='left', pad=40, fontsize=14)
        #plt.legend(bbox_to_anchor=(0.0, -0.05), loc='upper left', ncols=3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncols=1, borderaxespad=0)

        plt.text(1.01, 0, 'A: Water, Cellulose,\n    Starch, Lignin\nB: Protein, Nitrogen\nC: Sugar, Starch', color='k', transform = plt.gca().transAxes)
        #        bbox=dict(facecolor='none', edgecolor='0.8', boxstyle='round'))

        plt.show()

        return