import pickle
import os

def find_roi_files(root_dir):
    string_list = ['.pkl', 'roi']
    
    matching_files = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if all(string in filename.lower() for string in string_list):
                matching_files.append(os.path.join(root, filename))
    return matching_files

# Project codes for labeling ROI data
# **IMPORTANT**: ROIs should be named using the **same** naming convention used to label ASD files 

plant_codes = {
    'Ammo_bre': ['Ammophila', 'breviligulata', 'American Beachgrass', 'grass', 'https://en.wikipedia.org/wiki/Ammophila_breviligulata'],
    'Chas_lat': ['Chasmanthium', 'latifolium', 'River Oats', 'grass', 'https://en.wikipedia.org/wiki/Chasmanthium_latifolium'],
    'Pani_ama': ['Panicum', 'amarum', 'Coastal Panic Grass', 'grass', 'https://en.wikipedia.org/wiki/Panicum_amarum'],
    'Pani_vir': ['Panicum', 'virgatum', 'Switch Grass', 'grass', 'https://en.wikipedia.org/wiki/Panicum_virgatum'],
    'Soli_sem': ['Solidago', 'sempervirens', 'Seaside Goldenrod', 'succulent', 'https://en.wikipedia.org/wiki/Chasmanthium_latifolium'],
    'Robi_his': ['Robinia', 'hispida', 'Bristly locust', 'shrub', 'https://en.wikipedia.org/wiki/Robinia_hispida'],
    'More_pen': ['Morella', 'pennsylvanica', 'Bristly locust', 'shrub', 'https://en.wikipedia.org/wiki/Myrica_pensylvanica'],    
    'Rosa_rug': ['Rosa', 'rugosa', 'Sandy Beach Rose', 'shrub', 'https://en.wikipedia.org/wiki/Rosa_rugosa'],
    'Cham_fas': ['Chamaecrista', 'fasciculata', 'Partridge Pea', 'legume', 'https://en.wikipedia.org/wiki/Chamaecrista_fasciculata'],
    'Soli_rug': ['Solidago', 'rugosa', 'Wrinkleleaf goldenrod', 'shrub', 'https://en.wikipedia.org/wiki/Solidago_rugosa'],
    'Bacc_hal': ['Baccharis', 'halimifolia', 'Groundseltree', 'shrub', 'https://en.wikipedia.org/wiki/Baccharis_halimifolia'],
    'Iva_fru_': ['Iva', 'frutescens', 'Jesuits Bark ', 'shrub', 'https://en.wikipedia.org/wiki/Iva_frutescens'],
    'Ilex_vom': ['Ilex', 'vomitoria', 'Yaupon Holly', 'evergreen shrub', 'https://en.wikipedia.org/wiki/Ilex_vomitoria'],
    'Genus_spe': ['Genus', 'species', 'vegetation', 'background', '']
}  
age_codes = {  
    'PE': ['Post Germination Emergence', 'PE'],
	#'RE': ['Re-emergence', 'RE'],
    #'RE': ['Year 1 growth', '1G'],
	#'E': ['Emergence (from seed)', 'E'],
    'E': ['Post Germination Emergence', 'PE'],
	#'D': ['Dormant', 'D'],
	'1G': ['Year 1 growth', '1G'],
    '2G': ['Year 2 growth', '2G'],
	#'1F': ['Year 1 Flowering', '1F'],
    'J': ['Juvenile', 'J'],
	'M': ['Mature', 'M']
}
principal_part_codes = {  
    'MX': ['Mix', 'MX'],
    #'S': ['Seed', 'SE'],
	#'SA': ['Shoot Apex', 'SA'],
    'SA': ['Internode Stem', 'ST'],
	'L': ['Leaf/Blade', 'L'],
	#'IS': ['Internode Stem', 'IS'],
    'ST': ['Internode Stem', 'ST'],
    'SP': ['Sprout', 'SP'],
	#'CS': ['Colar Sprout', 'CS'],
    'CS': ['Sprout', 'SP'],
	#'RS': ['Root Sprout', 'RS'],
    'RS': ['Sprout', 'SP'],
	'LG': ['Lignin', 'LG'],
	'FL': ['Flower', 'FL'],
    #'B': ['Blade', 'B'],
	'B': ['Leaf/Blade', 'L'],
    'FR': ['Fruit', 'FR'],
	#'S': ['Seed', 'SE'], #moved above because 'S' is in other codes; this is an old code
    'SE': ['Seed', 'SE'],
	#'St': ['Stalk', 'St']
}
health_codes = {
    'MH': ['Healthy/Unhealthy Mix', 'MH'],
	'DS': ['Drought Stress', 'DS'],
	'SS': ['Salt Stress (soak)', 'SS'],
    'SY': ['Salt Stress (spray)', 'SY'],
	'S': ['Stressed', 'S'],
    'LLRZ': ['LLRZ Lab Stress', 'LLRZ'],
	#'D': ['Dormant', 'D'],
    'R': ['Rust', 'R'],
    'H': ['Healthy', 'H']
}

lifecycle_codes = { 
	'D': ['Dormant', 'D'],
    'RE': ['Re-emergence', 'RE'],
    'FLG': ['Flowering', 'FLG'],
    'FRG': ['Fruiting', 'FRG'],
    "FFG": ['Fruiting and Flowering', 'FFG'],
    'N': ['Neither', 'N']
}