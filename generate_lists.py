# Checks if the data_audit.txt file exists already and if not generates it

import os
import re
import pandas as pd
import numpy as np

# create parser
# parser = argparse.ArgumentParser()

# get the location of the running file as the defualt source location note for this to work the
# call is: python3 /home/shaun/Dropbox/projects/blackgrass_detection_GAN/black-grass_images/data_audit.py
source_dir = os.path.dirname(os.path.abspath("/mnt/yifan/data/blackgrass/blackgrass/blackgrass/"))
os.chdir(source_dir)

print('creating data_audit.txt')

# Make list of dictonary, append to list, then condense to data frame. this is many time faster than the alternatives
# when there is lots of data rows to append
row_list = []

# go into each directory, find the metadata file (.csv), parse that and save the file paths for each image file.
r = re.compile("^Field")
field_files = list(filter(r.match, os.listdir(source_dir)))

for field in field_files:

    field_dir = source_dir + '/' + field

    # look at each date folder in field
    field_dates = os.listdir(field_dir)

    for fd in field_dates:
        date_dir = field_dir + '/' + fd

        den_files = os.listdir(date_dir)
        # read in the .csv file at this level
        r = re.compile("\.csv$")
        metadata_file = list(filter(r.search, den_files))
        meta_dat = pd.read_csv(date_dir + '/' + metadata_file[0])

        # step through the density files
        for den in den_files:
            if re.match("^((?!csv).)*$", den):  # don't want to process the .csv file
                den_dir = date_dir + '/' + den

                # get the full list of files
                img_files = os.listdir(den_dir)

                # get red, green blue red_edge and NIR file names
                r = re.compile("^IMG_[0-9]+_3_")
                red_img = list(filter(r.match, img_files))

                r = re.compile("^IMG_[0-9]+_2_")
                green_img = list(filter(r.match, img_files))

                r = re.compile("^IMG_[0-9]+_1_")
                blue_img = list(filter(r.match, img_files))

                r = re.compile("^IMG_[0-9]+_5_")
                RE_img = list(filter(r.match, img_files))

                r = re.compile("^IMG_[0-9]+_4_")
                NIR_img = list(filter(r.match, img_files))

                # create dict for each row
                for j in range(len(red_img)):
                    dict1 = {'red': den_dir + '/' + red_img[j], 'green': den_dir + '/' + green_img[j],
                             'blue': den_dir + '/' + blue_img[j], 'red_edge': den_dir + '/' + RE_img[j],
                             'NIR': den_dir + '/' + NIR_img[j], 'field': meta_dat['Field_ID'][0],
                             'BG_class': den, 'season': meta_dat['season'][0], 'crop': meta_dat['crop_type'][0],
                             'soil': meta_dat['soil_type'][0]}
                    row_list.append(dict1)

df4 = pd.DataFrame(row_list,
                   columns=['red', 'green', 'blue', 'red_edge', 'NIR', 'field', 'BG_class', 'season', 'crop', 'soil'])

df4.to_csv(source_dir + '/data_table.txt', header=True, index=None, sep='\t')

