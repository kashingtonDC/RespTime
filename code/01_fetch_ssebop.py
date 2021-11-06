import os
import requests
import subprocess
from tqdm import tqdm
from bs4 import BeautifulSoup

def list_files(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    zipfiles = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    detfiles = [x for x in zipfiles if 'det' in x]
    return detfiles

def download_file(file, data_dir):
    ext = os.path.split(file)[-1]
    outfn = os.path.join(data_dir,ext)
    if not os.path.exists(outfn):
        dl_cmd = 'curl -o {} {}'.format(outfn,file)
        os.system(dl_cmd)
        return outfn
    else:
        return outfn

def untar(file, data_dir):
    tar_cmd = 'tar -xvf {} -C {}'.format(file, data_dir)
    os.system(tar_cmd)
    return file

def clip(intif,outfn, shapefile = "/Users/aakashahamed/Desktop/RS_GW/shape/argus_grace.shp"):

    if not os.path.exists(outfn): # Dont write if already there 
        cmd = '''gdalwarp -cutline {} {} {}'''.format(shapefile,intif, outfn)
        print(cmd)
        os.system(cmd)

    return outfn 


def main():

    # main params
    url = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads/'
    ext = 'zip'
    data_dir = '../data/ssebop/raw'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # setup write dir and check if already existss
    dst_dir = "../data/ssebop/processed"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # List remote files on URL 
    remote_files = list_files(url, ext) 

    # Setup write files 
    processed_tifs = [os.path.join(dst_dir,os.path.split(x)[1].replace(".zip",'.tif')) for x in remote_files ]

    # main loop 
    for remote_file, processed_file in tqdm(list(zip(remote_files,processed_tifs))[:]):
        # check if exists 
        if os.path.exists(processed_file):
            print("already processed {} ..... skipping ======= ".format(os.path.split(processed_file)[1]))

        # else main processing routine 
        else:
            # download
            localfile = download_file(remote_file, data_dir)

            print(remote_file)
            print(localfile)

            # unzip 
            localzip = untar(localfile, data_dir)
            # clean up 
            xml_fn = os.path.splitext(localfile)[0] + ".xml" #.splitext(".zip")
            zip_fn = os.path.splitext(localfile)[0] + ".zip" #.splitext(".zip")
            for oldfile in [zip_fn, xml_fn]:
                if os.path.exists(oldfile):
                    os.remove(oldfile)

            # clip tif and write to processed dir 
            dst_dir = "../data/ssebop/processed"
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            print(localfile)
            print(os.path.split(localfile)[0])
            tif_fn = os.path.splitext(localfile)[0] + ".tif" #.splitext(".zip")
            final_tif = clip(tif_fn,processed_file)

            # remove the raw tif 
            if os.path.exists(tif_fn):
                os.remove(tif_fn)

            print("finished processing {} =======================================".format(final_tif))

    return 

    # # loop through list 
    # for file in tqdm(files[:]):

    #     ext = os.path.split(file)[-1]
    #     outfn = os.path.join(data_dir,ext)


    #     localfile = download_file(file, data_dir)
    #     tif_fn = os.path.splitext(localfile)[0] + ".tif" 
        
    #     # check if exists already, if so, skip
    #     if os.path.exists(tif_fn):
    #         print("already processed {} ..... skipping ======= ".format(tif_fn))
    #         continue
        
    #     else:
    #         localzip = untar(localfile, data_dir)
            

    #         # clean up 
    #         xml_fn = os.path.splitext(localfile)[0] + ".xml" #.splitext(".zip")
    #         zip_fn = os.path.splitext(localfile)[0] + ".zip" #.splitext(".zip")
    #         for oldfile in [zip_fn, xml_fn]:
    #             if os.path.exists(oldfile):
    #                 os.remove(oldfile)

    #         # clip tif and write to processed dir 
    #         dst_dir = "../data/ssebop/processed"
    #         if not os.path.exists(dst_dir):
    #             os.mkdir(dst_dir)
    #         tif_fn = os.path.splitext(localfile)[0] + ".tif" #.splitext(".zip")
    #         final_tif = clip(tif_fn,dst_dir)

    #         # remove the raw tif 
    #         os.remove(tif_fn)

    #         print("finished processing {} =======================================".format(final_tif))

    
if __name__ == '__main__':
    main()