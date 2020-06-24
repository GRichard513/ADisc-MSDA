import urllib.request
import tarfile
import time
import os
import numpy as np

def download_amazon(output_path = './data/'):
    """
    Downloads unprocessed data for the Multi-Domain Sentiment Dataset
    Inputs:
        - output_path: path where to store the data
    """
    t0 = time.time()
    if not os.path.exists(output_path+'./unprocessed.tar.gz'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        print('Beginning file download with urllib2...')
        url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz'
        urllib.request.urlretrieve(url, output_path+'./unprocessed.tar.gz')
    else:
        print('Data already downloaded')
    t1 = time.time()
    print('Elapsed time: %.2f'%(t1-t0))
    return


def unzip_data(path='./data/'):
    """
    Unzips data
    Inputs:
        - path: path where data is stored
    """
    
    t0 = time.time()
    if not os.path.exists(path+'./sorted_data/'):
        print('Beginning unzipping...')
        fname = path+'unprocessed.tar.gz'
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(path)
            tar.close()
    else:
        print('Data already extracted')
    t1 = time.time()
    print('Elapsed time: %.2f s'%(t1-t0))
