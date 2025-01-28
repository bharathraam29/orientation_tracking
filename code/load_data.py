import pickle
import sys
import time 
import os

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

def load_dataset(dataset_idx):
    dataset=str(dataset_idx)
    cfile = "/home/bradhakrishnan/ECE276A_PR1/data/cam/cam" + dataset + ".p"
    ifile = "/home/bradhakrishnan/ECE276A_PR1/data/imu/imuRaw" + dataset + ".p"
    vfile = "/home/bradhakrishnan/ECE276A_PR1/data/vicon/viconRot" + dataset + ".p"
    imud=None
    vicd=None 
    camd=None
    
    ts = tic()
    if os.path.exists(cfile):
        camd = read_data(cfile)
    if os.path.exists(ifile):
        imud = read_data(ifile)
    if os.path.exists(vfile):
        vicd = read_data(vfile)
    toc(ts,"Data import")
    return imud, vicd, camd





