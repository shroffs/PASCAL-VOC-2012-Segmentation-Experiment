import tarfile

def extract_voc10(tar_file):
    '''Extracts tarfile to working directory.

    Args:
        Path to tarfile
    '''

    tar= tarfile.open(tar_file, 'r:')
    tar.extractall()

extract_voc10('VOCtrainval_11-May-2012.tar')
