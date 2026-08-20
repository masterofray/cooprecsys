#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-21"


import os
import zipfile
from getpass import getpass
from ..configs import logger


def Unzip(Zips       : str,
          password   : str = None,
          DirExtract : str = '.',
         ) -> None:
    Zips = str(Zips)
    if not os.path.exists(Zips):
        logger.error(f'Error: Zip file not found at {Zips}')
        return None
    os.makedirs(DirExtract, exist_ok = True)
    Hasil = False
    try:
        with zipfile.ZipFile(Zips, 'r') as zf:
            try:
                zf.extractall(path = DirExtract, pwd = password.encode() if password else None)
            except RuntimeError as arc:
                if 'encrypted' in str(arc):
                    if password is None:
                        password = getpass('Masukkan password: ')
                zf.extractall(path = DirExtract, pwd = password.encode())
        logger.info(f'Successfully unzipped {Zips} to {DirExtract}/')
        Hasil = True

    except zipfile.BadZipFile:
        logger.error(f'Error: "{Zips}" is not a valid zip file or is corrupted.')

    except RuntimeError as err:
        logger.error(f'Error unzipping {Zips}: {err}')
        if 'Bad password' in str(err):
            logger.warning('Check password anda lagi: ')

    except Exception as arc:
        logger.error(f'An unexpected error occurred: {arc}.')

    finally:
        return Hasil
        
if __name__ == '__main__':
    pass