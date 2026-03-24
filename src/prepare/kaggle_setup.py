import os
import json
from pathlib import Path

def KaggleSetup(username, Token):
    Success = False
    try :
        auth = {'username': username, 'key': Token}
        ConfDir = Path.home() / '.config' / 'kaggle'
        ConfDir.mkdir(parents=True, exist_ok=True)
        Dest = ConfDir / 'kaggle.json'
        with open(Dest, 'w') as thefile:
            json.dump(auth, thefile)
        os.chmod(ConfDir, 0o600)
        print(f'Kaggle API Dir: {ConfDir}')
        Success = True
    except Exception as Arr:
        print(f'Failed : {Arr}')
    finally:
        return Success

def KaggleDown(address : str = 'andrexibiza/grocery-sales-dataset'):
    Success = False
    try :
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f'Downloading {address} ...', end = '\r')
        api.dataset_download_files(address, path='.', unzip=True)
        print('Download Success')
        Success = True
    except Exception as Arr:
        print(f'Failed : {Arr}')
    finally:
        print('Files:', os.listdir('.'))
        return Success
