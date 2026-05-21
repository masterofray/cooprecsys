'''
Create by Aryanto
at 20251225
email me : aryanto.dandan@gmail.com
'''

import os
import zipfile

def Unzip(Zips       : str,
          password   : str = None,
          DirExtract : str = '.',
         ) -> None:
    if not os.path.exists(Zips):
        print(f'Error: Zip file not found at {Zips}')
        return None

    if password is None:
        password = getpass('Masukkan password: ')
    os.makedirs(DirExtract, exist_ok = True)
    Hasil = False
    try:
        with zipfile.ZipFile(Zips, 'r') as zf:
            if len(str(password)):
                zf.extractall(path=DirExtract, pwd = password.encode())
            else:
                zf.extractall(path=DirExtract)
        print(f'Successfully unzipped {Zips} to {DirExtract}/')
        Hasil = True

    except zipfile.BadZipFile:
        print(f'Error: "{Zips}" is not a valid zip file or is corrupted.')

    except RuntimeError as err:
        print(f'Error unzipping {Zips}: {err}')
        if 'Bad password' in str(err):
            print('Check password anda lagi: ')

    except Exception as arc:
        print(f'An unexpected error occurred: {arc}.')

    finally:
        return Hasil
        
if __name__ == '__main__':
    pass