'''
Create by Aryanto
at 20251225
email me : aryanto.dandan@gmail.com
'''

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC as Derived
from cryptography.hazmat.backends import default_backend
from getpass import getpass
import zipfile
import base64
import os

def ForKeys(password: str, salt: bytes) -> bytes:
    kdf = Derived(algorithm=hashes.SHA256(),
                length=32,
                salt = salt,
                iterations=100000,
                backend=default_backend(),
                )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key
    
def EncoderPass(message: str, password: str = None) -> str:
    if password is None:
        password = getpass("Password: ")
    salt = os.urandom(16)
    key = ForKeys(password, salt)
    f = Fernet(key)
    encrypted_data = f.encrypt(message.encode())
    Hasil = base64.urlsafe_b64encode(salt + encrypted_data).decode()
    return Hasil

def DecoderPass(Enkripsi: str, password: str = None) -> str:
    if password is None:
        password = getpass("Password: ")
    decoded_data = base64.urlsafe_b64decode(Enkripsi.encode())
    salt = decoded_data[:16]
    encrypted_data = decoded_data[16:]
    key = ForKeys(password, salt)
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    Hasil = decrypted_data.decode()
    return Hasil

if __name__ == '__main__':
    pass