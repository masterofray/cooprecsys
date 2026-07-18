#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-13"

#import sys
from pathlib  import Path

LocDir = Path(__file__).resolve().parents[2] / 'src'
#sys.path.append(str(LocDir))
from src.models import InferenceTest

def inference_test():
    Args      = {'Datapath'  : LocDir.parents[0]/'data'/'sampledata.parquet',
                 'configpath': LocDir/'configs'/'configuration.ini',
                 'QueryID'   : 'CustomerID',
                 'LabelID'   : 'CategoryID',
                 'FilterDF'  : ['CustomerID', 'ProductName', 'Class', 
                                'Resistant', 'IsAllergic', 'ProductPrice', 
                                'Quantity', 'Discount','TotalPrice', 
                                'relevance_score', 'rank', 'is_fallback'],
                 'odir'      : LocDir.parent/'artifacts'}
    TheResult = InferenceTest(**Args)
    print(TheResult.head())
    return TheResult


if __name__ == '__main__':
    inference_test()