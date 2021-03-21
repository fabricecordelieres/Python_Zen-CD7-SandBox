from CarrierOverview import CarrierOverview as co
from SlideCoverslipDetector import SlideCoverslipDetector as sc

input_dir = './datasets/'
filename = 'CarrierOverview4.czi'
co_obj = co(input_dir, filename)
sc_obj = sc(co_obj)
sc_obj.test_detect()

