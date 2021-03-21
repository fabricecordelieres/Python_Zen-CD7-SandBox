from CarrierOverview import CarrierOverview as co
from SlideCoverslipDetector import SlideCoverslipDetector as sc

input_dir = './datasets/'
filename = 'CarrierOverview.czi'
co_obj = co(input_dir, filename)
sc_obj = sc(co_obj)
#sc_obj.test_detect()
sc_obj.find_containers() # Options are available: width, height and tolerance
sc_obj.find_circular_coverslips() # Options are available: diameter and tolerance
sc_obj.show()

