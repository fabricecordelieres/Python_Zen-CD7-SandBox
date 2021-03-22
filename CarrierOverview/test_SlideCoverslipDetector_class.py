from CarrierOverview import CarrierOverview as co
from SlideCoverslipDetector import SlideCoverslipDetector as sc

input_dir = './datasets/'
filename = 'CarrierOverview.czi'
co_obj = co(input_dir, filename)
sc_obj = sc(co_obj)
#sc_obj.test_detect()
#sc_obj.find_containers() # Options are available: width, height and tolerance
#print(sc_obj.find_circular_coverslips(limit_to_area=(250, 0, 1250, 2030))) # Options are available: diameter and tolerance
#sc_obj.find_circular_coverslips(diameter= 20, tolerance=.2)
#sc_obj.find_circular_coverslips_in_containers(diameter_coverslip= 20, tolerance_coverslip=.2)

sc_obj.find_circular_coverslips_in_containers()

sc_obj.show()

