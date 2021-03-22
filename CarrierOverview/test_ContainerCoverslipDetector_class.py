from CarrierOverview import CarrierOverview as co
from ContainerCoverslipDetector import ContainerCoverslipDetector as cc

input_dir = './datasets/'
filename = 'CarrierOverview.czi'
co_obj = co(input_dir, filename)
cc_obj = cc(co_obj)
#cc_obj.test_detect()
#cc_obj.find_containers() # Options are available: width, height and tolerance
#print(cc_obj.find_circular_coverslips(limit_to_area=(250, 0, 1250, 2030))) # Options are available: diameter and tolerance
#cc_obj.find_circular_coverslips(diameter= 20, tolerance=.2)
#cc_obj.find_circular_coverslips_in_containers(diameter_coverslip= 20, tolerance_coverslip=.2)

cc_obj.find_circular_coverslips_in_containers()

cc_obj.show()

