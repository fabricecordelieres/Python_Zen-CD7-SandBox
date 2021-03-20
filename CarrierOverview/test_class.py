from CarrierOverview import CarrierOverview as co

dir = ''
filename = 'CarrierOverview.czi'
overview = co(dir, filename)

print('----- All metadata -----')
print(overview)
print('----- All metadata -----')

print('----- Positional data -----')
dimensions = ('X', 'Y')
# Build a header for the output table
output_string = 'Tile_Nb' + '\t'
for dimension in dimensions:
    items = (dimension + '_start', dimension + '_size', dimension + '_start_coordinate', dimension + '_stored_size',
             'Stage' + dimension + 'Position_mm')
    for item in items:
        output_string += item + '\t'
print(output_string)

# Log all values
for i in range(0, overview.metadata['SizeM']):
    output_string = str(i + 1) + '\t'
    for dimension in dimensions:
        items = (dimension + '_start', dimension + '_size', dimension + '_start_coordinate', dimension + '_stored_size',
                 'Stage' + dimension + 'Position_mm')
        for item in items:
            output_string += str(overview.metadata['Tile' + str(i) + '_' + item]) + '\t'
    print(output_string)

overview.save_image(dir, 'Output.tif')
overview.show('Ceci est un titre')
