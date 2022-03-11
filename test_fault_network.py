import os, sys
import hazard.rshalib as rshalib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from castorlib import (read_fault_source_model, project_folder, gis_folder)

gis_filename = "Chile-faults.tab"
gis_filespec = os.path.join(gis_folder, gis_filename)

allow_triple_junctions = False
#rms = 2.85
rms = 1.7
src_model = read_fault_source_model(gis_filespec, characteristic=False)
for flt in src_model:
	flt.rupture_mesh_spacing = rms
	print(flt.source_id, flt.name, flt.get_mean_strike(), flt.get_length(), flt.get_length() / rms)

flt_network = src_model.get_fault_network(allow_triple_junctions=allow_triple_junctions)
sys.exit()
flt_network.check_consistency()
print(flt_network.fault_links['0#09'])

print(src_model.get_linked_subfaults([['1#02', '1#01', '12#29']]))

#connections = flt_network.get_connections('0#01', 25, allow_triple_junctions=allow_triple_junctions)
connections = flt_network.get_all_connections(66, allow_triple_junctions=allow_triple_junctions)
print(len(connections), max([len(conn) for conn in connections]))
faults = []
mags = {}
for conn in connections:
	if len(conn) == 3:
	#if '8' in [subflt_id.split('#')[0] for subflt_id in conn]:
		#print conn
		fault_model = src_model.get_linked_subfaults([conn])
		[flt] = fault_model.sources
		faults.append(flt)
		#mag = flt.calc_Mmax_Wells_Coppersmith()
		#num_sections = len(conn)
		#if not num_sections in mags:
		#	mags[num_sections] = mag

#pylab.plot(mags.keys(), mags.values())
#pylab.show()

#exit()
fault_model = rshalib.source.SourceModel("", faults)
map = fault_model.get_plot()
map.plot()
