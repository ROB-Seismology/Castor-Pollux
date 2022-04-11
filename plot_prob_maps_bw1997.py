import os, sys
import numpy as np
import openquake.hazardlib as oqhazlib
#import mapping.layeredbasemap as lbm
#import hazard.rshalib as rshalib
from hazard.rshalib.source import SimpleUniformGridSourceModel
from hazard.rshalib.source_estimation import estimate_epicenter_location_and_magnitude_from_intensities

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from castorlib import (plot_gridsearch_map,
					   read_evidence_site_info_from_txt,
					   read_evidence_sites_from_gis,
					   TRT, USD, LSD, RAR, RMS,
					   project_folder, gis_folder,
					   data_points, fault_model,
					   base_fig_folder, watershed_file, subcatchments_file)


fig_folder = os.path.join(base_fig_folder)
if not os.path.exists(fig_folder):
	print('Figures folder does not exist')
	#os.mkdir(fig_folder)

output_format = 'png'

## Event names
event = "~4400 cal yrs BP"

## IPE and IMT
ipe_name = "BakunWentworth1997"
#ipe_name = "AtkinsonWald2007"
imt = oqhazlib.imt.MMI()

polygon_discretization = 2.5


## Construct grid source model
grid_outline = (-73.8, -71.2, -46.1, -44.9)
grid_spacing = 0.1

min_mag, max_mag, mag_bin_width = 4.5, 8.5, 0.2
depth = 10
strike, dip, rake = 20, 80, 180
point_msr = oqhazlib.scalerel.PointMSR()
wc1994_msr = oqhazlib.scalerel.WC1994()

grd_src_model = SimpleUniformGridSourceModel(grid_outline, grid_spacing,
			min_mag, max_mag + mag_bin_width, mag_bin_width, depth,
			strike, dip, rake, wc1994_msr, USD, LSD, RMS, RAR, TRT)
lon_grid, lat_grid = grd_src_model.lon_grid, grd_src_model.lat_grid



## Read MTD evidence
pe_site_models, ne_site_models = [], []
pe_thresholds, pe_sites, ne_thresholds, ne_sites = [], [], [], []

textfile = os.path.join(gis_folder, "%s.txt" % data_points)
(pe_thresholds, pe_site_models,
ne_thresholds, ne_site_models) = read_evidence_site_info_from_txt(textfile)

for pe_site_model, pe_threshold in zip(pe_site_models, pe_thresholds):
	print("+%s (n=%d): %s" % (pe_site_model.name.encode(errors='replace'), len(pe_site_model), pe_threshold))
for ne_site_model, ne_threshold in zip(ne_site_models, ne_thresholds):
	print("-%s (n=%d): %s" % (ne_site_model.name.encode(errors='replace'), len(ne_site_model), ne_threshold))

for pesm, pe_threshold in zip(pe_site_models, pe_thresholds):
	for pe_site in pesm.get_sites():
		pe_sites.append(pe_site)
		#print("+%s: %s" % (pe_site.name, pe_threshold))

for nesm, ne_threshold in zip(ne_site_models, ne_thresholds):
	for ne_site in nesm.get_sites():
		ne_sites.append(ne_site)
		#print("-%s: %s" % (ne_site.name, ne_threshold))

#print(sum([len(pesm) for pesm in pe_site_models]), sum([len(nesm) for nesm in ne_site_models]))
#print(len(pe_sites), len(ne_sites))
pe_thresholds = np.array(pe_thresholds)
ne_thresholds = np.array(ne_thresholds)

# Additional constraint: intensities must be 7.5 or higher in part of Aysén catchment
gis_file = os.path.join(gis_folder, subcatchments_file)
site_spacing = 1
# partial_pe_fraction aanpassen in functie van catchment en/of site_spacing verlagen
#subcatchments = ['Rio Simpson', 'Aysen Fjord', 'Esteros', 'Rio Blanco', 'Rio Condor', 'Rio Maninhuales', 'Los Palos']
subcatchments = ['Rio Simpson']
for subcatchment in subcatchments:
	print(subcatchment)
	partial_pe_site_model = read_evidence_sites_from_gis(gis_file, site_spacing, polygon_name=subcatchment)[0]
	partial_pe_sites = partial_pe_site_model.get_sites()
	#print('%s subcatchment: %d sites' % (subcatchment, len(partial_pe_sites)))
	partial_pe_thresholds = [7.5] * len(partial_pe_sites)
	partial_pe_fraction = 0.1
    		
	## Compute magnitudes and RMS errors at grid points
	method = 'probabilistic_highest'
	result = (estimate_epicenter_location_and_magnitude_from_intensities(
		ipe_name, imt, grd_src_model, pe_sites, pe_thresholds,
		ne_sites, ne_thresholds, method=method,
		partial_pe_sites=partial_pe_sites, partial_pe_intensities=partial_pe_thresholds,
		partial_pe_fraction=partial_pe_fraction, mag_pdf_idx=(list(np.around(grd_src_model.lats, 1)).index(-45.5), list(np.around(grd_src_model.lons, 1)).index(-72))))
	if method[:13] == 'probabilistic':
		fig_filespec = os.path.join(fig_folder, "%s_%s_%s+watershed_%s_maxprob.%s")
		fig_filespec %= (event, ipe_name, method, subcatchment, output_format)
		(mag_grid, rms_grid, mag_pdf) = result
		mag_pdf.plot(fig_filespec=fig_filespec)
	else:
		(mag_grid, rms_grid) = result
	
	#idx = np.unravel_index(rms_grid.argmin(), rms_grid.shape)
	#print(mag_grid[idx], lon_grid[idx], lat_grid[idx])
	
	rms_grid[np.isinf(rms_grid)] = 10.0
	
	## Plot map
	# TODO: blend alpha in function of rms
	# See: https://matplotlib.org/devdocs/gallery/images_contours_and_fields/image_transparency_blend.html
	text_box = "Event: %s\n" % event
	if 'probabilistic' in method:
		text_box += "P: %.2f - %.2f"
	else:
		text_box += "RMSE: %.2f - %.2f"
	try:
		text_box %= (rms_grid.min(), rms_grid[rms_grid < 10].max())
	except:
		pass
	
	rms_is_prob = ('probabilistic' in method)
	map = plot_gridsearch_map(grd_src_model, mag_grid, rms_grid,
							pe_site_models, ne_site_models,
							site_model_gis_file=None,
							text_box=text_box, catchment=subcatchment,
							plot_rms_as_alpha=False, rms_is_prob=rms_is_prob,
							plot_epicenter_as="area")
	
	
	#fig_filespec = os.path.join(fig_folder, "%s_bw1997_forward.%s" % (event, output_format))
	#fig_filespec = None
	fig_filespec = os.path.join(fig_folder, "%s_%s_%s+watershed_%s.%s")
	fig_filespec %= (event, ipe_name, method, subcatchment, output_format)
	
	dpi = 200 if fig_filespec else 90
	map.plot(fig_filespec=fig_filespec, dpi=dpi)