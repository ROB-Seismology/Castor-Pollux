import os, sys
import numpy as np

import hazard.rshalib as rshalib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from castorlib import *
from create_animated_gif import create_animated_gif

## Event names
event = "~4400 cal yrs BP"


## Selected magnitude for final figure in paper
event_mags = {'~4400 cal yrs BP':[4.05, 4.64, 4.99, 5.23, 5.58, 5.82, 6.09, 6.31, 6.51, 6.72, 6.92]}
#event_mags = {'~4400 cal yrs BP':[6.55]}

## IPE names
ipe_names = ["BakunWentworth1997WithSigma"]


## Parameters
#truncation_level = 2.5
truncation_level = 1
soil_params = rshalib.site.REF_SOIL_PARAMS
imt = oqhazlib.imt.MMI()
strict_intersection = True


## Map parameters
#map_region = (-74, -72, -46, -44.5)
map_region = (-73.8, -71.5, -46, -45)
#output_format = "pdf"
output_format = "png"

## Read fault source model

## Faults discretized as floating ruptures
"""
dM = 0.25
min_mag, max_mag = 6.0 - dM/2, 7.0
fault_mags = np.arange(min_mag, max_mag, dM) + dM/2
print(fault_mags)
for M in fault_mags:
	fault_filespec = os.path.join(gis_folder, LOFZ-faultmodel)
	source_model = read_fault_source_model_as_floating_ruptures(fault_filespec, M, M + dM/2, dM, depth=0.1)
"""

## Discretize faults as network
dM = 0.2
fault_mags, fault_networks = [], []
fault_filespec = os.path.join(gis_folder, fault_model)
for M, source_model in read_fault_source_model_as_network(fault_filespec, dM=dM):
	#print(fault_mags, fault_networks)
	fault_mags.append(M)
	fault_networks.append(source_model)
	
#calculations
max_prob_dict = {}
section_prob_dict = {}
max_prob_dict[event] = {}
section_prob_dict[event] = {}

fig_folder = os.path.join(base_fig_folder)
if not os.path.exists(fig_folder):
	os.mkdir(fig_folder)

## Read MTD evidence
pe_thresholds, pe_site_models, ne_thresholds, ne_site_models = [], [], [], []

textfile = os.path.join(gis_folder, "%s.txt" % data_points)
(pe_thresholds, pe_site_models,
ne_thresholds, ne_site_models) = read_evidence_site_info_from_txt(textfile)

for pe_site_model, pe_threshold in zip(pe_site_models, pe_thresholds):
	print("+%s (n=%d): %s" % (pe_site_model.name.encode(errors='replace'), len(pe_site_model), pe_threshold))
for ne_site_model, ne_threshold in zip(ne_site_models, ne_thresholds):
	print("-%s (n=%d): %s" % (ne_site_model.name.encode(errors='replace'), len(ne_site_model), ne_threshold))

## Additional constraint: intensities must be 7.5 or higher in Aysén catchment (or as point in catchment)

## Construct ground-motion model
for ipe_name in ipe_names:
	max_prob_dict[event][ipe_name] = []
	section_prob_dict[event][ipe_name] = {}
	trt_gsim_dict = {TRT: ipe_name}
	gmpe_system_def = {TRT: rshalib.pmf.GMPEPMF([ipe_name], [1])}
	integration_distance_dict = {}


for M, source_model in zip(fault_mags, fault_networks):
	print(M)
	## Compute rupture probabilities
	prob_dict = calc_rupture_probability_from_ground_motion_thresholds(
						source_model, gmpe_system_def, imt, pe_site_models,
						pe_thresholds, ne_site_models, ne_thresholds, truncation_level,
						integration_distance_dict=integration_distance_dict,
						strict_intersection=strict_intersection)
	probs = np.array(list(prob_dict.values()))
	probs = probs[:, 0]
	max_prob = probs.max()
	max_prob_dict[event][ipe_name].append(max_prob)
	print(M, max_prob)
	for rup_name, prob in zip(prob_dict.keys(), probs):
		for section in rup_name.split('+'):
			if not section in section_prob_dict[event][ipe_name]:
				section_prob_dict[event][ipe_name][section] = [prob]
			else:
				section_prob_dict[event][ipe_name][section].append(prob)

	## Plot
	if "WithSigma" in ipe_name:
		ipe_label = ipe_name[:ipe_name.find("WithSigma")]
	else:
		ipe_label = ipe_name

	#text_box = "Event: %s\nIPE: %s\nM: %.2f, Pmax: %.2f"
	text_box = "Event: %s\nM: %.2f\nPmax: %.2f"
	text_box %= (event, M, max_prob)

	#title = "Event: %s, IPE: %s, M=%.2f" % (event, ipe_name, M)
	title = ""

	fig_filename = "%s_%s_M=%.2f.%s" % (event, ipe_label, M, output_format)
	fig_filespec = os.path.join(fig_folder, fig_filename)
	
	#fig_filespec = None

	## Colormaps: RdBu_r, YlOrRd, BuPu, RdYlBu_r, Greys
	if np.isclose(M, magnitude, atol=0.01):
			plot_rupture_probabilities(source_model, prob_dict, pe_site_models, ne_site_models,
										map_region, plot_point_ruptures=True, colormap="RdYlBu_r",
										title=title, text_box=text_box,	fig_filespec=fig_filespec)

## Generate animated GIF

for ipe_name in ipe_names:
	img_basename = "%s_%s" % (event, ipe_label)
	create_animated_gif(fig_folder, img_basename)
#exit()


# ## Determine which sections have highest probability
# for event in events:
# 	print(event)
# 	for ipe_name in ipe_names:
# 		print(ipe_name)
# 		sections = list(section_prob_dict[event][ipe_name].keys())
# 		probs = [np.array(list(l)) for l in section_prob_dict[event][ipe_name].values()]
# 		#print(probs[0])
# 		mean_probs = np.array([p.mean() for p in probs])
# 		max_probs = np.array([p.max() for p in probs])
# 		idxs = np.argsort(mean_probs)[::-1]
# 		for idx in idxs[:10]:
# 			print("  %s: %.2f, %.2f" % (sections[idx], mean_probs[idx], max_probs[idx]))



# ## Plot max_prob vs magnitude for different IPEs per event
# colors = ['r', 'b', 'g', 'm', 'k']
# for event in events:
# 	pylab.cla()
# 	for ipe_name, color in zip(ipe_names, colors):
# 		if "WithSigma" in ipe_name:
# 			label = ipe_name[:ipe_name.find("WithSigma")]
# 		else:
# 			label = ipe_name
# 		pylab.plot(fault_mags, max_prob_dict[event][ipe_name], 'x-', color=color, label=label)
# 	pylab.xlim(fault_mags[0], fault_mags[-1])
# 	pylab.ylim(0, 1)
# 	pylab.xlabel("Magnitude")
# 	pylab.ylabel("Max. normalized probability")
# 	pylab.title("Event: %s" % event)
# 	pylab.legend(loc=3)

# 	fig_folder = os.path.join(base_fig_folder, event)
# 	fig_filename = "%s_M_vs_prob.%s" % (event, output_format)
# 	#fig_filespec = os.path.join(fig_folder, fig_filename)
# 	fig_filespec = None
# 	if fig_filespec:
# 		pylab.savefig(fig_filespec, dpi=200)
# 	else:
# 		pylab.show()