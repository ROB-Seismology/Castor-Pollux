# -*- coding: iso-Latin-1 -*-

import os
import numpy as np
import matplotlib
import pylab
from mapping.geotools.read_gis import read_gis_file
import mapping.layeredbasemap as lbm
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib
from hazard.rshalib.source_estimation import calc_rupture_probability_from_ground_motion_thresholds
from hazard.rshalib.source.read_from_gis import import_source_model_from_gis


## Folder locations
login_name = os.getlogin()
if login_name == 'kris':
	project_folder = r"C:\Users\kris\Documents\Projects\2022 - Castor-Pollux"
	data_points = "Castor-points"
	LOFZ_model = "LOFZ_breukenmodel4.TAB"
elif login_name == 'kwils':
	project_folder = r"C:\Users\kwils.UGENT\OneDrive - UGent\Ground motions"
	data_points = "Castor-points"
	LOFZ_model = "LOFZ-faultmodel.TAB"
gis_folder = os.path.join(project_folder, "Input data", "GIS")


## Common parameters for area and fault sources
TRT = "ASC"
USD = 0
#LSD = 12.5
LSD = 15
RAR = 1
MSR = "WC1994"
RMS = 2.5

## Override fault dip
DIP = 89


def create_point_source(lon, lat, mag, depth, strike, dip, rake, id=""):
	"""
	Create point source from given parameters

	Note: other parameters (upper and lower seismogenic depth, aspect ratio,
	magnitude scaling relationship and rupture mesh spacing) are defined,
	at the module level, but could be overridden

	:param lon:
		float, longitude (in degrees)
	:param lat:
		float, latitude (in degrees)
	:param mag:
		float, (moment) magnitude
	:param depth:
		float, focal depth (in km)
	:param strike:
		int, fault strike (in degrees)
	:param dip:
		int, fault dip (in degrees)
	:param rake:
		int, fault rake (in degrees)
	:param id:
		str, source ID
		(default: '')

	:return:
		instance of :class:`rshalib.source.PointSource`
	"""
	point = rshalib.geo.Point(lon, lat)
	name = "%.2f, %.2f" % (lon, lat)

	dM = 0.1
	mfd = rshalib.mfd.EvenlyDiscretizedMFD(mag, dM, np.ones(1))

	nopl = rshalib.geo.NodalPlane(strike, dip, rake)
	npd = rshalib.pmf.NodalPlaneDistribution([nopl], [1])

	hdd = rshalib.pmf.HypocentralDepthDistribution([depth], [1])

	source = rshalib.source.PointSource(id, name, TRT, mfd, RMS, MSR, RAR,
										USD, LSD, point, npd, hdd)

	return source


def create_uniform_grid_source_model(grid_outline, grid_spacing, min_mag,
									max_mag, dM, depth, strike, dip, rake):
	"""
	Create uniform grid source model with same parameters at each node

	Note: other parameters (upper and lower seismogenic depth, aspect ratio,
	magnitude scaling relationship and rupture mesh spacing) are defined,
	at the module level, but could be overridden

	:param grid_outline:
		(min_lon, max_lon, min_lat, max_lat) tuple
	:param grid_spacing:
		float, grid spacing (in degrees)
	:param min_mag:
		float, minimum magnitude
	:param max_mag:
		float, maximum magnitude
	:param dM:
		float, magnitude bin width
	:param depth:
		float, focal depth (in km)
	:param strike:
	:param dip:
	:param rake:
		ints, fault strike, dip and rake (in degrees)

	:return:
		instance of :class:`rshalib.source.SourceModel`, containing grid of
		point sources
	"""
	num_mags = int(round((max_mag - min_mag) / dM))
	mfd = rshalib.mfd.EvenlyDiscretizedMFD(min_mag + dM/2, dM, np.ones(num_mags)/float(num_mags))

	nopl = rshalib.geo.NodalPlane(strike, dip, rake)
	npd = rshalib.pmf.NodalPlaneDistribution([nopl], [1])

	hdd = rshalib.pmf.HypocentralDepthDistribution([depth], [1])

	lons = np.arange(grid_outline[0], grid_outline[1] + grid_spacing, grid_spacing)
	lats = np.arange(grid_outline[2], grid_outline[3] + grid_spacing, grid_spacing)
	sources = []
	i = 0
	for lon in lons:
		for lat in lats:
			point = rshalib.geo.Point(lon, lat)
			name = "%.2f, %.2f" % (lon, lat)
			source = rshalib.source.PointSource(i, name, TRT, mfd, RMS, MSR, RAR,
												USD, LSD, point, npd, hdd)
			sources.append(source)
			i += 1
	return rshalib.source.SourceModel("Grid", sources)


def read_fault_source_model(gis_filespec, characteristic=True):
	"""
	Read fault source model from GIS file

	:param gis_filespec:
		str, full path to GIS file
	:param characteristic:
		bool, whether fault sources should be simple faults (False)
		or characteristic faults (True)
		(default: True)

	:return:
		instance of :class:`rshalib.source.SourceModel`, containing fault sources
	"""
	## Note: set fault dip to 89 degrees to avoid crash
	## in rupture.surface.get_joyner_boore_distance function in oqhazlib
	column_map = {
		'id': '#',
		'name': 'Name',
		'tectonic_region_type': TRT,
		'rupture_aspect_ratio': RAR,
		'upper_seismogenic_depth': USD,
		'lower_seismogenic_depth': LSD,
		'magnitude_scaling_relationship': MSR,
		'rupture_mesh_spacing': RMS,
		#'dip': 'Dip',
		'dip': DIP,
		'min_mag': None,
		'max_mag': None,
		'rake': 'Rake',
		'slip_rate': 1,
		'bg_zone': 'Id'
	}

	fault_ids = []

	somo = import_source_model_from_gis(gis_filespec, column_map=column_map,
										source_ids=fault_ids)
	if characteristic:
		for i, flt in enumerate(somo.sources):
			somo.sources[i] = flt.to_characteristic_source(convert_mfd=True)
			somo.sources[i].mfd.set_num_sigma(0)
			somo.sources[i].mfd.modify_set_occurrence_rates([1])

	return somo


def read_fault_source_model_as_floating_ruptures(gis_filespec, min_mag, max_mag,
															dM, depth, aspect_ratio=None):
	"""
	Read fault source model from GIS file, but split into overlapping rupture
	planes, extending from the top edge and with lengths and widths scaled
	by magnitude

	:param gis_filespec:
		str, full path to GIS file
	:param min_mag:
		float, minimum magnitude for generated ruptures
	:param max_mag:
		float, maximum magnitude for generated ruptures
	:param dM:
		float, magnitude bin width
	:param depth:
		obsolete, currently ignored
	:param aspect_ratio:
		float, aspect ratio for generated rupture planes
		(default: None = use default ratio defined at module level)

	:return:
		instance of :class:`rshalib.source.SourceModel`, containing fault sources
	"""
	from copy import deepcopy
	from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface

	num_mags = int(round((max_mag - min_mag) / dM))
	mfd = rshalib.mfd.EvenlyDiscretizedMFD(min_mag + dM/2, dM, np.ones(num_mags)/float(num_mags))

	fault_somo = read_fault_source_model(gis_filespec, characteristic=False)
	## Divide fault trace in points
	sources = []
	for f, flt in enumerate(fault_somo.sources):
		if aspect_ratio:
			flt.rupture_aspect_ratio = aspect_ratio
		for mag in mfd.get_center_magnitudes():
			subfaults = flt.get_top_edge_rupture_faults(mag)
			sources.extend(subfaults)
		"""
		fault_surface = SimpleFaultSurface.from_fault_data(flt.fault_trace, USD, LSD, flt.dip, RMS)
		fault_mesh = fault_surface.get_mesh()
		surface_locations = fault_mesh[0:1]
		#print surface_locations.lons
		#print surface_locations.lats
		#print flt.get_length()
		#print (len(surface_locations) - 1) * RMS
		mesh_rows, mesh_cols = fault_mesh.shape
		hypo_depths = []
		for j in range(1, mesh_rows-1):
			mesh = fault_mesh[j-1:j+2,:]
			hypocenter = mesh.get_middle_point()
			hypo_depths.append(hypocenter.depth)
		hypo_idx = np.abs(np.array(hypo_depths) - depth).argmin()
		hypo_idx += 1
		print hypo_idx
		for i in range(1, mesh_cols-1):
			mesh = fault_mesh[hypo_idx-1:hypo_idx+1, i-1:i+1]
			dip, strike = mesh.get_mean_inclination_and_azimuth()
			hypocenter = mesh.get_middle_point()
			distance_to_start = i * RMS - RMS/2.
			distance_to_end = (mesh_cols - i) * RMS - RMS/2.
			nodal_plane = rshalib.geo.NodalPlane(strike, dip, flt.rake)
			npd = rshalib.pmf.NodalPlaneDistribution([nodal_plane], [1])
			hdd = rshalib.pmf.HypocentralDepthDistribution([depth], [1])
			name = "%s #%02d" % (flt.name, i+1)
			ID = flt.source_id + "#%02d" % (i+1)

			point_source = rshalib.source.PointSource(ID, name, TRT, mfd, RMS, MSR, RAR,
												USD, LSD, hypocenter, npd, hdd)
			## Check if rupture stays within fault limits
			for mag in mfd.get_center_magnitudes():
				rup_length, rup_width = point_source._get_rupture_dimensions(mag, nodal_plane)
				if rup_length / 2 <= min(distance_to_start, distance_to_end):
					pt_col = i + 0.5
					rup_col_num = int(round(rup_length / RMS))
					start_col = min(i, int(i - (rup_col_num / 2.)))
					end_col = max(i, int(i + (rup_col_num / 2.)))
					#rup_row_num = int(round(rup_width / RMS))
					subfault_trace = list(surface_locations)[start_col:end_col+1]
					subfault_trace = oqhazlib.geo.Line(subfault_trace)
					subfault_mfd = rshalib.mfd.EvenlyDiscretizedMFD(mag, mfd.bin_width, [1])
					subfault_source_id = ID + "_M=%s" % mag
					subfault_lsd =  rup_width * np.cos(np.radians(90 - flt.dip))
					subfault = rshalib.source.SimpleFaultSource(subfault_source_id, flt.name,
									TRT, subfault_mfd, RMS, MSR, RAR, USD, subfault_lsd,
									subfault_trace, flt.dip, flt.rake)

					#subfault = deepcopy(point_source)
					#subfault.mfd = rshalib.mfd.EvenlyDiscretizedMFD(mag, mfd.bin_width, [1])
					#subfault.source_id = ID + "_M=%s" % mag

					sources.append(subfault)
				#print mag, rup_length/2, distance_to_start, distance_to_end
		"""
	somo_name = fault_somo.name + "_pts"
	return rshalib.source.SourceModel(somo_name, sources)


def read_fault_source_model_as_network(gis_filespec, section_len=2.85, dM=0.2,
					num_sections=None, max_strike_delta=60, characteristic=True):
	"""
	Read fault source model as network, containing all possible connections

	:param gis_filespec:
		str, full path to GIS file
	:param section_len:
		float, minimum length to split fault traces in (in km);
		depending on the magnitude, multiple sections may be combined
		(default: 2.85)
	:param dM:
		float, magnitude bin width
		(default: 0.2)
	:param num_sections:
		int, if given, generate only the combinations involving this
		number of sections
		(default: None)
	:param max_strike_delta:
		int, maximum change in strike (in degrees) to allow for combining sections
		(default: 60)
	:param characteristic:
		bool, whether fault sources should be simple faults (False)
		or characteristic faults (True)
		(default: True)

	:return:
		generator, yielding instances of :class:`rshalib.source.SourceModel`
		for each magnitude
	"""
	#import eqgeology.Scaling.WellsCoppersmith1994 as wc
	wc = oqhazlib.scalerel.WC1994()


	## Read fault model
	fault_somo = read_fault_source_model(gis_filespec, characteristic=False)

	## Construct fault network
	print("Constructing fault network...")
	for flt in fault_somo:
		flt.rupture_mesh_spacing = section_len
	allow_triple_junctions = False
	flt_network = fault_somo.get_fault_network(allow_triple_junctions=allow_triple_junctions, max_strike_delta=max_strike_delta)
	flt_network.check_consistency()
	print("Determining all possible connections...")
	connections = flt_network.get_all_connections(100, allow_triple_junctions=allow_triple_junctions)
	max_num_sections = max([len(conn) for conn in connections])
	min_aspect_ratio = 0.67

	## Determine relation between section length/area and magnitude
	section_nums = np.arange(1, max_num_sections + 1)
	lengths = section_nums * section_len
	widths = np.minimum(lengths / min_aspect_ratio, (LSD - USD) * np.sin(np.radians(DIP)))
	areas = lengths * widths
	#┬mags = np.array([wc.GetMagFromRuptureParams(RA=ra)['RA'].val for ra in areas])
	mags = np.array([wc.get_median_mag(ra, None) for ra in areas])

	if num_sections is None:
		## Determine magnitudes (spaced at least dM) and corresponding number of sections
		idxs = [0]
		for m, M in enumerate(mags[1:]):
			if M - mags[idxs[-1]] >= dM:
				idxs.append(m+1)
		Mrange, num_sections = mags[idxs], section_nums[idxs]
	else:
		if isinstance(num_sections, int):
			num_sections = [num_sections]
		idxs = np.array(num_sections) - 1
		Mrange = mags[idxs]

	## OBSOLETE: used to interpolate number of sections for a range of mags
	#min_mag, max_mag = np.round(mags[0], 1), np.round(mags[-1], 1)
	#Mrange = np.arange(min_mag, max_mag + dM/2, dM)

	## Extract all possible ruptures with number of sections corresponding to
	## given magnitude
	#for M in Mrange:
		#num_sections = int(round(rshalib.utils.interpolate(mags, section_nums, M)))

	## Extract all possible ruptures with given number of sections
	for M, num_sec in zip(Mrange, num_sections):
		linked_ruptures = [conn for conn in connections if len(conn) == num_sec]
		print("M=%.2f, len=%d, n=%d" % (M, num_sec, len(linked_ruptures)))
		rupture_model = fault_somo.get_linked_subfaults(linked_ruptures,
				min_aspect_ratio=min_aspect_ratio, characteristic=characteristic)
		for flt in rupture_model:
			flt.mfd = rshalib.mfd.CharacteristicMFD(M, 1, 0.1, num_sigma=0)
		yield (M, rupture_model)


def read_evidence_site_info_from_txt(filespec):
	"""
	Read shaking evidence from different sites from text file

	:param filespec:
		str, full path to text file containing shaking evidence

	:return:
		(pe_thresholds, pe_site_models, ne_thresholds, ne_site_models) tuple:
		- pe_thresholds: list of floats, intensity thresholds for positive evidence
		- pe_site_models: list with instances of :class:`rshalib.site.SoilSiteModel`,
		  corresponding soil site models for positivie evidence
		- ne_thresholds: list of floats, intensity thresholds for negative evidence
		- ne_site_models: corresponding soil site models for negative evidence
	"""
	pe_thresholds, ne_thresholds = [], []
	pe_sites, ne_sites = [],[]

	lons, lats = [], []
	with open(filespec) as f:
		for line in f:
			line = line.strip()
			if line and line[0] in ('<', '>'):
				intensity = float(line[2:])
				points = [oqhazlib.geo.Point(lon, lat) for (lon, lat) in zip(lons, lats)]
				[datasites] = points
				if line[0] == '<':
					ne_thresholds.append(intensity)
					ne_sites.append(datasites)
				else:
					pe_thresholds.append(intensity)
					pe_sites.append(datasites)
				lons, lats = [], []
			elif line:
				if line[-1] in ('W', 'E', 'N', 'S'):
					deg = float(line[:-1])

					if line[-1] in ('W', 'S'):
						deg = -deg
					if line[-1] in ('W', 'E'):
						lons.append(deg)
					else:
						lats.append(deg)

	pe_site_models = []
	for p, pe_sites in enumerate(pe_sites):
		name = "Positive evidence #%d (I>%.1f)" % (p+1, pe_thresholds[p])
		site = rshalib.site.SoilSite(pe_sites.longitude, pe_sites.latitude)
		site_model = rshalib.site.SoilSiteModel([site], name)
		pe_site_models.append(site_model)

	ne_site_models = []
	for n, ne_sites in enumerate(ne_sites):
		name = "Negative evidence #%d (I<%.1f)" % (n+1, ne_thresholds[n])
		site = rshalib.site.SoilSite(ne_sites.longitude, ne_sites.latitude)
		site_model = rshalib.site.SoilSiteModel([site], name)
		ne_site_models.append(site_model)

	pe_thresholds, ne_thresholds = np.array(pe_thresholds), np.array(ne_thresholds)
	return pe_thresholds, pe_site_models, ne_thresholds, ne_site_models


def plot_rupture_probabilities(source_model, prob_dict, pe_site_models, ne_site_models,
								region, plot_point_ruptures=True, colormap="RdBu_r",
								title=None, text_box=None, site_model_gis_file=None,
								prob_min=0., prob_max=1., highlight_max_prob_section=True,
								max_prob_mag_precision=1, legend_label="Normalized probability",
								neutral_site_models=[], fig_filespec=None):
	"""
	Generate map of rupture probabilities

	:param source_model:
		instance of :class:`rshalib.source.SourceModel`
	:param prob_dict:
		dict, mapping source IDs to probabilities (corresponding to center
		magnitudes of their MFD)
	:param pe_site_models:
	:param ne_site_models:
		list with instances of :class:`rshalib.site.SoilSiteModel`,
		site models with positive/negative evidence
	:param region:
		(min_lon, max_lon, min_lat, max_lat), map region
	:param plot_point_ruptures:
		bool, whether to plot point sources as rupture planes (True)
		or as points (False)
		(default: True)
	:param colormap:
		str, name of matplotlib colormap to use for probabilities
		(default: "RdBu_r")
	:param title:
		str, plot title
		(default: None)
	:param text_box:
		str, text to add in separate box
		(default: None)
	:param site_model_gis_file:
		str, full path to GIS file containing sites with shaking evidence
		(necessary to plot polygons that have been discretized into points)
		(default: None)
	:param prob_min:
		float, minimum probability for color scale
		(default: 0.)
	:param prob_max:
		float, maximum probability for color scale
		(default: 1.)
	:param highlight_max_prob_section:
		bool, whether or not to highlight the section with the highest probability
		(default: True)
	:param max_prob_mag_precision:
		int, precision (number of decimals) to round magnitude corresponding
		to maximum probability
		(default: 1)
	:param legend_label:
		str, label to use in map legend
		(default: "Normalized probability")
	:param neutral_site_models:
		list with instances of :class:`rshalib.site.SoilSiteModel`,
		site models with no evidence
		(default: [])
	:param fig_filespec:
		str, full path to output file
		(default: None, will plot on screen)
	"""
	## Extract source locations
	x, y = [], []
	values = {'mag': [], 'prob': []}
	PROB_MIN = 1E-5
	if source_model.get_point_sources():
		## Point sources and discretized fault sources
		for source_id, probs in prob_dict.items():
			source = source_model[source_id]
			center_magnitudes = source.mfd.get_center_magnitudes()
			idx = probs.argmax()
			max_prob = probs[idx]
			## Select non-zero probability rupture locations to be plotted
			if max_prob > PROB_MIN:
				values['prob'].append(max_prob)
				#values['prob'].append(max_prob / ref_prob)
				mag = source.mfd.get_center_magnitudes()[idx]
				values['mag'].append(mag)

				if not plot_point_ruptures:
					x.append(source.location.longitude)
					y.append(source.location.latitude)
					print(x[-1], y[-1], values['mag'][-1], max_prob)
				else:
					## Not sure this is correct if fault is not vertical
					## Point source ruptures
					[nodal_plane] = source.nodal_plane_distribution.nodal_planes
					hypocenter = source.location
					#hypocenter.depth = 0.1
					rup_surface = source._get_rupture_surface(mag, nodal_plane, hypocenter)
					top_left = rup_surface.top_left
					top_right = rup_surface.top_right
					# TODO: extract all top coordinates!
					x.append([top_left.longitude, top_right.longitude])
					y.append([top_left.latitude, top_right.latitude])

		max_prob = np.max(values['prob'])
		print("Max. probability: %.3f" % max_prob)

	else:
		## Fault sources
		for fault_id, [prob] in prob_dict.items():
			fault = source_model[fault_id]
			lons = np.array([pt.longitude for pt in fault.fault_trace.points])
			lats = np.array([pt.latitude for pt in fault.fault_trace.points])
			values['mag'].append(fault.mfd.get_center_magnitudes()[0])
			values['prob'].append(prob)
			#values['prob'].append(prob / ref_prob)
			x.append(lons)
			y.append(lats)

	## Reorder from lowest to highest probability
	idxs = np.argsort(values['prob'])
	values['prob'] = [values['prob'][idx] for idx in idxs]
	values['mag'] = [values['mag'][idx] for idx in idxs]
	x = [x[idx] for idx in idxs]
	y = [y[idx] for idx in idxs]

	if source_model.get_point_sources() and not plot_point_ruptures:
		source_data = lbm.MultiPointData(x, y, values=values)
	else:
		source_data = lbm.MultiLineData(x, y, values=values)
		## Find rupture with highest probability for each magnitude
		if highlight_max_prob_section and source_model.get_fault_sources():
			mag_max_prob_id_dict = {}
			for idx in idxs:
				mag = np.round(values['mag'][idx], max_prob_mag_precision)
				prob = values['prob'][idx]
				if prob and (not mag in mag_max_prob_id_dict or prob > mag_max_prob_id_dict[mag][1]):
					mag_max_prob_id_dict[mag] = (idx, prob)
			max_source_data = None
			for mag, (idx, prob) in mag_max_prob_id_dict.items():
				fault_id = list(prob_dict.keys())[idx]
				#print("Max. prob.: %s (M=%.2f) %.2f" % (fault_id, mag, prob))
				line_data = source_data[int(idx)]
				if not max_source_data:
					max_source_data = line_data.to_multi_line()
				else:
					max_source_data.append(line_data)


	## Plot histogram of probabilities
	"""
	bin_width = 0.02
	xmax = np.ceil(max_prob / bin_width) * bin_width
	num_bins = xmax / bin_width + 1
	bins = np.linspace(0, xmax, num_bins)
	pylab.hist(values['prob'], bins=bins, log=False)
	pylab.show()
	"""


	layers = []

	## Coastlines
	data = lbm.BuiltinData("coastlines")
	style = lbm.LineStyle()
	layer = lbm.MapLayer(data, style)
	layers.append(layer)

	## Sources
	colorbar_style = lbm.ColorbarStyle(legend_label, format="%.1f")
	#thematic_color = lbm.ThematicStyleGradient([1E-3, 1E-2, 1E-1, 1], "RdBu_r", value_key='prob', colorbar_style=colorbar_style)
	#thematic_color = lbm.ThematicStyleGradient([0.01, 0.05, 0.125, 0.25, 0.5, 1], "RdBu_r", value_key='prob', colorbar_style=colorbar_style)
	#thematic_color = lbm.ThematicStyleColormap("Reds", vmin=0.001, vmax=max_prob_color, value_key='prob', colorbar_style=colorbar_style, alpha=1)
	#if not max_prob_color:
	#	max_prob_color = 1./ref_prob
	#norm = matplotlib.colors.LogNorm(vmin=1./max_prob_color, vmax=max_prob_color)
	norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
	thematic_color = lbm.ThematicStyleColormap(colormap, norm=norm, value_key='prob', colorbar_style=colorbar_style, alpha=1)
	## zero probabilities
	thematic_color.color_map.set_bad(thematic_color.color_map(0))

	if source_model.get_point_sources() and not plot_point_ruptures:
		edge_magnitudes = np.concatenate([source.mfd.get_magnitude_bin_edges(), [center_magnitudes[-1]+dM/2]])
		mag_sizes = (center_magnitudes - 4) ** 2
		thematic_size = lbm.ThematicStyleRanges(edge_magnitudes, mag_sizes, value_key='mag', labels=center_magnitudes)
		#thematic_size = 8
		thematic_legend_style = lbm.LegendStyle("Magnitude", location="upper left")
		style = lbm.PointStyle(fill_color=thematic_color, size=thematic_size, thematic_legend_style=thematic_legend_style)
	else:
		style = lbm.LineStyle(line_color=thematic_color, line_width=2, thematic_legend_style="main")
		layer = lbm.MapLayer(source_data, style)
	layers.append(layer)

	## Highlight fault section with highest probability
	if highlight_max_prob_section and source_model.get_fault_sources():
		front_style = lbm.FrontStyle("asterisk", size=10, num_sides=1, angle=-45, interval=[0,1], alternate_sides=True)
		style = lbm.LineStyle(line_width=0, front_style=front_style)
		layer = lbm.MapLayer(max_source_data, style)
		layers.append(layer)

		front_style = lbm.FrontStyle("asterisk", size=12, num_sides=1, angle=-135, interval=[0,1], alternate_sides=True)
		style = lbm.LineStyle(line_width=0, front_style=front_style)
		layer = lbm.MapLayer(max_source_data, style)
		layers.append(layer)

	## Observation sites
	## Read polygons from GIS file if specified
	site_polygons = {}
	if site_model_gis_file:
		site_data = lbm.GisData(site_model_gis_file, label_colname='Name')
		site_data = site_data.get_data()[-1]
		for polygon in site_data:
			site_polygons[polygon.label] = polygon

	## Positive evidence
	for pe_site_model in pe_site_models:
		site_name = pe_site_model.name.split('(')[0].strip()
		if site_name in site_polygons:
			pe_data = site_polygons[site_name]
			pe_style = lbm.PolygonStyle(line_width=0, fill_color='m', alpha=0.5)
		else:
			pe_style = lbm.PointStyle('+', size=8, line_width=1, line_color='m')
			pe_data = lbm.MultiPointData(pe_site_model.lons, pe_site_model.lats)
		layer = lbm.MapLayer(pe_data, pe_style)
		layers.append(layer)

	## Negative evidence
	for ne_site_model in ne_site_models:
		site_name = ne_site_model.name.split('(')[0].strip()
		if site_name in site_polygons:
			ne_data = site_polygons[site_name]
			ne_style = lbm.PolygonStyle(line_width=0, fill_color='c', alpha=0.5)
		else:
			ne_style = lbm.PointStyle('_', size=8, line_width=1, line_color='c')
			ne_data = lbm.MultiPointData(ne_site_model.lons, ne_site_model.lats)
		layer = lbm.MapLayer(ne_data, ne_style)
		layers.append(layer)

	## Neutral
	for site_model in neutral_site_models:
		site_name = site_model.name.split('(')[0].strip()
		if site_name in site_polygons:
			site_data = site_polygons[site_name]
			site_style = lbm.PolygonStyle(line_width=0, fill_color='gray', alpha=0.5)
		else:
			site_style = lbm.PointStyle('x', size=6, line_width=1, line_color='dimgrey')
			site_data = lbm.MultiPointData(site_model.lons, site_model.lats)
		layer = lbm.MapLayer(site_data, site_style)
		layers.append(layer)


	scalebar_style = lbm.ScalebarStyle((-72.25, -44.9), 25, font_size=12, yoffset=2000)
	map = lbm.LayeredBasemap(layers, title, "merc", region=region,
							graticule_interval=(1, 0.5), resolution='h',
							scalebar_style=scalebar_style)

	## Add text box
	if text_box:
		pos = (0.965, 0.035)
		if max([len(s) for s in text_box.split('\n')]) > 15:
			font_size = 12
		else:
			font_size = 14
		text_style = lbm.TextStyle(font_size=font_size, horizontal_alignment='right',
							vertical_alignment='bottom', multi_alignment='left',
							background_color='w', border_color='k', border_pad=0.5)
		map.draw_text_box(pos, text_box, text_style, zorder=10000)

	if fig_filespec:
		dpi = 200
	else:
		dpi = 90
	map.plot(fig_filespec=fig_filespec, dpi=dpi)
