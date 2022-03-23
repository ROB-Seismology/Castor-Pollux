# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:24:56 2022

@author: kris
"""

import numpy as np
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib


def create_point_source(lon, lat, depth, mag, sdr, name='',
								trt='Active Shallow Crust', msr='WC1994',
								rms=1, rar=1, usd=0, lsd=20):
	"""
	Create OpenQuake point source

	:param lon:
		float, earthquake longitude (in degrees)
	:param lat:
		float, earthquake latitude (in degrees)
	:param depth:
		float, focal depth (in km)
	:param mag:
		float, moment magnitude
	:param sdr:
		(strike, dip, rake) tuple
	:param name:
		str, earthquake name
		(default: '')
	:param trt:
		str, tectonic region type
		(default: 'Active Shallow Crust')
	:param msr:
		str, magnitude scaling relationship
		(default: 'WC1994')
	:param rms:
		float, rupture mesh spacing (in km)
		(default: 1)
	:param rar:
		float rupture aspect ratio
		(default: 1)
	:param usd:
		float, upper seismogenic depth (in km)
		(default: 0)
	:param lsd:
		float, lower seismogenic depth (in km)
		(default: 20)

	:return:
		instance of :class:`rshalib.source.PointSource`
	"""
	location = rshalib.geo.Point(lon, lat)
	dM = 0.1
	mfd = rshalib.mfd.EvenlyDiscretizedMFD(mag - dM/2., dM, [1])
	hdd = rshalib.pmf.HypocentralDepthDistribution([depth], [1])
	strike, dip, rake = sdr
	nodal_plane = rshalib.geo.NodalPlane(strike, dip, rake)
	ndd = rshalib.pmf.NodalPlaneDistribution([nodal_plane], [1])
	pt_src = rshalib.source.PointSource('', name, trt, mfd, rms, msr, rar,
												usd, lsd, location, ndd, hdd)

	return pt_src


eq_lon = -149.955
eq_lat = 61.346
# TODO: set correct focal depth
eq_depth = 10
eq_mag = 7.1
eq_name = '2018 Anchorage'
sdr = (0, 30, -90)

site_lon = -149.044
site_lat = 61.380
site_depth = 0

pt_src = create_point_source(eq_lon, eq_lat, eq_depth, eq_mag, sdr, eq_name)
[Rrup] = pt_src.calc_rrup(site_lon, site_lat, site_depth)
[Rjb] = pt_src.calc_rjb(site_lon, site_lat, site_depth)
[Rhypo] = pt_src.calc_rhypo(site_lon, site_lat, site_depth)
[Repi] = pt_src.calc_repi(site_lon, site_lat, site_depth)
print(Rrup, Rjb, Rhypo, Repi)
