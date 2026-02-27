from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import pandas as pd
from ambiance import Atmosphere
from geographiclib.geodesic import Geodesic

def interpolated_pressure_to_alt(baropress, groundtruth):
    baropress = float(baropress[0])
    if baropress >= groundtruth[0]:
        return float(ATMOSPHERE_GROUNDTRUTH_M[0])
    if baropress <= groundtruth[-1]:
        return ATMOSPHERE_GROUNDTRUTH_M[-1]
    for i in range(len(groundtruth) - 1):
        z0 = groundtruth[i]
        z1 = groundtruth[i + 1]
        if z0 >= baropress >= z1:
            r0 = ATMOSPHERE_GROUNDTRUTH_M[i]
            r1 = ATMOSPHERE_GROUNDTRUTH_M[i + 1]
            t = (baropress - z0) / (z1 - z0)
            return float(r0 + t * (r1 - r0))

def interpolated_at_altitude(alt_m, groundtruth):
    if alt_m <= ATMOSPHERE_GROUNDTRUTH_M[0]:
        return float(groundtruth[0])
    if alt_m >= ATMOSPHERE_GROUNDTRUTH_M[-1]:
        return groundtruth[-1]
    for i in range(len(ATMOSPHERE_GROUNDTRUTH_M) - 1):
        z0 = ATMOSPHERE_GROUNDTRUTH_M[i]
        z1 = ATMOSPHERE_GROUNDTRUTH_M[i + 1]
        if z0 <= alt_m <= z1:
            r0 = groundtruth[i]
            r1 = groundtruth[i + 1]
            t = (alt_m - z0) / (z1 - z0)
            return float(r0 + t * (r1 - r0))


# Ciddor refractive index

def ciddor_Z(T_k, p_pa, xw):
    t_c = T_k - 273.15
    a0 = 1.58123e-6
    a1 = -2.9331e-8
    a2 = 1.1043e-10
    b0 = 5.707e-6
    b1 = -2.051e-8
    c0 = 1.9898e-4
    c1 = -2.376e-6
    d = 1.83e-11
    e = -0.765e-8

    return 1 - (p_pa / T_k) * (
        a0 + a1 * t_c + a2 * t_c**2
        + (b0 + b1 * t_c) * xw
        + (c0 + c1 * t_c) * xw**2
    ) + (p_pa / T_k) ** 2 * (d + e * xw**2)


def ciddor(lambda_nm, t_c, p_pa, rh_percent, xc_ppm):
    """
    Ciddor refractive index model (moist air).
    """
    lam_um = lambda_nm / 1000.0
    h = rh_percent / 100.0
    sigma = 1.0 / lam_um

    T = t_c + 273.15
    R = 8.314510

    k0 = 238.0185
    k1 = 5792105
    k2 = 57.362
    k3 = 167917

    w0 = 295.235
    w1 = 2.6422
    w2 = -0.032380
    w3 = 0.004028

    A = 1.2378847e-5
    B = -1.9121316e-2
    C = 33.93711047
    D = -6.3431645e3

    alpha = 1.00062
    beta = 3.14e-8
    gamma = 5.6e-7

    # saturation vapor pressure of water vapor in air at temperature T (Pa)
    if t_c >= 0:
        svp = math.exp(A * T**2 + B * T + C + D / T)
    else:
        svp = 10 ** (-2663.5 / T + 12.537)

    # enhancement factor of water vapor in air
    f = alpha + beta * p_pa + gamma * t_c**2

    # molar fraction of water vapor in moist air
    xw = (f * h * svp / p_pa) if p_pa > 0 else 0.0

    # refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, 450 ppm CO2
    nas = 1 + (k1 / (k0 - sigma**2) + k3 / (k2 - sigma**2)) * 1e-8
    #refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, xc ppm CO2
    naxs = 1 + (nas - 1) * (1 + 0.534e-6 * (xc_ppm - 450))
    #refractive index of water vapor at standard conditions (20 °C, 1333 Pa)
    nws = 1 + 1.022 * (w0 + w1 * sigma**2 + w2 * sigma**4 + w3 * sigma**6) * 1e-8
    #molar mass of dry air, kg/mol
    Ma = 1e-3 * (28.9635 + 12.011e-6 * (xc_ppm - 400))
    #molar mass of water vapor, kg/mol
    Mw = 0.018015
    
    #compressibility of dry air
    Za = ciddor_Z(288.15, 101325, 0.0)
    #compressibility of pure water vapor
    Zw = ciddor_Z(293.15, 1333, 1.0)
    
    #density of standard air
    paxs = 101325 * Ma / (Za * R * 288.15)
    #density of standard water vapor
    pws = 1333 * Mw / (Zw * R * 293.15)

    Z = ciddor_Z(T, p_pa, xw)
    #density of the dry component of the moist air
    pa = (p_pa * Ma / (Z * R * T) * (1 - xw)) if p_pa > 0 else 0.0
    #density of the water vapor component
    pw = (p_pa * Mw / (Z * R * T) * xw) if p_pa > 0 else 0.0

    return 1.0 + (pa / paxs) * (naxs - 1.0) + (pw / pws) * (nws - 1.0)



# WGS-84 ellipsoid constants (meters)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def wgs84_radii_of_curvature(lat_deg: float):
    lat = math.radians(lat_deg)
    s = math.sin(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * s * s)
    N = WGS84_A / denom
    M = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)
    return M, N

def wgs84_normal_section_radius(lat_deg: float, azimuth_deg: float) -> float:
    M, N = wgs84_radii_of_curvature(lat_deg)
    A = math.radians(azimuth_deg)
    invR = (math.cos(A) ** 2) / M + (math.sin(A) ** 2) / N
    return 1.0 / invR

def wgs84_gauss_radius(lat_deg: float) -> float:
    M, N = wgs84_radii_of_curvature(lat_deg)
    return math.sqrt(M * N)

#Constants
#Earth's radius
SPHERE_RADIUS_M = 6371000  
EARTH_RADIUS_M = SPHERE_RADIUS_M  

# ambiance validity
AMB_MIN_ALT_M = -5_000.0
AMB_MAX_ALT_M = 80_000.0


def haversine_central_angle_rad(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2) ** 2)
    a = min(1.0, max(0.0, a))
    return 2.0 * math.asin(math.sqrt(a))

# Atmosphere via ambiance

class RefractionSettings:
    use_wgs84_ellipsoid = True  
    wavelength_nm = 550.0
    co2_ppm = 450.0
    segment_m = 50.0
    top_of_atmosphere_m = 80000.0


class AmbianceAtmosphereWithRH:
    """
    n(alt) = Ciddor(T(alt), P(alt), RH_profile(alt), CO2, wavelength)
    - T,P from ambiance for -5..80 km (clamped)
    - Above 80 km: treat as vacuum -> n = 1
    - RH from RH profile (0..22 km), interpolated; above 22 km -> 0%
    """
    def __init__(self, settings=RefractionSettings, cache_quant_m=RefractionSettings.segment_m):
        self.settings = RefractionSettings
        self.cache_quant_m = cache_quant_m
        self._cache = {}

    @staticmethod
    def _ambiance_TP(alt_m):
        if alt_m > ATMOSPHERE_GROUNDTRUTH_MAXALT:
            a = Atmosphere([alt_m])
            T = float(a.temperature[0])  # K
            P = float(a.pressure[0])     # Pa
        else:
            T = interpolated_at_altitude(alt_m,ATMOSPHERE_GROUNDTRUTH_T_C)+273.15 #Convert to K
            P = interpolated_at_altitude(alt_m,ATMOSPHERE_GROUNDTRUTH_HPA)*100 #Convert to Pa
        return T, P

    def n_at(self, alt_m: float) -> float:
        z = float(alt_m)

        if z >= AMB_MAX_ALT_M:
            return 1.0
        if z < AMB_MIN_ALT_M:
            z = AMB_MIN_ALT_M

        k = int(round(z / self.cache_quant_m))
        if k in self._cache:
            return self._cache[k]

        T, P = self._ambiance_TP(z)
        if z > ATMOSPHERE_GROUNDTRUTH_MAXALT:
            RH = 0
        else:
            RH = interpolated_at_altitude(z,ATMOSPHERE_GROUNDTRUTH_RH)

        n = ciddor(
            lambda_nm=self.settings.wavelength_nm,
            t_c=T - 273.15,
            p_pa=P,
            rh_percent=RH,
            xc_ppm=self.settings.co2_ppm,
        )

        self._cache[k] = float(n)
        return float(n)

# Ray tracing

def _trace_to_radius(
    atm: AmbianceAtmosphereWithRH,
    b,
    observer_alt_m: float,
    segment_m: float,
    r_target: float,
    top_of_atmosphere_m: float,
    s_max: Optional[float],
    earth_radius_m: float,
):
    r0 = earth_radius_m + observer_alt_m
    r = r0
    phi = 0.0
    s = 0.0
    ds = float(segment_m)

    while True:
        alt = r - earth_radius_m
        if alt >= top_of_atmosphere_m:
            return phi, r, s, False
        if s_max is not None and s >= s_max:
            return phi, r, s, False
        if r >= r_target:
            return phi, r, s, True

        n = atm.n_at(alt)
        arg = b / (n * r)
        if arg > 1.0:
            return phi, r, s, False
        arg = max(-1.0, min(1.0, arg))
        alpha = math.acos(arg)

        dr = ds * math.sin(alpha)
        dphi = (ds * math.cos(alpha)) / r

        if r + dr >= r_target:
            frac = (r_target - r) / max(1e-12, dr)
            phi += dphi * frac
            r = r_target
            s += ds * frac
            return phi, r, s, True

        r += dr
        phi += dphi
        s += ds

        if phi > 800.0 or r > (earth_radius_m + top_of_atmosphere_m) * 2:
            return phi, r, s, False


def _trace_infinite_exit_direction(atm, alpha_app_rad, observer_alt_m, segment_m, top_of_atmosphere_m, earth_radius_m: float):
    r0 = earth_radius_m + observer_alt_m
    n0 = atm.n_at(observer_alt_m)
    b = n0 * r0 * math.cos(alpha_app_rad)

    r_top = earth_radius_m + top_of_atmosphere_m
    phi_top, r_hit, _s, ok = _trace_to_radius(atm=atm, b=b, observer_alt_m=observer_alt_m, segment_m=segment_m, r_target=r_top, top_of_atmosphere_m=top_of_atmosphere_m, s_max=None, earth_radius_m=earth_radius_m)
    if not ok:
        return 0.0, 0.0, False
    # vacuum at boundary
    arg = b / r_hit
    if arg > 1.0:
        return 0.0, 0.0, False
    arg = max(-1.0, min(1.0, arg))
    alpha_exit = math.acos(arg)

    radial_hat = (math.cos(phi_top), math.sin(phi_top))
    tangent_hat = (-math.sin(phi_top), math.cos(phi_top))
    dx = math.sin(alpha_exit) * radial_hat[0] + math.cos(alpha_exit) * tangent_hat[0]
    dy = math.sin(alpha_exit) * radial_hat[1] + math.cos(alpha_exit) * tangent_hat[1]
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return 0.0, 0.0, False
    return dx / norm, dy / norm, True

#apparent -> geometric (stars)

def geometric_from_apparent_infinite(apparent_elev_deg, observer_lat_deg, observer_alt_m, settings):
    #In WGS-84 mode, uses latitude-dependent Gauss radius sqrt(M*N).
    #In spherical mode, uses SPHERE_RADIUS_M.

    atm = AmbianceAtmosphereWithRH(settings=settings, cache_quant_m=settings.segment_m)
    alpha_app = math.radians(apparent_elev_deg)

    if getattr(settings, "use_wgs84_ellipsoid", True):
        earth_radius_m = wgs84_gauss_radius(observer_lat_deg)
    else:
        earth_radius_m = float(SPHERE_RADIUS_M)

    dx, dy, ok = _trace_infinite_exit_direction(
        atm=atm,
        alpha_app_rad=alpha_app,
        observer_alt_m=observer_alt_m,
        segment_m=settings.segment_m,
        top_of_atmosphere_m=settings.top_of_atmosphere_m,
        earth_radius_m=earth_radius_m,
    )
    if not ok:
        return float("nan")

    alpha_geom = math.atan2(dx, dy)
    return math.degrees(alpha_geom)


# Finite target (Latitude Longitude Altitude): apparent + geometric + uplift

def apparent_geometric_uplift_target_lla(observer_lat_deg, observer_lon_deg, observer_alt_m, target_lat_deg, target_lon_deg, target_alt_m, settings,  max_total_path_m):
    max_iter = int(80)
    atm = AmbianceAtmosphereWithRH(settings=settings, cache_quant_m=settings.segment_m)

    use_wgs84 = getattr(settings, "use_wgs84_ellipsoid", True)

    if use_wgs84:
        inv = Geodesic.WGS84.Inverse(observer_lat_deg, observer_lon_deg, target_lat_deg, target_lon_deg)
        ground_m = float(inv["s12"])
        az1_deg = float(inv["azi1"])

        earth_radius_m = wgs84_normal_section_radius(observer_lat_deg, az1_deg)
        phi_t = ground_m / earth_radius_m
        phi_t_deg = math.degrees(phi_t)
    else:
        earth_radius_m = float(SPHERE_RADIUS_M)
        phi_t = haversine_central_angle_rad(observer_lat_deg, observer_lon_deg, target_lat_deg, target_lon_deg)
        ground_m = earth_radius_m * phi_t
        phi_t_deg = math.degrees(phi_t)

    r0 = earth_radius_m + observer_alt_m
    r_t = earth_radius_m + target_alt_m
    r_top = earth_radius_m + settings.top_of_atmosphere_m
    if r_t > r_top:
        r_t = r_top

    # geometric elevation
    x_obs, y_obs = r0, 0.0
    x_t = r_t * math.cos(phi_t)
    y_t = r_t * math.sin(phi_t)
    vx = x_t - x_obs
    vy = y_t - y_obs
    alpha_geom = math.atan2(vx, vy)
    geom_elev_deg = math.degrees(alpha_geom)

    #shoot for alpha_app so phi_hit == phi_t at r=r_t
    n0 = atm.n_at(observer_alt_m)
    eps = math.radians(0.0001)

    def phi_hit(alpha_app):
        alpha_app = max(alpha_app, eps)
        b = n0 * r0 * math.cos(alpha_app)
        ph, _r, _s, ok = _trace_to_radius(
            atm=atm,
            b=b,
            observer_alt_m=observer_alt_m,
            segment_m=settings.segment_m,
            r_target=r_t,
            top_of_atmosphere_m=settings.top_of_atmosphere_m,
            s_max=max_total_path_m,
            earth_radius_m=earth_radius_m,
        )
        return ph if ok else float("nan")

    def g(alpha_app):
        ph = phi_hit(alpha_app)
        if not math.isfinite(ph):
            return float("nan")
        return ph - phi_t

    #bracketing around geometric elevation
    center = max(alpha_geom, eps)
    span = math.radians(0.5)
    max_span = math.radians(20.0)

    lo = max(eps, center - span)
    hi = min(math.radians(89.9), center + span)
    g_lo = g(lo)
    g_hi = g(hi)

    while (not math.isfinite(g_lo) or not math.isfinite(g_hi) or g_lo * g_hi > 0) and span < max_span:
        span *= 1.7
        lo = max(eps, center - span)
        hi = min(math.radians(89.9), center + span)
        g_lo = g(lo)
        g_hi = g(hi)

    #coarse scan fallback
    if (not math.isfinite(g_lo) or not math.isfinite(g_hi) or g_lo * g_hi > 0):
        samples = []
        for k in range(40):
            samples.append(max(eps, center - k * math.radians(0.5)))
        for k in range(1, 100):
            samples.append(min(math.radians(89.9), center + k * math.radians(0.5)))

        prev_a = None
        prev_g = None
        bracket = None
        for a in samples:
            ga = g(a)
            if not math.isfinite(ga):
                continue
            if prev_a is not None and prev_g is not None and prev_g * ga <= 0:
                bracket = (prev_a, a, prev_g, ga)
                break
            prev_a, prev_g = a, ga

        if bracket is None:
            return float("nan"), geom_elev_deg, float("nan"), ground_m, phi_t_deg

        lo, hi, g_lo, g_hi = bracket

    #bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        if not math.isfinite(g_mid):
            lo = mid
            continue
        if abs(g_mid) < 1e-12:
            lo = hi = mid
            break
        if g_lo * g_mid <= 0:
            hi = mid
            g_hi = g_mid
        else:
            lo = mid
            g_lo = g_mid

    alpha_app = 0.5 * (lo + hi)
    apparent_deg = math.degrees(alpha_app)
    uplift_deg = apparent_deg - geom_elev_deg
    return apparent_deg, geom_elev_deg, uplift_deg, ground_m, phi_t_deg

#Calculate refraction for airplane or other target & stars behind it

def target_and_star_geometry_bundle(observer_lat_deg, observer_lon_deg, observer_alt_m, airplane_lat_deg, airplane_lon_deg, airplane_alt_m,settings,max_total_path_m,ATMOSPHERE_GROUNDTRUTH_HPA1,ATMOSPHERE_GROUNDTRUTH_M1,ATMOSPHERE_GROUNDTRUTH_T_C1,ATMOSPHERE_GROUNDTRUTH_RH1,ATMOSPHERE_GROUNDTRUTH_MAXALT1):
    global ATMOSPHERE_GROUNDTRUTH_HPA
    global ATMOSPHERE_GROUNDTRUTH_M
    global ATMOSPHERE_GROUNDTRUTH_T_C
    global ATMOSPHERE_GROUNDTRUTH_RH
    global ATMOSPHERE_GROUNDTRUTH_MAXALT
    
    ATMOSPHERE_GROUNDTRUTH_HPA=ATMOSPHERE_GROUNDTRUTH_HPA1
    ATMOSPHERE_GROUNDTRUTH_M=ATMOSPHERE_GROUNDTRUTH_M1
    ATMOSPHERE_GROUNDTRUTH_T_C=ATMOSPHERE_GROUNDTRUTH_T_C1
    ATMOSPHERE_GROUNDTRUTH_RH=ATMOSPHERE_GROUNDTRUTH_RH1
    ATMOSPHERE_GROUNDTRUTH_MAXALT=ATMOSPHERE_GROUNDTRUTH_MAXALT1
    
    plane_app, plane_geom, plane_uplift, ground_m, phi_deg = apparent_geometric_uplift_target_lla(
        observer_lat_deg=observer_lat_deg,
        observer_lon_deg=observer_lon_deg,
        observer_alt_m=observer_alt_m,
        target_lat_deg=airplane_lat_deg,
        target_lon_deg=airplane_lon_deg,
        target_alt_m=airplane_alt_m,
        settings=settings,
        max_total_path_m=max_total_path_m,
    )
    # Slant range
    if getattr(settings, 'use_wgs84_ellipsoid', True):
        inv = Geodesic.WGS84.Inverse(observer_lat_deg, observer_lon_deg, airplane_lat_deg, airplane_lon_deg)
        ground_s12 = float(inv['s12'])
        az1_deg = float(inv['azi1'])
        R_eff = wgs84_normal_section_radius(observer_lat_deg, az1_deg)
    else:
        ground_s12 = float(ground_m)
        R_eff = float(SPHERE_RADIUS_M)
    phi = ground_s12 / R_eff
    r_obs = R_eff + observer_alt_m
    r_tgt = R_eff + airplane_alt_m
    slantrange = math.sqrt(r_obs*r_obs + r_tgt*r_tgt - 2.0*r_obs*r_tgt*math.cos(phi))
    star_geom = geometric_from_apparent_infinite(
        apparent_elev_deg=plane_app,
        observer_lat_deg=observer_lat_deg,
        observer_alt_m=observer_alt_m,
        settings=settings,
    )
    star_uplift = (plane_app - star_geom) if math.isfinite(star_geom) else float("nan")
    
    return(star_geom, ground_m,plane_app,star_uplift,plane_geom,slantrange,phi_deg)
