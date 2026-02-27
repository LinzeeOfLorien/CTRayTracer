import pandas as pd
import numpy as np
import ephem
import math
import time
import os
import datetime
import refraction

starcat = pd.read_csv('HIPCatalogue.csv', sep=',', header=0, encoding="utf-8")
J2000time = datetime.datetime(2000, 1, 1, 12, 0, 0)  # January 1st, 2000 at 12:00 UTC

class RefractionSettings:
    wavelength_nm = 550.0
    co2_ppm = 450.0
    segment_m = 50.0
    top_of_atmosphere_m = 80_000.0

def equatorial_to_horizon(dec, ra, lat, lst):
    hour = math.radians(math.degrees(lst) - math.degrees(ra))
    if math.degrees(hour) < 0:
        hour = math.radians(math.degrees(hour) + 360)
    elif math.degrees(hour) > 360:
        hour = math.radians(math.degrees(hour) - 360)
    alt = math.asin(math.sin(dec)*math.sin(lat)+math.cos(dec)*math.cos(lat)*math.cos(hour))
    az = math.acos((math.sin(dec)-math.sin(lat)*math.sin(alt))/(math.cos(lat)*math.cos(alt)))
    if math.sin(hour)>0:
        az = (360 - (az * 180/math.pi))*(math.pi/180)
    az = math.radians(math.degrees(az))
    if math.degrees(az) > 360:
        az = math.radians(math.degrees(az) - 360)
    alt = math.degrees(alt)
    az = math.degrees(az)
    return(alt, az)

def calc_obliquity(T):
    #T = number of julian centuries since 2000 Jan 1.5
    epsilon = 23.43929167 - 0.013004167 * T - 0.000000167 * T**2 + 0.000000502778 * T**3
    return(epsilon)

def calc_nutation(current_time):
    #T = number of julian centuries since 1900 Jan 0.5
    B1900time = datetime.datetime(1900,1,1,00,00,00)
    time_difference = (current_time - B1900time).total_seconds()
    T = (time_difference / (365.25 * 24 * 3600))/100
    A =  100.002136 * T
    L =  279.6967 + 360.0 * (A - int(A))
    B =  5.372617 * T
    Omega =  259.1833 - 360.0 * (B - int(B))
    ecllonDelta = (-17.2*math.sin(math.radians(Omega)) - 1.3*math.sin(math.radians(2*L)))/3600
    ecloblDelta = (9.2*math.cos(math.radians(Omega)) + 0.5*math.cos(math.radians(2*L)))/3600
    return(ecllonDelta,ecloblDelta)
    

def correct_aberration(ra, dec, current_time):
    #This corrects for both stellar aberration AND nutation using the function above, be advised if you want to use this function elsewhere
    J2000time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    time_difference = (current_time - J2000time).total_seconds()
    J2000T = (time_difference / (365.25 * 24 * 3600))/100
    obliquity = calc_obliquity(J2000T)
    ecllonDelta,ecloblDelta = calc_nutation(current_time)
    obliquity +=ecloblDelta
    sun = ephem.Sun(current_time)
    sunra = sun.g_ra
    sundec = sun.g_dec
    suneq = ephem.Equatorial(sunra, sundec, epoch=current_time)
    sunlon = ephem.Ecliptic(suneq).lon
    deltaRA = (-20.5*((math.cos(math.radians(ra))*math.cos((sunlon))*math.cos(math.radians(obliquity))+math.sin(math.radians(ra))*math.sin((sunlon)))/math.cos(math.radians(dec))))/3600
    deltaDec = (-20.5*(math.cos((sunlon))*math.cos(math.radians(obliquity))*(math.tan(math.radians(obliquity))*math.cos(math.radians(dec))-math.sin(math.radians(ra))*math.sin(math.radians(dec)))+math.cos(math.radians(ra))*math.sin(math.radians(dec))*math.sin((sunlon))))/3600
    deltaRAsec = deltaRA*3600
    deltaDecsec = deltaDec*3600
    correctedRA = ra + deltaRA
    correctedDec = dec + deltaDec
    correctedStar = ephem.Equatorial(math.radians(correctedRA), math.radians(correctedDec), epoch=current_time)
    correctedStarEcl = ephem.Ecliptic(correctedStar, epoch=current_time)
    nutdeltaRa = (math.cos(math.radians(obliquity))+math.sin(math.radians(obliquity))*math.sin(math.radians(correctedRA))*math.tan(math.radians(correctedDec)))*ecllonDelta-math.cos(math.radians(correctedRA))*math.tan(math.radians(correctedDec))*ecloblDelta
    nutdeltaDec = math.cos(math.radians(correctedRA))*math.sin(math.radians(obliquity))*ecllonDelta+math.sin(math.radians(correctedRA))*ecloblDelta
    fullycorrectedRA = correctedRA + nutdeltaRa
    fullycorrectedDec = correctedDec + nutdeltaDec
    return(fullycorrectedRA, fullycorrectedDec)

df = pd.read_csv('CelestialTheodoliteData.csv', sep=',', encoding="utf-8")
df_out = pd.DataFrame(columns=df.columns)
totalgood = -1
totalbad = 0
globeerrors = []
celtheoerrors = []
globeangleprediction_list = []
surface_distance_list = []
target_apparent_elevation_list = []
globedelta_list = []
riserunalt_list = []
celtheoerror_list = []
FEerror_list = []
riserunerror_list = []
starname_list = []
starcorrectedRA_list = []
starcorrectedDec_list = []
stargeometricaltobserver_list = []
stargeometricazobserver_list = []
stargeometricaltpeak_list = []
stargeometricazpeak_list = []
targetaltoverobserver_list = []
targetrefraction_list = []
starrefraction_list = []
globetimedelta_list = []
celtheotimedelta_list = []
globeheighterror_list = []
FEheighterror_list = []
winner_list = []
for index, row in df.iterrows():
    if row['Target']=='Weather Balloon':
        totalgood+=1
        df_out.loc[totalgood] = row
        #Correct for geopotential height
        EarthRadius = 6371000.0
        row['Target Alt'] = (EarthRadius * row['Target Alt']) / (EarthRadius - row['Target Alt'])
        #df_out = pd.concat([df_out, row], ignore_index=True)
        currenttime = datetime.datetime(row['Year'],row['Month'],row['Day'],row['Hour'],row['Minute'],row['Second'])
        globeobserver = ephem.Observer()
        globeobserver.lat = math.radians(row['Observer Lat'])
        globeobserver.lon = math.radians(row['Observer Lon'])
        globeobserver.date = currenttime
        globeobserver.epoch = globeobserver.date
        balloon = pd.read_csv(row['Weather Balloon Data File'])
        starname_list.append('')
        starcorrectedRA_list.append('')
        starcorrectedDec_list.append('')
        stargeometricaltobserver_list.append('')
        stargeometricazobserver_list.append('')
        stargeometricaltpeak_list.append('')
        stargeometricazpeak_list.append('')
        globetimedelta_list.append('')
        celtheotimedelta_list.append('')
        balloon["geometric height"] = (EarthRadius * balloon['HGHT m']) / (EarthRadius - balloon['HGHT m'])
        ATMOSPHERE_GROUNDTRUTH_HPA1 = balloon['PRES hPa'].values
        ATMOSPHERE_GROUNDTRUTH_M1 = balloon['geometric height'].values
        ATMOSPHERE_GROUNDTRUTH_T_C1 = balloon['TEMP C'].values
        ATMOSPHERE_GROUNDTRUTH_RH1 = balloon['RELH %'].values
        ATMOSPHERE_GROUNDTRUTH_MAXALT1 = ATMOSPHERE_GROUNDTRUTH_M1[-1]
        globepredictedstargeometricangle, surface_distance,target_apparent_elevation,star_uplift,target_geom,globeslantrange,phi_deg = refraction.target_and_star_geometry_bundle(
            observer_lat_deg=row['Observer Lat'],
            observer_lon_deg=row['Observer Lon'],
            observer_alt_m=row['Observer Alt'],
            airplane_lat_deg=row['Target Lat'],
            airplane_lon_deg=row['Target Lon'],
            airplane_alt_m=row['Target Alt'],
            settings=RefractionSettings,
            max_total_path_m=None,
            ATMOSPHERE_GROUNDTRUTH_HPA1=ATMOSPHERE_GROUNDTRUTH_HPA1,
            ATMOSPHERE_GROUNDTRUTH_M1=ATMOSPHERE_GROUNDTRUTH_M1,
            ATMOSPHERE_GROUNDTRUTH_T_C1=ATMOSPHERE_GROUNDTRUTH_T_C1,
            ATMOSPHERE_GROUNDTRUTH_RH1=ATMOSPHERE_GROUNDTRUTH_RH1,
            ATMOSPHERE_GROUNDTRUTH_MAXALT1=ATMOSPHERE_GROUNDTRUTH_MAXALT1
        )
        starrefraction_list.append(star_uplift)
        globeangleprediction_list.append(globepredictedstargeometricangle)
        target_apparent_elevation_list.append(target_apparent_elevation)
        staralt = float(row['Apparent Alt'])
        globedelta = abs(target_apparent_elevation-staralt)
        globeresidual = (staralt-target_apparent_elevation)
        globedelta_list.append(globeresidual)
        print(globeresidual)
        targetaltaboveobserver = row['Target Alt']-row['Observer Alt']
        targetaltoverobserver_list.append(targetaltaboveobserver)
        riserunalt = math.degrees(math.atan(targetaltaboveobserver/surface_distance))
        riserunalt_list.append(riserunalt)
        celtheodelta = abs(staralt-riserunalt)
        celtheoresidual = (staralt-riserunalt)
        #Append FE error to separate rise/run compared to measured apparent angle list
        FEerror_list.append(celtheoresidual)
        celtheoerror_list.append('')
        globeerrors.append(globedelta)
        FEheighterror = surface_distance*math.tan(math.radians(celtheoresidual))
        FEheighterror_list.append(abs(FEheighterror))
        AngleC = 180-(phi_deg+(target_geom+90))
        AngleA = 180-(AngleC+globeresidual)
        globeheighterror = (globeslantrange*math.sin(math.radians(globeresidual)))/math.sin(math.radians(AngleA))
        globeheighterror_list.append(abs(globeheighterror))
        surface_distance_km = round(surface_distance/1000,2)
        surface_distance_list.append(surface_distance_km)
        targetrefraction = target_apparent_elevation - target_geom
        targetrefraction_list.append(targetrefraction)
        if celtheodelta>globedelta:
            print(row['Target'],globeobserver.date,'Surface Distance: ',surface_distance_km,' rise/run error:',celtheodelta,' globe raytracing error:',globeresidual,' Globe Wins!')
            winner_list.append('Globe Wins!')
        else:
            print(row['Target'],globeobserver.date,'Surface Distance: ',surface_distance_km,' rise/run error:',celtheodelta,' globe raytracing error:',globeresidual,' Cel Theo Wins!')
            winner_list.append('Flat Earth Wins!')
    else:
        RAstardeg = float(row['RA'])
        Decstardeg = float(row['Dec'])
        #Match the star to a known star in the HIP catalogue
        lowestsep = 180
        currenttime = datetime.datetime(row['Year'],row['Month'],row['Day'],row['Hour'],row['Minute'],row['Second'])
        time_difference = (currenttime - J2000time).total_seconds()
        fractional_year = time_difference / (365.25 * 24 * 3600)
        for i2, starrow in starcat.iterrows():
            try:
                totalraprop = (float(starrow['pmRA'])*fractional_year)/(1000*3600)
                totaldecprop = (float(starrow['pmDE'])*fractional_year)/(1000*3600)
                raproped = float(starrow['_RAJ2000'])+(totalraprop)
                decproped = float(starrow['_DEJ2000'])+(totaldecprop)
                starj2000 = ephem.Equatorial(math.radians(raproped), math.radians(decproped), epoch=ephem.J2000)
                starEOD = ephem.Equatorial(starj2000, epoch=currenttime)
                currentsep = ephem.separation((starEOD.ra,starEOD.dec),(math.radians(RAstardeg),math.radians(Decstardeg)))
                if math.degrees(currentsep)<lowestsep:
                    #New best match so go ahead and calculate for aberration, nutation, etc
                    correctedRA, correctedDec = correct_aberration(math.degrees(starEOD.ra), math.degrees(starEOD.dec), currenttime)
                    correctedStar = ephem.Equatorial(math.radians(correctedRA), math.radians(correctedDec), epoch=currenttime)
                    lowestsep = math.degrees(currentsep)
                    name = starrow['HIP']
                    mag = starrow['Vmag']
            except Exception as e:
                pass
                #print(e)
        if lowestsep > 0.1:
            totalbad+=1
        else:
            totalgood+=1
            starname = str('HIP '+str(name))
            print(starname)
            starname_list.append(starname)
            starcorrectedRA_list.append(correctedRA)
            starcorrectedDec_list.append(correctedDec)
            df_out.loc[totalgood] = row
            #Now that we have a match, calculate alt/az and compare.
            globeobserver = ephem.Observer()
            globeobserver.lat = math.radians(row['Observer Lat'])
            globeobserver.lon = math.radians(row['Observer Lon'])
            globeobserver.date = currenttime
            globeobserver.epoch = globeobserver.date
            starDec = correctedStar.dec
            starRA = correctedStar.ra
            staralt, staraz = equatorial_to_horizon(starDec, starRA, globeobserver.lat, globeobserver.sidereal_time())
            stargeometricaltobserver_list.append(staralt)
            stargeometricazobserver_list.append(staraz)
            #Now predict for "celestial theodolite," no refraction, observer at peak
            celtheoobserver = ephem.Observer()
            celtheoobserver.lat = math.radians(row['Target Lat'])
            celtheoobserver.lon = math.radians(row['Target Lon'])
            celtheoobserver.date = currenttime
            celtheoobserver.epoch = celtheoobserver.date
            starDec = correctedStar.dec
            starRA = correctedStar.ra
            celtheoalt, celtheoaz = equatorial_to_horizon(starDec, starRA, celtheoobserver.lat, celtheoobserver.sidereal_time())
            stargeometricaltpeak_list.append(celtheoalt)
            stargeometricazpeak_list.append(celtheoaz)
            
            balloon = pd.read_csv(row['Weather Balloon Data File'])
            EarthRadius = 6371000.0
            balloon["geometric height"] = (EarthRadius * balloon['HGHT m']) / (EarthRadius - balloon['HGHT m'])
            ATMOSPHERE_GROUNDTRUTH_HPA1 = balloon['PRES hPa'].values
            ATMOSPHERE_GROUNDTRUTH_M1 = balloon['geometric height'].values
            ATMOSPHERE_GROUNDTRUTH_T_C1 = balloon['TEMP C'].values
            ATMOSPHERE_GROUNDTRUTH_RH1 = balloon['RELH %'].values
            ATMOSPHERE_GROUNDTRUTH_MAXALT1 = ATMOSPHERE_GROUNDTRUTH_M1[-1]
            #try:
            globepredictedstargeometricangle, surface_distance,target_apparent_elevation,star_uplift,target_geom,globeslantrange,phi_deg = refraction.target_and_star_geometry_bundle(
                observer_lat_deg=row['Observer Lat'],
                observer_lon_deg=row['Observer Lon'],
                observer_alt_m=row['Observer Alt'],
                airplane_lat_deg=row['Target Lat'],
                airplane_lon_deg=row['Target Lon'],
                airplane_alt_m=row['Target Alt'],
                settings=RefractionSettings,
                max_total_path_m=None,
                ATMOSPHERE_GROUNDTRUTH_HPA1=ATMOSPHERE_GROUNDTRUTH_HPA1,
                ATMOSPHERE_GROUNDTRUTH_M1=ATMOSPHERE_GROUNDTRUTH_M1,
                ATMOSPHERE_GROUNDTRUTH_T_C1=ATMOSPHERE_GROUNDTRUTH_T_C1,
                ATMOSPHERE_GROUNDTRUTH_RH1=ATMOSPHERE_GROUNDTRUTH_RH1,
                ATMOSPHERE_GROUNDTRUTH_MAXALT1=ATMOSPHERE_GROUNDTRUTH_MAXALT1
            )
            starrefraction_list.append(star_uplift)
            globeangleprediction_list.append(globepredictedstargeometricangle)
            target_apparent_elevation_list.append(target_apparent_elevation)
            globedelta = abs(globepredictedstargeometricangle-staralt)
            globeerrors.append(globedelta)
            globeresidual = (staralt-globepredictedstargeometricangle)
            print(globeresidual)
            globedelta_list.append(globeresidual)
            targetaltaboveobserver = row['Target Alt']-row['Observer Alt']
            targetaltoverobserver_list.append(targetaltaboveobserver)
            riserunalt = math.degrees(math.atan(targetaltaboveobserver/surface_distance))
            riserunalt_list.append(riserunalt)
            celtheodelta = abs(celtheoalt-riserunalt)
            celtheoerrors.append(celtheodelta)
            celtheoresidual = (celtheoalt-riserunalt)
            FEerror_list.append('')
            celtheoerror_list.append(celtheoresidual)
            surface_distance_km = round(surface_distance/1000,2)
            surface_distance_list.append(surface_distance_km)
            targetrefraction = target_apparent_elevation - target_geom
            targetrefraction_list.append(targetrefraction)
            #From observer to peak position star will be 1 degree higher per 60 nautical miles surface distance
            degreestoincrease = surface_distance_km/111.12
            perfectstargeometric = target_apparent_elevation-star_uplift
            celtheotheoreticalminimum = riserunalt-((perfectstargeometric)+(degreestoincrease))
            FEheighterror = surface_distance*math.tan(math.radians(celtheoresidual))
            FEheighterror_list.append(abs(FEheighterror))
            AngleC = 180-(phi_deg+(target_geom+90))
            AngleA = 180-(AngleC+globeresidual)
            globeheighterror = (globeslantrange*math.sin(math.radians(globeresidual)))/math.sin(math.radians(AngleA))
            globeheighterror_list.append(abs(globeheighterror))
            #Now find time delta
            currenttime = datetime.datetime(row['Year'],row['Month'],row['Day'],row['Hour'],row['Minute'],row['Second'])
            duration_to_add = datetime.timedelta(seconds=1, milliseconds=0)
            new_time = currenttime + duration_to_add
            globeobserver = ephem.Observer()
            globeobserver.lat = math.radians(row['Observer Lat'])
            globeobserver.lon = math.radians(row['Observer Lon'])
            globeobserver.date = new_time
            globeobserver.epoch = globeobserver.date
            starDec = correctedStar.dec
            starRA = correctedStar.ra
            staralt2, staraz2 = equatorial_to_horizon(starDec, starRA, globeobserver.lat, globeobserver.sidereal_time())
            staraltrate = staralt-staralt2
            staraltlast = staralt
            timetocorrect=(globeresidual/staraltrate)
            secondstocorrect = int(timetocorrect)
            millisecondstocorrect = round((timetocorrect-secondstocorrect),3)*1000
            duration_to_add = datetime.timedelta(seconds=secondstocorrect, milliseconds=millisecondstocorrect)
            new_time = currenttime + duration_to_add
            last_time = new_time
            while globeresidual > 0.00028:
                last_time = new_time
                globeobserver.date = last_time
                globeobserver.epoch = globeobserver.date
                staralt3, staraz3 = equatorial_to_horizon(starDec, starRA, globeobserver.lat, globeobserver.sidereal_time())
                globeresidual = (staralt3-globepredictedstargeometricangle)
                staraltrate = (staraltlast-staralt3)/timetocorrect
                timetocorrect=(globeresidual/staraltrate)
                secondstocorrect = int(timetocorrect)
                millisecondstocorrect = round((timetocorrect-secondstocorrect),3)*1000
                duration_to_add = datetime.timedelta(seconds=secondstocorrect, milliseconds=millisecondstocorrect)
                new_time = last_time + duration_to_add
            timedeltaglobe = (last_time - currenttime).total_seconds()
            globetimedelta_list.append(timedeltaglobe)
            currenttime = datetime.datetime(row['Year'],row['Month'],row['Day'],row['Hour'],row['Minute'],row['Second'])
            duration_to_add = datetime.timedelta(seconds=1, milliseconds=0)
            new_time = currenttime + duration_to_add
            celtheoobserver = ephem.Observer()
            celtheoobserver.lat = math.radians(row['Target Lat'])
            celtheoobserver.lon = math.radians(row['Target Lon'])
            celtheoobserver.date = new_time
            celtheoobserver.epoch = celtheoobserver.date
            celtheoalt2, celtheoaz2 = equatorial_to_horizon(starDec, starRA, celtheoobserver.lat, celtheoobserver.sidereal_time())
            celtheoaltrate = celtheoalt-celtheoalt2
            celtheoaltlast = celtheoalt
            timetocorrect=(celtheoresidual/celtheoaltrate)
            secondstocorrect = int(timetocorrect)
            millisecondstocorrect = round((timetocorrect-secondstocorrect),3)*1000
            duration_to_add = datetime.timedelta(seconds=secondstocorrect, milliseconds=millisecondstocorrect)
            new_time = currenttime + duration_to_add
            last_time = new_time
            while celtheoresidual > 0.00028:
                last_time = new_time
                celtheoobserver.date = last_time
                celtheoobserver.epoch = celtheoobserver.date
                celtheoalt3, celtheoaz3 = equatorial_to_horizon(starDec, starRA, celtheoobserver.lat, celtheoobserver.sidereal_time())
                celtheoresidual = (celtheoalt3-riserunalt)
                celtheoaltrate = (celtheoaltlast-celtheoalt3)/timetocorrect
                timetocorrect=(celtheoresidual/celtheoaltrate)
                secondstocorrect = int(timetocorrect)
                millisecondstocorrect = round((timetocorrect-secondstocorrect),3)*1000
                duration_to_add = datetime.timedelta(seconds=secondstocorrect, milliseconds=millisecondstocorrect)
                new_time = last_time + duration_to_add
            timedeltaFE = (last_time - currenttime).total_seconds()
            celtheotimedelta_list.append(timedeltaFE)
            if celtheodelta>globedelta:
                print(row['Target'],globeobserver.date,'Surface Distance: ',surface_distance_km,' cel theo error:',celtheodelta,' globe raytracing error:',globedelta,' Globe Wins!')
                winner_list.append('Globe Wins!')
            else:
                print(row['Target'],globeobserver.date,'Surface Distance: ',surface_distance_km,' cel theo error:',celtheodelta,' globe raytracing error:',globedelta,' Cel Theo Wins!')
                winner_list.append('Celestial Theodolite Wins!')
            df_out = df_out.fillna('')
            print(df_out)
        
df_out['Star Name'] = starname_list
df_out['Star RA (Equinox of Date corrected for proper motion/aberration/nutation)'] = starcorrectedRA_list
df_out['Star Dec (Equinox of Date corrected for proper motion/aberration/nutation)'] = starcorrectedDec_list
df_out['Star Geometric Alt From Observer Location'] = stargeometricaltobserver_list
df_out['Star Geometric Az From Observer Location'] = stargeometricazobserver_list
df_out['Star Geometric Alt From Target Location'] = stargeometricaltpeak_list
df_out['Star Geometric Az From Target Location'] = stargeometricazpeak_list
df_out['Target Surface Distance'] = surface_distance_list
df_out['Target Altitude - Observer Altitude'] = targetaltoverobserver_list
df_out['Flat Earth Rise/Run Angle'] = riserunalt_list
df_out['Target Refraction'] = targetrefraction_list
df_out['Target Apparent Elevation Globe Prediction With Refraction'] = target_apparent_elevation_list
df_out['Star Refraction'] = starrefraction_list
df_out['Star Alt With Refraction Subtracted'] = globeangleprediction_list
df_out['Globe Error (Degrees)'] = globedelta_list
df_out['Celestial Theodolite Error (Degrees)'] = celtheoerror_list
df_out['Flat Earth Rise/Run Error (Degrees)'] = FEerror_list
df_out['Globe Occultation Time Error (Seconds)'] = globetimedelta_list
df_out['Celestial Theodolite Occultation Time Error (Seconds)'] = celtheotimedelta_list
df_out['Globe Target Height Error (meters)'] = globeheighterror_list
df_out['Flat Earth Target Height Error (meters)'] = FEheighterror_list
df_out['Winner'] = winner_list
df_out = df_out.fillna('')
df_out.to_csv('CelestialTheodoliteResults.csv', sep=',', encoding='utf-8', index=False, header=True)
