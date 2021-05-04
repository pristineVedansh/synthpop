# Important Variables
STATE = "Maharashtra"
DISTRICTS = {"Pune": 3132143}
#DISTRICTS = {"Bellary": 2452595}


import time
from ctgan import CTGANSynthesizer
import pandas as pd
from math import ceil
import places
from opencage.geocoder import OpenCageGeocode
import numpy as np
import gc


ctgan_model = CTGANSynthesizer()


print("Loading pretrained model...")
ctgan_gen = ctgan_model.load("{}/{}.pkl".format(STATE, STATE))

print("Loading state details...")
state_details = pd.read_csv("{}/{}_Details.csv".format(STATE, STATE))


state_details = state_details[26:30]
#state_details.astype({'DD_Code': 'int32', 'Population': 'int32'}).dtypes
#DISTRICTS = dict(zip(state_details.District, state_details.Population))

for district, population_counts in DISTRICTS.items():
    gc.collect()
    
    print("Generating {} of population for {}...".format(population_counts, district))
    # Generate samples on the basis of dictionary
    samples = ctgan_gen.sample(population_counts)
    gc.collect()
    print("Generated {} of population for {}...".format(population_counts, district))

    # Split combined caste religion into caste and religion
    caste_religion = pd.DataFrame(samples.CasteReligion.str.split(' ',1).tolist(), columns = ['Religion','Caste'])
    samples = pd.concat([samples, caste_religion], axis=1)
    del samples['CasteReligion']
    gc.collect()

    # Add district name in the generated data
    samples['District'] = district
    print("ADDING JOB COLUMNS...")
    # Generate job columns
    job_generator = CTGANSynthesizer()
    job_generator = job_generator.load("job.pkl")
    jobs = job_generator.sample(population_counts)

    # Split Job into JobLabel and JobID
    job_label_id = pd.DataFrame(jobs.Job.str.rsplit(' ',1).tolist(), columns = ['JobLabel','JobID'])
    samples = pd.concat([samples, job_label_id], axis=1)
    gc.collect()

    #samples['Job']

    # Essential worker columns list
    print("Adding essential workers...")
    essential_list = ['Police', 'Sweepers', 'Sales, shop', 'Shopkeepers', 'Boilermen', 'Nursing', 'Journalists', 'Electrical', 'Food', 'Physicians', 'Mail distributors', 'Loaders', 'Village officials', 'Govt officials', 'Telephone op']
    samples['essential_worker'] = np.where(samples['JobLabel'].isin(essential_list), 1, 0)
    samples.loc[samples['Age'] <=16, ['essential_worker', 'JobLabel', 'JobID']] = 0, "Student", 150

    samples.loc[samples['JobID'] == 'X1', 'JobID'] = 151
    samples.loc[samples['JobID'] == 'EE', 'JobID'] = 152
    samples.loc[samples['JobID'] == 'X9', 'JobID'] = 153
    samples.loc[samples['JobID'] == 'AA', 'JobID'] = 154


    # Public Transport
    print("Adding public transport...")
    unique_jobs = samples.JobLabel.unique()
    transport_dict = {key:1 for key in unique_jobs}
    privTransport_Jobs = ['Police','Govt Officials','Teachers','Engineers','Managerial nec','Production nec','Textile','Professional nec','Journalists','Economists','Jewellery','Shopkeepers','Physicians','Computing op','Mgr manf','Technical sales']

    for job in transport_dict.keys():
        if job in privTransport_Jobs:
            transport_dict[job]=0

    samples['PublicTransport_Jobs'] = np.nan
    for index, row in samples.iterrows():
        val = transport_dict[row['JobLabel']]
        samples.at[index,'PublicTransport_Jobs'] = val 

    print("ADDING LAT LONG FOR VILLAGES...")
    # Adding the Village Names and Lat Long from a file including all these information
    dist_info = pd.read_csv("{}/{}.csv".format(STATE, district))
    dist_info.astype({'TOT_P': 'int32'}).dtypes
    new_df = []
    
    for index, row in dist_info.iterrows():
        for i in range(int(row['TOT_P'])):
            new_df.append([row['Name'], row['Latitude'], row['Longitude']])

    # New dataframe with village name, lat, and lon
    vill_lat_lon = pd.DataFrame(new_df, columns=['VillTownName', 'VillTownLatitude', 'VillTownLongitude'])
    samples = pd.concat([samples, vill_lat_lon], axis=1)
    del vill_lat_lon
    gc.collect()


    print("Adding HHIDs...")
    # HHID addition
    villages_names = samples['VillTownName'].unique().tolist()
    groups = samples.groupby(['VillTownName'])
    subsets = []
    num_members = 6
    hhid = 1
    dist_id = str(int(state_details[state_details.District==district]['DD_Code'].tolist()[0]))

    for village in villages_names:

        each_village = groups.get_group(village)
        count = ceil(len(each_village['Age'].tolist())/6)
        #hhid = each_village['HHID'].unique()
        u_lat = each_village['VillTownLatitude'].unique()
        u_lat = u_lat[0]
        u_lon = each_village['VillTownLongitude'].unique()
        u_lon = u_lon[0]

        # subsets: list of dataframes of each family. We'll merge all these later
        for i in range(count):
            # For the remaining population
            if i == (count-1):
                subdf = each_village.iloc[(i*num_members):]
                u_lat = u_lat + 0.000100
                subdf['H_Lat'] = u_lat
                u_lon = u_lon + 0.000100
                subdf['H_Lon'] = u_lon
                #subdf['HHID'] = u_lon
                hh = dist_id + str(hhid).zfill(7)
                hh = int(hh)
                subdf['HHID'] = hh
                subsets.append(subdf)

            else:
                subdf = each_village.iloc[(i*num_members):(i+1)*num_members]
                u_lat = u_lat + 0.000100
                subdf['H_Lat'] = u_lat
                u_lon = u_lon + 0.000100
                subdf['H_Lon'] = u_lon
                hh = dist_id + str(hhid).zfill(7)
                hh = int(hh)
                subdf['HHID'] = hh
                subsets.append(subdf)

            hhid = hhid + 1
    
    samples = pd.concat(subsets)
    del subsets, groups
    gc.collect()

    # Agent ID
    ll = list(range(1, len(samples) + 1))
    ll = list(map(str, ll)) 
    ll = [dist_id + idt.zfill(8) for idt in ll]
    agnt_id = list(map(int, ll))
    samples['Agent_ID'] = agnt_id

    samples = samples.dropna()
    gc.collect()
    # Workplace ID allocation
    key = "453c9e8c783b49449afc4e0b68bd5287"  # get api key from:  https://opencagedata.com
    geocoder = OpenCageGeocode(key)
    query = str(district) + ", " +  str(STATE) + ", " + str("India")
    geo_results = geocoder.geocode(query)
    geo_lat = geo_results[0]['geometry']['lat']
    geo_long = geo_results[0]['geometry']['lng']

    print("Got Lat Lon for {}".format(district))

    places_object = places.Places(int(dist_id), geo_lat, geo_long, population_counts, 8)
    print("Generating workplaces...")
    places_object.generate_workplaces(list(samples['JobLabel']))
    print("Generating schools...")
    places_object.generate_schools()
    print("Generating public places...")
    places_object.generate_public_places()
    print("Save places...")
    places_object.save_places()
    #print(places_object.public_places)
    adults = samples[samples['Age']>18]
    adults = places_object.assign_workplaces(adults)

    children = samples[samples['Age']<19]
    children = places_object.assign_schools(children)

    total_population = pd.concat([adults,children], axis=0)
    del adults, children
    gc.collect()
    samples = places_object.assign_public_places(total_population)
    del total_population
    gc.collect()
    samples['WorkPlaceID'] = samples['WorkPlaceID'].fillna(0)
    samples['public_place_id'] = samples['public_place_id'].fillna(0)
    samples['school_id'] = samples['school_id'].fillna(0)
    del samples['WorksAtSameCategory']

    
    # Adding adherence to lockdown
    def f(row):
        if (35 <= row['Age'] <=  40):
            val = 0.0

        elif (30 <= row['Age'] <  35):
            val = 0.1

        elif (25 <= row['Age'] <  30):
            val = 0.2

        elif (20 <= row['Age'] <  25):
            val = 0.3

        elif (15 <= row['Age'] <  20):
            val = 0.4

        elif (10 <= row['Age'] <  15):
            val = 0.8

        elif (0 <= row['Age'] <  10):
            val = 1.0
        
        elif (60 <= row['Age'] <  100):
            val = 1.0
        
        elif (40 <= row['Age'] <  60):
            val = 0.9

        else:
            val = 1.0
        return val

    samples['Adherence_to_Intervention'] = samples.apply(f, axis=1)
    samples = samples.astype({'WorkPlaceID': 'int64', 'school_id': 'int64', 'PublicTransport_Jobs': 'int64', 'Age' : 'int64', 'essential_worker' : 'int64'})
    gc.collect()

    # Save the file to csv
    print("Saving file..")
    samples.to_csv("{}/Synthetic/{}_{}.csv".format(STATE, STATE, district), index=False)
    gc.collect()
