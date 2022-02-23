
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np

st.markdown('''
# 🚗 **New Vehicle Recommender Engine** 🛻

##### Please fill in all the feature fields to have the algorithm suggest the top vehicles for you.
''')
df = pd.read_csv('https://raw.githubusercontent.com/Jeepofun/Final-Project/main/Data/ordinal2.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.rename(columns={'Base_ Engine': 'Base_Engine'})

from sklearn.preprocessing import StandardScaler

X = df[['Base_price', 'Body_style', 'Seats', 'Base_Engine', 'Base_Power', 'Mileage', 'Drive', 'Towing', 'Min_cargo_L', 'Front_headroom', 
        'Front_legroom', 'Rear_legroom', 'Rear_shoulder', 'Fuel_capacity', 'Batt', 'Range']].values

# Standardize the features so that no feature dominates the
# distance computations due to unit scale
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

from sklearn.metrics.pairwise import euclidean_distances
# Car the user is looking at

with st.form(key='feature-input'):
        price = st.number_input(label='Price: 10398-599341')
        body = st.selectbox('Body Style:  SUV-1, Sedan-2, Coupe-3, Hatchback-4, Pickup-5, Van-6, Convertible-7, Wagon-8', (1,2,3,4,5,6,7,8))
        seats = st.selectbox('Seats', (2,3,4,5,6,7,8,12))
        engine = st.selectbox('Engine Type:  Gasoline-1, Hybrid-2, Electric-3, Diesel-4, Hydrogen-5',(1,2,3,4,5))
        hp = st.number_input(label='Horsepower: 78-999')
        mileage = st.slider('L/100km', 1.0, 20.5, .1)
        drive = st.selectbox('Drive:  AWD-1, FWD-2, RWD-3, 4WD-4' , (1,2,3,4))
        towing = st.number_input(label='Towing: 0.0-31200.0')
        storage = st.number_input(label='Min interior storage (L): 0.0-7400.0')
        fr_head = st.number_input(label='Front headroom (in.): 34.7-56.4')
        fr_leg = st.number_input(label='Front legroom (in.): 35.8-46.3')
        r_leg = st.number_input(label='Rear legroom (in.): 0.0-44.3')
        shoulder = st.number_input(label='Rear shoulder room (in.): 0.0-71.4')
        fuel = st.number_input(label='Fuel Capacity (L): 0.0-157.0')
        batt = st.number_input(label='Battery Size (kWh): 0.0-135.0')
        range = st.slider('Range (km)', 25, 1138, 1)
        vehicle_features = ""
        submit = st.form_submit_button('Submit')

if submit:
        st.success("Thank you, we are calculating your results")
        vehicle_features = (price, body, seats, engine, hp, mileage, drive,
        towing, storage, fr_head, fr_leg, r_leg, shoulder, fuel,
        batt, range) 

#vehicle_features = [55000,       # Price: 10398-599341
 #                 5,           # Body style (numeric): SUV-1, Sedan-2, Coupe-3, Hatchback-4, Pickup-5, Van-6, Convertible-7, Wagon-8 
  #                5,           # Seats: 2, 3, 4, 5, 6, 7, 8, 12
  #                1,           # Engine Type (numeric): Gasoline-1, Hybrid-2, Electric-3, Diesel-4, Hydrogen-5
   #               300.0,       # Horsepower: 78-999
   #               13.0,        # L/100km: 1.0-20.5
    #              4,           # Drive (numeric): AWD-1, FWD-2, RWD-3, 4WD-4
    #              5000.0,     # Towing: 0.0-31200.0
    #              300.0,         # Min interior storage (L): 0.0-7400.0
    #              40.8,        # Front Headroom(in.): 34.7-56.4
    #              43.9,        # Front Legroom(in.): 35.8-46.3
    #              38.0,         # Rear Legroom(in.): 0.0-44.3 
     #             55.0,         # Rear Shoulderroom(in.): 0.0-71.4
      #            50.0,        # Fuel Capacity (L): 0.0-157.0
     #             1.90,         # Battery Size (kWh): 0.0-135.0
     #             790          # Range (kms): 25-1138 
    #             ]
vehicle_features = scaler.transform([vehicle_features])

# Distance from all other cars
distances = euclidean_distances(X, vehicle_features)
distances = distances.reshape(-1)   # Before it was (n_cars, 1)

# Find the 5 indices with the minimum distance (highest similarity) to the car we're looking at
ordered_indices = distances.argsort()
closest_indices = ordered_indices[:5]

# Get the cars for these indices
closest_vehicles = df.iloc[closest_indices]
st.header('Top 5 Recommended Vehicles')
closest_vehicles