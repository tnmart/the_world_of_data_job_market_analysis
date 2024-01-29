import streamlit as st
import pandas as pd
import numpy as np
import pickle

cost_of_living = pd.read_csv('../data/cleaned/cost_of_living_cleanedV1.csv')

def input_data_streamlit(cost_of_living_dataframe):
    # Values for dropdown menus
    columns = {
        'work_year': [2020, 2021, 2022, 2023, 2024, 2025],

        'employee_residence': ['Germany', 'United States', 'United Kingdom', 'Canada', 'Spain',
       'Ireland', 'South Africa', 'Poland', 'France', 'Czech Republic',
       'Netherlands', 'Pakistan', 'Ukraine', 'Lithuania', 'Portugal',
       'Australia', 'Uganda', 'Colombia', 'Italy', 'Slovenia', 'Romania',
       'Greece', 'India', 'Latvia', 'Mauritius', 'Armenia', 'Croatia',
       'Thailand', 'South Korea', 'Estonia', 'Turkey', 'Philippines',
       'Brazil', 'Qatar', 'Russia', 'Kenya', 'Tunisia', 'Ghana',
       'Belgium', 'Switzerland', 'Ecuador', 'Peru', 'Mexico', 'Moldova',
       'Nigeria', 'Saudi Arabia', 'Argentina', 'Egypt', 'Georgia',
       'Central African Republic', 'Finland', 'Austria', 'Singapore',
       'Sweden', 'Kuwait', 'Cyprus', 'Bosnia and Herzegovina', 'Iran',
       'China', 'Costa Rica', 'Chile', 'Denmark', 'Bolivia',
       'Dominican Republic', 'Indonesia', 'United Arab Emirates',
       'Malaysia', 'Japan', 'Honduras', 'Algeria', 'Vietnam', 'Iraq',
       'Bulgaria', 'Serbia', 'New Zealand', 'Hong Kong', 'Luxembourg',
       'Malta'],

        'experience_level': ['Entry-level', 'Mid-level', 'Senior', 'Executive'],

        'employment_type': ['Full-time', 'Part-time', 'Contract', 'Freelance'],

        'work_setting': ['Hybrid', 'In-person', 'Remote'],

        'company_location': ['Germany', 'United States', 'United Kingdom', 'Canada', 'Spain',
       'Ireland', 'South Africa', 'Poland', 'France', 'Netherlands',
       'Luxembourg', 'Lithuania', 'Portugal', 'Gibraltar', 'Australia',
       'Colombia', 'Ukraine', 'Slovenia', 'Romania', 'Greece', 'India',
       'Latvia', 'Mauritius', 'Russia', 'Italy', 'South Korea', 'Estonia',
       'Czech Republic', 'Brazil', 'Qatar', 'Kenya', 'Denmark', 'Ghana',
       'Sweden', 'Turkey', 'Switzerland', 'Ecuador', 'Mexico', 'Israel',
       'Nigeria', 'Saudi Arabia', 'Argentina', 'Japan',
       'Central African Republic', 'Finland', 'Singapore', 'Croatia',
       'Armenia', 'Bosnia and Herzegovina', 'Pakistan', 'Iran', 'Austria',
       'American Samoa', 'Thailand', 'Philippines', 'Belgium', 'Egypt',
       'Indonesia', 'United Arab Emirates', 'Malaysia', 'Honduras',
       'Algeria', 'Iraq', 'New Zealand', 'Moldova', 'Malta'], 

        'company_size': ['L', 'M', 'S'],

        'job_field': ['Data Engineering', 'Data Science', 'Data Analysis', 'Other']
    }

    # Dictionary to save the inputs
    data = {}

    # Use Streamlit widgets to get user input
    for column, options in columns.items():
        if options:  # If options are provided, use a select box
            data[column] = st.selectbox(f"Select {column}:", options)
        else:  # For numerical input
            data[column] = st.number_input(f"Enter {column}:", step=1.0)

    # Button to add data to DataFrame
    if st.button('Predict'):
        new_row = pd.DataFrame([data], columns=columns.keys())

        # Perform the merge operation here, after the new entry is added
        df_merged = new_row.merge(cost_of_living_dataframe, left_on='employee_residence', right_on='country', how='left').drop(columns='country')
        experience_level = {
        'Entry-level': 1,
        'Mid-level': 2,
        'Senior': 3,
        'Executive': 4,
        }
        df_merged['experience_level'] = df_merged['experience_level'].replace(experience_level)
        employment_type = {
        'Full-time': 4,
        'Contract': 3,
        'Part-time': 2,
        'Freelance':1,
        }
        df_merged['employment_type'] = df_merged['employment_type'].replace(employment_type)
        work_setting = {
        'In-person': 3,
        'Hybrid': 2,
        'Remote': 1,
        }
        df_merged['work_setting'] = df_merged['work_setting'].replace(work_setting)
        company_size = {
        'L': 3,
        'M': 2,
        'S': 1,
        }
        df_merged['company_size'] = df_merged['company_size'].replace(company_size)
        df_num = df_merged.select_dtypes(np.number)
        df_cat = df_merged.select_dtypes(object)


        # Importing scaler I used in the model
        scaler_file = '../ml/scalers/standard_scaler.pkl'

        with open(scaler_file, 'rb') as file:
            loaded_scaler = pickle.load(file)

        # loading the scaler on the user dataframe
        scaled_data = loaded_scaler.transform(df_num)
        sc_df = pd.DataFrame(X_train_num_transformed, columns=X_train_num.columns , index=X_train_num.index)

        return scaled_data


    # Return an empty DataFrame if button is not pressed
    return pd.DataFrame(columns=columns.keys())

# Use the function in Streamlit
st.title('Jobs in Data salary predictor')
user_df = input_data_streamlit(cost_of_living)

if not user_df.empty:
    st.write('Data Entered:')
    st.write(user_df)