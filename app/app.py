import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PowerTransformer
st.set_page_config(layout="centered")

cost_of_living = pd.read_csv('../data/cleaned/cost_of_living_cleanedV1.csv')

def input_data_streamlit(cost_of_living_dataframe):
    # Values for dropdown menus
    columns = {
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

    # Creating a dictionary to save the inputs
    data = {}

    # creating dropdown menus
    for column, options in columns.items():
        column_formatted = column.replace("_", " ")
        if options:
            data[column] = st.selectbox(f"Select {column_formatted}:", options)
        else:
            data[column] = st.number_input(f"Enter {column_formatted}:", step=1.0)

    # actions after pressing the button predict
    if st.button('Predict'):
        new_row = pd.DataFrame([data], columns=columns.keys())

        # Performing the merge operation
        df_merged = new_row.merge(cost_of_living_dataframe, left_on='employee_residence', right_on='country', how='left').drop(columns='country')

        #applying feature transformations to match the saved transformers, encoders and models
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

        #Splitting data into numerical and categorical
        df_num = df_merged.select_dtypes(np.number)
        df_cat = df_merged.select_dtypes(object)

        #loading saved power transformer for the X
        pickle_file_path = '../ml/transformers/power_transformer_x.pkl'
        with open(pickle_file_path, 'rb') as file:
            pt_x = pickle.load(file)

        #fitting and transforming the numerical df using the saved power transformer
        df_num_pt = pt_x.transform(df_num)

        #creating transformed dataframe
        df_num_transformed = pd.DataFrame(df_num_pt, columns=df_num.columns , index=df_num.index)


        #loading one hot encoder
        pickle_file_path = '../ml/encoders/one_hot_encoder.pkl'
        with open(pickle_file_path, 'rb') as file:
            ohe = pickle.load(file)

        #fitting and transforming the categorical df using the saved one hot encoder
        df_cat_ohe = ohe.transform(df_cat).toarray()

        #creating transformed dataframe
        encoded_feature_names = ohe.get_feature_names_out(df_cat.columns)
        df_cat_encoded = pd.DataFrame(df_cat_ohe, columns=encoded_feature_names, index=df_cat.index)

        #concatenating dataframes
        df_concat = pd.concat([df_num_transformed, df_cat_encoded], axis=1)

        #loading the saved min max scaler
        pickle_file_path = '../ml/scalers/MinMaxScaler_concat.pkl'
        with open(pickle_file_path, 'rb') as file:
            mms = pickle.load(file)

        #applying another Min Max scaler to the dataframe
        concat_scaled = mms.transform(df_concat)

        df_concat_scaled = pd.DataFrame(concat_scaled, columns=df_concat.columns , index=df_concat.index)

        #loading the saved model
        pickle_file_path = '../ml/models/gradient_boost_regressor.pkl'
        with open(pickle_file_path, 'rb') as file:
            gb_reg = pickle.load(file)

        # predicting the transformed y
        y_pred_transf = gb_reg.predict(df_concat_scaled)

        #loading the power transformer for the y
        pickle_file_path = '../ml/transformers/power_transformer_y.pkl'
        with open(pickle_file_path, 'rb') as file:
            pt_y = pickle.load(file)

        #reshapping the transformed data
        y_train_2d = y_pred_transf.reshape(-1, 1)
        y_pred = pt_y.inverse_transform(y_train_2d)

        #extracting and rounding the predicted value
        salary = round ((y_pred[0][0])/100) * 100

        #output header
        st.write('The expected salary for the selected features is approximately :')
        # st.write(f'{user_df} €')

        #output target
        st.markdown("""
        <style>
        .big-font {
            font-size:50px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<p class="big-font">{salary} €</p>', unsafe_allow_html=True)

    # Return an empty DataFrame if button is not pressed
    return pd.DataFrame(columns=columns.keys())

# Use the function in Streamlit
st.title('Jobs in Data - Wage Prediction')
user_df = input_data_streamlit(cost_of_living)
