## specyficzne ustawienia dla google cloud
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow_probability as tfp
import openpyxl
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import meridian
from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
#from meridian.analysis import optimizer
#from meridian.analysis import analyzer
#from meridian.analysis import visualizer
#from meridian.analysis import summarizer
#from meridian.analysis import formatter

import IPython
import pickle as pkl
import max27_func
from max27_func import func_27
import max27_opti_func
from max27_opti_func import opti_func_27

# data loader structure
coord_to_columns = load.CoordToColumns(
    time='time_period',
    geo='geo',
    controls=['competition_impressions'],
    population='population',
    kpi='tot_sale',
    revenue_per_kpi='car_price',
    media=[
        'Youtube_impressions',
        'Facebook_impressions',
        'Instagram_impressions',
        'Display_impressions',
    ],
    media_spend=[
        'YouTube_cost',
        'Facebook_cost',
        'Instagram_cost',
        'Display_cost',
    ]
)

correct_media_to_channel = {
    'Youtube_impressions': 'Youtube',
    'Facebook_impressions': 'Facebook',
    'Instagram_impressions': 'Instagram',
    'Display_impressions': 'Display',
}
correct_media_spend_to_channel = {
    'YouTube_cost': 'Youtube',
    'Facebook_cost': 'Facebook',
    'Instagram_cost': 'Instagram',
    'Display_cost': 'Display',
}

# Data source
file_path = "/home/michal_a_lesniewski/mmm_dump.pkl"
with open(file_path, 'rb') as file:
    mmm = pkl.load(file)

# Configure page
st.set_page_config(layout="wide")
# Create a two-column layout
col1, col2 = st.columns([2, 1])  # Adjust the proportions as needed

with col2:  # Right column
    st.image("/home/michal_a_lesniewski/Cap_logo.png", width=350)
    
with col1:  # Left column
    # Custom style for the header
    st.markdown("<h2 style='font-size:46px; color:black; " \
    "text-decoration:underline red;'>Marketing Mix Modelling Dashboard</h2>", unsafe_allow_html=True)
    
# Styl zakładki bocznej
st.html("""<style>[data-testid="stSidebarContent"] {color: black; background-color: #0072AF} </style>""")
st.sidebar.subheader('Input data & parameters', divider="red") 
if st.sidebar.button('Load new CSV Data', key='but1'):
    loader = load.CsvDataLoader(
        csv_path="/home/michal_a_lesniewski/simulated_data.csv",
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=correct_media_to_channel,
        media_spend_to_channel=correct_media_spend_to_channel,
    )
    data = loader.load()
    #pkl.dump(mmm, open('../data_dump.pkl', 'ab'))   
    #st.sidebar.subheader('Meridian model configuration', divider="red")
    roi_mu = 0.2 #st.sidebar.number_input("Insert a number for roi_mu", min_value=0.01, max_value=1.00, value=0.2, step=0.01)
    roi_sigma = 0.9 #st.sidebar.number_input("Insert a number for roi_sigma", min_value=0.01, max_value=1.00, value=0.9, step=0.01)
    prior = prior_distribution.PriorDistribution(roi_m = tfp.distributions.LogNormal(roi_mu, roi_sigma, name = constants.ROI_M))
    model_spec = spec.ModelSpec(prior = prior)
    mmm = model.Meridian(input_data = data, model_spec=model_spec)
    st.sidebar.subheader('New csv data loaded', divider="grey")

if st.sidebar.button('Run Meridian Model', key='but2'):
    prior_sample = 500 #st.sidebar.number_input("Insert a number of prior sample", value=500)
    numer_of_chains = 7 #st.sidebar.number_input("Insert a number of numer_of_chains", value=7)
    numer_of_adapt = 500 #st.sidebar.number_input("Insert a number of numer_of_adapt", value=500)
    numer_of_burnin = 500 #st.sidebar.number_input("Insert a number of numer_of_burnin", value=500)
    numer_of_keep = 1000 #t.sidebar.number_input("Insert a number of numer_of_keep", value=1000)
    mmm.sample_prior(prior_sample)
    mmm.sample_posterior(n_chains=numer_of_chains, n_adapt=numer_of_adapt, n_burnin=numer_of_burnin, n_keep=numer_of_keep)
    
    output_path = '/home/michal_a_lesniewski/mmm_dump.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'ab') as f:
        pkl.dump(mmm, f)

    #pkl.dump(mmm, open('../mmm_1704.pkl', 'ab')) 
    st.sidebar.subheader('Meridian model ready', divider="grey")
  
if st.sidebar.button('Data Preparation', key='but3'):
    func_27(mmm)
    opti_func_27(mmm)
    st.sidebar.subheader('Data prepared', divider="grey") 

# definicja kontentu strony
show_results = st.sidebar.checkbox('Model Results Summary') #, value=True
if show_results:
    rr_texx = pd.read_pickle('rr_tex.pkl')
    rr_text = rr_texx.loc[0, "Formated RR"]
    st.markdown(f"<h1 style='font-size:26px; color:black; text-decoration:underline; text-decoration-color:blue;'>Model's R² {rr_text}</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Expected revenue vs. actual revenue', divider="blue") 
    from max27_func import fit_chart
    fit_tabel = pd.read_pickle('fit_tabel.pkl')
    fit_geo_list = set(fit_tabel['geo'])
    fit_geo_radio = list(fit_geo_list)
    fit_m = st.radio('Choose a region:', fit_geo_radio, horizontal=True, key='fit_radio')
    fit_chart(fit_m)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Channel Contribution', divider="blue") 
    from max27_func import marketing_contribution_chart
    chart_col, table_col = st.columns([1.8, 1.2])
    
    with chart_col:
        marketing_contribution_chart()
    
    with table_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('Impressions and spend per media channel')
        media_data = pd.read_pickle('media_data.pkl')
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(media_data, use_container_width=True)
    
    from max27_func import prior_posterior_chart
    combined_df = pd.read_pickle('combined_df.pkl')
    channel_list = [column for column in combined_df.columns if column != 'distribution']
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Channel Prior and Posterior Distributions', divider="blue") 
    prior = st.radio('Choose a channel:', channel_list, horizontal=True, key='prior_radio')
    prior_posterior_chart(prior)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Hill saturation curves', divider="blue") 
    from max27_func import hill_curves_chart
    K_hill_chart_data = pd.read_pickle('K_hill_chart_data.pkl')
    hill_channel_list = set(K_hill_chart_data['channel'])
    hill = st.radio('Choose a channel:', list(hill_channel_list), horizontal=True, key='hill_radio')
    hill_curves_chart(hill)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Adstock saturation curves', divider="blue") 
    from max27_func import adstock_curves_chart
    K_adstock_chart_data = pd.read_pickle('K_adstock_chart_data.pkl')
    adstock_channel_list = set(K_adstock_chart_data['channel'])
    adstock = st.radio('Choose a channel:', list(adstock_channel_list), horizontal=True, key='adstock_radio')
    adstock_curves_chart(adstock)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('ROI vs Effectiveness', divider="blue") 
    from max27_func import cont_chart
    co_data = pd.read_pickle('co_data.pkl')
    cont_chart()

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Response curves by marketing channel', divider="blue") 
    from max27_func import response_hill_chart
    res_hi = pd.read_pickle('res_hi.pkl')
    response_hill_chart()

show_opti = st.sidebar.checkbox('Recommended budget allocation', value=False , key='key_opti')
if show_opti:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Standard optimization scenario', divider="blue") 
    from max27_opti_func import chacha_data_chart, chacha_pie_chart, incremental_reve_chart 
    chart_col, tab_col = st.columns([2, 2])  # Dwie kolumny: lewa większa
    with chart_col:
        #st.markdown("<br>", unsafe_allow_html=True)
        chacha_data_chart()
        #st.markdown("<br>", unsafe_allow_html=True)
        chacha_pie_chart()

    with tab_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('**Optimized budget allocation**')
        st.markdown("<br><br>", unsafe_allow_html=True)
        budget_all_tab = pd.read_pickle('budget_all_tab.pkl')
        st.dataframe(budget_all_tab) #, use_container_width=True
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        incremental_reve_chart()

    st.subheader('Optimized budget on response curves', divider="green")
    from max27_opti_func import response_curves_chart
    response_curves_chart()

show_custom_opti = st.sidebar.checkbox('Customized budget allocation', value=False , key='key_custom_opti')
if show_custom_opti:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Customized optimization scenario', divider="blue") 
    #from max26_opti_func import chacha_data, chacha_data_chart, opti_budget_tab

    with st.sidebar.form("my_form"):
        st.write("Enter new budget value")
        budget_num = st.number_input(' ', value=1000000, placeholder="Type a number...", format="%d")       
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("This is new budget for custom optimalization")
    