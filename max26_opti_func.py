## specyficzne ustawienia dla google cloud
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import tensorflow_probability as tfp
import arviz as az

import IPython
import pickle as pkl
import meridian

from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter
 
def chacha_data(mmm):
    from meridian.analysis.optimizer import BudgetOptimizer
    
    optimizer = BudgetOptimizer(mmm)  # mmm = twój model/analyzer
    results = optimizer.optimize()  # budżet np. 1 mln
    box_c = results.optimized_data 
    output_box_df_c = box_c.to_dataframe().reset_index()
    change_in_channel = output_box_df_c[['channel', 'metric', 'spend', 'incremental_outcome']]
    chacha_data = change_in_channel[change_in_channel['metric'] == 'mean']
    output_optimized =  chacha_data[['channel','spend','incremental_outcome']]
    cha_data = chacha_data[['channel', 'spend']]
    cha_data = cha_data.rename(columns={"spend": "Optimized_spend"})
    cha_data.to_pickle('cha_data.pkl')
    output_optimized.to_pickle('output_optimized.pkl')
    
def chacha_data_chart():
    cha_data = pd.read_pickle('cha_data.pkl')
    # Tworzenie wykresu kolumnowego
    fig_cha = px.bar(
        cha_data,
        x="channel",
        y="Optimized_spend",
        title="Change in optimized spend for each channel",
        labels={"Optimized_spend": "$", "channel": "Channel"},
        color_discrete_map={"#4ECDE6": "#4ECDE6", "#F28B82": "#F28B82"},  # Niebieski i czerwony kolor
    )

    # Aktualizacja wyglądu wykresu
    fig_cha.update_layout(
        title_font=dict(size=18, family="Google Sans Display", color="#3C4043"),
        xaxis_title="",
        yaxis_title="$",
        xaxis_tickangle=-45,
        font=dict(family="Roboto", size=12, color="#5F6368"),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_cha, use_container_width=True)

def opti_budget_tab(mmm):
    from meridian.analysis.optimizer import BudgetOptimizer
    optimizer = BudgetOptimizer(mmm)
    results = optimizer.optimize()  # This runs the optimization
    box_b = results.nonoptimized_data  # This gets the response 
    output_box_df_b = box_b.to_dataframe().reset_index()
    nonopti_budget = output_box_df_b[['channel', 'metric', 'spend', 'incremental_outcome']]
    nopti_budget = nonopti_budget[nonopti_budget['metric'] == 'mean']
    nop_budget = nopti_budget[['channel', 'spend']]
    output_nonoptimized = nopti_budget[['channel','spend','incremental_outcome']]
    nop_budget = nop_budget.rename(columns={"spend": "Non-optimized Spend"})
    cha_data = pd.read_pickle('cha_data.pkl')
    budget_allocation_tab = pd.merge(nop_budget, cha_data[['channel', 'Optimized_spend']], on='channel', how='left')
      
    # Formatowanie danych w tabeli
    budget_allocation_tab['Non-optimized Spend'] = budget_allocation_tab['Non-optimized Spend'].apply(lambda x: '{:,.0f}'.format(x).replace(',', ' '))
    budget_allocation_tab['Optimized_spend'] = budget_allocation_tab['Optimized_spend'].apply(lambda x: '{:,.0f}'.format(x).replace(',', ' '))
    budget_all_tab = budget_allocation_tab[['channel', 'Non-optimized Spend', 'Optimized_spend']]
    budget_all_tab.reset_index(drop=True, inplace=True)  # Usunięcie indeksu

    output_nonoptimized.to_pickle('output_nonoptimized.pkl')
    
    return budget_all_tab
    
def chacha_pie_chart():
    cha_data = pd.read_pickle('cha_data.pkl')
    fig_pie = px.pie(
        cha_data,
        names='channel',
        values='Optimized_spend',
        title='Optimized budget allocation',
        color_discrete_sequence=px.colors.qualitative.Set3,  # Możesz dobrać inną paletę
    )
    
    fig_pie.update_traces(textinfo='percent+label', pull=[0.02]*len(cha_data))
    fig_pie.update_layout(
        title_font=dict(size=18, family='Google Sans Display', color='#3C4043'),
        legend_title_text='',
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig_pie, use_container_width=True)

def incremental_reve():
    output_nonoptimized = pd.read_pickle('output_nonoptimized.pkl')
    output_optimized = pd.read_pickle('output_optimized.pkl')
    row_start = pd.DataFrame([{'channel': 'non_optimized', 'incremental_outcome': output_nonoptimized['spend'].sum()}])
    merged = pd.merge(output_optimized[['channel', 'incremental_outcome']], output_nonoptimized[['channel', 'incremental_outcome']],
        on='channel', suffixes=('_opt', '_non'))   
    diff_rows = pd.DataFrame({'channel': merged['channel'],
                              'incremental_outcome': merged['incremental_outcome_opt'] - merged['incremental_outcome_non']})   
    row_end = pd.DataFrame([{'channel': 'optimized','incremental_outcome': output_optimized['spend'].sum()}])
    inc_reve = pd.concat([row_start, diff_rows, row_end], ignore_index=True)
    
    return inc_reve
    
def incremental_reve_chart(inc_reve):
    fig_reve = go.Figure(go.Waterfall(
        name="Optimized incremental revenue across all channels",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(inc_reve) - 2) + ["total"],
        x=inc_reve["channel"],
        textposition="outside",
        text=[f"{x:,.0f}" for x in inc_reve["incremental_outcome"]],
        y=inc_reve["incremental_outcome"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#F28B82"}},
        increasing={"marker": {"color": "#4ECDE6"}},
        totals={"marker": {"color": "#4285F4"}}
    ))
    
    fig_reve.update_layout(
        title="Optimized incremental revenue across all channels",
        title_font=dict(size=20, family="Google Sans Display", color="#3C4043"),
        waterfallgap=0.3,
        yaxis=dict(
            title="Incremental revenue",
            tickformat=",.0f",
            gridcolor="#DADCE0",       # szare linie
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#DADCE0",
            zerolinewidth=1
        ),
        xaxis=dict(
            tickangle=-45
        ),
        font=dict(family="Roboto", size=12, color="#5F6368"),
        plot_bgcolor="white",         # białe tło wykresu
        paper_bgcolor="white"         # białe tło całego obszaru
    )
    st.plotly_chart(fig_reve, use_container_width=True)

def response_curves(mmm):    
    from meridian.analysis.optimizer import BudgetOptimizer
    optimizer = BudgetOptimizer(mmm)
    results = optimizer.optimize()  # This runs the optimization
    box_a = results.get_response_curves()  # This gets the response curves dataset
    output_box_df = box_a.to_dataframe().reset_index()
    output_df = output_box_df[output_box_df['metric'] =='mean']
    output_df.to_pickle('output_df.pkl')

def response_curves_chart():
    output_df = pd.read_pickle('output_df.pkl')
    channels = output_df['channel'].unique()
    
    fig_rc = go.Figure()
    
    # Dodajemy krzywą reakcji dla każdego kanału
    for channel in channels:
        channel_data = output_df[output_df['channel'] == channel]
        
        # Krzywa reakcji
        fig_rc.add_trace(go.Scatter(
            x=channel_data['spend'],
            y=channel_data['incremental_outcome'],
            mode='lines',
            name=f'{channel} Response Curve',
            line=dict(width=3),  # Zwiększona grubość linii
        ))
    
        # Dodajemy punkt aktualnych wydatków dla każdego kanału
        current_spend = channel_data[channel_data['spend'] == channel_data['spend'].iloc[0]]
        fig_rc.add_trace(go.Scatter(
            x=current_spend['spend'],
            y=current_spend['incremental_outcome'],
            mode='markers',
            name=f'{channel} Current Spend',
            marker=dict(size=12, symbol='circle', color='red', line=dict(width=2))  # Większy punkt i czerwona kropka
        ))
    
    # Konfiguracja wykresu
    fig_rc.update_layout(
        title="Response Curves with Current Spend",
        xaxis_title="Spend ($)",
        yaxis_title="Incremental Revenue ($)",
        plot_bgcolor='white',
        hovermode="x unified",
        title_font=dict(family="Roboto", size=20, color="#3C4043"),  # Większy tytuł
        xaxis=dict(
            title_font=dict(size=14, family="Roboto", color="#5F6368"),
            tickfont=dict(size=12),
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title_font=dict(size=14, family="Roboto", color="#5F6368"),
            tickfont=dict(size=12),
            gridcolor="lightgrey",
        ),
        width=1000,  # Szerokość wykresu
        height=800,  # Wysokość wykresu
    )

    st.plotly_chart(fig_rc, use_container_width=True)


