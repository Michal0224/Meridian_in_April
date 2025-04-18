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
import openpyxl
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

# visualizer.ModelDiagnostics(mmm).plot_prior_and_posterior_distribution()
def prior_posterior_data(mmm):
    media_channels = mmm.input_data.media_channel.to_numpy()

    prior_data = mmm.inference_data.prior['roi_m'].to_numpy()
    prior_data_draws = prior_data.shape[1]
    prior_data = prior_data.reshape(prior_data_draws, len(media_channels))
    prior_df = pd.DataFrame(prior_data, columns=media_channels).assign(distribution='prior')

    posterior_data = mmm.inference_data.posterior['roi_m'].to_numpy()
    posterior_data_draws = posterior_data.shape[0] * posterior_data.shape[1]
    posterior_data = posterior_data.reshape(posterior_data_draws, len(media_channels))
    posterior_df = pd.DataFrame(posterior_data, columns=media_channels).assign(distribution='posterior')
    combined_df = pd.concat([prior_df, posterior_df])

    combined_df.to_pickle('combined_df.pkl')

def prior_posterior_chart(prior):  
    combined_df = pd.read_pickle('combined_df')
    # Filtracja danych dla 'prior' i 'posterior'
    prior_data = combined_df[combined_df['distribution'] == 'prior']
    posterior_data = combined_df[combined_df['distribution'] == 'posterior']
    
    # Tworzenie wykresu density plot dla 'prior' i 'posterior'
    fig_pp = ff.create_distplot(
        [prior_data[prior], posterior_data[prior]], 
        ['Prior Distribution', 'Posterior Distribution'], 
        show_hist=True,  # Wyłącza histogram
        show_rug=False    # Wyłącza paski poniżej wykresu
    )
    
    # Dodanie siatki poziomej i pionowej z linią przerywaną
    fig_pp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', griddash='dash')  # Siatka pionowa
    fig_pp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', griddash='dash')  # Siatka pozioma
    fig_pp.update_layout( paper_bgcolor="white",  # Tło poza obszarem wykresu 
                      plot_bgcolor="white"    # Tło samego wykresu
                     )
    fig_pp.update_xaxes(title_text='ROI')
    # Wyświetlanie wykresu
    st.plotly_chart(fig_pp, key='fig_pp')

def marketing_contribution(mmm):
    Kontrybucja_total = visualizer.MediaSummary(mmm).summary_table()
    filtered_df = Kontrybucja_total[Kontrybucja_total['distribution'] == 'posterior'][['channel', '% contribution']]
    filtered_df['contribution'] = filtered_df['% contribution'].str.split(' ').str[0].str.replace('%', '', regex=False)
    filtered_df = filtered_df[filtered_df['channel'] != 'All Channels']
    baseline_contribution = 100 - filtered_df['contribution'].astype(float).sum()
    baseline_row = pd.DataFrame({'channel': ['Baseline'], 'contribution': [baseline_contribution]})
    processed_df = pd.concat([filtered_df[['channel', 'contribution']], baseline_row], ignore_index=True)
    # Sortowanie danych
    baseline_row = processed_df[processed_df['channel'] == 'Baseline']
    df_without_baseline = processed_df[processed_df['channel'] != 'Baseline']
    sorted_df = df_without_baseline.sort_values(by='contribution', ascending=False)
    final_df = pd.concat([baseline_row, sorted_df], ignore_index=True)
    # Konwersja na typ float dla kolumny 'contribution' (zapewniamy poprawność)
    final_df['contribution'] = final_df['contribution'].astype(float)
    
    final_df.to_pickle('final_df.pkl')

def marketing_contribution_chart():
    df = pd.read_pickle('final_df.pkl')
    # Obliczanie pozycji bazowej (kaskadowy efekt)
    df['start'] = df['contribution'].cumsum() - df['contribution']
    df['end'] = df['contribution'].cumsum()
    
    # Kolory dla każdego paska
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    
    # Tworzenie wykresu w Plotly
    fig_mc = go.Figure()
    
    for index, row in df.iterrows():
        fig_mc.add_trace(go.Bar(
            x=[row['contribution']],  # Wartość contribution na osi X
            y=[row['channel']],       # Kanał na osi Y
            orientation='h',          # Poziomy układ słupków
            base=row['start'],        # Punkt bazowy dla każdego paska
            marker=dict(color=colors[index], opacity=0.8),  # Styl słupków
            name=row['channel']
        ))
    
    # Dodanie wartości liczbowych na końcu każdego paska
    for index, row in df.iterrows():
        fig_mc.add_trace(go.Scatter(
            x=[row['end']], 
            y=[row['channel']],
            text=f"{row['contribution']:.1f}%",  # Formatowanie wartości
            mode="text",
            textfont=dict(size=14, color="black"),  # Zmniejszenie czcionki
            showlegend=False
        ))
    
    # Konfiguracja osi i układu
    fig_mc.update_layout(
        title="Contribution by Baseline and Marketing Channels",
        xaxis_title="Contribution (%)",
        yaxis=dict(
            title="Channel",
            categoryorder="array",  # Ręczna kolejność kategorii
            categoryarray=df['channel'][::-1],  # Odwrócenie kolejności (Baseline na górze)
        ),
        xaxis=dict(range=[0, 100]),  # Skala osi X od 0 do 100%
        template="plotly_white",
        height=400
    )
    
    # Wyświetlenie wykresu
    st.plotly_chart(fig_mc, key='fig_mc')

def media_activities(mmm):
    # Pobieranie danych i ich agregacja
    media = mmm.input_data.media.to_dataframe().reset_index().groupby('media_channel').agg({'media': 'sum'})
    media_spend = mmm.input_data.media_spend.to_dataframe().reset_index().groupby('media_channel').agg({'media_spend': 'sum'})
    media_data = pd.merge(media, media_spend, 'left', 'media_channel')
    
    # Obliczanie całkowitych wydatków
    total_spend = media_data['media_spend'].sum()
    media_data['% of media spend'] = (media_data['media_spend'] / total_spend) * 100
    
    # Formatowanie kolumn
    media_data['media'] = media_data['media'].apply(lambda x: f"{x:,.0f}")  # Format: 100 000
    media_data['media_spend'] = media_data['media_spend'].apply(lambda x: f"{x:,.2f}")  # Format: 100,000.00
    media_data['% of media spend'] = media_data['% of media spend'].apply(lambda x: f"{x:.2f}%")  # Format: 34.85%
    
    media_data.to_pickle('media_data.pkl')

def hill_curves(mmm):
    Krzywe_hill = summarizer.visualizer.MediaEffects(mmm).hill_curves_dataframe()
    K_hill = Krzywe_hill.fillna(0) # lepiej usunąć NaN
    K_hill_chart_data = K_hill[['channel', 'media_units', 'distribution', 'ci_hi', 'ci_lo', 'mean']]

    K_hill_chart_data.to_pickle('K_hill_chart_data.pkl')
    
def hill_curves_chart(hill):
    K_hill_chart_data = pd.read_pickle('K_hill_chart_data.pkl')
    hill_data = K_hill_chart_data[K_hill_chart_data['channel'] == hill]
    fig_hill = go.Figure()
    
    # Linie wypełnienia dla 'prior'
    prior_data = hill_data[hill_data['distribution'] == 'prior']
    fig_hill.add_trace(go.Scatter(
        x=prior_data['media_units'],
        y=prior_data['ci_lo'],
        fill=None,
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Prior (CI Low)'
    ))
    fig_hill.add_trace(go.Scatter(
        x=prior_data['media_units'],
        y=prior_data['ci_hi'],
        fill='tonexty',  # Wypełnienie do poprzedniej linii (CI Low)
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Prior (CI High)'
    ))
    
    # Linie wypełnienia dla 'posterior'
    posterior_data = hill_data[hill_data['distribution'] == 'posterior']
    fig_hill.add_trace(go.Scatter(
        x=posterior_data['media_units'],
        y=posterior_data['ci_lo'],
        fill=None,
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Posterior (CI Low)'
    ))
    fig_hill.add_trace(go.Scatter(
        x=posterior_data['media_units'],
        y=posterior_data['ci_hi'],
        fill='tonexty',  # Wypełnienie do poprzedniej linii (CI Low)
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Posterior (CI High)'
    ))
    
    # Linie 'mean' dla 'prior'
    fig_hill.add_trace(go.Scatter(
        x=prior_data['media_units'],
        y=prior_data['mean'],
        mode='lines+markers',
        name='Prior (Mean)',
        line=dict(color='blue', dash='solid')
    ))
    
    # Linie 'mean' dla 'posterior'
    fig_hill.add_trace(go.Scatter(
        x=posterior_data['media_units'],
        y=posterior_data['mean'],
        mode='lines+markers',
        name='Posterior (Mean)',
        line=dict(color='red', dash='solid')
    ))
    
    # Konfiguracja osi i układu
    fig_hill.update_layout(
        title= f"Comparison of Prior and Posterior for {hill}",
        xaxis_title="Media Units",
        yaxis_title="Values",
        template="plotly_white",
        legend_title="Legend",
        height=500,
        width=1200
    )
    
    st.plotly_chart(fig_hill, key='fig_hill')

def adstock_curves(mmm):
    Krzywe_adstock = summarizer.visualizer.MediaEffects(mmm).adstock_decay_dataframe()
    K_adstock = Krzywe_adstock.fillna(0) # lepiej usunąć NaN
    K_adstock_chart_data = K_adstock[['channel', 'time_units', 'distribution', 'ci_hi', 'ci_lo', 'mean']]

    K_adstock_chart_data.to_pickle('K_adstock_chart_data.pkl')

def adstock_curves_chart(adstock):
    K_adstock_chart_data = pd.read_pickle('K_adstock_chart_data.pkl')
    adstock_data = K_adstock_chart_data[K_adstock_chart_data['channel'] == adstock]
    fig_adstock = go.Figure()
    
    # Linie wypełnienia dla 'prior'
    prior_data = adstock_data[adstock_data['distribution'] == 'prior']
    fig_adstock.add_trace(go.Scatter(
        x=prior_data['time_units'],
        y=prior_data['ci_lo'],
        fill=None,
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Prior (CI Low)'
    ))
    fig_adstock.add_trace(go.Scatter(
        x=prior_data['time_units'],
        y=prior_data['ci_hi'],
        fill='tonexty',  # Wypełnienie do poprzedniej linii (CI Low)
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Prior (CI High)'
    ))
    
    # Linie wypełnienia dla 'posterior'
    posterior_data = adstock_data[adstock_data['distribution'] == 'posterior']
    fig_adstock.add_trace(go.Scatter(
        x=posterior_data['time_units'],
        y=posterior_data['ci_lo'],
        fill=None,
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Posterior (CI Low)'
    ))
    fig_adstock.add_trace(go.Scatter(
        x=posterior_data['time_units'],
        y=posterior_data['ci_hi'],
        fill='tonexty',  # Wypełnienie do poprzedniej linii (CI Low)
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Posterior (CI High)'
    ))
    
    # Linie 'mean' dla 'prior'
    fig_adstock.add_trace(go.Scatter(
        x=prior_data['time_units'],
        y=prior_data['mean'],
        mode='lines+markers',
        name='Prior (Mean)',
        line=dict(color='blue', dash='solid')
    ))
    
    # Linie 'mean' dla 'posterior'
    fig_adstock.add_trace(go.Scatter(
        x=posterior_data['time_units'],
        y=posterior_data['mean'],
        mode='lines+markers',
        name='Posterior (Mean)',
        line=dict(color='red', dash='solid')
    ))
    
    # Konfiguracja osi i układu
    fig_adstock.update_layout(
        title= f"Comparison of Prior and Posterior for {adstock}",
        xaxis_title="Time Units",
        yaxis_title="Values",
        template="plotly_white",
        legend_title="Legend",
        height=500,
        width=1200
    )
    
    st.plotly_chart(fig_adstock, key='fig_adstock')

def fit_data(mmm):
    fit_tabel = summarizer.visualizer.ModelFit(mmm)._model_fit_data.to_dataframe().reset_index()
    fit_tabel.to_pickle('fit_tabel.pkl')

def fit_chart(fit_m):
    fit_tabel = pd.read_pickle('fit_tabel.pkl')
    filtered_data = fit_tabel[fit_tabel['geo'] == fit_m]
    filtered_data = filtered_data[filtered_data['metric'] == 'mean']
    fig_line = go.Figure()
    
    fig_line.add_trace(go.Scatter(x=filtered_data['time'], y=filtered_data['expected'], 
                                  mode='lines', name='Expected'))
    fig_line.add_trace(go.Scatter(x=filtered_data['time'], y=filtered_data['baseline'], 
                                  mode='lines', name='Baseline'))
    fig_line.add_trace(go.Scatter(x=filtered_data['time'], y=filtered_data['actual'], 
                                  mode='lines', name='Actual'))
    
    fig_line.update_layout(
        title=f'Expected revenu vs. actual revenue for {fit_m}',
        xaxis_title='Time',
        yaxis_title='Values',
        template='plotly_white',
        legend_title='Metrics'
    )
    
    st.plotly_chart(fig_line, use_container_width=True)

def rr_write(mmm):
    RR = analyzer.Analyzer(mmm).predictive_accuracy().to_array()[0,0,1].to_numpy()
    rr_text = f"{RR.item():.4f}"  # Formatuje liczbę z sześcioma miejscami po przecinku
    rr_tex = pd.DataFrame([rr_text], columns=["Formated RR"])  # Tworzenie DataFrame
    rr_tex.to_pickle('rr_tex.pkl')

def cont_total(mmm):
    Cont_total = visualizer.MediaSummary(mmm).summary_table()
    cont_data = Cont_total[['channel', 'distribution', 'spend', 'roi', 'effectiveness']]
    cont_data = cont_data[cont_data['distribution'] == 'posterior']
    co_data = cont_data
    
    def extract_first_number(value):
        try:
            if isinstance(value, float):
                return value
            return float(value.split(" ")[0]) if "(" in value else 0.0
        except (ValueError, AttributeError):
            
            return 0.0
    
    co_data['roi'] = co_data['roi'].apply(extract_first_number)
    co_data['effectiveness'] = co_data['effectiveness'].apply(extract_first_number)
    co_data['spend'] = co_data['spend'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    co_data.fillna(0)
    
    co_data.to_pickle('co_data.pkl')

def cont_chart():
    co_data = pd.read_pickle('co_data.pkl')
    # Tworzenie wykresu w Plotly
    fig_con = px.scatter(
        co_data,
        x='roi',  # Oś X
        y='effectiveness',  # Oś Y
        size='spend',  # Wielkość kółka zależna od 'spend'
        color='channel',  # Kolor kółka według 'channel'
        hover_data=['channel', 'spend', 'roi', 'effectiveness'],  # Dodatkowe informacje w podpowiedzi
        labels={
            'roi': 'ROI',
            'effectiveness': 'Effectiveness',
            'spend': 'Spend'
        },
        title='ROI vs. Effectiveness',
        size_max=50,  # Maksymalny rozmiar kropek
        color_continuous_scale='Viridis'  # Wyraźniejsza skala kolorów (możesz zmienić na inną np. 'Jet', 'RdBu')
    )
    
    # Dostosowanie wyglądu osi, tła i siatki
    fig_con.update_layout(
        plot_bgcolor='white',  # Białe tło
        title=dict(
            font=dict(family="Roboto", size=18, color="#3C4043"),  # Styl tytułu
            x=0.05  # Pozycja tytułu względem osi
        ),
        xaxis=dict(
            title="ROI",  # Tytuł osi X
            tickfont=dict(size=14, family="Roboto", color="#5F6368"),
            gridcolor="lightgrey",  # Kolor siatki
            griddash="dash",  # Linia przerywana
            zeroline=False
        ),
        yaxis=dict(
            title="Effectiveness",  # Tytuł osi Y
            tickfont=dict(size=14, family="Roboto", color="#5F6368"),
            gridcolor="lightgrey",
            griddash="dash",
            zeroline=False
        )
    )
    st.plotly_chart(fig_con, use_container_width=True)

def response_hill(mmm):
    resp_hill = summarizer.visualizer.MediaEffects(mmm).hill_curves_dataframe()
    res_hill = resp_hill[['channel', 'media_units', 'distribution', 'mean']].fillna(0)
    res_hill = res_hill[res_hill['distribution'] == 'posterior']
    res_hill['media spent'] = res_hill['media_units'] * 10000
    res_hill['mean value'] = res_hill['mean'] * 10000
    res_hi = res_hill[['channel','mean value', 'media spent']]
    
    res_hi.to_pickle('res_hi.pkl')

def response_hill_chart():
    res_hi = pd.read_pickle('res_hi.pkl')
    fig_rh = px.line(
        res_hi,
        x='media spent',  # Oś X
        y='mean value',  # Oś Y
        color='channel',  # Linie w różnych kolorach dla każdego kanału
        markers=True,  # Punkty na linii dla każdej wartości
        labels={
            'media spent': 'Media Spent',
            'mean value': 'Mean Value',
            'channel': 'Channel'
        },
        title='Response curves by marketing channel'
    )
    
    # Dostosowanie wyglądu wykresu
    fig_rh.update_layout(
        plot_bgcolor='white',  # Białe tło
        title=dict(
            font=dict(family="Roboto", size=18, color="#3C4043"),  # Styl tytułu
            x=0.05  # Pozycja tytułu względem osi
        ),
        xaxis=dict(
            title="Media Spent",  # Tytuł osi X
            tickfont=dict(size=14, family="Roboto", color="#5F6368"),
            gridcolor="lightgrey",  # Kolor poziomej siatki
            griddash="dash",  # Linia przerywana dla osi X
            zeroline=False,
            showline=True
        ),
        yaxis=dict(
            title="Mean Value",  # Tytuł osi Y
            tickfont=dict(size=14, family="Roboto", color="#5F6368"),
            gridcolor="lightgrey",  # Kolor pionowej siatki
            griddash="dash",  # Linia przerywana dla osi Y
            zeroline=False,
            showline=True
        )
    )

    st.plotly_chart(fig_rh, use_container_width=True)

def func_27(mmm):
    prior_posterior_data(mmm)
    marketing_contribution(mmm)
    media_activities(mmm)
    hill_curves(mmm)
    adstock_curves(mmm)
    fit_data(mmm)
    cont_total(mmm)
    response_hill(mmm)
    rr_write(mmm)
    




   




    



