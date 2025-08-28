import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title = 'Single-Ticker',
    page_icon='stock-market.png',
    layout= 'wide'
)


st.title("Single-Ticker Risk Metric")

VaR_by_ticker = pd.read_csv('VaR-by-ticker.csv')
volatility_by_ticker = pd.read_csv('volatility-by-ticker.csv')
CVaR_by_ticker = pd.read_csv('CVaR-by-Ticker.csv')
drawdown_csv = pd.read_csv('data-with-drawdown.csv')

tickers = drawdown_csv['Ticker'].unique()

selected_ticker = st.selectbox(
    'Choose a Company: ',
    tickers
)


col1, col2, col3 = st.columns(3)

volatility = volatility_by_ticker[volatility_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]
VaR = VaR_by_ticker[VaR_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]
CVaR = CVaR_by_ticker[CVaR_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]

with col1:
    st.metric(
        'Volatility',
        f'{volatility * 100:.2f}%'
    )
with col2:
    st.metric(
        'VaR',
        f'{VaR * 100:.2f}%',
    )
with col3:
    st.metric(
        'CVaR',
        f'{CVaR * 100:.2f}%',
    )

# Drawdown-Visualization

graph1, graph2 = st.columns(2)

with graph1:
    st.markdown(f'##### Drawndown For {selected_ticker}')

    drawdown_for_ticker = drawdown_csv.groupby('Ticker')
    drawdown_for_ticker = drawdown_for_ticker.get_group(selected_ticker)

    # color code
    color_map = plt.get_cmap('tab10')
    color_index = list(tickers).index(selected_ticker) % 10
    color = color_map(color_index)

    fig, ax = plt.subplots(figsize=(25,8), facecolor='grey')
    ax.plot(drawdown_for_ticker['Date'],drawdown_for_ticker['Drawdown'], label=selected_ticker, marker= 'o', color=color)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_facecolor('#babfc2')
    ax.set_xticklabels(drawdown_for_ticker['Date'], rotation=30)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Plotting the returns in an histogram

with graph2:
    st.markdown(f'##### Histogram of Daily Returns for {selected_ticker}')

    color2 = color_map(color_index+1)

    return_by_ticker = drawdown_csv.groupby('Ticker')
    return_by_ticker = return_by_ticker.get_group(selected_ticker)

    fig1, ax = plt.subplots(figsize=(25,8.3), facecolor='#5e5152')
    ax.hist(return_by_ticker['Return'] * 100, bins=10, color=color2, edgecolor='black')
    ax.set_facecolor('#897678')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    st.pyplot(fig1)

# Plotting Risk Heapmap


risk_df = pd.concat([
    VaR_by_ticker.set_index('Ticker')['Return'].rename('VaR'),
    CVaR_by_ticker.set_index('Ticker')['Return'].rename('CVaR'),
    volatility_by_ticker.set_index('Ticker')['Return'].rename('Volatility')
], axis=1).reset_index()

#color code

color3 = color_map(color_index+4)

fig, ax = plt.subplots(figsize=(12,20))
sns.heatmap(
    risk_df.set_index('Ticker'),
    annot= True,
    cmap='coolwarm',
    fmt='.2f',
    ax=ax
)

plt.gcf().set_facecolor(color3)

ax.set_title("Risk heatmap")
st.pyplot(fig)


# Portfolio Analysis

