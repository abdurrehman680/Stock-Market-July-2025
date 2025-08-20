import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title = 'Portfolio Analysis',
    page_icon = 'stock-market.png',
    layout= 'wide'
)

st.title("Risk Porfolio")

VaR_by_ticker = pd.read_csv('VaR-by-ticker.csv')
volatility_by_ticker = pd.read_csv('volatility-by-ticker.csv')
CVaR_by_ticker = pd.read_csv('CVaR-by-Ticker.csv')
drawdown_csv = pd.read_csv('data-with-drawdown.csv')
drawdown_csv = drawdown_csv.dropna(subset=['Return'])

tickers = drawdown_csv['Ticker'].unique()

selected_tickers = st.sidebar.multiselect(
    "Chose Companies:",
    tickers,
    default=tickers[:4]
)

if selected_tickers:
    portfolio_data_1 = drawdown_csv[drawdown_csv['Ticker'].isin(selected_tickers)]
    portfolio_data = portfolio_data_1.groupby(['Date', 'Ticker'])['Return'].mean().reset_index()
    returns_pivot = portfolio_data.pivot(index='Date', columns='Ticker', values='Return')
    
    returns_pivot = returns_pivot.fillna(0)

    weights = []

    st.sidebar.markdown("### Set Portfolio weights")

    for ticker in selected_tickers:
        w = st.sidebar.number_input(f"Weight for {ticker}", min_value= 0.0, max_value=1.0, value=round(1/len(selected_tickers), 2), step=0.01, format="%.2f")
        weights.append(w)

    weights = np.array(weights)

    if not np.isclose(weights.sum(), 1.0, atol=1e-3):
        st.sidebar.warning(f"Sum of weights is {weights.sum():.2f}, should be 1.00")
    else:

        # caculation of metrices
        mean_returns = returns_pivot.mean()
        portfolio_return = np.dot(weights, mean_returns)

        cov_matrix = returns_pivot.cov()

        port_return = returns_pivot.dot(weights)
        port_return_df = port_return.to_frame(name='Returns')

        port_volatility = port_return.std()
        port_VaR_95 = np.percentile(port_return, 5)
        port_CVaR_95 = port_return[port_return <= port_VaR_95].mean()



        # Metrices

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            'Portfolio Return',
            f'{portfolio_return*100:.2f}%'
        )
        col2.metric(
            'Portfolio Volatility',
            f'{port_volatility*100:.2f}%'
        )
        col3.metric(
            'Portfolio VaR(%)',
            f'{port_VaR_95*100:.2f}%'
        )
        col4.metric(
            'Portfolio CVaR(%)',
            f'{port_CVaR_95*100:.2f}%'
        )
        # Visualizing

        graph1, graph2 = st.columns(2)
        
        # Cumulaitve Returns
        port_return_df['Cumulative Returns'] = (1+port_return_df['Returns']).cumprod() - 1

        with graph1:
            st.markdown('#### Cumulative Returns')

            fig, ax = plt.subplots(figsize=(25,9), facecolor='#deffd7')
            ax.plot(port_return_df.index, port_return_df['Cumulative Returns']*100, color="red")
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Returns')
            ax.set_facecolor('lightblue')
            ax.legend()
            ax.set_xticklabels(port_return_df.index, rotation=45)
            ax.grid(True)
            st.pyplot(fig)

        # Scatter plot
        with graph2:
            st.markdown('#### Risk vs Returns')
            fig_size = (10, 7) if len(selected_tickers) > 12 else (10, 3.77)
            fig, ax = plt.subplots(figsize=fig_size, facecolor='#deffd7')

            for t in selected_tickers:
                ax.scatter(returns_pivot[t].std()*100, returns_pivot[t].mean()*100, label=t, s=100)

            ax.scatter(port_volatility*100, portfolio_return*100, color='olive', marker='o', s=100, label='Portfolio')
            ax.set_title('Risk-Return Scatter Plot')    
            ax.set_facecolor('#deffd7')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Return')
            ax.legend(selected_tickers, bbox_to_anchor=(1.01,1), loc='upper left', fontsize='small')
            ax.grid(True)
            st.pyplot(fig)

        # Calculating Rolling Sharpe Ratio

        window = 10
        rolling_volatility = port_return.rolling(window).std()
        sharpe_ratio = port_return.rolling(window).mean()/rolling_volatility
        sharpe_ratio = sharpe_ratio.dropna()
        rolling_volatility = rolling_volatility.dropna()

        # huihui
        # if st.checkbox("huihui"):
        #     st.write(port_return.mean())
        #     st.write(sharpe_ratio)
            

        # Date column for x-axis
        date_for_graph = pd.Series(sharpe_ratio.index)
        date_for_graph = pd.to_datetime(date_for_graph)
        date_for_graph = date_for_graph.dt.strftime('%b-%d')

        # Visualizing Sharpe Ratio
        figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6), facecolor='#deffd7')
        ax1.plot(date_for_graph, sharpe_ratio, color='yellow', label='Rolling Sharpe Ratio')
        ax1.set_title('Rolling Sharpe Ratio')
        ax1.set_facecolor('grey')

        ax2.plot(date_for_graph, rolling_volatility, color='orange', label='Rolling Volatility')
        ax2.set_title('Rolling Volatility')
        ax2.set_xticklabels(date_for_graph, rotation=45)
        ax2.set_xlabel('Date')
        ax2.set_facecolor('grey')
        ax.legend()
        st.pyplot(figure)
