import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from matplotlib.ticker import MultipleLocator


st.set_page_config(
    page_title = 'Portfolio Analysis',
    page_icon = 'stock-market.png',
    layout= 'wide'
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

haha1, haha2 = st.columns(2)

with haha1:
    st.title("Risk Porfolio")

    VaR_by_ticker = pd.read_csv('VaR-by-ticker.csv')
    volatility_by_ticker = pd.read_csv('volatility-by-ticker.csv')
    CVaR_by_ticker = pd.read_csv('CVaR-by-Ticker.csv')
    drawdown_csv = pd.read_csv('data-with-drawdown.csv')
    dropped_drawdown_csv = drawdown_csv.dropna(subset=['Return'])

    tickers = dropped_drawdown_csv['Ticker'].unique()

    selected_tickers = st.sidebar.multiselect(
        "Chose Companies:",
        tickers,
        default=tickers[:4]
    )

    if selected_tickers:
        portfolio_data_1 = dropped_drawdown_csv[dropped_drawdown_csv['Ticker'].isin(selected_tickers)]
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

            # calculation of metrices
                # single mean value for each ticker
            mean_of_returns = returns_pivot.mean()
            portfolio_return = np.dot(weights, mean_of_returns)

            cov_matrix = returns_pivot.cov()

                # Series of portfolio-returns
            port_return = returns_pivot.dot(weights)
                # df with cumulative returns as well
            port_return_df = port_return.to_frame(name='Returns')

            port_volatility = port_return.std()
            port_VaR_95 = np.percentile(port_return, 5)
            port_CVaR_95 = port_return[port_return <= port_VaR_95].mean()



            # Metrices

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                f"""
                    <div style="background-color:#141b60; border:0px solid black; box-shadow: 2px 2px 2px rgba(255,255,255,0.4); padding:2px;border-radius:62px;">
                        <h4 style="color:#ffe6d6; font-size:12px; text-align:center;">Port Return</h4>
                        <p style="font-size:15px;color:#ffe6d6;text-align:center;">{portfolio_return*100:.2f}%</p>
                    </div>
                """,
                unsafe_allow_html=True
            )

            with col2:
                st.markdown(
                f"""
                    <div style="background-color:#141b60; border:0px solid black; box-shadow: 2px 2px 2px rgba(255,255,255,0.4); padding:2px;border-radius:62px;">
                        <h4 style="color:#ffe6d6;font-size:12px; text-align:center;">Port Volatility</h4>
                        <p style="font-size:15px;color:#ffe6d6;text-align:center;">{port_volatility*100:.2f}%</p>
                    </div>
                """,
                unsafe_allow_html=True
            )

            with col3:
                st.markdown(
                f"""
                    <div style="background-color:#141b60; border:0px solid black; box-shadow: 2px 2px 2px rgba(255,255,255,0.4); padding:2px;border-radius:62px;">
                        <h4 style="color:#ffe6d6;font-size:12px; text-align:center;">Port VaR</h4>
                        <p style="font-size:15px;color:#ffe6d6;text-align:center;">{port_VaR_95*100:.2f}%</p>
                    </div>
                """,
                unsafe_allow_html=True
            )

            with col4:
                st.markdown(
                f"""
                    <div style="background-color:#141b60; border:0px solid black; box-shadow: 2px 2px 2px rgba(255,255,255,0.4); padding:2px;border-radius:62px;">
                        <h4 style="color:#ffe6d6;font-size:12px; text-align:center;">Port CVaR</h4>
                        <p style="font-size:15px;color:#ffe6d6;text-align:center;">{port_CVaR_95*100:.2f}%</p>
                    </div>
                """,
                unsafe_allow_html=True
            )

            # col1.metric(
            #     'Portfolio Return',
            #     f'{portfolio_return*100:.2f}%'
            # )
            # col2.metric(
            #     'Portfolio Volatility',
            #     f'{port_volatility*100:.2f}%'
            # )
            # col3.metric(
            #     'Portfolio VaR(%)',
            #     f'{port_VaR_95*100:.2f}%'
            # )
            # col4.metric(
            #     'Portfolio CVaR(%)',
            #     f'{port_CVaR_95*100:.2f}%'
            # )
            # Visualizing

            st.markdown("")
            graph1, graph2 = st.columns(2)
            
            # Cumulaitve Returns
            port_return_df['Cumulative Returns'] = (1+port_return_df['Returns']).cumprod() - 1

            with graph1:
                st.markdown('###### Cumulative Returns')

                fig, ax = plt.subplots(figsize=(25,9), facecolor='#b0a2a1')
                ax.plot(port_return_df.index, port_return_df['Cumulative Returns']*100, color="white")
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Returns')
                ax.set_facecolor('#3c85c3')
                ax.set_title("Cumulaitve Returns (%)")
                ax.legend()
                ax.set_xticks(port_return_df.index[::3])
                ax.set_xticklabels(port_return_df.index[::3], rotation=45)
                ax.grid(True)
                st.pyplot(fig)

            # Scatter plot
            with graph2:
                st.markdown('###### Risk vs Returns')
                fig_size = (10, 7) if len(selected_tickers) > 12 else (10, 3.77)
                fig, ax = plt.subplots(figsize=fig_size, facecolor='#675756')

                for t in selected_tickers:
                    ax.scatter(returns_pivot[t].std()*100, returns_pivot[t].mean()*100, label=t, s=100)

                ax.scatter(port_volatility*100, portfolio_return*100, color='olive', marker='o', s=100, label='Portfolio')
                ax.set_title('Risk-Return Scatter Plot')    
                ax.set_facecolor('#93bcde')
                ax.set_xlabel('Volatility')
                ax.set_ylabel('Return')
                ax.legend(selected_tickers, bbox_to_anchor=(1.01,1), loc='upper left', fontsize='small')
                ax.grid(True)
                st.pyplot(fig)

            # Sharpe Ratio

            sharpe_ratio = mean_of_returns.mean()/port_volatility

            # Sortino Ratio
            downside = port_return[port_return < 0]
            downside_deviation = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 0
            if downside_deviation == 0:
                if mean_of_returns.mean() > 0:
                    sortino_ratio = np.inf
                elif mean_of_returns.mean() < 0:
                    sortino_ratio = -np.inf
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = mean_of_returns.mean()/downside_deviation

            # Visualizing Sharpe Ratio and Sortino Ratio

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background:#6c3520;padding:10px;border-radius:10px; box-shadow: 10px 10px 10px rgba(0,0,0,0.7);text-align:center'>
                    <h8 style='color:white;'>Sharpe Ratio </h8>
                    <h8 style='color:yellow;'>{round(sharpe_ratio, 3)}</h8>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='background:#3e4cd7; padding:10px; box-shadow: 10px 10px 10px rgba(0,0,0,0.7); border-radius:10px;text-align:center'>
                    <h8 style='color:#ffe6d6;'>Sortino Ratio</h8>
                    <h8 style='color:yellow;'>{round(sortino_ratio, 3)}</h8>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("")

            # Calculating Rolling Sharpe Ratio

            window = 10
                # Rolling volatility
            rolling_volatility = port_return.rolling(window).std()
            rolling_sharpe_ratio = port_return.rolling(window).mean()/rolling_volatility
            rolling_sharpe_ratio = rolling_sharpe_ratio.dropna()
            rolling_volatility = rolling_volatility.dropna()

                
            # Calculating Rolling Sortino Ratio
            
                # def downside_dev(roll_returns):
                #     downside = roll_returns[roll_returns < 0]
                #     return np.sqrt(np.mean(roll_returns**2)) if len(downside) > 0 else 0
            
            roll_downside_dev = port_return.rolling(window).apply(lambda x: np.sqrt(np.mean(x[x < 0]**2)) if np.any(x < 0) else 0)
            roll_avg = port_return.rolling(window).mean()


            def cal_sortino(roll_avg, roll_downside_dev, indx=0):
                if indx == len(roll_avg):
                    return pd.Series(dtype=float)
                if roll_downside_dev.iloc[indx] == 0:
                    if roll_avg.iloc[indx] > 0:
                        value = np.inf
                    elif roll_avg.iloc[indx] < 0:
                        value = -np.inf
                    else:
                        value = 0.0
                else:
                    value = roll_avg.iloc[indx]/roll_downside_dev.iloc[indx]
                return_data = cal_sortino(roll_avg, roll_downside_dev, indx+1)
                return pd.concat([pd.Series([value]), return_data], ignore_index=True)

            roll_sortino_ratio = cal_sortino(roll_avg, roll_downside_dev)

            # Date column for x-axis
            date_for_graph = pd.Series(rolling_sharpe_ratio.index)
            date_for_graph = pd.to_datetime(date_for_graph)
            date_for_graph = date_for_graph.dt.strftime('%b-%d')

            # Visualizing Rolling Sharpe Ratio
                # Rolling Sharpe Ratio
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15,12), facecolor='orange')
            plt.subplots_adjust(hspace=0.18)
            ax1.plot(date_for_graph, rolling_sharpe_ratio, color='yellow')
            ax1.set_title('Rolling Sharpe Ratio')
            ax1.set_facecolor('#420743')
            # ax1.set_yticks(np.arange(-0.2,0.4,0.05))
            ax1.yaxis.set_major_locator(MultipleLocator(0.05))
            ax1.grid(True)

                # Rolling Volatility
            ax2.plot(date_for_graph, rolling_volatility*100, color='orange')
            ax2.set_title('Rolling Volatility (%)')
            ax2.set_facecolor('#420743')
            # ax2.set_yticks(np.arange(0.010,0.040, 0.002))
            ax2.yaxis.set_major_locator(MultipleLocator(0.2))
            ax2.grid(True)

                # Sortino Ratio
            ax3.plot(date_for_graph, roll_sortino_ratio.dropna(), color='lightblue')
            ax3.set_title('Rolling Sortino')
            ax3.set_xticklabels(date_for_graph, rotation=45)
            ax3.set_xlabel('Date')
            ax3.set_facecolor('#420743')
            # ax3.set_yticks(np.arange(-0.2,0.4,0.05))
            ax3.yaxis.set_major_locator(MultipleLocator(0.05))
            ax3.grid(True)
            st.pyplot(figure)


# SIngle-Ticker Analysis
with haha2:
    co1, co2 = st.columns(2)
    with co1:
        st.markdown("""
                <h4 style="padding-top:7rem; color:yellow ">Single-Ticker Risk Metric</h4>
                """, unsafe_allow_html=True)
        tickers = dropped_drawdown_csv['Ticker'].unique()

        selected_ticker = st.selectbox(
            'Choose a Company',
            tickers
        )
        # Drawdown For Ticker
        st.markdown(f'###### Drawndown per day for {selected_ticker}')

        drawdown_for_ticker = dropped_drawdown_csv.groupby('Ticker')
        drawdown_for_ticker = drawdown_for_ticker.get_group(selected_ticker)

        # color code
        color_map = plt.get_cmap('tab10')
        color_index = list(tickers).index(selected_ticker) % 10
        color = color_map(color_index+2)

        fig, ax = plt.subplots(figsize=(25,8), facecolor='#ffe6d6')
        ax.plot(drawdown_for_ticker['Date'],drawdown_for_ticker['Drawdown'], marker= 'o', color="#ffe6d6")
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_facecolor(color)
        ax.set_xticklabels(drawdown_for_ticker['Date'], rotation=30)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("---------------------")

        # Donut CHart for percentage of return of each stock
        st.markdown('###### Return (%) per Ticker')
        weights = pd.Series(weights, index=selected_tickers)
        weighted_returns = mean_of_returns * weights
        percentage_of_returns = weighted_returns/weighted_returns.sum()*100

        fig = go.Figure(data=[go.Pie(
                labels= selected_tickers,
                values=abs(percentage_of_returns),
                hole=0.5,
                hovertext=[f"{'+' if val>0 else ''}{val:.2f}%" for ticker, val in percentage_of_returns.items()],
                hoverinfo ='text',
                textinfo='none'
        )])
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),width=80, height=130,
            legend=dict(
                x=0,
                y=0.5,
                xanchor='right',
                yanchor='top'
            )
        )

        st.plotly_chart(fig)


    with co2:
        # Volatility, VaR, Cvar Calculation
        avg_return_per_stock = drawdown_csv[drawdown_csv['Ticker'] == selected_ticker]['Return'].mean()
        volatility = volatility_by_ticker[volatility_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]
        VaR = VaR_by_ticker[VaR_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]
        CVaR = CVaR_by_ticker[CVaR_by_ticker['Ticker'] == selected_ticker]['Return'].values[0]
        st.markdown(f"""
            <style>
                .custom-table {{
                    margin-top:7rem;
                    border-collapse: collapse;
                    width: 100%;
                }}
                .custom-table th, .custom-table td{{
                    border: 1px solid #141b60;
                    padding: 8px;
                    text-align: center;
                }}
                .custom-table th{{
                    background-color: yellow;
                    color: #141b60;
                }}
                .custom-table td{{
                    background-color: #3e4cd7;
                }}
            </style>
            <table class="custom-table">
                <tr>
                    <th>Return</th>
                    <th>Volatility</th>
                    <th>VaR</th>
                    <th>CVaR</th>
                </tr>
                <tr>
                    <td>{avg_return_per_stock:.4f}</td>
                    <td>{volatility:.2f}</td>
                    <td>{VaR:.2f}</td>
                    <td>{CVaR:.2f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True
        )

        # Histogram of Returns
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown(f'###### Distribution of Daily Returns - {selected_ticker}')

        color2 = color_map(color_index+3)

        return_by_ticker = dropped_drawdown_csv.groupby('Ticker')
        return_by_ticker = return_by_ticker.get_group(selected_ticker)

        fig1, ax = plt.subplots(figsize=(25,9), facecolor='#5e5152')
        ax.hist(return_by_ticker['Return'] * 100, bins=10, color=color2, edgecolor='black')
        ax.set_facecolor('#897678')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True)

        st.pyplot(fig1)

        st.markdown("---------------------")

        # HEatmap VaR
        vol_selected_tickers = volatility_by_ticker[volatility_by_ticker['Ticker'].isin(selected_tickers)]
        var_selected_tickers = VaR_by_ticker[VaR_by_ticker['Ticker'].isin(selected_tickers)]
        cvar_selected_tickers = CVaR_by_ticker[CVaR_by_ticker['Ticker'].isin(selected_tickers)]

        heatmap_df = pd.concat([
            vol_selected_tickers.set_index('Ticker')['Return'].rename('Volatility'),
            var_selected_tickers.set_index('Ticker')['Return'].rename('VaR'),
            cvar_selected_tickers.set_index('Ticker')['Return'].rename('CVaR')
        ], axis=1)

        # Visualizing heatmap
        fig, ax = plt.subplots(figsize=(7,4), facecolor="#a684a7")
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='plasma',
            fmt='.2f',
            ax=ax
        )
        ax.set_title("Risk Heatmap")
        st.pyplot(fig)
    

    # Potfolio Divident yeild 

    df_for_yield = drawdown_csv[drawdown_csv['Ticker'].isin(selected_tickers)]
    df_for_yield = df_for_yield[['Date', 'Ticker', 'Dividend Yield']]
    df_for_yield['Weights'] = df_for_yield['Ticker'].map(weights)
    df_for_yield['Weighted_Yield'] = df_for_yield['Weights'] * df_for_yield['Dividend Yield']
    port_div_yield = df_for_yield.groupby('Date')['Weighted_Yield'].sum()
    
    #plot for div yeild

    # fig, ax = subplots()
    # ax.plot(port_div_yield.index, port_div_yield['Weighted_Yield'])
    # st.pyplot()
    st.markdown("###### Portfolio Dividend Yeild")
    fig = go.Figure(data=[go.Scatter(
        x=port_div_yield.index,
        y=port_div_yield,
        mode='lines+markers',
        line=dict(color="#a9e733"),
    )])
    fig.update_layout(margin=dict(t=0,b=0),width=600,height=210,xaxis_title='Date',yaxis_title='Dividend Yield')
    st.plotly_chart(fig)

    # huihui
    # st.write(avg_return_per_stock)