# portfolio_utils.py

# Contains all the helper functions for portfolio optimization and strategies.
import pandas as pd
import yfinance as yf
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.dates as mdates

### Functions for get stock info
def get_stock_data(stock_symbols, start_date, end_date):
    stock_data = {}
    for symbol in stock_symbols:
        stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def get_stock_price(stock_symbols, start_date, end_date):
    stock_data = {}
    for symbol in stock_symbols:
        stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
    return pd.DataFrame(stock_data)

### Functions to calculate technical factors
def calculate_moving_average(stock_prices, short_window, long_window):
    short_ma = stock_prices.rolling(window=short_window).mean()
    long_ma = stock_prices.rolling(window=long_window).mean()
    return short_ma, long_ma

# Calculate Z-score for mean reversion
def calculate_z_score(stock_prices, window):
    rolling_mean = stock_prices.rolling(window=window).mean()
    rolling_std = stock_prices.rolling(window=window).std()
    z_score = (stock_prices - rolling_mean) / rolling_std
    return z_score

def calculate_diff_z_score(etf_prices, index_prices, window):
    price_diff = etf_prices - index_prices
    # 计算历史价差的滚动均值和标准差
    rolling_mean = price_diff.rolling(window=window).mean()
    rolling_std = price_diff.rolling(window=window).std()
    diff_z_score = (price_diff - rolling_mean) / rolling_std
    return diff_z_score

# Adjust Threshold Dynamically
def dynamic_threshold(SSINDX, window):
    std_SSINDX = (SSINDX - SSINDX.mean())/SSINDX.std()
    rolling_std = std_SSINDX.rolling(window).std()
    return 2.0 + 2 * rolling_std

# Portfolio optimization
def portfolio_optimization(returns, cov_matrix):
    num_assets = len(returns)

    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def constraint(weights):
        return np.sum(weights) - 1

    cons = ({'type': 'eq', 'fun': constraint})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(objective, num_assets * [1. / num_assets,], bounds=bounds, constraints=cons)
    
    return result.x

### Display Results
# Calculate max drawdown
def calculate_max_drawdown(portfolio_values):
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = np.max(drawdown) * 100
    return max_drawdown

def calculate_turnover(positions, stock_prices, window):
    """
    Calculate the turnover ratio based on position changes and stock prices.
    
    Parameters:
    - positions: DataFrame or array where each row represents the positions of all assets at a given time
    - stock_prices: DataFrame or array of stock prices corresponding to positions
    
    Returns:
    - turnovers: The average turnover ratio over all periods
    """
    # 计算每个时间段的市值变化
    portfolio_values = positions[1:] * stock_prices[window:]
    weight_changes = np.abs(portfolio_values.diff()).dropna()
    
    # 计算每个时间段的总权重变化
    turnovers = weight_changes.sum(axis=1) / portfolio_values.sum(axis=1).shift(1)
    
    return turnovers.mean()  # 返回平均Turnover

# Summarize PnL analysis
def pnl_summary(portfolio_values, dates, buy_count, sell_count):
    df = pd.DataFrame({'Date': dates, 'Portfolio Value': portfolio_values})
    df.set_index('Date', inplace=True)
    df['Return'] = df['Portfolio Value'].pct_change()
    df['Year'] = df.index.year

    summary = df.groupby('Year').agg({
        'Portfolio Value': ['first', 'last'],
        'Return': ['std']
    })
    summary.columns = ['Start Value', 'End Value', 'Return Std']
    summary['Annual Return %'] = ((summary['End Value'] / summary['Start Value']) - 1) * 100
    summary['Sharpe Ratio'] = summary['Annual Return %']  / (summary['Return Std'] * np.sqrt(252) * 100)
    
    # Calculate Max Drawdown per year
    max_drawdowns = []
    for year in summary.index:
        year_values = df[df['Year'] == year]['Portfolio Value']
        max_drawdowns.append(calculate_max_drawdown(year_values.values))

    summary['Max Drawdown %'] = max_drawdowns
    summary['Buy Count'] = buy_count
    summary['Sell Count'] = sell_count
    
    # Round to 2 decimal places
    summary = summary.round(2)

    return summary

# Summarize total PnL analysis
def pnl_summary_total(portfolio_values, dates, positions, stock_prices, window):
    df = pd.DataFrame({'Date': dates, 'Portfolio Value': portfolio_values})
    df.set_index('Date', inplace=True)
    df['Return'] = df['Portfolio Value'].pct_change()
    
    # 计算投资总年数
    num_years = (dates[-1] - dates[0]).days / 365.0
    # 计算回报率
    total_return = ((df['Portfolio Value'].iloc[-1] / df['Portfolio Value'].iloc[0]) - 1) * 100
    # 计算总体年化回报率
    returns = (df['Portfolio Value'].iloc[-1]/df['Portfolio Value'].iloc[0]) ** (1/num_years) - 1
    # 计算夏普比率
    sharpe_ratio = df['Return'].mean() / df['Return'].std() * np.sqrt(252)
    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(df['Portfolio Value'].values)
    # 计算Turnover
    turnover = calculate_turnover(positions, stock_prices, window=window)
    # 计算Fitness
    fitness = sharpe_ratio * np.sqrt(np.abs(returns)/np.maximum(turnover, 0.125))

    summary_total = pd.DataFrame({
        'Total Return %': [np.round(total_return, 2)],
        'Annualized Return %': [np.round(returns * 100, 2)],
        'Sharpe Ratio': [sharpe_ratio],
        'Max Drawdown %': [max_drawdown],
        'Turnover %': [np.round(turnover * 100, 2)],
        'Fitness': [np.round(fitness, 2)]
    })

    # Round to 2 decimal places
    summary_total = summary_total.round(2)

    return summary_total

### Visualization
# PnL Visualization
def pnl_visualization(dates, portfolio_values, benchmark):
    # Interactive Plot for Portfolio Value
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value'))
    fig.add_trace(go.Scatter(x=dates, y=benchmark, mode='lines',name='SSINDX PnL'))
    fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Portfolio Value', hovermode='x unified')
    fig.show()
    
# Historical Position
def plot_historical_positions(dates, positions, stock_names, stock_prices, window):
    dates = dates[1:]
    # 计算每个股票价值
    stock_values = positions[1:] * stock_prices[window:]
    # print(stock_values)
    
    # 计算总资产
    total_portfolio_value = stock_values.sum(axis=1).values.reshape(-1, 1)  # 移除 keepdims 并调整形状
    
    # 计算每个股票在总资产中的权重
    positions_weight = stock_values / total_portfolio_value * 100
   # 累积求和以形成堆积图效果
    cumulative_weights = np.cumsum(positions_weight, axis=1)
    fig = go.Figure()

    # 创建堆积面积图
    for i, stock in enumerate(stock_names):
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_weights.iloc[:, i],
            mode='none',  # 不显示线，只显示堆积部分
            fill='tonexty' if i > 0 else 'tozeroy',  # 第一个是从零开始填充，后面的堆积在前面的基础上
            name=stock,
            hoverinfo='text',  # 使用自定义的 hover 信息
            text=[f"{stock}: {p:.2f}%" for p in positions_weight.iloc[:, i]],  # 生成每个点的具体信息
            hovertemplate="%{text}<extra></extra>"  # 显示自定义的文本信息
        ))
    
    fig.update_layout(
        title="Historical Stock Positions Over Time",
        xaxis_title="Date",
        yaxis_title="Position Percentage (%)",
        yaxis=dict(range=[0, 100]),  # 确保Y轴从0到100%
        showlegend=True,
        hovermode="x unified"  # 使得悬停时显示统一的标签框
    )
    
    fig.show()
    
# 当前仓位的饼图
def plot_current_positions(weights, stock_names, threshold=0.03):
    # 合并小份额
    other_weight = 0
    filtered_weights = []
    filtered_symbols = []
    for weight, symbol in zip(weights, stock_names):
        if weight < threshold:
            other_weight += weight
        else:
            filtered_weights.append(weight)
            filtered_symbols.append(symbol)
    
    if other_weight > 0:
        filtered_weights.append(other_weight)
        filtered_symbols.append("Other")
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_symbols,
        values=filtered_weights,
        textinfo='label+percent',
        hoverinfo='label+percent+value',
        textfont=dict(size=12),
        hole=0.3  # 如果你想要甜甜圈图效果
    )])
    
    fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
    
    fig.update_layout(
        title_text="Current Portfolio Positions",
        annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    
    fig.show()
  
# 仓位调整提示的热力图
def plot_adjustment_heatmap(dates, adjustments, stock_names):
    disp_dates = pd.to_datetime(dates[-30:])
    disp_adjustments = adjustments[-30:]
    
    # 设置图像大小
    plt.figure(figsize=(12, 8))
    # 创建热力图
    sns.heatmap(disp_adjustments.T, cmap="coolwarm", center=0, linewidths=0.1, linecolor='black', 
                cbar_kws={'label': 'Adjustment Signal'}, yticklabels=stock_names, xticklabels=disp_dates.strftime('%m-%d'))
    
    plt.title("Position Adjustment Signals Over Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Stock Symbol")
   
    plt.show()
    
# 仓位调整表格
def display_position_table(stock_names, positions, stock_prices, adjustments, transaction_cost):
    # 持仓
    current_position = positions[-1]
    previous_position = positions[-2]
    initial_position = positions[0]
    
    # 今日盈亏
    current_prices = stock_prices.iloc[-1]
    previous_prices = stock_prices.iloc[-2]
    initial_prices = stock_prices.iloc[0]
    
    pnl_today = []
    for i, position in enumerate(current_position):
        if position == previous_position[i]:  # 如果仓位没有改变
            pnl = (current_prices.iloc[i] - previous_prices.iloc[i]) * position
        else:  # 如果仓位改变
            pnl = (current_prices.iloc[i] - previous_prices.iloc[i]) * previous_position[i]  # 旧仓位的盈亏
            pnl -= transaction_cost * abs(position - previous_position[i]) * current_prices.iloc[i]  # 扣除手续费
        pnl_today.append(pnl)
    
    # 持仓盈亏（累计总盈亏）
    pnl_total = []
    for i, position in enumerate(current_position):
        cumulative_pnl = (current_prices.iloc[i] - initial_prices.iloc[i]) * position
        cumulative_pnl -= transaction_cost * abs(position - initial_position[i]) * current_prices.iloc[i]  # 扣除手续费
        pnl_total.append(cumulative_pnl)
    
    # 仓位
    current_stock_val = positions[-1] * stock_prices.iloc[-1]
    previous_stock_val = positions[-2] * stock_prices.iloc[-2]
    total_current = current_stock_val.sum(axis=0)
    total_previous = previous_stock_val.sum(axis=0)
    
    current_weights = current_stock_val / total_current
    previous_weights = previous_stock_val / total_previous
    weight_changes = (current_weights - previous_weights) * 100
    data = {
        'Stock Name': stock_names,
        'Holding': current_position.astype(int),
        'PnL Today': np.round(pnl_today, 2),
        'PnL Total': np.round(pnl_total, 2),
        'Y_Weight (%)': np.round(previous_weights * 100, 2),
        'T_Weight (%)': np.round(current_weights * 100, 2),
        'Adjustment Signal': adjustments[-1],
        'Weight Change (%)': np.round(weight_changes, 2)
    }
    df = pd.DataFrame(data)
    
    df = df.sort_values(by='T_Weight (%)', ascending=False)
    return df   