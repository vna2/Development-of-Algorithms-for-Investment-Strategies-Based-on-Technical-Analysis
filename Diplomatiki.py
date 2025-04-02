import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv

display_global = False
by_cap_only = True


def make_input(input_file, output_folder, start_date, end_date, top_x=25):
    def load_input_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading input data: {e}")
            return pd.DataFrame()

    def get_asset_data(ticker, start_date, end_date):
        try:
            asset = yf.Ticker(ticker)
            asset_info = asset.info
            sector = asset_info.get('sector', 'Unknown')
            industry = asset_info.get('industry', 'Unknown')
            company_name = asset_info.get('shortName', 'Unknown')
            historical_data = asset.history(start=start_date, end=end_date)

            if historical_data.empty:
                return None, None, sector, industry, company_name

            avg_volume = historical_data['Volume'].mean()
            avg_close_price = historical_data['Close'].mean()
            shares_outstanding = asset_info.get('sharesOutstanding', 0)
            avg_market_cap = avg_close_price * shares_outstanding if shares_outstanding > 0 else None

            return avg_market_cap, avg_volume, sector, industry, company_name
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None, None, 'Unknown', 'Unknown', 'Unknown'

    def format_market_cap(market_cap):
        if market_cap is None:
            return None
        if market_cap >= 1e12:
            return f"{market_cap / 1e12:.0f}T".replace('.', ',')
        elif market_cap >= 1e9:
            return f"{market_cap / 1e9:.0f}B".replace('.', ',')
        elif market_cap >= 1e6:
            return f"{market_cap / 1e6:.0f}M".replace('.', ',')
        else:
            return f"{market_cap:,.0f}"

    def format_volume(volume):
        if volume is None:
            return None
        return f"{volume:,.0f}"

    input_data = load_input_data(input_file)

    if 'Sector' not in input_data.columns:
        print(f"Warning: 'Sector' column not found in {input_file}. Using 'Unknown' as default sector.")
        input_data['Sector'] = 'Unknown'

    performance_data = []

    for _, row in input_data.iterrows():
        ticker = row[0]
        market_cap, volume, sector, industry, company_name = get_asset_data(ticker, start_date, end_date)

        if market_cap is None or volume is None:
            sector = 'Unknown'

        performance_data.append({
            'Symbol': ticker,
            'Company Name': company_name,
            'Market Cap': market_cap,
            'Volume': volume,
            'Sector': sector,
            'Industry': industry,
        })

    df = pd.DataFrame(performance_data)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if 'Sector' not in df.columns:
        print(f"Warning: 'Sector' column is missing after data collection.")
        return  # Avoid proceeding if 'Sector' column is missing

    sectors = df['Sector'].unique()

    # Process data for each sector
    for sector in sectors:
        sector_df = df[df['Sector'] == sector]

        sector_by_cap = sector_df.sort_values(by='Market Cap', ascending=False).head(top_x)
        sector_file_cap = os.path.join(output_folder, f"{sector.replace(' ', '_')}_sector_by_cap_top{top_x}.csv")
        sector_by_cap['Market Cap'] = sector_by_cap['Market Cap'].apply(format_market_cap)
        sector_by_cap['Volume'] = sector_by_cap['Volume'].apply(format_volume)
        sector_by_cap.to_csv(sector_file_cap, index=False)
        print(f"Output for sector '{sector}' sorted by market capitalization saved to {sector_file_cap}")
        if not by_cap_only:
            sector_by_volume = sector_df.sort_values(by='Volume', ascending=False).head(top_x)
            sector_file_vol = os.path.join(output_folder, f"{sector.replace(' ', '_')}_sector_by_volume_top{top_x}.csv")
            sector_by_volume['Market Cap'] = sector_by_volume['Market Cap'].apply(format_market_cap)
            sector_by_volume['Volume'] = sector_by_volume['Volume'].apply(format_volume)
            sector_by_volume.to_csv(sector_file_vol, index=False)
            print(f"Output for sector '{sector}' sorted by volume saved to {sector_file_vol}")


def extract_daily_open_close(base_input_path, base_output_path, start_date, end_date):
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if os.path.isdir(region_path):
            for file in os.listdir(region_path):
                if file.endswith(".csv"):
                    input_file_path = os.path.join(region_path, file)
                    try:
                        tickers = pd.read_csv(input_file_path)['Symbol'].tolist()
                    except KeyError:
                        print(f"Error: No 'Symbol' column in {input_file_path}")
                        continue

                    output_dir = os.path.join(base_output_path, region, file.replace(".csv", ""))
                    os.makedirs(output_dir, exist_ok=True)

                    for ticker in tickers:
                        try:
                            ticker_obj = yf.Ticker(ticker)
                            info = ticker_obj.info
                            category = info.get('sector', 'Unknown')
                            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                            if not data.empty:
                                daily_data = data[['Open', 'Close']].reset_index()
                                daily_data['Ticker'] = ticker
                                daily_data['Category'] = category
                                daily_data['Input'] = region

                                ticker_file_path = os.path.join(output_dir, f"{ticker}_daily_data.csv")
                                daily_data.to_csv(ticker_file_path, index=False)
                                print(f"Saved data for {ticker} (Category: {category}) to {ticker_file_path}")
                            else:
                                print(f"No data found for {ticker} between {start_date} and {end_date}")
                        except Exception as e:
                            print(f"Error processing {ticker}: {e}")


def read_asset_list_CSV(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if 'Ticker' in df.columns:
            return df['Ticker'].tolist()
        else:
            return df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Σφάλμα στην ανάγνωση λίστας assets από CSV: {e}")
        return []


def delete_csv_files(path, asset_list):
    for ticker in asset_list:
        filename = os.path.join(path, f"{ticker}_signals.csv")
        if os.path.exists(filename):
            os.remove(filename)

def calculate_final_statistics(df, date_range_folder, algorithm_folder, subfolder, csv_file, by_type):
    if df.empty:
        print(f"Empty File {csv_file} . Skiped.")
        return None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    buy_profits = []
    strong_buy_profits = []
    buy_prices = []
    strong_buy_prices = []
    holding_periods = []

    i = 0
    while i < len(df) - 1:
        if df.iloc[i]['type'] == 'Buy' and df.iloc[i + 1]['type'] == 'Sell':
            buy_price = df.iloc[i]['price']
            sell_price = df.iloc[i + 1]['price']
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            buy_profits.append(profit_percentage)
            buy_prices.append(buy_price)
            holding_periods.append(
                (df.iloc[i + 1]['date'] - df.iloc[i]['date']).days)
            i += 2

        elif df.iloc[i]['type'] == 'Strong Buy' and df.iloc[i + 1]['type'] == 'Sell':
            strong_buy_price = df.iloc[i]['price']
            strong_sell_price = df.iloc[i + 1]['price']
            strong_profit_percentage = ((strong_sell_price - strong_buy_price) / strong_buy_price) * 100
            strong_buy_profits.append(strong_profit_percentage)
            strong_buy_prices.append(strong_buy_price)
            holding_periods.append(
                (df.iloc[i + 1]['date'] - df.iloc[i]['date']).days)
            i += 2

        else:
            i += 1

    buy_trade_count = len(buy_profits)
    strong_buy_trade_count = len(strong_buy_profits)

    buy_wins = len([profit for profit in buy_profits if profit > 0])
    strong_buy_wins = len([profit for profit in strong_buy_profits if profit > 0])

    win_rate_buy = (buy_wins / buy_trade_count * 100) if buy_trade_count > 0 else 0
    win_rate_strong_buy = (strong_buy_wins / strong_buy_trade_count * 100) if strong_buy_trade_count > 0 else 0

    total_trades = buy_trade_count + strong_buy_trade_count
    total_wins = buy_wins + strong_buy_wins
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    total_profit_buy_percentage = sum(buy_profits)
    total_profit_strong_buy_percentage = sum(strong_buy_profits)
    overall_profit_percentage = total_profit_buy_percentage + total_profit_strong_buy_percentage

    avg_profit_buy_percentage = total_profit_buy_percentage / buy_trade_count if buy_trade_count > 0 else 0
    avg_profit_strong_buy_percentage = total_profit_strong_buy_percentage / strong_buy_trade_count if strong_buy_trade_count > 0 else 0

    max_profit_buy = max(buy_profits) if buy_profits else 0
    min_profit_buy = min(buy_profits) if buy_profits else 0
    max_profit_strong_buy = max(strong_buy_profits) if strong_buy_profits else 0
    min_profit_strong_buy = min(strong_buy_profits) if strong_buy_profits else 0

    all_profits = buy_profits + strong_buy_profits
    overall_max_profit = max(all_profits) if all_profits else 0
    overall_min_profit = min(all_profits) if all_profits else 0

    volatility_buy = np.std(buy_prices) if len(buy_prices) > 1 else 0
    volatility_strong_buy = np.std(strong_buy_prices) if len(strong_buy_prices) > 1 else 0
    mean_buy = np.mean(buy_prices) if len(buy_prices) > 0 else 0
    mean_strong_buy = np.mean(strong_buy_prices) if len(strong_buy_prices) > 0 else 0
    relative_volatility_buy = (volatility_buy / mean_buy * 100) if mean_buy > 0 else 0
    relative_volatility_strong_buy = (volatility_strong_buy / mean_strong_buy * 100) if mean_strong_buy > 0 else 0

    all_prices = buy_prices + strong_buy_prices
    overall_volatility = np.std(all_prices) if len(all_prices) > 1 else 0
    overall_mean = np.mean(all_prices) if len(all_prices) > 0 else 0
    overall_relative_volatility = (overall_volatility / overall_mean * 100) if overall_mean > 0 else 0

    avg_holding_period = np.mean(holding_periods) if holding_periods else 0

    trade_periods = (df['date'].max() - df['date'].min()).days  # Ολική περίοδος
    trade_frequency = total_trades / (trade_periods / 30) if trade_periods > 0 else 0  # Συντελεστής για μήνες

    time_weighted_returns = np.prod([1 + (profit / 100) for profit in all_profits]) - 1 if all_profits else 0

    stats = {
        "Date Range": date_range_folder,
        "Algorithm": algorithm_folder,
        "Subfolder": subfolder,
        "CSV File": csv_file,
        "By Type": by_type,
        "Buy Trade Count": buy_trade_count,
        "Strong Buy Trade Count": strong_buy_trade_count,
        "Total Trades": total_trades,
        "Buy Win Rate (%)": win_rate_buy,
        "Strong Buy Win Rate (%)": win_rate_strong_buy,
        "Overall Win Rate (%)": overall_win_rate,
        "Total Profit Buy (%)": total_profit_buy_percentage,
        "Total Profit Strong Buy (%)": total_profit_strong_buy_percentage,
        "Avg Profit Buy (%)": avg_profit_buy_percentage,
        "Avg Profit Strong Buy (%)": avg_profit_strong_buy_percentage,
        "Max Profit Buy (%)": max_profit_buy,
        "Min Profit Buy (%)": min_profit_buy,
        "Max Profit Strong Buy (%)": max_profit_strong_buy,
        "Min Profit Strong Buy (%)": min_profit_strong_buy,
        "Mean Buy": mean_buy,
        "Volatility Buy": volatility_buy,
        "Relative Volatility Buy (%)": relative_volatility_buy,
        "Mean Strong Buy": mean_strong_buy,
        "Volatility Strong Buy": volatility_strong_buy,
        "Relative Volatility Strong Buy (%)": relative_volatility_strong_buy,
        "Overall Profit (%)": overall_profit_percentage,
        "Overall Max Profit (%)": overall_max_profit,
        "Overall Min Profit (%)": overall_min_profit,
        "Overall Mean": overall_mean,
        "Overall Volatility": overall_volatility,
        "Overall Relative Volatility (%)": overall_relative_volatility,
        "Avg Holding Period (Days)": avg_holding_period,
        "Trade Frequency (Trades per Month)": trade_frequency,
        "Time-Weighted Returns (%)": time_weighted_returns
    }

    return stats

def process_and_generate_final_statistics(base_path, output_file):
    statistics = []

    for date_range_folder in os.listdir(base_path):
        date_range_path = os.path.join(base_path, date_range_folder)

        if os.path.isdir(date_range_path) and date_range_folder != "Merged_Files":
            for algorithm_folder in os.listdir(date_range_path):
                algorithm_path = os.path.join(date_range_path, algorithm_folder)

                if os.path.isdir(algorithm_path):
                    for subfolder in os.listdir(algorithm_path):
                        subfolder_path = os.path.join(algorithm_path, subfolder)

                        for root, dirs, files in os.walk(subfolder_path):
                            if "By_cap" in root:
                                by_type = "By_cap"
                            elif "By_volume" in root:
                                by_type = "By_volume"
                            else:
                                by_type = "Unknown"

                            for csv_file in files:
                                csv_file_path = os.path.join(root, csv_file)

                                if csv_file.endswith(".csv") and not csv_file.startswith("Final_Summary"):
                                    if os.path.getsize(csv_file_path) > 0:
                                        try:
                                            df = pd.read_csv(csv_file_path)
                                            #print(f"First few rows of {csv_file_path}:")
                                            #print(df.head())
                                            if "price" not in df.columns or "type" not in df.columns:
                                                print(f"Missing columns in {csv_file_path}, skipping.")
                                                continue

                                            stats = calculate_final_statistics(
                                                df,
                                                date_range_folder,
                                                algorithm_folder,
                                                subfolder,
                                                csv_file,
                                                by_type
                                            )

                                            if stats:
                                                print(f"Statistics for {csv_file}: {stats}")
                                                statistics.append(stats)
                                            else:
                                                print(f"No statistics returned for {csv_file}.")
                                        except pd.errors.EmptyDataError:
                                            print(f"File {csv_file_path} is empty, skipping.")
                                    else:
                                        print(f"File {csv_file_path} is empty, skipping.")

    if statistics:
        stats_df = pd.DataFrame(statistics)

        stats_df.to_csv(output_file, index=False)
        print(f"Summary statistics have been saved to {output_file}")
    else:
        print("No statistics were generated, please check the input files.")

class Wallet:
    def __init__(self, display=display_global):
        self.buy_count = 0
        self.sell_count = 0
        self.buy_not_executed = 0
        self.sell_not_executed = 0
        self.win_count = 0
        self.loss_count = 0
        self.display = display
        self.stock = 0
        self.buy_price = 0
        self.results = []

    def buy(self, price, amount):
        self.stock += amount
        self.buy_price = price
        self.buy_count += 1
        if self.display:
            print(f"Bought {amount} Asset(s) at ${price}")
        return True

    def sell(self, price, amount):
        if self.stock < amount:
            if self.display:
                print("Not enough asset to sell")
            self.sell_not_executed += 1
            return False

        self.stock -= amount
        self.sell_count += 1

        profit = (price - self.buy_price) * amount
        percentage_change = ((price - self.buy_price) / self.buy_price) * 100

        if profit > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.results.append({
            'buy_price': self.buy_price,
            'sell_price': price,
            'profit': profit,
            'percentage_change': percentage_change
        })

        if self.display:
            print(f"Sold {amount} asset(s) at ${price} , Profit: ${profit:.2f} ({percentage_change:.2f}%)")
        return True

    def get_summary(self):
        return {
            'buy_count': self.buy_count,
            'sell_count': self.sell_count,
            'buy_not_executed': self.buy_not_executed,
            'sell_not_executed': self.sell_not_executed,
            'win_count': self.win_count,
            'loss_count': self.loss_count
        }



class Assets:
    def __init__(self, ticker, display=True):
        self.ticker = ticker
        self.category = None
        self.data = None
        self.display = display

    def get_category(self, csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            if 'Ticker' in df.columns:
                df = df[df['Ticker'] == self.ticker]
            if not df.empty:
                self.category = df['Category'].iloc[0] if 'Category' in df.columns else "N/A"
            else:
                print(f"No data for  {self.ticker} , file {csv_path}")
                self.category = "N/A"
            return self.category
        except Exception as e:
            print(f"Erro at find category {self.ticker} from CSV: {e}")
            self.category = "N/A"
            return self.category

    def get_data(self, csv_path, start_date, end_date):

        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            if 'Ticker' in df.columns:
                df = df[df['Ticker'] == self.ticker]
            df = df.loc[start_date:end_date]
            self.data = df.sort_index()
            if self.data.empty:
                if display_global:
                    print(f"No data {self.ticker} from {start_date} - {end_date} at {csv_path}")
            else:
                if self.display:
                    if display_global:
                        print(
                            f"No data {self.ticker} from {start_date} - {end_date} load data succesfully {csv_path}")
        except Exception as e:
            print(f"Error data load {self.ticker} from: {e}")

    def get_price(self, date):
        try:
            return self.data.loc[date, 'Close']
        except KeyError:
            print(f"Δεν υπάρχουν δεδομένα για την ημερομηνία {date}")
            return None

class RSI_Strategy:
    def __init__(self, start_date, end_date, asset, wallet, period=14, display=display_global):
        self.asset = asset
        self.wallet = wallet
        self.display = display
        self.period = period
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_date = start_date
        self.end_date = end_date
        self.start_price = None
        self.end_price = None

        self.calculate_indicators()
        self.apply_signals()

        if self.start_date in self.asset.data.index:
            self.start_price = self.asset.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):
        data = self.asset.data.copy()

        delta = data['Close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss

        # Calculate RSI
        data['RSI'] = 100 - (100 / (1 + rs))

        self.asset.data = data

    def apply_signals(self):
        data = self.asset.data
        data['Buy_Signal'] = False
        data['Sell_Signal'] = False

        for i in range(1, len(data)):
            current_date = data.index[i]
            current_rsi = data['RSI'].iloc[i]

            if current_rsi < 30:
                if not self.has_open_position:
                    data.loc[current_date, 'Buy_Signal'] = True
                    buy_price = data['Close'].iloc[i]
                    if self.wallet.buy(buy_price, 1):
                        self.has_open_position = True
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Buy',
                            'price': buy_price,
                            'profit': None,
                            'percentage_change': None
                        })

            if current_rsi > 70:
                if self.has_open_position:
                    data.loc[current_date, 'Sell_Signal'] = True
                    sell_price = data['Close'].iloc[i]
                    if self.wallet.sell(sell_price, 1):
                        self.has_open_position = False
                        last_trade = self.wallet.results[-1]
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Sell',
                            'price': sell_price,
                            'profit': last_trade['profit'],
                            'percentage_change': last_trade['percentage_change']
                        })

        if self.buy_sell_signals:
            self.end_price = self.buy_sell_signals[-1]['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0
        )
        wallet_summary = self.wallet.get_summary()

        # Count signal types
        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.asset.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, output_dir):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.asset.ticker}_RSI_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_RSI_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "RSI")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = RSI_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class MACD_Strategy:
    def __init__(self, start_date, end_date, asset, wallet, display=display_global, fast_period=12, slow_period=26, signal_period=9):
        self.asset = asset
        self.wallet = wallet
        self.display = display
        self.start_date = start_date
        self.end_date = end_date
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_price = None
        self.end_price = None
        self.calculate_macd()
        self.apply_signals()

        if self.start_date in self.asset.data.index:
            self.start_price = self.asset.data.loc[self.start_date, 'Close']

    def calculate_macd(self):
        self.asset.data['MACD'] = self.asset.data['Close'].ewm(span=self.fast_period, adjust=False).mean() - self.asset.data['Close'].ewm(span=self.slow_period, adjust=False).mean()

        self.asset.data['MACD_Signal'] = self.asset.data['MACD'].ewm(span=self.signal_period, adjust=False).mean()

        self.asset.data['MACD_Histogram'] = self.asset.data['MACD'] - self.asset.data['MACD_Signal']

    def apply_signals(self):
        data = self.asset.data
        data['Buy_Signal'] = False
        data['Sell_Signal'] = False

        for i in range(1, len(data)):
            current_date = data.index[i]
            current_macd = data['MACD'].iloc[i]
            prev_macd = data['MACD'].iloc[i - 1]
            current_macd_signal = data['MACD_Signal'].iloc[i]
            prev_macd_signal = data['MACD_Signal'].iloc[i]

            if current_macd > current_macd_signal and prev_macd <= prev_macd_signal:
                if not self.has_open_position:
                    self.asset.data.loc[current_date, 'Buy_Signal'] = True
                    buy_price = data['Close'].iloc[i]
                    if self.wallet.buy(buy_price, 1):
                        self.has_open_position = True
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Buy',
                            'price': buy_price,
                            'profit': None,
                            'percentage_change': None
                        })

            if current_macd < current_macd_signal and prev_macd >= prev_macd_signal:
                if self.has_open_position:
                    self.asset.data.loc[current_date, 'Sell_Signal'] = True
                    sell_price = data['Close'].iloc[i]
                    if self.wallet.sell(sell_price, 1):
                        self.has_open_position = False
                        last_trade = self.wallet.results[-1]
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Sell',
                            'price': sell_price,
                            'profit': last_trade['profit'],
                            'percentage_change': last_trade['percentage_change']
                        })

        if self.buy_sell_signals:
            self.end_price = self.buy_sell_signals[-1]['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0
        )
        wallet_summary = self.wallet.get_summary()

        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.asset.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, output_dir):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.asset.ticker}_MACD_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_MACD_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "MACD")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = MACD_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class SMA_Strategy:
    def __init__(self, start_date, end_date, stock, wallet, display=display_global, short_period=50, long_period=200):
        self.stock = stock
        self.wallet = wallet
        self.display = display
        self.start_date = start_date
        self.end_date = end_date
        self.short_period = short_period
        self.long_period = long_period
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_price = None
        self.end_price = None
        self.calculate_sma()
        self.apply_signals()
        if self.start_date in self.stock.data.index:
            self.start_price = self.stock.data.loc[self.start_date, 'Close']

    def calculate_sma(self):
        # Calculate short-term SMA
        self.stock.data['SMA_Short'] = self.stock.data['Close'].rolling(window=self.short_period).mean()

        # Calculate long-term SMA
        self.stock.data['SMA_Long'] = self.stock.data['Close'].rolling(window=self.long_period).mean()

    def apply_signals(self):
        self.stock.data['Buy_Signal'] = False
        self.stock.data['Sell_Signal'] = False

        for i in range(1, len(self.stock.data)):
            current_date = self.stock.data.index[i]
            prev_date = self.stock.data.index[i - 1]

            current_sma_short = self.stock.data['SMA_Short'].iloc[i]
            prev_sma_short = self.stock.data['SMA_Short'].iloc[i - 1]
            current_sma_long = self.stock.data['SMA_Long'].iloc[i]
            prev_sma_long = self.stock.data['SMA_Long'].iloc[i - 1]

            if current_sma_short > current_sma_long and prev_sma_short <= prev_sma_long:
                if not self.has_open_position:
                    self.stock.data.loc[current_date, 'Buy_Signal'] = True
                    buy_price = self.stock.data['Close'].iloc[i]
                    if self.wallet.buy(buy_price, 1):
                        self.has_open_position = True
                        self.buy_sell_signals.append({
                            'ticker': self.stock.ticker,
                            'category': self.stock.category,
                            'date': current_date,
                            'type': 'Buy',
                            'price': buy_price,
                            'profit': None,
                            'percentage_change': None
                        })

            if current_sma_short < current_sma_long and prev_sma_short >= prev_sma_long:
                if self.has_open_position:
                    self.stock.data.loc[current_date, 'Sell_Signal'] = True
                    sell_price = self.stock.data['Close'].iloc[i]
                    if self.wallet.sell(sell_price, 1):
                        self.has_open_position = False
                        last_trade = self.wallet.results[-1]
                        self.buy_sell_signals.append({
                            'ticker': self.stock.ticker,
                            'category': self.stock.category,
                            'date': current_date,
                            'type': 'Sell',
                            'price': sell_price,
                            'profit': last_trade['profit'],
                            'percentage_change': last_trade['percentage_change']
                        })

        if self.buy_sell_signals:
            last_trade = self.buy_sell_signals[-1]
            self.end_price = last_trade['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0
        )
        wallet_summary = self.wallet.get_summary()

        # Count signal types
        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.stock.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, path):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{self.stock.ticker}_SMA_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_SMA_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "SMA")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = SMA_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class Dimbeta_Strategy:
    def __init__(self, start_date, end_date, asset, wallet, display=display_global):
        self.asset = asset
        self.wallet = wallet
        self.display = display
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_date = start_date
        self.end_date = end_date
        self.start_price = None
        self.end_price = None

        self.calculate_indicators()
        self.apply_signals()

        if self.start_date in self.asset.data.index:
            self.start_price = self.asset.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):
        data = self.asset.data.copy()
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['Dimbeta'] = ((data['Close'] - data['SMA20']) / data['SMA20']) * 100
        data['EMA20_Dimbeta'] = data['Dimbeta'].ewm(span=20, adjust=False).mean()
        data['STD10_Dimbeta'] = data['Dimbeta'].rolling(window=10).std()
        data['MA10_STD10_Dimbeta'] = data['STD10_Dimbeta'].rolling(window=10).mean()
        data['-MA10_STD10_Dimbeta'] = -data['MA10_STD10_Dimbeta']
        self.asset.data = data

    def apply_signals(self):
        data = self.asset.data
        data['Buy_Signal'] = False
        data['Sell_Signal'] = False

        for i in range(1, len(data)):
            current_date = data.index[i]
            current_dimbeta = data['Dimbeta'].iloc[i]
            prev_dimbeta = data['Dimbeta'].iloc[i - 1]
            current_ema20 = data['EMA20_Dimbeta'].iloc[i]
            prev_ema20 = data['EMA20_Dimbeta'].iloc[i - 1]
            current_std10 = data['STD10_Dimbeta'].iloc[i]
            current_ma10_std10 = data['MA10_STD10_Dimbeta'].iloc[i]
            current_neg_ma10_std10 = data['-MA10_STD10_Dimbeta'].iloc[i]

            if current_dimbeta > current_ema20 and prev_dimbeta <= prev_ema20:
                if not self.has_open_position:
                    if current_ema20 < 0:
                        data.loc[current_date, 'Buy_Signal'] = True
                        buy_price = data['Close'].iloc[i]
                        if self.wallet.buy(buy_price, 1):
                            self.has_open_position = True
                            self.buy_sell_signals.append({
                                'ticker': self.asset.ticker,
                                'category': self.asset.category,
                                'date': current_date,
                                'type': 'Strong Buy',
                                'price': buy_price,
                                'profit': None,
                                'percentage_change': None
                            })
                    else:
                        data.loc[current_date, 'Buy_Signal'] = True
                        buy_price = data['Close'].iloc[i]
                        if self.wallet.buy(buy_price, 1):
                            self.has_open_position = True
                            self.buy_sell_signals.append({
                                'ticker': self.asset.ticker,
                                'category': self.asset.category,
                                'date': current_date,
                                'type': 'Buy',
                                'price': buy_price,
                                'profit': None,
                                'percentage_change': None
                            })

            if current_dimbeta < current_ema20 and prev_dimbeta >= prev_ema20:
                if self.has_open_position:
                    if current_std10 < current_neg_ma10_std10:
                        data.loc[current_date, 'Sell_Signal'] = True
                        sell_price = data['Close'].iloc[i]
                        if self.wallet.sell(sell_price, 1):
                            self.has_open_position = False
                            last_trade = self.wallet.results[-1]
                            self.buy_sell_signals.append({
                                'ticker': self.asset.ticker,
                                'category': self.asset.category,
                                'date': current_date,
                                'type': 'Strong Sell',
                                'price': sell_price,
                                'profit': last_trade['profit'],
                                'percentage_change': last_trade['percentage_change']
                            })
                    else:
                        data.loc[current_date, 'Sell_Signal'] = True
                        sell_price = data['Close'].iloc[i]
                        if self.wallet.sell(sell_price, 1):
                            self.has_open_position = False
                            last_trade = self.wallet.results[-1]
                            self.buy_sell_signals.append({
                                'ticker': self.asset.ticker,
                                'category': self.asset.category,
                                'date': current_date,
                                'type': 'Sell',
                                'price': sell_price,
                                'profit': last_trade['profit'],
                                'percentage_change': last_trade['percentage_change']
                            })

        if self.buy_sell_signals:
            self.end_price = self.buy_sell_signals[-1]['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0)
        wallet_summary = self.wallet.get_summary()

        strong_buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Strong Buy'])
        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        strong_sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Strong Sell'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.asset.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'strong_buy_count': strong_buy_count,
            'buy_count': buy_count,
            'strong_sell_count': strong_sell_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, output_dir):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.asset.ticker}_DimBeta_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_dimbeta_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "Dimbeta")

    # Results storage
    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = Dimbeta_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class Dimbeta_RSI_Strategy:
    def __init__(self, start_date, end_date, asset, wallet, display=True):
        self.asset = asset
        self.wallet = wallet
        self.display = display
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_date = start_date
        self.end_date = end_date
        self.start_price = None
        self.end_price = None

        self.calculate_indicators()
        self.apply_signals()

        if self.start_date in self.asset.data.index:
            self.start_price = self.asset.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):
        data = self.asset.data.copy()

        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['Dimbeta'] = ((data['Close'] - data['SMA20']) / data['SMA20']) * 100
        data['EMA20_Dimbeta'] = data['Dimbeta'].ewm(span=20, adjust=False).mean()

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        self.asset.data = data

    def apply_signals(self):
        data = self.asset.data
        data['Buy_Signal'] = False
        data['Sell_Signal'] = False

        for i in range(1, len(data)):
            current_date = data.index[i]
            current_dimbeta = data['Dimbeta'].iloc[i]
            prev_dimbeta = data['Dimbeta'].iloc[i - 1]
            current_ema20 = data['EMA20_Dimbeta'].iloc[i]
            prev_ema20 = data['EMA20_Dimbeta'].iloc[i - 1]
            current_rsi = data['RSI'].iloc[i]

            if current_dimbeta > current_ema20 and prev_dimbeta <= prev_ema20 and current_rsi < 40:
                if not self.has_open_position:
                    data.loc[current_date, 'Buy_Signal'] = True
                    buy_price = data['Close'].iloc[i]
                    if self.wallet.buy(buy_price, 1):
                        self.has_open_position = True
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Dimbeta RSI Buy',
                            'price': buy_price,
                            'profit': None,
                            'percentage_change': None
                        })

            if current_dimbeta < current_ema20 and prev_dimbeta >= prev_ema20 and current_rsi > 60:
                if self.has_open_position:
                    data.loc[current_date, 'Sell_Signal'] = True
                    sell_price = data['Close'].iloc[i]
                    if self.wallet.sell(sell_price, 1):
                        self.has_open_position = False
                        last_trade = self.wallet.results[-1]
                        self.buy_sell_signals.append({
                            'ticker': self.asset.ticker,
                            'category': self.asset.category,
                            'date': current_date,
                            'type': 'Dimbeta RSI Sell',
                            'price': sell_price,
                            'profit': last_trade['profit'],
                            'percentage_change': last_trade['percentage_change']
                        })

        if self.buy_sell_signals:
            self.end_price = self.buy_sell_signals[-1]['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0)
        wallet_summary = self.wallet.get_summary()

        dimbeta_rsi_buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Dimbeta RSI Buy'])
        dimbeta_rsi_sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Dimbeta RSI Sell'])

        return {
            'category': self.asset.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'dimbeta_rsi_buy_count': dimbeta_rsi_buy_count,
            'dimbeta_rsi_sell_count': dimbeta_rsi_sell_count
        }

    def save_signals_to_csv(self, output_dir):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.asset.ticker}_Dimbeta_RSI_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_dimbeta_rsi_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "Dimbeta_RSI")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = Dimbeta_RSI_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class DimRMO_Strategy:
    def __init__(self, start_date, end_date, stock, wallet, display=display_global):
        self.stock = stock
        self.wallet = wallet
        self.display = display
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_date = start_date
        self.end_date = end_date
        self.start_price = None
        self.end_price = None
        self.calculate_indicators()
        self.apply_signals()
        if self.start_date in self.stock.data.index:
            self.start_price = self.stock.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):
        # Calculate the percentage change from the previous day over a rolling window of 6 days
        self.stock.data['dimRMO'] = (self.stock.data['Close'].pct_change(periods=1) * 100).rolling(window=6).mean()

        # **EMA Calculation for dimRMO (3-day Exponential Moving Average)**
        self.stock.data['EMA5_dimRMO'] = self.stock.data['dimRMO'].ewm(span=3, adjust=False).mean()

        # **Standard Deviation Calculation for dimRMO (5-day)**
        self.stock.data['STD10_dimRMO'] = self.stock.data['dimRMO'].rolling(window=5).std()

        # **Moving Average of the 5-day STD of dimRMO (5-day moving average)**
        self.stock.data['MA5_STD10_dimRMO'] = self.stock.data['STD10_dimRMO'].rolling(window=5).mean()

        # **Deviation of STD from its 5-day moving average**
        self.stock.data['dev_STD10_from_MA5'] = self.stock.data['STD10_dimRMO'] - self.stock.data['MA5_STD10_dimRMO']

    def apply_signals(self):
        self.stock.data['Buy_Signal'] = False
        self.stock.data['Sell_Signal'] = False

        for i in range(1, len(self.stock.data)):
            current_date = self.stock.data.index[i]
            current_dimRMO = self.stock.data['dimRMO'].iloc[i]
            prev_dimRMO = self.stock.data['dimRMO'].iloc[i - 1]
            current_ema5 = self.stock.data['EMA5_dimRMO'].iloc[i]
            prev_ema5 = self.stock.data['EMA5_dimRMO'].iloc[i - 1]
            current_dev_STD10 = self.stock.data['dev_STD10_from_MA5'].iloc[i]
            prev_dev_STD10 = self.stock.data['dev_STD10_from_MA5'].iloc[i - 1]
            current_std10 = self.stock.data['STD10_dimRMO'].iloc[i]
            current_ma5_std10 = self.stock.data['MA5_STD10_dimRMO'].iloc[i]

            # **Buy Signal Logic**
            if current_ema5 > current_dev_STD10 and prev_ema5 <= prev_dev_STD10 and current_dimRMO > current_ma5_std10:
                if not self.has_open_position:
                    self.stock.data.loc[current_date, 'Buy_Signal'] = True
                    buy_price = self.stock.data['Close'].iloc[i]
                    if self.wallet.buy(buy_price, 1):
                        self.has_open_position = True
                        self.buy_sell_signals.append({
                            'ticker': self.stock.ticker,
                            'category': self.stock.category,
                            'date': current_date,
                            'type': 'Buy',
                            'price': buy_price,
                            'profit': None,
                            'percentage_change': None
                        })

            # **Sell Signal Logic**
            if self.has_open_position:
                if current_ema5 < current_dev_STD10:
                    if prev_ema5 >= prev_dev_STD10:
                        sell_price = self.stock.data['Close'].iloc[i]

                        self.stock.data.loc[current_date, 'Sell_Signal'] = True
                        if self.wallet.sell(sell_price, 1):
                            self.has_open_position = False  # Close position
                            last_trade = self.wallet.results[-1]
                            self.buy_sell_signals.append({
                                'ticker': self.stock.ticker,
                                'category': self.stock.category,
                                'date': current_date,
                                'type': 'Sell',
                                'price': sell_price,
                                'profit': last_trade['profit'],
                                'percentage_change': last_trade['percentage_change']
                            })

        if self.buy_sell_signals:
            last_trade = self.buy_sell_signals[-1]
            self.end_price = last_trade['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0
        )
        wallet_summary = self.wallet.get_summary()

        # Count signal types
        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.stock.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, path):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{self.stock.ticker}_dimRMO_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_DimRMO_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "DimRMO")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = DimRMO_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category

class DimLAMDA_Strategy:
    def __init__(self, start_date, end_date, stock, wallet, display=display_global):
        self.stock = stock
        self.wallet = wallet
        self.display = display
        self.start_date = start_date
        self.end_date = end_date
        self.has_open_position = False
        self.buy_sell_signals = []
        self.signal_code = 0
        self.total_profit = 0
        self.wins = 0
        self.losses = 0
        self.start_price = None
        self.end_price = None
        self.calculate_indicators()
        self.apply_signals()
        if self.start_date in self.stock.data.index:
            self.start_price = self.stock.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):

        # Calculate the 20-day Simple Moving Average (SMA20) for Dimbeta
        self.stock.data['SMA20'] = self.stock.data['Close'].rolling(window=20).mean()

        # Calculate Dimbeta (percentage difference between Close price and SMA20)
        self.stock.data['Dimbeta'] = (self.stock.data['Close'] - self.stock.data['SMA20']) / self.stock.data['SMA20']

        # Calculate the 10-day Standard Deviation for Dimbeta
        self.stock.data['STD10_Dimbeta'] = self.stock.data['Dimbeta'].rolling(window=10).std()

        # Calculate DimLAMDA (the deviation of Dimbeta from its 10-day standard deviation)
        self.stock.data['DimLAMDA'] = ((self.stock.data['Dimbeta'] - self.stock.data['STD10_Dimbeta']) /
                                       self.stock.data['STD10_Dimbeta']) * 100

        # Calculate the 10-day Moving Average of the 10-day Standard Deviation of Dimbeta
        self.stock.data['MA10_STD10_Dimbeta'] = self.stock.data['STD10_Dimbeta'].rolling(window=10).mean()

        # Calculate DimbetaRate (the deviation of the 10-day Standard Deviation from its 10-day moving average)
        self.stock.data['DimbetaRate'] = ((self.stock.data['STD10_Dimbeta'] - self.stock.data['MA10_STD10_Dimbeta']) /
                                          self.stock.data['MA10_STD10_Dimbeta']) * 100

        # Calculate the 20-day Exponential Moving Average (EMA20) for Dimbeta
        self.stock.data['EMA20_Dimbeta'] = self.stock.data['Dimbeta'].ewm(span=20, adjust=False).mean()

    def apply_signals(self):
        self.stock.data['Buy_Signal'] = False
        self.stock.data['Sell_Signal'] = False

        for i in range(1, len(self.stock.data)):
            current_date = self.stock.data.index[i]

            current_dimbeta = self.stock.data['Dimbeta'].iloc[i]
            prev_dimbeta = self.stock.data['Dimbeta'].iloc[i - 1]
            current_ema20 = self.stock.data['EMA20_Dimbeta'].iloc[i]
            prev_ema20 = self.stock.data['EMA20_Dimbeta'].iloc[i - 1]
            current_dimlamda = self.stock.data['DimLAMDA'].iloc[i]
            prev_dimlamda = self.stock.data['DimLAMDA'].iloc[i - 1]
            current_dimbeta_rate = self.stock.data['DimbetaRate'].iloc[i]
            prev_dimbeta_rate = self.stock.data['DimbetaRate'].iloc[i - 1]

            # Buy Signal: When DimbetaRate crosses below DimLAMDA
            if current_dimbeta_rate < current_dimlamda and prev_dimbeta_rate >= prev_dimlamda:
                # When Dimbeta crosses above EMA20
                if current_dimbeta > current_ema20 and prev_dimbeta <= prev_ema20:
                    if not self.has_open_position:
                        self.stock.data.loc[current_date, 'Buy_Signal'] = True
                        buy_price = self.stock.data['Close'].iloc[i]
                        if self.wallet.buy(buy_price, 1):
                            self.has_open_position = True
                            self.signal_code += 1
                            self.buy_sell_signals.append({
                                'ticker': self.stock.ticker,
                                'category': self.stock.category,
                                'date': current_date,
                                'type': 'Buy',
                                'price': buy_price,
                                'profit': None,
                                'percentage_change': None
                            })

            # Sell Signal: When DimbetaRate crosses above DimLAMDA
            if current_dimbeta_rate > current_dimlamda and prev_dimbeta_rate <= prev_dimlamda:
                # When Dimbeta crosses below EMA20
                if current_dimbeta < current_ema20 and prev_dimbeta >= prev_ema20:
                    if self.has_open_position:
                        self.stock.data.loc[current_date, 'Sell_Signal'] = True
                        sell_price = self.stock.data['Close'].iloc[i]
                        if self.wallet.sell(sell_price, 1):
                            self.has_open_position = False
                            last_trade = self.wallet.results[-1]
                            self.buy_sell_signals.append({
                                'ticker': self.stock.ticker,
                                'category': self.stock.category,
                                'date': current_date,
                                'type': 'Sell',
                                'price': sell_price,
                                'profit': last_trade['profit'],
                                'percentage_change': last_trade['percentage_change']
                            })

            if self.buy_sell_signals:
                last_trade = self.buy_sell_signals[-1]
                self.end_price = last_trade['price']

    def get_results(self):
        total_profit = sum([result['profit'] for result in self.wallet.results])
        total_percentage_change = (
            sum([result['percentage_change'] for result in self.wallet.results]) / len(self.wallet.results)
            if self.wallet.results else 0
        )
        wallet_summary = self.wallet.get_summary()

        # Count signal types
        buy_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Buy'])
        sell_count = len([signal for signal in self.buy_sell_signals if signal['type'] == 'Sell'])

        return {
            'category': self.stock.category,
            'total_profit': total_profit,
            'total_percentage_change': total_percentage_change,
            'total_buy_orders': wallet_summary['buy_count'],
            'total_sell_orders': wallet_summary['sell_count'],
            'buy_orders_not_executed': wallet_summary['buy_not_executed'],
            'sell_orders_not_executed': wallet_summary['sell_not_executed'],
            'win_count': wallet_summary['win_count'],
            'loss_count': wallet_summary['loss_count'],
            'start_price': self.start_price,
            'end_price': self.end_price,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def save_signals_to_csv(self, path):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{self.stock.ticker}_DimLAMDA_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")

def run_DimLAMDA_backtest(start_date, end_date, interval):
    base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
    base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "DimLAMDA")

    results_by_category = {}

    for region in os.listdir(base_input_path):
        region_path = os.path.join(base_input_path, region)
        if not os.path.isdir(region_path):
            continue

        for sector_folder in os.listdir(region_path):
            sector_path = os.path.join(region_path, sector_folder)
            if not os.path.isdir(sector_path):
                continue

            all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

            asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

            output_dir = os.path.join(base_output_path, region, sector_folder)
            os.makedirs(output_dir, exist_ok=True)
            delete_csv_files(output_dir, asset_list)

            category_results = []

            for ticker_file in all_daily_files:
                ticker = ticker_file.replace("_daily_data.csv", "")
                ticker_data_path = os.path.join(sector_path, ticker_file)

                asset = Assets(ticker)
                asset.get_data(ticker_data_path, start_date, end_date)
                asset.get_category(ticker_data_path)

                wallet = Wallet(display=False)
                strategy = DimLAMDA_Strategy(start_date, end_date, asset, wallet)
                result = strategy.get_results()

                category_results.append((ticker, result))
                strategy.save_signals_to_csv(output_dir)

            results_by_category[sector_folder] = category_results

    return results_by_category


class ROC_PRIC_Strategy:
    def __init__(self, start_date, end_date, stock, wallet, display=display_global):
        self.stock = stock
        self.wallet = wallet
        self.display = display
        self.has_open_position = False
        self.buy_sell_signals = []
        self.start_date = start_date
        self.end_date = end_date
        self.current_buy_price = None
        self.trailing_high = None
        self.start_price = None
        self.end_price = None

        self.calculate_indicators()

        self.apply_signals()

        if self.start_date in self.stock.data.index:
            self.start_price = self.stock.data.loc[self.start_date, 'Close']

    def calculate_indicators(self):
        self.stock.data['ROC_PRIC'] = self.stock.data['Close'].pct_change() * 100
        self.stock.data['EMA5_ROC_PRIC'] = self.stock.data['ROC_PRIC'].ewm(span=5, adjust=False).mean()
        self.stock.data['STD10_ROC_PRIC'] = self.stock.data['ROC_PRIC'].rolling(window=10).std()
        self.stock.data['MA10_STD10_ROC_PRIC'] = self.stock.data['STD10_ROC_PRIC'].rolling(window=10).mean()
        self.stock.data['dev_STD10_from_MA10'] = self.stock.data['STD10_ROC_PRIC'] - self.stock.data[
            'MA10_STD10_ROC_PRIC']
        self.stock.data['MA28_ROC_PRIC'] = self.stock.data['ROC_PRIC'].rolling(window=28).mean()

        self.stock.data['dimSigma'] = (self.stock.data['EMA5_ROC_PRIC'] - self.stock.data['STD10_ROC_PRIC']) / \
                                      self.stock.data['STD10_ROC_PRIC'] * 100

        self.stock.data['MA10_ROC_PRIC'] = self.stock.data['ROC_PRIC'].rolling(window=10).mean()
        self.stock.data['sRatio'] = self.stock.data['MA10_ROC_PRIC'] / self.stock.data['MA10_STD10_ROC_PRIC']

        self.stock.data['MA20_Close'] = self.stock.data['Close'].ewm(span=20, adjust=False).mean()
        self.stock.data['dimbeta'] = (self.stock.data['Close'] - self.stock.data['MA20_Close']) / self.stock.data[
            'MA20_Close'] * 100
        self.stock.data['EMA20_dimbeta'] = self.stock.data['dimbeta'].ewm(span=20, adjust=False).mean()
        self.stock.data['STD10_dimbeta'] = self.stock.data['dimbeta'].rolling(window=10).std()

        self.stock.data['dimKappa'] = (self.stock.data['dimbeta'] - self.stock.data['STD10_dimbeta']) / self.stock.data[
            'STD10_dimbeta'] * 100

    def apply_signals(self):
        self.stock.data['Buy_Signal'] = False
        self.stock.data['Sell_Signal'] = False

        for i in range(2, len(self.stock.data)):
            current = self.stock.data.iloc[i]
            prev = self.stock.data.iloc[i - 1]
            current_date = self.stock.data.index[i]

            # Buy Conditions
            crossover_buy = (current['EMA5_ROC_PRIC'] > current['dev_STD10_from_MA10']) and (
                        prev['EMA5_ROC_PRIC'] <= prev['dev_STD10_from_MA10'])
            roc_condition = current['ROC_PRIC'] > current['MA10_STD10_ROC_PRIC']
            sratio_condition = current['sRatio'] > 0
            dimkappa_condition = (current['dimKappa'] > current['dimSigma']) and (prev['dimKappa'] <= prev['dimSigma'])

            # Sell Conditions
            crossover_sell = (current['EMA5_ROC_PRIC'] < current['dev_STD10_from_MA10']) and (
                        prev['EMA5_ROC_PRIC'] >= prev['dev_STD10_from_MA10'])
            dimsigma_condition = current['dimSigma'] > 0
            sratio_negative = current['sRatio'] < 0
            dimbeta_crossover = (current['EMA20_dimbeta'] < current['dimbeta']) and (
                        prev['EMA20_dimbeta'] >= prev['dimbeta'])

            if not self.has_open_position:
                if crossover_buy and roc_condition and sratio_condition and dimkappa_condition:
                    self.execute_buy(i)

            if self.has_open_position:
                if crossover_sell or dimsigma_condition or sratio_negative or dimbeta_crossover:
                    self.execute_sell(i, 'Conditional Sell')

                if current['Close'] > self.trailing_high:
                    self.trailing_high = current['Close']
                if current['Close'] <= self.trailing_high * 0.95:
                    self.execute_sell(i, 'Trailing Stop')

    def execute_buy(self, idx):
        buy_price = self.stock.data.iloc[idx]['Close']
        if self.wallet.buy(buy_price, 1):
            self.has_open_position = True
            self.current_buy_price = buy_price
            self.trailing_high = buy_price
            self.record_signal('Buy', idx, buy_price)

    def execute_sell(self, idx, sell_type):
        sell_price = self.stock.data.iloc[idx]['Close']
        if self.wallet.sell(sell_price, 1):
            self.has_open_position = False
            self.end_price = sell_price
            self.record_signal(f'Sell ({sell_type})', idx, sell_price)

    def record_signal(self, signal_type, idx, price):
        signal_data = {
            'date': self.stock.data.index[idx],
            'type': signal_type,
            'price': price,
            'ROC_PRIC': self.stock.data.iloc[idx]['ROC_PRIC'],
            'EMA5_ROC_PRIC': self.stock.data.iloc[idx]['EMA5_ROC_PRIC'],
            'dev_STD10': self.stock.data.iloc[idx]['dev_STD10_from_MA10'],
            'dimSigma': self.stock.data.iloc[idx]['dimSigma'],
            'sRatio': self.stock.data.iloc[idx]['sRatio'],
            'dimKappa': self.stock.data.iloc[idx]['dimKappa'],
            'dimbeta': self.stock.data.iloc[idx]['dimbeta']
        }
        self.buy_sell_signals.append(signal_data)

    def get_results(self):
        total_profit = sum(trade['profit'] for trade in self.wallet.results if trade['profit'] is not None)
        total_percentage = sum(
            trade['percentage_change'] for trade in self.wallet.results if trade['percentage_change'] is not None)
        avg_percentage = total_percentage / len(self.wallet.results) if self.wallet.results else 0

        return {
            'ticker': self.stock.ticker,
            'start_price': self.start_price,
            'end_price': self.end_price,
            'total_profit': total_profit,
            'avg_percentage_change': avg_percentage,
            'total_trades': len(self.wallet.results),
            'winning_trades': sum(1 for t in self.wallet.results if t['profit'] > 0),
            'losing_trades': sum(1 for t in self.wallet.results if t['profit'] < 0),
            'max_profit': max((t['profit'] for t in self.wallet.results), default=0),
            'max_loss': min((t['profit'] for t in self.wallet.results), default=0),
            'buy_signals': sum(1 for s in self.buy_sell_signals if 'Buy' in s['type']),
            'sell_signals': sum(1 for s in self.buy_sell_signals if 'Sell' in s['type'])
        }

    def save_signals_to_csv(self, path):
        df = pd.DataFrame(self.buy_sell_signals)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{self.stock.ticker}_ROC_PRIC_signals.csv")
        df.to_csv(filename, index=False)
        if self.display:
            print(f"Saved signals to {filename}")
def run_ROC_PRIC_backtest(start_date, end_date, interval):
            base_input_path = os.path.join("Daily_Open_Close", f"{start_date}-{end_date}")
            base_output_path = os.path.join("Output", f"{start_date}-{end_date}", "ROC_PRIC")

            results_by_category = {}

            for region in os.listdir(base_input_path):
                region_path = os.path.join(base_input_path, region)
                if not os.path.isdir(region_path):
                    continue

                for sector_folder in os.listdir(region_path):
                    sector_path = os.path.join(region_path, sector_folder)
                    if not os.path.isdir(sector_path):
                        continue

                    all_daily_files = [f for f in os.listdir(sector_path) if f.endswith("_daily_data.csv")]

                    asset_list = [f.replace("_daily_data.csv", "") for f in all_daily_files]

                    output_dir = os.path.join(base_output_path, region, sector_folder)
                    os.makedirs(output_dir, exist_ok=True)
                    delete_csv_files(output_dir, asset_list)

                    category_results = []

                    for ticker_file in all_daily_files:
                        ticker = ticker_file.replace("_daily_data.csv", "")
                        ticker_data_path = os.path.join(sector_path, ticker_file)

                        asset = Assets(ticker)
                        asset.get_data(ticker_data_path, start_date, end_date)
                        asset.get_category(ticker_data_path)

                        wallet = Wallet(display=False)
                        strategy = ROC_PRIC_Strategy(start_date, end_date, asset, wallet)
                        result = strategy.get_results()

                        category_results.append((ticker, result))
                        strategy.save_signals_to_csv(output_dir)

                    results_by_category[sector_folder] = category_results

            return results_by_category

def main(start_date, end_date, interval):

    make_input("Europe_full_INPUT.csv", "Input_data/" + start_date + "-" + end_date + "/Algorithm_input/Europe_assets", start_date, end_date, 5)
    make_input("USA_full_INPUT.csv", "Input_data/" + start_date + "-" + end_date + "/Algorithm_input/USA_assets", start_date, end_date, 5)
    make_input("INDEX_INPUT.csv", "Input_data/" + start_date + "-" + end_date + "/Algorithm_input/INDEX", start_date, end_date, 5)
    make_input("Crypto_INPUT.csv", "Input_data/" + start_date + "-" + end_date + "/Algorithm_input/Crypto", start_date, end_date, 5)
    make_input("ETFDB_INPUT.csv", "Input_data/" + start_date + "-" + end_date + "/Algorithm_input/ETFS", start_date, end_date, 5)


    extract_daily_open_close("Input_data/" + start_date + "-" + end_date + "/Algorithm_input", "Daily_Open_Close/" + start_date + "-" + end_date, start_date, end_date)

    run_RSI_backtest(start_date, end_date, interval)
    run_MACD_backtest(start_date, end_date, interval)
    run_SMA_backtest(start_date, end_date, interval)
    run_dimbeta_backtest(start_date, end_date, interval)
    #run_dimbeta_rsi_backtest(start_date, end_date, interval)
    run_DimRMO_backtest(start_date, end_date, interval)
    run_DimLAMDA_backtest(start_date, end_date, interval)
    #run_ROC_PRIC_backtest(start_date, end_date, interval)


if __name__ == "__main__":

    main('2022-11-01', '2023-11-01', '1d')

    process_and_generate_final_statistics("Output", "Final_Statistics_Summary.csv")
