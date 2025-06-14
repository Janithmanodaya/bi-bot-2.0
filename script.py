import os
import time
import threading
import json
import logging # Added logging import
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance import ThreadedWebsocketManager

# ======== Logging Configuration ========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======== Config File ========
CONFIG_PATH = 'config.json'

# Load and save config

def load_config():
    default = {
        "api_key": "",
        "api_secret": "",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "leverage": 20,
        "risk_per_trade": 0.01,
        "max_trades_per_hour": 2,
        "tp_percent": 0.007,
        "sl_percent": 0.005
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            default.update(json.load(f))
    return default


def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)

# ======== Initialize ========
cfg = load_config()
API_KEY = cfg['api_key']
API_SECRET = cfg['api_secret']
SYMBOL = cfg['symbol']
INTERVAL = {'1m': Client.KLINE_INTERVAL_1MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE}.get(cfg['interval'], Client.KLINE_INTERVAL_1MINUTE)
LEVERAGE = cfg['leverage']
RISK_PER_TRADE = cfg['risk_per_trade']
MAX_TRADES_PER_HOUR = cfg['max_trades_per_hour']
TP_PERCENT = cfg['tp_percent']
SL_PERCENT = cfg['sl_percent']

# Strategy settings
HISTORY_LEN = 200
EMA_SHORT, EMA_MED, EMA_LONG = 9, 20, 50
RSI_PERIOD = 14
VOL_LOOKBACK, VOL_MULT = 20, 1.5

# ======== State ========
history = pd.DataFrame(columns=['open','high','low','close','volume','time'])
trades_last_hour = []
in_session = False
trade_history = []
current_trade = None
client = None

# ======== Client init ========
def init_client():
    global client
    logging.info("Initializing Binance client.")
    if not API_KEY or not API_SECRET:
        logging.error("Missing API credentials. Cannot initialize client.")
        # print("[ERROR] Missing API credentials.") # Replaced by logging
        return
    client = Client(API_KEY, API_SECRET, testnet=False)
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        logging.info(f"Leverage set to {LEVERAGE} for {SYMBOL}.")
    except Exception as e:
        logging.warning(f"Leverage set failed for {SYMBOL} with leverage {LEVERAGE}: {e}")
        # print(f"[WARN] Leverage set failed: {e}") # Replaced by logging
    logging.info("Binance client initialized successfully.")


# ======== Indicators (no TA-Lib) ========
def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    if avg_loss.eq(0).any(): # Avoid division by zero if all losses are 0
        return pd.Series(100.0, index=series.index) if avg_gain.gt(0).any() else pd.Series(50.0, index=series.index)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, period):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df, period=10, mult=3):
    atr_s = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr_s
    lower = hl2 - mult * atr_s
    st = [True] * len(df) # Initialize with True
    if len(df) == 0: return pd.Series(st, index=df.index) # Handle empty DataFrame

    for i in range(1, len(df)):
        if df['close'].iat[i] > upper.iat[i-1]:
            st[i] = True
        elif df['close'].iat[i] < lower.iat[i-1]:
            st[i] = False
        else:
            st[i] = st[i-1]
    return pd.Series(st, index=df.index)

# ======== Trade utils ========
def can_trade():
    now = time.time()
    while trades_last_hour and now - trades_last_hour[0] > 3600:
        trades_last_hour.pop(0)
    return len(trades_last_hour) < MAX_TRADES_PER_HOUR

def in_position():
    if not client: return False # Ensure client is initialized
    pos = client.futures_position_information(symbol=SYMBOL)
    return float(next(p['positionAmt'] for p in pos if p['symbol']==SYMBOL)) != 0

def calc_qty(price):
    if not client: return 0 # Ensure client is initialized
    bal = float(client.futures_account_balance()[0]['balance']) # Consider error handling here too if needed
    return round((bal * RISK_PER_TRADE * LEVERAGE) / price, 3)

def record_trade(exit_p=None):
    global current_trade
    if not current_trade or exit_p is None: return
    pnl = ((exit_p - current_trade['entry']) if current_trade['side']=='BUY' else (current_trade['entry'] - exit_p)) * current_trade['qty']
    current_trade.update({'exit': exit_p, 'exit_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'pnl': pnl})
    trade_history.append(current_trade.copy()) # Use .copy() to avoid issues with dict reuse
    logging.info(f"Trade recorded: {current_trade}")
    current_trade = None

def enter_order(side, price):
    global current_trade
    if not client:
        logging.error("Cannot enter order: Binance client not initialized.")
        return
    qty = calc_qty(price)
    if qty <= 0:
        logging.warning(f"Calculated quantity is {qty}. Cannot place order for {SYMBOL}.")
        return
    try:
        client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET', quantity=qty)
        trades_last_hour.append(time.time())
        current_trade = {'side': side, 'entry': price, 'entry_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'qty': qty}
        logging.info(f"Entering {side} order for {SYMBOL} at {price}, quantity {qty}.")
    except Exception as e:
        logging.error(f"Failed to enter {side} order for {SYMBOL} at {price}, qty {qty}: {e}")


def exit_position():
    if not client:
        logging.error("Cannot exit position: Binance client not initialized.")
        return
    pos = client.futures_position_information(symbol=SYMBOL)
    amt = float(next(p['positionAmt'] for p in pos if p['symbol']==SYMBOL))
    if amt == 0:
        logging.info(f"No position to exit for {SYMBOL}.")
        return
    side = 'SELL' if amt > 0 else 'BUY'
    price = float(client.futures_mark_price(symbol=SYMBOL)['markPrice'])
    try:
        client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET', quantity=abs(amt))
        logging.info(f"Exiting position for {SYMBOL}. Side: {side}, Quantity: {abs(amt)}, Price: {price}.")
        record_trade(price) # record_trade is called after successful exit
    except Exception as e:
        logging.error(f"Failed to exit {side} position for {SYMBOL}, qty {abs(amt)}: {e}")


# ======== Core strategy ========
def on_new_candle(c):
    global history
    logging.debug(f"Processing new candle: {c}")
    if not in_session or not c or 'time' not in c: return

    # Ensure c is a dictionary before attempting to append
    if not isinstance(c, dict):
        logging.warning(f"on_new_candle received non-dict candle data: {c}")
        return

    history = history.append(c, ignore_index=True)
    if len(history) > HISTORY_LEN: # Keep history to specified length
        history = history.iloc[-HISTORY_LEN:]

    if len(history) < max(EMA_LONG, RSI_PERIOD, VOL_LOOKBACK): return

    df = history.copy().reset_index(drop=True)
    df['ema9'], df['ema20'], df['ema50'] = ema(df['close'], EMA_SHORT), ema(df['close'], EMA_MED), ema(df['close'], EMA_LONG)
    df['rsi'], df['vol_avg'], df['supertrend'] = rsi(df['close'], RSI_PERIOD), df['volume'].rolling(VOL_LOOKBACK).mean(), supertrend(df)

    if df.empty:
        logging.debug("DataFrame is empty in on_new_candle after indicator calculation.")
        return

    last, prev = df.iloc[-1], df.iloc[-2]
    p = last['close']

    # exit
    if in_position() and current_trade:
        tp_price_buy = current_trade['entry'] * (1 + TP_PERCENT)
        sl_price_buy = current_trade['entry'] * (1 - SL_PERCENT)
        tp_price_sell = current_trade['entry'] * (1 - TP_PERCENT)
        sl_price_sell = current_trade['entry'] * (1 + SL_PERCENT)

        if ((current_trade['side']=='BUY' and (p >= tp_price_buy or p <= sl_price_buy)) or
           (current_trade['side']=='SELL' and (p <= tp_price_sell or p >= sl_price_sell))):
            logging.info(f"Exit condition met for {current_trade['side']} {SYMBOL}. Current price: {p}, Entry: {current_trade['entry']}")
            exit_position()
    # entry
    if can_trade() and not in_position():
        if (prev['ema9']<prev['ema20']<prev['ema50'] and last['ema9']>last['ema20']>last['ema50'] and last['supertrend'] and 30<last['rsi']<70 and last['volume']>VOL_MULT*last['vol_avg']):
            logging.info(f"Entry condition met for BUY signal on {SYMBOL} at price {p}")
            enter_order('BUY', p)
        if (prev['ema9']>prev['ema20']>prev['ema50'] and last['ema9']<last['ema20']<last['ema50'] and not last['supertrend'] and 30<last['rsi']<70 and last['volume']>VOL_MULT*last['vol_avg']):
            logging.info(f"Entry condition met for SELL signal on {SYMBOL} at price {p}")
            enter_order('SELL', p)

# ======== WebSocket ========
def start_ws():
    logging.info("Starting WebSocket connection.")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    def handle_msg(msg):
        logging.debug(f"WebSocket message received: {msg}")
        k = msg.get('k', {})
        if k.get('x'): # Is candle closed?
            candle_data = {'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']), 'close': float(k['c']), 'volume': float(k['v']), 'time': k['t']}
            on_new_candle(candle_data)
    twm.start_kline_socket(callback=handle_msg, symbol=SYMBOL, interval=INTERVAL)
    # twm.join() # This would block, not suitable for Flask thread

# ======== Flask App ========
app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET', 'change_me')

@app.route('/')
def dashboard():
    """
    Render the main dashboard with bot status, trade stats, balance, and connection info.
    """
    # Determine connection status
    connection_status = bool(client and API_KEY and API_SECRET)
    balance_error_message = None

    # Fetch USDT futures balance if connected
    balance = None
    if connection_status:
        try:
            balances = client.futures_account_balance()
            usdt = next((b for b in balances if b['asset'] == 'USDT'), None)
            balance = float(usdt['balance']) if usdt else None
        except BinanceAPIException as e:
            balance = None
            balance_error_message = f"API Error: Could not fetch balance. Please check API credentials. (Error: {e.message})"
            # print(f"[ERROR] Fetching balance failed (API Exception): {e}") # Replaced by logging
            logging.error(f"Failed to fetch account balance (API Exception): {e.message}")
        except BinanceRequestException as e:
            balance = None
            balance_error_message = f"Network Error: Could not fetch balance. Please check your internet connection. (Error: {e.message})"
            # print(f"[ERROR] Fetching balance failed (Request Exception): {e}") # Replaced by logging
            logging.error(f"Failed to fetch account balance (Request Exception): {e.message}")
        except Exception as e:
            balance = None
            balance_error_message = f"An unexpected error occurred while fetching balance: {e}"
            # print(f"[ERROR] Fetching balance failed (Unexpected Exception): {e}") # Replaced by logging
            logging.error(f"Failed to fetch account balance (Unexpected Exception): {e}", exc_info=True)


    # Compute total PnL
    total_pnl = sum(t.get('pnl', 0) for t in trade_history)

    return render_template(
        'dashboard.html',
        cfg=cfg,
        running=in_session,
        trades_count=len(trades_last_hour),
        last_trade=current_trade or {},
        history=trade_history,
        total_pnl=total_pnl,
        balance=balance,
        connection_status=connection_status,
        balance_error_message=balance_error_message
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        new_cfg = {}
        for key in ['api_key', 'api_secret', 'symbol', 'interval']:
            new_cfg[key] = request.form.get(key)
        # Numeric values
        try:
            new_cfg['leverage'] = int(request.form.get('leverage'))
            new_cfg['risk_per_trade'] = float(request.form.get('risk_per_trade'))
            new_cfg['max_trades_per_hour'] = int(request.form.get('max_trades_per_hour'))
            new_cfg['tp_percent'] = float(request.form.get('tp_percent'))
            new_cfg['sl_percent'] = float(request.form.get('sl_percent'))
        except (ValueError, TypeError) as e:
            flash(f'Invalid numeric input: {e}', 'danger')
            return redirect(url_for('settings'))


        # Store old values for comparison
        old_symbol = cfg['symbol']
        old_leverage = cfg['leverage']
        old_api_key = cfg['api_key']
        old_api_secret = cfg['api_secret']
        old_interval = cfg['interval']

        save_config(new_cfg)

        # Reload global cfg object
        global cfg
        cfg = load_config()

        # Re-initialize global operational variables
        global API_KEY, API_SECRET, SYMBOL, INTERVAL, LEVERAGE, RISK_PER_TRADE, MAX_TRADES_PER_HOUR, TP_PERCENT, SL_PERCENT
        API_KEY = cfg['api_key']
        API_SECRET = cfg['api_secret']
        SYMBOL = cfg['symbol']
        INTERVAL = {'1m': Client.KLINE_INTERVAL_1MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE}.get(cfg['interval'], Client.KLINE_INTERVAL_1MINUTE)
        LEVERAGE = cfg['leverage']
        RISK_PER_TRADE = cfg['risk_per_trade']
        MAX_TRADES_PER_HOUR = cfg['max_trades_per_hour']
        TP_PERCENT = cfg['tp_percent']
        SL_PERCENT = cfg['sl_percent']

        logging.info(f"Settings updated and configuration reloaded: {cfg}")

        # Handle API key changes
        if API_KEY != old_api_key or API_SECRET != old_api_secret:
            global client
            client = None # Reset client to force reinitialization
            init_client() # Attempt to reinitialize client with new keys
            if client:
                flash('Settings saved. API keys changed and client reinitialized. Consider restarting the bot for WebSocket to use new symbol/interval if changed.', 'success')
            else:
                flash('Settings saved. API keys changed but client failed to reinitialize. Check credentials. Restart required.', 'danger')
        else:
            # Attempt to update leverage if client is initialized and SYMBOL or LEVERAGE changed
            if client and (SYMBOL != old_symbol or LEVERAGE != old_leverage):
                try:
                    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
                    logging.info(f"Leverage updated to {LEVERAGE} for {SYMBOL} via settings change.")
                    flash_message = f'Settings saved. Leverage updated to {LEVERAGE} for {SYMBOL}.'
                    if SYMBOL != old_symbol or cfg['interval'] != old_interval:
                        flash_message += ' Restart bot for symbol/interval changes to take full effect (WebSocket).'
                    flash(flash_message, 'success')
                except Exception as e:
                    logging.warning(f"Failed to update leverage via settings change: {e}")
                    flash_message = f'Settings saved, but failed to update leverage on Binance: {e}.'
                    if SYMBOL != old_symbol or cfg['interval'] != old_interval:
                        flash_message += ' Restart bot for symbol/interval changes to take full effect (WebSocket).'
                    flash(flash_message, 'warning')
            else:
                flash_message = 'Settings saved.'
                if SYMBOL != old_symbol or cfg['interval'] != old_interval:
                    flash_message += ' Restart bot for symbol/interval changes to take full effect (WebSocket).'
                elif API_KEY == old_api_key and API_SECRET == old_api_secret and SYMBOL == old_symbol and LEVERAGE == old_leverage and cfg['interval'] == old_interval : # Only if no major changes
                    flash_message = 'Settings saved. No major changes detected that require a restart.'
                flash(flash_message, 'success' if 'Restart bot' not in flash_message else 'info')


        return redirect(url_for('settings'))

    return render_template('settings.html', cfg=cfg)

@app.route('/start')
def start_bot():
    global in_session
    if not client: # Check if client was initialized
        init_client() # Attempt to initialize if not already
        if not client: # If still not initialized (e.g. missing keys)
             flash('Cannot start bot: Binance client not initialized. Check API keys.', 'danger')
             logging.error("Bot start failed: Binance client could not be initialized.")
             return redirect(url_for('dashboard'))

    in_session = True
    logging.info("Bot started.")
    flash('Bot started successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/stop')
def stop_bot():
    global in_session
    in_session = False
    logging.info("Bot stopped.")
    flash('Bot stopped.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/exit')
def manual_exit():
    logging.info("Manual exit initiated.")
    exit_position()
    flash('Manual exit attempt processed.', 'info') # Message updated for clarity
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    logging.info("Application starting.")
    init_client() # Initialize client on startup
    if client: # Only start WebSocket if client initialized
        threading.Thread(target=start_ws, daemon=True).start()
    else:
        logging.warning("WebSocket not started because Binance client failed to initialize.")
    app.run(host='0.0.0.0', port=5000)


@app.route('/api/balance')
def api_balance():
    if not client or not API_KEY or not API_SECRET:
        return jsonify({'error': 'Binance client not configured or not connected'}), 400

    try:
        balances = client.futures_account_balance()
        usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
        
        if usdt_balance and 'balance' in usdt_balance:
            balance_value = float(usdt_balance['balance'])
            return jsonify({'balance': balance_value})
        else:
            logging.warning("USDT balance asset not found or 'balance' key missing in futures_account_balance response.")
            return jsonify({'error': 'USDT balance not found'}), 404

    except BinanceAPIException as e:
        logging.error(f"Binance API Exception while fetching balance for /api/balance: {e}")
        return jsonify({'error': 'Failed to fetch balance due to API error', 'details': str(e)}), 500
    except BinanceRequestException as e:
        logging.error(f"Binance Request Exception while fetching balance for /api/balance: {e}")
        return jsonify({'error': 'Failed to fetch balance due to request error', 'details': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error while fetching balance for /api/balance: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred while fetching balance', 'details': str(e)}), 500
