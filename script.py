import os
import time
import threading
import json
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, flash
from binance.client import Client
from binance import ThreadedWebsocketManager

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
    if not API_KEY or not API_SECRET:
        print("[ERROR] Missing API credentials.")
        return
    client = Client(API_KEY, API_SECRET, testnet=True)
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    except Exception as e:
        print(f"[WARN] Leverage set failed: {e}")

# ======== Indicators (no TA-Lib) ========
def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
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
    st = [True]
    for i in range(1, len(df)):
        if df['close'].iat[i] > upper.iat[i-1]:
            st.append(True)
        elif df['close'].iat[i] < lower.iat[i-1]:
            st.append(False)
        else:
            st.append(st[i-1])
    return pd.Series(st, index=df.index)

# ======== Trade utils ========
def can_trade():
    now = time.time()
    while trades_last_hour and now - trades_last_hour[0] > 3600:
        trades_last_hour.pop(0)
    return len(trades_last_hour) < MAX_TRADES_PER_HOUR

def in_position():
    pos = client.futures_position_information(symbol=SYMBOL)
    return float(next(p['positionAmt'] for p in pos if p['symbol']==SYMBOL)) != 0

def calc_qty(price):
    bal = float(client.futures_account_balance()[0]['balance'])
    return round((bal * RISK_PER_TRADE * LEVERAGE) / price, 3)

def record_trade(exit_p=None):
    global current_trade
    if not current_trade or exit_p is None: return
    pnl = ((exit_p - current_trade['entry']) if current_trade['side']=='BUY' else (current_trade['entry'] - exit_p)) * current_trade['qty']
    current_trade.update({'exit': exit_p, 'exit_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'pnl': pnl})
    trade_history.append(current_trade)
    current_trade = None

def enter_order(side, price):
    global current_trade
    qty = calc_qty(price)
    client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET', quantity=qty)
    trades_last_hour.append(time.time())
    current_trade = {'side': side, 'entry': price, 'entry_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'qty': qty}

def exit_position():
    pos = client.futures_position_information(symbol=SYMBOL)
    amt = float(next(p['positionAmt'] for p in pos if p['symbol']==SYMBOL))
    if amt == 0: return
    side = 'SELL' if amt>0 else 'BUY'
    price = float(client.futures_mark_price(symbol=SYMBOL)['markPrice'])
    client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET', quantity=abs(amt))
    record_trade(price)

# ======== Core strategy ========
def on_new_candle(c):
    global history
    if not in_session or not c or 'time' not in c: return
    history = history.append(c, ignore_index=True).iloc[-HISTORY_LEN:]
    if len(history) < max(EMA_LONG, RSI_PERIOD, VOL_LOOKBACK): return
    df = history.copy().reset_index(drop=True)
    df['ema9'], df['ema20'], df['ema50'] = ema(df['close'], EMA_SHORT), ema(df['close'], EMA_MED), ema(df['close'], EMA_LONG)
    df['rsi'], df['vol_avg'], df['supertrend'] = rsi(df['close'], RSI_PERIOD), df['volume'].rolling(VOL_LOOKBACK).mean(), supertrend(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    p = last['close']
    # exit
    if in_position() and current_trade:
        if ((current_trade['side']=='BUY' and (p>=current_trade['entry']*(1+TP_PERCENT) or p<=current_trade['entry']*(1-SL_PERCENT))) or
           (current_trade['side']=='SELL' and (p<=current_trade['entry']*(1-TP_PERCENT) or p>=current_trade['entry']*(1+SL_PERCENT)))):
            exit_position()
    # entry
    if can_trade() and not in_position():
        if (prev['ema9']<prev['ema20']<prev['ema50'] and last['ema9']>last['ema20']>last['ema50'] and last['supertrend'] and 30<last['rsi']<70 and last['volume']>VOL_MULT*last['vol_avg']):
            enter_order('BUY', p)
        if (prev['ema9']>prev['ema20']>prev['ema50'] and last['ema9']<last['ema20']<last['ema50'] and not last['supertrend'] and 30<last['rsi']<70 and last['volume']>VOL_MULT*last['vol_avg']):
            enter_order('SELL', p)

# ======== WebSocket ========
def start_ws():
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    def handle_msg(msg):
        k = msg.get('k', {})
        if k.get('x'):
            on_new_candle({'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']), 'close': float(k['c']), 'volume': float(k['v']), 'time': k['t']})
    twm.start_kline_socket(callback=handle_msg, symbol=SYMBOL, interval=INTERVAL)

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

    # Fetch USDT futures balance if connected
    balance = None
    if connection_status:
        try:
            balances = client.futures_account_balance()
            usdt = next((b for b in balances if b['asset'] == 'USDT'), None)
            balance = float(usdt['balance']) if usdt else None
        except Exception:
            balance = None

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
        connection_status=connection_status
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        new_cfg = {}
        for key in ['api_key', 'api_secret', 'symbol', 'interval']:
            new_cfg[key] = request.form.get(key)
        # Numeric values
        new_cfg['leverage'] = int(request.form.get('leverage'))
        new_cfg['risk_per_trade'] = float(request.form.get('risk_per_trade'))
        new_cfg['max_trades_per_hour'] = int(request.form.get('max_trades_per_hour'))
        new_cfg['tp_percent'] = float(request.form.get('tp_percent'))
        new_cfg['sl_percent'] = float(request.form.get('sl_percent'))

        save_config(new_cfg)
        flash('Settings saved. Please restart the bot to apply.', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html', cfg=cfg)

@app.route('/start')
def start_bot():
    global in_session
    init_client()
    in_session = True
    return redirect(url_for('dashboard'))

@app.route('/stop')
def stop_bot():
    global in_session
    in_session = False
    return redirect(url_for('dashboard'))

@app.route('/exit')
def manual_exit():
    exit_position()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    init_client()
    threading.Thread(target=start_ws, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
