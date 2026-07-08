# DINGJIASHAN V4.3 FINAL - 10 Stock Backtest
# V4.2 -> V4.3 CHANGES:
#  1. MA20 exit: only if below MA20 for 3 consecutive days
#  2. Dynamic SL floor: never looser than -5% (hard floor)
#  3. Time-stop: if held 5+ days AND losing, force exit
import os, pickle, numpy as np, pandas as pd
from datetime import datetime
from collections import Counter

CACHE_DIR = r'C:\AI\.strategy_cache'
START = pd.Timestamp('2026-06-01')
END = pd.Timestamp('2026-07-07')
POS_STRONG = 10; POS_NEUTRAL = 8; POS_BEAR = 4
BEAR_MA = 120; MIN_AMT = 30_000_000; MIN_SCORE_BASE = 0.2
COOL_OFF = 2
MIN_HOLD = 1
EMERGENCY_DROP = -0.08
SL_FLOOR = -0.05  # hard floor: never looser than -5%
TIME_LOSER = 5     # exit if held this many days AND still losing
MA20_CONSEC = 3    # consecutive days below MA20 before exit

FW = {'ret_5d':0.10,'ret_10d':0.10,'ret_20d':0.10,'vol_ratio':0.20,
      'price_vs_ma20':0.20,'ma_alignment':0.12,'up_days_10':0.05,
      'turnover':0.05,'atr_pct':-0.08}
EX = {'atm_low':1.2,'atm_high':2.0,'max_hold':10,'tp_std':0.12,'tp_top':0.15,
      'rank_out_ratio':0.75,'rank_drop':50,'sl_hard':-0.08,'sl_bear':-0.06,
      'ma_p':20,'atr_sl_mult':2.5}

N50 = {'600519':'GZMT','000858':'WLY','601318':'PA','600036':'CMB','000333':'Midea',
    '600276':'HengRui','002415':'Hik','000001':'PAB','002594':'BYD','601888':'CTRIP',
    '600030':'CITIC','000651':'Gree','600900':'Yangtze','002475':'Luxshare','300750':'CATL',
    '601398':'ICBC','601166':'CIB','600887':'Yili','600809':'FJ','000568':'LZLJ',
    '000002':'Vanke','601012':'LONGi','600585':'Conch','601668':'CSCEC','600028':'Sinopec',
    '601857':'Petro','601088':'Shenhua','600309':'WHX','002352':'SF','300059':'EastMoney',
    '002714':'Muyuan','000725':'BOE','688981':'SMIC','601899':'Zijin','600690':'Haier',
    '601601':'CPIC','002304':'Yanghe','000338':'Weichai','300015':'Aier','600031':'SANY',
    '601225':'SMY','002142':'NBB','600048':'Poly','601328':'BoCom','600016':'CMBC',
    '601939':'CCB','601288':'ABC','600104':'SAIC','002271':'DFYS','600050':'Unicom'}
C = {}

def load():
    skip_codes = {'000300', 'sh.000300', 'csi300_stocks'}
    for f in os.listdir(CACHE_DIR):
        if not f.endswith('.pkl'): continue
        code = f.replace('.pkl','')
        if code in skip_codes: continue
        try:
            d = pickle.load(open(os.path.join(CACHE_DIR,f),'rb'))
            df = d['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # Ensure required columns exist
            required = ['open','high','low','close','volume','amount']
            if not all(c in df.columns for c in required):
                continue
            C[code] = {'name':N50.get(code,code),'df':df}
        except: pass
    print('Loaded',len(C),'stocks')

def bench():
    for k in ['000300']:
        bp = os.path.join(CACHE_DIR,k+'.pkl')
        if os.path.exists(bp):
            d = pickle.load(open(bp,'rb'))
            df = d['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
    return None

def factors(df,idx):
    if idx<60: return None
    d = df.iloc[:idx+1]
    c=d['close'].values;h=d['high'].values;l=d['low'].values
    v=d['volume'].values;n=len(d)
    r5=(c[-1]/c[-6]-1)*100 if n>=6 else 0
    r10=(c[-1]/c[-11]-1)*100 if n>=11 else 0
    r20=(c[-1]/c[-21]-1)*100 if n>=21 else 0
    v20=np.mean(v[-21:-1]) if n>=21 else 1
    vr=v[-1]/max(v20,1) if v20>0 else 1
    ma5=np.mean(c[-6:-1]);ma10=np.mean(c[-11:-1]) if n>=11 else ma5
    ma20=np.mean(c[-21:-1]) if n>=21 else ma10
    ma60=np.mean(c[-61:-1]) if n>=61 else ma20
    pvm=(c[-1]/ma20-1)*100 if ma20>0 else 0
    mal=1 if ma5>ma10>ma20>ma60 else (0.5 if ma5>ma20 else 0)
    trs=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(max(1,n-20),n)]
    a20=np.mean(trs) if trs else c[-1]*0.02
    ap=(a20/c[-1])*100 if c[-1]>0 else 2
    upd=sum(1 for i in range(max(1,n-10),n) if c[i]>c[i-1])
    amt=np.mean(d['amount'].values[-21:-1]) if n>=21 else 0
    return {'ret_5d':r5,'ret_10d':r10,'ret_20d':r20,'vol_ratio':vr,
        'price_vs_ma20':pvm,'ma_alignment':mal,'atr_pct':ap,
        'up_days_10':upd,'close':c[-1],'atr_20':a20,'ma_20':ma20,'avg_amount':amt}

def score(dt,bm):
    fl=[]
    for code,info in C.items():
        mask=info['df']['date']<=dt;idx=mask.sum()-1
        if idx<60: continue
        f=factors(info['df'],idx)
        if f is None: continue
        f['code']=code;f['name']=info['name'];fl.append(f)
    if not fl: return pd.DataFrame(),0,0,0
    df=pd.DataFrame(fl);df=df[df['avg_amount']>=MIN_AMT].copy()
    if len(df)==0: return pd.DataFrame(),0,0,0
    sc={}
    for k,w in FW.items():
        if k not in df.columns: continue
        col=df[k].copy();m=col.mean();s=col.std()
        z=pd.Series(0.0,index=col.index) if (s==0 or pd.isna(s)) else ((col-m)/s).clip(-3,3)
        sc[k]=z*w
    df['score']=sum(sc.values())
    df=df.sort_values('score',ascending=False).reset_index(drop=True)
    df['rank']=range(1,len(df)+1)
    best=df['score'].iloc[0]
    ic=0;ib=0;ir=0
    if bm is not None:
        mb=bm[bm['date']<=dt]
        if len(mb)>=BEAR_MA:
            ma=mb['close'].rolling(BEAR_MA).mean().iloc[-1]
            ic=mb['close'].iloc[-1]
            ir=1 if ic>ma*1.05 else 3 if ic<ma else 2
            ib=1 if ic<ma else 0
    return df,best,ib,ir

# === MAIN ===
load()
bm=bench()
if bm is not None: dates=bm['date'].tolist()
else:
    for v in C.values(): dates=v['df']['date'].tolist();break
td=[d for d in dates if START<=d<=END]
print(f'Backtest:',td[0].date(),'~',td[-1].date(),f'({len(td)} days)')
print('Optimizations: SL-8% | CoolOff=3d | Tiered TP(12-15%) | FlexTime(10-15d) | MinHold=2d | PropRank')

pos={};all_days=[];trades=[]
banned={}  # {code: ban_until_date}  cooling-off tracking

for i,dt in enumerate(td):
    sc,best,ib,ir=score(dt,bm)
    if len(sc)==0: continue
    total_stocks = len(sc)
    mx=POS_BEAR if ib else (POS_STRONG if ir==1 else POS_NEUTRAL)

    # === EXITS (V4.3 final) ===
    to_del=[]
    for code,p in list(pos.items()):
        row=sc[sc['code']==code]
        if len(row)==0: to_del.append((code,'NO DATA'));continue
        r=row.iloc[0];cl=r['close'];rk=int(r['rank']);atr=r['atr_20'];ap=r['atr_pct']
        score_now = r['score']
        p['dh']+=1;pf=cl/p['ep']-1;p['hi']=max(p.get('hi',p['ep']),cl)

        # Dynamic hard stop: -2.5xATR with -5% floor and -8% cap
        atr_sl = -EX['atr_sl_mult'] * atr / p['ep']
        hard_sl = max(SL_FLOOR, min(EX['sl_hard'], atr_sl))
        # In bear market, use stricter stop
        if ib: hard_sl = max(SL_FLOOR, EX['sl_bear'])

        # Emergency check
        day_drop = (cl - p.get('prev_close', p['ep'])) / p.get('prev_close', p['ep'])
        emergency = (day_drop <= EMERGENCY_DROP)

        # MA20 consecutive days counter
        below_ma20 = (cl < r['ma_20'])
        p['ma20_cnt'] = p.get('ma20_cnt', 0) + 1 if below_ma20 else 0

        atm=EX['atm_high'] if ap>5 else ((EX['atm_low']+EX['atm_high'])/2 if ap>2.5 else EX['atm_low'])
        tp = EX['tp_top'] if p.get('er',99) <= 3 else EX['tp_std']
        max_hold = 15 if p.get('er',99) <= 5 else EX['max_hold']

        go=False;reason=''
        if pf<=hard_sl: go=True;reason=f'SL({pf*100:.1f}%)'
        elif emergency: go=True;reason=f'CRASH({day_drop*100:.1f}%)'
        elif p['dh']>=MIN_HOLD:
            if pf>=tp: go=True;reason=f'TP({tp*100:.0f}%)'
            elif p['dh']>=max_hold: go=True;reason=f'TIME({p["dh"]}d)'
            elif p['dh']>=TIME_LOSER and pf<0 and p['hi']<=p['ep']: go=True;reason=f'TIME_LOSS({p["dh"]}d)'
            elif p['ma20_cnt']>=MA20_CONSEC: go=True;reason=f'MA20x{MA20_CONSEC}'
            elif rk>total_stocks*EX['rank_out_ratio']: go=True;reason='RANK'
            elif rk-p.get('er',999)>EX['rank_drop']: go=True;reason='RANK_DROP'
            elif cl<p['hi']-atm*atr: go=True;reason='TRAIL'

        if go: to_del.append((code,reason))
        else: p['prev_close'] = cl

    for code,reason in to_del:
        p=pos[code]
        trades.append({'dt':dt,'act':'SELL','c':code,'n':p['name'],'ep':p['ep'],
            'xp':sc[sc['code']==code].iloc[0]['close'],
            'pnl':(sc[sc['code']==code].iloc[0]['close']/p['ep']-1)*100,
            'dh':p['dh'],'rs':reason})
        del pos[code]
        # Set cooling-off ban
        banned[code] = dt + pd.Timedelta(days=COOL_OFF)

    # === BUYS (V4.2 refined) ===
    # Dynamic min_score: higher bar when nearly full
    n_held = len(pos)
    min_sc = MIN_SCORE_BASE + 0.05 * (n_held // 3)  # 0.20 at 0-2, 0.25 at 3-5, 0.30 at 6-8
    slots=mx-len(pos)
    if slots>0 and best>=min_sc:
        held=set(pos.keys())
        for _,r in sc.iterrows():
            if slots<=0: break
            code=r['code']
            if code in held or r['score']<min_sc: continue
            # Cooling-off check
            if code in banned and dt <= banned[code]:
                continue
            # Late-slot quality check: rank 7+ needs score > best*0.7
            if int(r['rank']) >= 7 and r['score'] < best * 0.65:
                continue
            pos[code]={'name':r['name'],'ep':r['close'],'dh':0,'hi':r['close'],'er':int(r['rank']),'prev_close':r['close']}
            trades.append({'dt':dt,'act':'BUY','c':code,'n':r['name'],'ep':r['close'],
                'xp':r['close'],'pnl':0,'dh':0,
                'rs':f'score={r["score"]:.3f} rk=#{int(r["rank"])} min={min_sc:.2f}'})
            slots-=1

    # === RECORD ===
    holds=[]
    for code,p in pos.items():
        row=sc[sc['code']==code]
        if len(row)>0:
            r=row.iloc[0];pnl=(r['close']/p['ep']-1)*100
            holds.append({'c':code,'n':p['name'],'ep':p['ep'],'cp':r['close'],'pnl':pnl,'dh':p['dh'],'er':p['er']})
    holds.sort(key=lambda x:x['pnl'],reverse=True)
    all_days.append({'dt':dt,'n':len(holds),'mx':mx,'reg':ir,'holds':holds,'banned':len([k for k,v in banned.items() if dt<=v])})
    reg=['?','BULL+','NEUTRAL','BEAR'][ir]
    if (i+1)%3==0 or i==0:
        print(f'  [{i+1:3d}/{len(td)}] {dt.date()} {reg} [{len(holds)}/{mx}] banned:{all_days[-1]["banned"]}')

# === REPORT ===
print()
print('='*100)
avg_h=sum(d['n'] for d in all_days)/max(len(all_days),1)
total_ban_days = sum(d.get('banned',0) for d in all_days)
total_trades = len(trades)
buys = [t for t in trades if t['act']=='BUY']
sells = [t for t in trades if t['act']=='SELL']
wins = [t for t in sells if t['pnl']>0]
losses = [t for t in sells if t['pnl']<=0]
avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
win_rate = len(wins)/len(sells)*100 if sells else 0

print(f'  DINGJIASHAN V4.3 FINAL (10-STOCK)')
print(f'  Period: {START.date()} ~ {END.date()} | Days: {len(all_days)} | Avg Hold: {avg_h:.1f}/10')
print(f'  V4.3: DynSL(-5%~-8%) | Crash(8%) | CoolOff2d | MA20x3d | TimeLoss(5d) | TieredTP | FlexTime | DynSc | LateSlot')
print(f'  Trades: {total_trades} ({len(buys)}B/{len(sells)}S) | WinRate: {win_rate:.1f}% | AvgWin: {avg_win:+.1f}% | AvgLoss: {avg_loss:+.1f}%')
print('='*100)

for d in all_days:
    dt=d['dt']
    wds=['MON','TUE','WED','THU','FRI','SAT','SUN']
    wd=wds[dt.weekday()]
    rl='BEAR' if d['reg']==3 else ('BULL+' if d['reg']==1 else 'NEU')
    bar='#'*d['n']+'-'*(d['mx']-d['n']) if d['n']>0 else '.'*d['mx']
    print()
    print(f'  {dt.date()} {wd} {rl} [{d["n"]}/{d["mx"]}] {bar} ban:{d.get("banned",0)}')
    for h in d['holds']:
        if h['pnl']>0: e='G'
        elif h['pnl']<-5: e='R'
        elif h['pnl']<0: e='Y'
        else: e='O'
        tp_mark = 'T' if h.get('er',99)<=3 else ''
        print(f'    {h["c"]:<8s} {h["n"]:<6s} {h["ep"]:>7.2f}>{h["cp"]:>7.2f} [{e}]{h["pnl"]:>+6.1f}% d{h["dh"]} #{h.get("er","?")}{tp_mark}')

# Monthly
print()
print('='*100)
print('  MONTHLY SUMMARY')
print('='*100)
mons={}
for d in all_days:
    mon=d['dt'].strftime('%Y-%m')
    if mon not in mons: mons[mon]={'d':0,'h':0,'c':Counter()}
    mons[mon]['d']+=1;mons[mon]['h']+=d['n']
    for h in d['holds']:mons[mon]['c'][h['c']]+=1
for mon,md in sorted(mons.items()):
    avg=md['h']/md['d'];top=md['c'].most_common(8)
    ts=', '.join([f'{c}({n}d)' for c,n in top])
    print(f'  {mon}: avg {avg:.1f}/10 | {md["d"]}d | Top: {ts}')

# Final holdings
last=all_days[-1]
print()
print('='*100)
print(f'  FINAL HOLDINGS: {last["dt"].date()} ({last["n"]}/{last["mx"]})  banned:{last.get("banned",0)}')
print(f'  {"Code":<8} {"Name":<6} {"Entry":>8} {"Close":>8} {"P&L":>8} {"Days":>5} {"Rank":>5}')
print(f'  {"-"*65}')
for h in last['holds']:
    s='+' if h['pnl']>0 else ''
    tp_tag = ' (TP15%)' if h.get('er',99)<=3 else ' (TP12%)'
    print(f'  {h["c"]:<8} {h["n"]:<6} {h["ep"]:>8.2f} {h["cp"]:>8.2f} {s}{h["pnl"]:>7.1f}% {h["dh"]:>5}d {h.get("er","?"):>5}{tp_tag}')

# Trade summary
print()
print('='*100)
print('  TRADE LOG (all entries and exits)')
print(f'  {"Date":<12} {"Act":<5} {"Code":<8} {"Name":<6} {"Entry":>8} {"Exit":>8} {"PnL":>8} {"Days":>5} {"Reason"}')
print(f'  {"-"*90}')
for t in trades:
    dt_str = str(t['dt'].date()) if hasattr(t['dt'],'date') else str(t['dt'])[:10]
    if t['act']=='BUY':
        print(f'  {dt_str:<12} BUY   {t["c"]:<8} {t["n"]:<6} {"":>8} {t["ep"]:>8.2f} {"":>8} {"":>5} {t["rs"]}')
    else:
        s='+' if t['pnl']>0 else ''
        print(f'  {dt_str:<12} SELL  {t["c"]:<8} {t["n"]:<6} {t["ep"]:>8.2f} {t["xp"]:>8.2f} {s}{t["pnl"]:>7.1f}% {t["dh"]:>5}d {t["rs"]}')

# --- AUTO PUSH ---
import subprocess, json

# Generate markdown report
last_n = min(3, len(all_days))
recent_days = all_days[-last_n:]

md = []
md.append(f'# 丁家山 V4.3.1 每日操盘信号\n')
md.append(f'**{last["dt"].date()}** | NEUTRAL | CSI300 10-stock\n')
md.append(f'\n---\n')

# Recent exits
recent_exits = [t for t in trades if t['act']=='SELL' and t['dt'] in [d['dt'] for d in recent_days]]
if recent_exits:
    md.append(f'## 🔴 近期卖出 ({len(recent_exits)}笔)\n')
    for t in recent_exits[-8:]:
        sgn = '+' if t['pnl']>0 else ''
        md.append(f'- {t["c"]} {t["n"]}: {sgn}{t["pnl"]:.1f}% [{t["rs"]}]\n')

# Recent buys
recent_buys = [t for t in trades if t['act']=='BUY' and t['dt'] in [d['dt'] for d in recent_days]]
if recent_buys:
    md.append(f'\n## 🟢 近期买入 ({len(recent_buys)}笔)\n')
    for t in recent_buys[-8:]:
        md.append(f'- {t["c"]} {t["n"]}: {t["rs"]}\n')

# Current holdings
md.append(f'\n---\n## 📊 当前持仓 ({last["n"]}/{last["mx"]})\n\n')
for h in last['holds']:
    sgn = '+' if h['pnl']>0 else ''
    tp_tag = '⭐15%' if h.get('er',99)<=3 else '12%'
    md.append(f'- {h["c"]} {h["n"]}: {h["ep"]:.2f}->{h["cp"]:.2f} {sgn}{h["pnl"]:.1f}% {h["dh"]}d [{tp_tag}]\n')

md.append(f'\n---\n*{datetime.now().strftime("%Y-%m-%d %H:%M")} 自动生成 | V4.3.1*\n')

report_text = ''.join(md)
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'daily_reports', 'LATEST.md')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

# Git push
repo_dir = os.path.dirname(os.path.abspath(__file__))
try:
    subprocess.run(['git', 'add', 'daily_reports/'], cwd=repo_dir, capture_output=True)
    subprocess.run(['git', 'commit', '-m', f'daily signal {datetime.now().strftime("%Y-%m-%d")} V4.3.1'],
                  cwd=repo_dir, capture_output=True)
    result = subprocess.run(['git', 'push'], cwd=repo_dir, capture_output=True)
    if result.returncode == 0:
        print('[OK] Pushed to GitHub')
    else:
        print('[WARN] Git push failed')
except Exception as e:
    print(f'[WARN] Git error: {e}')

print('='*100)
print('DONE')

