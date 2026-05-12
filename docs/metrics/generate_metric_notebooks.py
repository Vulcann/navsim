#!/usr/bin/env python3
"""
Generate 8 PDM metric demonstration notebooks.
Usage: conda run -n navsim python docs/generate_metric_notebooks.py
"""
import json, os

DOCS = os.path.dirname(os.path.abspath(__file__))
_ctr = [0]

def uid():
    _ctr[0] += 1
    n = _ctr[0]
    return "%04x%04x" % (n, (n * 13 + 7) % 65536)

def _lines(src): return src.splitlines(keepends=True)
def md(src): return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": _lines(src)}
def code(src): return {"cell_type": "code", "id": uid(), "metadata": {}, "source": _lines(src), "outputs": [], "execution_count": None}
def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.0"}}, "nbformat": 4, "nbformat_minor": 5}
def save(notebook, name):
    p = os.path.join(DOCS, name)
    with open(p, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"  Saved: {p}")


IMPORTS = """\
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import savgol_filter

matplotlib.rcParams.update({"figure.dpi": 100, "font.size": 9})

DT = 0.1; T = 6.0
time = np.arange(0, T + DT, DT); N = len(time)
VEH_L, VEH_W = 4.91, 2.10
HL, HW = VEH_L/2, VEH_W/2
_BASE = np.array([[-HL,-HW],[HL,-HW],[HL,HW],[-HL,HW]])

def vcorn(x,y,h):
    c,s=np.cos(h),np.sin(h)
    return _BASE @ np.array([[c,-s],[s,c]]).T + [x,y]

def fpt(x,y,h): return x+HL*np.cos(h), y+HL*np.sin(h)

def ei(rate,v0=0.):
    v=np.zeros(N); v[0]=v0
    for i in range(1,N): v[i]=v[i-1]+rate[i-1]*DT
    return v

def ixy(spd,hdg,x0=0.,y0=0.):
    x,y=np.zeros(N),np.zeros(N); x[0],y[0]=x0,y0
    for i in range(1,N):
        x[i]=x[i-1]+spd[i-1]*np.cos(hdg[i-1])*DT
        y[i]=y[i-1]+spd[i-1]*np.sin(hdg[i-1])*DT
    return x,y

def sat(c1,c2):
    for poly in [c1,c2]:
        for i in range(len(poly)):
            e=poly[(i+1)%len(poly)]-poly[i]
            ax=np.array([-e[1],e[0]]); ax/=(np.linalg.norm(ax)+1e-10)
            if (c1@ax).max()<(c2@ax).min() or (c2@ax).max()<(c1@ax).min(): return False
    return True

def collide(x1,y1,h1,x2,y2,h2): return sat(vcorn(x1,y1,h1),vcorn(x2,y2,h2))

def draw_road(ax,cx,cy,L,W=7.5,hdg=0.):
    c,s=np.cos(hdg),np.sin(hdg); n=np.array([-s,c])
    pts=np.array([[cx-L/2*c-W/2*n[0],cy-L/2*s-W/2*n[1]],[cx+L/2*c-W/2*n[0],cy+L/2*s-W/2*n[1]],
                  [cx+L/2*c+W/2*n[0],cy+L/2*s+W/2*n[1]],[cx-L/2*c+W/2*n[0],cy-L/2*s+W/2*n[1]]])
    ax.add_patch(mpatches.Polygon(pts,closed=True,fc='#444',ec='none',alpha=.35,zorder=0))
    ax.plot([cx-L/2*c,cx+L/2*c],[cy-L/2*s,cy+L/2*s],color='yellow',lw=1,ls='--',alpha=.7,zorder=1)
    for sg in [-1,1]:
        ex,ey=cx+sg*W/2*n[0],cy+sg*W/2*n[1]
        ax.plot([ex-L/2*c,ex+L/2*c],[ey-L/2*s,ey+L/2*s],color='white',lw=1.5,alpha=.8,zorder=1)

def mpatch(ax,x,y,h,fc,alpha=.88,zorder=4):
    p=mpatches.Polygon(vcorn(x,y,h),closed=True,fc=fc,ec='k',alpha=alpha,lw=.8,zorder=zorder)
    ax.add_patch(p); return p

def blimits(ax,xs,ys,pad=8.):
    cx=(xs.max()+xs.min())/2; cy=(ys.max()+ys.min())/2
    side=max(xs.max()-xs.min(),ys.max()-ys.min(),VEH_L*4)/2+pad
    ax.set_xlim(cx-side,cx+side); ax.set_ylim(cy-side,cy+side)
    ax.set_aspect('equal'); ax.grid(True,alpha=.2)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    return cx,cy,side

print("Common helpers loaded.")
"""


# ── Notebook 1: No At-Fault Collision ────────────────────────────────────────
def make_collision_nb():
    helpers = """\
def first_col(ex,ey,eh,ox,oy,oh):
    for i in range(N):
        if collide(ex[i],ey[i],eh[i],ox[i],oy[i],oh[i]): return i
    return None

def anim_col(ex,ey,eh,ox,oy,oh,oc='#CC0000',sb=1.,sh=0.,bttl='',mttl='',interval=100):
    cf=first_col(ex,ey,eh,ox,oy,oh)
    sc=np.where(np.arange(N)<(cf if cf else N),sb,sh)
    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    ax=np.concatenate([ex,ox]); ay=np.concatenate([ey,oy])
    cx,cy,side=blimits(ab,ax,ay,pad=8); draw_road(ab,cx,cy,side*2.5)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.plot(ox,oy,color=oc,lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    op=mpatch(ab,ox[0],oy[0],oh[0],oc,zorder=4)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))
    ctxt=ab.text(.5,.55,'',transform=ab.transAxes,fontsize=14,ha='center',
                  va='center',fontweight='bold',color='red',alpha=0)
    am.set_xlim(time[0],time[-1]); am.set_ylim(-.15,1.25)
    for yv,lc,lb in [(1.,'green','1.0 — pass'),(0.5,'orange','0.5 — static object'),(0.,'red','0.0 — agent hit')]:
        am.axhline(yv,color=lc,lw=1,ls=':',alpha=.5,label=lb)
    sl,=am.plot([],[],color='#1565C0',lw=2.5,zorder=4)
    sd,=am.plot([],[],'o',color='#1565C0',ms=8,zorder=5)
    sv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=13,va='top',ha='right',fontweight='bold')
    am.set_xlabel('time [s]'); am.set_ylabel('collision score')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8,loc='lower right'); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); op.set_xy(vcorn(ox[f],oy[f],oh[f]))
        efd.set_data(*fpt(ex[f],ey[f],eh[f])); tt.set_text(f't = {time[f]:.1f} s')
        is_c=cf is not None and f>=cf
        op.set_fc('#FF0000' if is_c else oc)
        ctxt.set_alpha(.85 if is_c else 0); ctxt.set_text('COLLISION!' if is_c else '')
        sl.set_data(time[:f+1],sc[:f+1]); sd.set_data([time[f]],[sc[f]])
        sv.set_xdata([time[f],time[f]]); s=sc[f]
        slbl.set_text(f'score = {s:.1f}')
        slbl.set_color('green' if s>=1. else 'orange' if s>=.5 else 'red')
        return [ep,op,efd,tt,ctxt,sl,sd,sv,slbl]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Collision helpers loaded.")
"""
    sc1 = """\
# Right lane y=-1.875, left lane y=+1.875 (7.5 m road)
es1=np.full(N,50/3.6); ex1,_=ixy(es1,np.zeros(N),x0=-20.); ey1=np.full(N,-1.875); eh1=np.zeros(N)
os1=np.full(N,35/3.6); ox1,_=ixy(os1,np.zeros(N),x0=0.);   oy1=np.full(N, 1.875); oh1=np.zeros(N)
anim=anim_col(ex1,ey1,eh1,ox1,oy1,oh1,oc='#388E3C',sb=1.,sh=1.,
    bttl='BEV | ego (blue,50 km/h) passes slower car (green,35 km/h) in adjacent lane',
    mttl='Score = 1.0 | adjacent-lane pass — not at fault')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Same lane, stopped agent ahead — rear-end collision
es2=np.full(N,50/3.6); ex2,_=ixy(es2,np.zeros(N),x0=-25.); ey2=np.zeros(N); eh2=np.zeros(N)
ox2=np.zeros(N); oy2=np.zeros(N); oh2=np.zeros(N)
anim=anim_col(ex2,ey2,eh2,ox2,oy2,oh2,oc='#CC0000',sb=1.,sh=0.,
    bttl='BEV | ego (50 km/h) rear-ends stopped agent (red)',
    mttl='Score: 1.0 → 0.0 | STOPPED_TRACK_COLLISION')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Static object (non-agent) — debris in road
es3=np.full(N,30/3.6); ex3,_=ixy(es3,np.zeros(N),x0=-18.); ey3=np.zeros(N); eh3=np.zeros(N)
ox3=np.full(N,4.); oy3=np.zeros(N); oh3=np.zeros(N)
anim=anim_col(ex3,ey3,eh3,ox3,oy3,oh3,oc='#F57C00',sb=1.,sh=0.5,
    bttl='BEV | ego (30 km/h) hits static road barrier (orange, non-agent)',
    mttl='Score: 1.0 → 0.5 | non-agent static object')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# No At-Fault Collision\n\n**Source:** `pdm_scorer.py · _calculate_no_at_fault_collision()`\n\n"
           "**Role:** multiplicative factor — a 0 here zeroes the entire PDM score.\n\n"
           "| Event | Score |\n|---|---|\n"
           "| No collision | 1.0 |\n"
           "| Hit non-agent static object | 0.5 |\n"
           "| Hit agent (car/pedestrian) in at-fault manner | 0.0 |\n\n"
           "**At-fault conditions (navsim):**\n"
           "- `ACTIVE_FRONT_COLLISION` — ego moving forward hits anything ahead\n"
           "- `STOPPED_TRACK_COLLISION` — ego hits a stopped tracked agent\n"
           "- `ACTIVE_LATERAL_COLLISION` while ego is in multiple lanes or off drivable area\n\n"
           "**Not at-fault:** side-swipe while correctly in a single lane (other vehicle assumed at fault)."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Safe Adjacent-Lane Pass (Score = 1.0)\n\n"
           "Ego (blue, 50 km/h) passes a slower vehicle (green, 35 km/h) in the adjacent lane.\n"
           "Polygons never overlap — score stays 1.0.\n\n"
           "> **Real world:** standard highway overtake. Navsim correctly assigns no fault — the vehicles are in separate lanes."),
        code(sc1),
        md("---\n## Scenario 2 — Rear-End Collision with Agent (Score → 0.0)\n\n"
           "Ego (blue, 50 km/h) rear-ends a stopped agent (red). This is a `STOPPED_TRACK_COLLISION`.\n"
           "Score drops to 0.0 at the frame of first polygon overlap.\n\n"
           "> **Real world:** highway approach to a jam — ego failed to brake. Full penalty because a tracked agent was hit at speed."),
        code(sc2),
        md("---\n## Scenario 3 — Hit Static Non-Agent Object (Score → 0.5)\n\n"
           "Ego (blue, 30 km/h) hits a road barrier (orange) that is **not** a tracked agent.\n"
           "Score drops to 0.5 — penalised but less severely.\n\n"
           "> **Real world:** clipping a road cone or hitting stray debris. Ego is at fault, but no road user was endangered directly."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Fault Type | Score |\n|---|---|---|\n"
           "| Adjacent-lane pass | None | 1.0 |\n"
           "| Rear-end stopped agent | STOPPED_TRACK_COLLISION | 0.0 |\n"
           "| Hit static barrier | Non-agent collision | 0.5 |"),
    ])


# ── Notebook 2: TTC ──────────────────────────────────────────────────────────
def make_ttc_nb():
    helpers = """\
# TTC = front-to-rear gap / closing speed (capped at 3 s for display)
# navsim: if ego speed < 5e-3 m/s → skip TTC check (stopped ego)
# violation if lead is ahead AND TTC < 1.0 s horizon

def compute_ttc(ex,ey,es, ox,oy,os_arr):
    ttc=np.full(N,3.)
    gap=np.maximum((ox-ex)-VEH_L,0.)   # front-of-ego to rear-of-other (same lane)
    rel=es-os_arr                        # closing speed
    for i in range(N):
        if es[i]<5e-3: ttc[i]=3.; continue  # stopped ego → no check
        if ox[i]>ex[i] and rel[i]>1e-3:     # lead is ahead and closing
            ttc[i]=min(3.,gap[i]/rel[i])
    return ttc

def anim_ttc(ex,ey,eh,es, ox,oy,oh,os_arr,bttl='',mttl='',interval=100):
    ttc=compute_ttc(ex,ey,es,ox,oy,os_arr)
    sc=np.where(ttc>=1.,1.,0.)
    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    ax_=np.concatenate([ex,ox]); ay_=np.concatenate([ey,oy])
    cx,cy,side=blimits(ab,ax_,ay_,pad=8); draw_road(ab,cx,cy,side*2.5)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.plot(ox,oy,color='#CC0000',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    op=mpatch(ab,ox[0],oy[0],oh[0],'#CC0000',zorder=4)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))
    gap_line,=ab.plot([],[],color='cyan',lw=2,zorder=3)
    gap_txt=ab.text(.5,.06,'',transform=ab.transAxes,fontsize=9,ha='center',
                     bbox=dict(boxstyle='round',fc='cyan',alpha=.6))
    am.set_xlim(time[0],time[-1]); am.set_ylim(-.1,3.3)
    am.axhline(1.,color='red',lw=1.5,ls='--',label='TTC threshold = 1.0 s')
    am.axhline(3.,color='gray',lw=.8,ls=':',alpha=.5,label='display cap = 3.0 s')
    tl,=am.plot([],[],color='#1565C0',lw=2,zorder=4,label='TTC')
    td,=am.plot([],[],'o',color='#1565C0',ms=7,zorder=5)
    tv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=12,va='top',ha='right',fontweight='bold')
    am.set_xlabel('time [s]'); am.set_ylabel('TTC [s]')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); op.set_xy(vcorn(ox[f],oy[f],oh[f]))
        efd.set_data(*fpt(ex[f],ey[f],eh[f])); tt.set_text(f't = {time[f]:.1f} s')
        fx=ex[f]+HL*np.cos(eh[f]); rx=ox[f]-HL*np.cos(oh[f])
        if rx>fx: gap_line.set_data([fx,rx],[ey[f],oy[f]]); gap_txt.set_text(f'gap={rx-fx:.1f}m')
        else: gap_line.set_data([],[]);gap_txt.set_text('')
        tl.set_data(time[:f+1],ttc[:f+1]); td.set_data([time[f]],[ttc[f]])
        tv.set_xdata([time[f],time[f]]); s=sc[f]
        slbl.set_text('PASS' if s>=1. else 'FAIL'); slbl.set_color('green' if s>=1. else 'red')
        return [ep,op,efd,tt,gap_line,gap_txt,tl,td,tv,slbl]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("TTC helpers loaded.")
"""
    sc1 = """\
# Safe following: ego 50 km/h, lead 50 km/h (same speed), 25 m gap → TTC = inf
es1=np.full(N,50/3.6); ex1,_=ixy(es1,np.zeros(N),x0=-30.); ey1=np.zeros(N); eh1=np.zeros(N)
os1=np.full(N,50/3.6); ox1,_=ixy(os1,np.zeros(N),x0=-5.);  oy1=np.zeros(N); oh1=np.zeros(N)
anim=anim_ttc(ex1,ey1,eh1,es1,ox1,oy1,oh1,os1,
    bttl='BEV | ego 50 km/h following lead 50 km/h — constant 25 m gap',
    mttl='TTC = ∞ (same speed) → score = 1.0 throughout')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Approaching: ego 50 km/h, lead 20 km/h, starting gap 10 m → TTC ≈ 1.2 s → dips below 1 s
es2=np.full(N,50/3.6); ex2,_=ixy(es2,np.zeros(N),x0=-15.); ey2=np.zeros(N); eh2=np.zeros(N)
os2=np.full(N,20/3.6); ox2,_=ixy(os2,np.zeros(N),x0=-5.);  oy2=np.zeros(N); oh2=np.zeros(N)
anim=anim_ttc(ex2,ey2,eh2,es2,ox2,oy2,oh2,os2,
    bttl='BEV | ego 50 km/h closing on lead 20 km/h — gap shrinking',
    mttl='TTC drops below 1.0 s → score flips to 0.0')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Stopped ego: speed < 5e-3 m/s → no TTC penalty regardless of proximity
es3=np.zeros(N)  # completely stopped
ex3=np.full(N,-2.); ey3=np.zeros(N); eh3=np.zeros(N)
ox3=np.full(N,3.); oy3=np.zeros(N); oh3=np.zeros(N)  # 5 m ahead, also stopped
anim=anim_ttc(ex3,ey3,eh3,es3,ox3,oy3,oh3,np.zeros(N),
    bttl='BEV | ego stopped — only 5 m from lead (also stopped)',
    mttl='Ego speed < 5e-3 m/s → TTC check skipped → score = 1.0')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Time-to-Collision (TTC)\n\n**Source:** `pdm_scorer.py · _calculate_ttc()`\n\n"
           "**Role:** weighted metric (weight = 5.0 — same as progress, highest weight)\n\n"
           "| Condition | Score |\n|---|---|\n"
           "| No TTC violation in 1 s horizon | 1.0 |\n"
           "| TTC violation detected | 0.0 |\n\n"
           "**How navsim computes TTC:**\n"
           "For each ego pose, project the vehicle's front corners forward at current speed\n"
           "for `{0, 0.3, 0.6, 0.9}` seconds. If any projection intersects a tracked object\n"
           "that is **ahead** of ego (and ego is not stopped), TTC violation → score = 0.0.\n\n"
           "**Key rule:** if `ego_speed < 5e-3 m/s` → TTC check is skipped entirely.\n"
           "A stopped vehicle cannot be at fault for a forward collision."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Safe Following Distance (Score = 1.0)\n\n"
           "Ego (blue, 50 km/h) follows lead (red, 50 km/h) at 25 m gap.\n"
           "Relative speed = 0 → TTC = ∞ → always passes.\n\n"
           "> **Real world:** cruising on a motorway at same speed as the car ahead — safe lane-hold."),
        code(sc1),
        md("---\n## Scenario 2 — Closing In, TTC Violation (Score → 0.0)\n\n"
           "Ego (blue, 50 km/h) approaches a slow lead (red, 20 km/h) from 10 m behind.\n"
           "Closing speed = 30 km/h = 8.33 m/s. TTC drops below 1 s quickly → score 0.0.\n\n"
           "> **Real world:** failing to adjust speed when approaching slow traffic — a near-miss\n"
           "> situation. The ego should have braked earlier to maintain a safe gap."),
        code(sc2),
        md("---\n## Scenario 3 — Stopped Ego (Score = 1.0 despite tiny gap)\n\n"
           "Ego is completely stopped, only 5 m from a stationary lead.\n"
           "Because `ego_speed < 5e-3 m/s`, the TTC check is **skipped** — score = 1.0.\n\n"
           "> **Real world:** bumper-to-bumper traffic jam stop. No TTC penalty — a stopped\n"
           "> vehicle is not creating a collision risk by being close to another stopped vehicle."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Ego speed | Lead speed | Gap | TTC | Score |\n|---|---|---|---|---|---|\n"
           "| Safe follow | 50 km/h | 50 km/h | 25 m | ∞ | 1.0 |\n"
           "| Closing in | 50 km/h | 20 km/h | 10 m | < 1 s | 0.0 |\n"
           "| Stopped ego | 0 km/h | 0 km/h | 5 m | N/A | 1.0 |\n\n"
           "> TTC has weight **5.0** — tied with Ego Progress as the highest-weighted metric.\n"
           "> A single TTC violation halves the weighted sub-score even before the multiplicative metrics apply."),
    ])


# ── Notebook 3: Lane Keeping ─────────────────────────────────────────────────
def make_lane_keeping_nb():
    helpers = """\
# Lane keeping: lateral deviation from centerline > 0.5 m for >= 20 consecutive steps (2.0 s) → fail
# Centerline = x-axis (y=0). Lateral deviation = abs(ego_y).
# Intersections excluded in real navsim; omitted here (straight road, no intersections).

LANE_LIMIT   = 0.5   # [m]
CONT_STEPS   = 20    # 2.0 s at DT=0.1

def compute_lane_score(ey):
    dev=np.abs(ey); consec=0; score=1.
    for i in range(N):
        if dev[i]>LANE_LIMIT: consec+=1
        else: consec=0
        if consec>=CONT_STEPS: score=0.; break
    return score, dev

def anim_lane(ex,ey,eh,es,lane_w=3.5,bttl='',mttl='',interval=100):
    score,dev=compute_lane_score(ey)
    sc_arr=np.ones(N); consec=0
    for i in range(N):
        if dev[i]>LANE_LIMIT: consec+=1
        else: consec=0
        if consec>=CONT_STEPS: sc_arr[i:]=0.; break

    fig,(ab,am,ac)=plt.subplots(1,3,figsize=(17,5))
    cx,cy,side=blimits(ab,ex,ey,pad=8); draw_road(ab,cx,cy,side*2.5,W=lane_w)
    # Lane limit lines
    for sg in [-1,1]:
        ab.axhline(sg*LANE_LIMIT,color='red',lw=1.2,ls='--',alpha=.8,zorder=2)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    lat_arrow=ab.annotate('',xy=(ex[0],0.),xytext=(ex[0],ey[0]),
                           arrowprops=dict(arrowstyle='<->',color='cyan',lw=1.5),zorder=3)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))

    # Deviation panel
    am.axhline(LANE_LIMIT,color='red',lw=1.5,ls='--',label=f'limit = {LANE_LIMIT} m')
    am.plot(time,dev,color='#1565C0',lw=1,alpha=.18)
    dl,=am.plot([],[],color='#1565C0',lw=2,zorder=4,label='|lat. error|')
    dd,=am.plot([],[],'o',color='#1565C0',ms=7,zorder=5)
    dv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    am.set_xlim(time[0],time[-1]); am.set_ylim(-.05,max(dev.max(),LANE_LIMIT)*1.3)
    am.set_xlabel('time [s]'); am.set_ylabel('lateral deviation [m]')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=12,va='top',ha='right',fontweight='bold')

    # Consecutive steps panel
    cons_arr=np.zeros(N); c2=0
    for i in range(N):
        if dev[i]>LANE_LIMIT: c2+=1
        else: c2=0
        cons_arr[i]=c2
    ac.axhline(CONT_STEPS,color='red',lw=1.5,ls='--',label=f'threshold = {CONT_STEPS} steps (2 s)')
    cl,=ac.plot([],[],color='#E65100',lw=2,zorder=4,label='consecutive steps > 0.5 m')
    cd,=ac.plot([],[],'o',color='#E65100',ms=7,zorder=5)
    cv=ac.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    ac.set_xlim(time[0],time[-1]); ac.set_ylim(-1,CONT_STEPS*1.5)
    ac.set_xlabel('time [s]'); ac.set_ylabel('consecutive steps')
    ac.set_title('Duration counter (≥20 steps = 2 s → FAIL)',fontsize=8)
    ac.legend(fontsize=8); ac.grid(True,alpha=.3)
    plt.tight_layout()

    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); efd.set_data(*fpt(ex[f],ey[f],eh[f]))
        tt.set_text(f't = {time[f]:.1f} s')
        lat_arrow.set_position((ex[f],0.)); lat_arrow.xy=(ex[f],ey[f])
        dl.set_data(time[:f+1],dev[:f+1]); dd.set_data([time[f]],[dev[f]])
        dv.set_xdata([time[f],time[f]])
        cl.set_data(time[:f+1],cons_arr[:f+1]); cd.set_data([time[f]],[cons_arr[f]])
        cv.set_xdata([time[f],time[f]])
        s=sc_arr[f]; slbl.set_text('PASS' if s>=1. else 'FAIL')
        slbl.set_color('green' if s>=1. else 'red')
        return [ep,efd,tt,dl,dd,dv,cl,cd,cv,slbl]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Lane keeping helpers loaded.")
"""
    sc1 = """\
# Perfect centering
es1=np.full(N,30/3.6); ex1=ei(es1,v0=0.); ey1=np.zeros(N); eh1=np.zeros(N)
anim=anim_lane(ex1,ey1,eh1,es1,
    bttl='BEV | perfectly centered — lateral error = 0',
    mttl='Lateral deviation well below 0.5 m → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Brief drift: exceeds 0.5 m for 1.5 s (15 steps) then returns — under 2 s threshold
ey2=np.where((time>=1.)&(time<2.5), 0.65, 0.05)   # 1.5 s above limit
es2=np.full(N,30/3.6); ex2=ei(es2,v0=0.); eh2=np.zeros(N)
anim=anim_lane(ex2,ey2,eh2,es2,
    bttl='BEV | ego drifts 0.65 m for 1.5 s then returns to lane',
    mttl='Exceeds 0.5 m for 15 steps < 20 threshold → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Sustained drift: > 0.5 m for 3 s (30 steps) — exceeds 2 s threshold → fail
ey3=np.where(time>=1.5, 0.75, 0.0)   # starts drifting at t=1.5 s, stays there
es3=np.full(N,30/3.6); ex3=ei(es3,v0=0.); eh3=np.zeros(N)
anim=anim_lane(ex3,ey3,eh3,es3,
    bttl='BEV | ego drifts 0.75 m starting at t=1.5 s — stays off-center',
    mttl='Consecutive steps > 0.5 m reaches 20 (2.0 s) → score = 0.0')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Lane Keeping\n\n**Source:** `pdm_scorer.py · _calculate_lane_keeping()`\n\n"
           "**Role:** weighted metric (weight = 2.0)\n\n"
           "| Condition | Score |\n|---|---|\n"
           "| Lateral deviation ≤ 0.5 m at all times | 1.0 |\n"
           "| Deviation > 0.5 m for < 2.0 s continuously | 1.0 |\n"
           "| Deviation > 0.5 m for ≥ 2.0 s continuously | 0.0 |\n\n"
           "**Key parameters:**\n"
           "- `lane_keeping_deviation_limit` = 0.5 m (lateral distance from centerline)\n"
           "- `lane_keeping_horizon_window` = 2.0 s (continuous violation window)\n"
           "- `continuous_steps_required` = 20 steps (at DT=0.1 s)\n"
           "- **Intersections excluded** — the check is skipped at intersection poses\n\n"
           "The counter **resets** if the vehicle returns to within 0.5 m. A brief excursion\n"
           "followed by correction does not fail."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Perfect Centering (Score = 1.0)\n\nEgo drives exactly on the centerline. Deviation = 0 everywhere.\n\n"
           "> **Real world:** textbook lane-hold on a straight road — cruise control, no distractions."),
        code(sc1),
        md("---\n## Scenario 2 — Brief Drift, Self-Corrected (Score = 1.0)\n\n"
           "Ego drifts 0.65 m off-center for 1.5 s (15 steps), then returns.\n"
           "Since 15 < 20 (threshold), the counter never reaches the limit — score stays 1.0.\n\n"
           "> **Real world:** momentary distraction (checking mirror) then correcting. The system\n"
           "> is forgiving of brief deviations that are self-corrected within 2 seconds."),
        code(sc2),
        md("---\n## Scenario 3 — Sustained Drift (Score → 0.0)\n\n"
           "Ego drifts 0.75 m off-center starting at t = 1.5 s and stays there.\n"
           "After 20 consecutive steps (2.0 s) above the limit → score = 0.0.\n\n"
           "> **Real world:** falling asleep at the wheel, or following a curve poorly.\n"
           "> The 2-second window distinguishes real lane departure from brief road irregularity."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Max deviation | Duration above limit | Score |\n|---|---|---|---|\n"
           "| Perfect centering | 0.0 m | 0 s | 1.0 |\n"
           "| Brief drift | 0.65 m | 1.5 s | 1.0 |\n"
           "| Sustained drift | 0.75 m | > 2.0 s | 0.0 |"),
    ])


# ── Notebook 4: Drivable Area ────────────────────────────────────────────────
def make_drivable_area_nb():
    helpers = """\
# Drivable area: ALL 4 vehicle corners must lie within drivable polygons.
# Any corner outside → score = 0.0 immediately.
# Simulated here as a straight road: drivable if |y_corner| <= road_half_width.

ROAD_HW = 3.75  # half-width of drivable road [m] (7.5 m total, 2 lanes)

def corner_in_road(x,y,h):
    c=vcorn(x,y,h)
    return np.all(np.abs(c[:,1])<=ROAD_HW)

def compute_da_score(ex,ey,eh):
    sc=np.ones(N)
    for i in range(N):
        if not corner_in_road(ex[i],ey[i],eh[i]): sc[i:]=0.; break
    return sc

def anim_da(ex,ey,eh,es,bttl='',mttl='',interval=100):
    sc=compute_da_score(ex,ey,eh)
    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    cx,cy,side=blimits(ab,ex,ey,pad=8); draw_road(ab,cx,cy,side*2.5,W=ROAD_HW*2)
    # Off-road zone shading
    for sg in [-1,1]:
        ab.axhspan(sg*(ROAD_HW),(sg*(ROAD_HW+6)),color='#8B4513',alpha=.25,zorder=0)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    # Corner dots
    cdots,=[],
    for _ in range(4): cdots.append(ab.plot([],[],'+',ms=8,mew=2,color='lime',zorder=7)[0])
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))
    am.set_xlim(time[0],time[-1]); am.set_ylim(-.15,1.25)
    am.axhline(1.,color='green',lw=1,ls=':',alpha=.5,label='pass = 1.0')
    am.axhline(0.,color='red',lw=1,ls=':',alpha=.5,label='fail = 0.0')
    sl,=am.plot([],[],color='#1565C0',lw=2.5,zorder=4)
    sd,=am.plot([],[],'o',color='#1565C0',ms=8,zorder=5)
    sv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=13,va='top',ha='right',fontweight='bold')
    am.set_xlabel('time [s]'); am.set_ylabel('drivable area score')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); efd.set_data(*fpt(ex[f],ey[f],eh[f]))
        tt.set_text(f't = {time[f]:.1f} s')
        c=vcorn(ex[f],ey[f],eh[f])
        in_road=np.abs(c[:,1])<=ROAD_HW
        for k,cd in enumerate(cdots):
            cd.set_data([c[k,0]],[c[k,1]])
            cd.set_color('lime' if in_road[k] else 'red')
        sl.set_data(time[:f+1],sc[:f+1]); sd.set_data([time[f]],[sc[f]])
        sv.set_xdata([time[f],time[f]]); s=sc[f]
        slbl.set_text('PASS' if s>=1. else 'FAIL'); slbl.set_color('green' if s>=1. else 'red')
        return [ep,efd,tt,sl,sd,sv,slbl]+cdots
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Drivable area helpers loaded.")
"""
    sc1 = """\
es1=np.full(N,30/3.6); ex1=ei(es1); ey1=np.zeros(N); eh1=np.zeros(N)
anim=anim_da(ex1,ey1,eh1,es1,
    bttl='BEV | ego centered on road — all corners (green +) inside drivable area',
    mttl='All 4 corners within road bounds → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Gradual drift off road: ego drifts slowly to y=5.0 m (road edge at 3.75 m)
drift_rate=1.2/3.  # m/s lateral
ey2=np.minimum(np.arange(N)*DT*drift_rate, 5.5)
es2=np.full(N,30/3.6); ex2=ei(es2); eh2=np.zeros(N)
anim=anim_da(ex2,ey2,eh2,es2,
    bttl='BEV | ego gradually drifts off road — corners turn red as boundary crossed',
    mttl='Score = 1.0 → 0.0 when first corner crosses road boundary')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Sharp turn: ego turns sharply, one rear corner exits road
turn_yaw=np.where(time<1.,0.5,0.)*time
ey3=np.cumsum(np.r_[0,np.full(N-1,30/3.6)*np.sin(turn_yaw[:-1])*DT])
ex3=np.cumsum(np.r_[0,np.full(N-1,30/3.6)*np.cos(turn_yaw[:-1])*DT])
# For visualization: use a simple diagonal drift
ey3b=np.minimum(np.arange(N)*DT*2.0,7.)
ex3b=np.cumsum(np.r_[0,np.full(N-1,25/3.6)*DT])
anim=anim_da(ex3b,ey3b,np.zeros(N),np.full(N,25/3.6),
    bttl='BEV | ego drives diagonally off road at 25 km/h',
    mttl='Score = 0.0 once any corner exits road boundary (brown = off-road zone)')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Drivable Area Compliance\n\n**Source:** `pdm_scorer.py · _calculate_drivable_area_compliance()`\n\n"
           "**Role:** multiplicative factor\n\n"
           "| Condition | Score |\n|---|---|\n"
           "| All 4 corners within drivable area | 1.0 |\n"
           "| Any corner outside drivable area | 0.0 |\n\n"
           "**Drivable layers checked (navsim):** ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK_AREA\n\n"
           "Corner indicators: **green +** = inside road, **red +** = outside.\n"
           "Brown shading = non-drivable area (grass, footpath, etc.)"),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Stays on Road (Score = 1.0)\n\nEgo drives straight, centered on road. All 4 corners well within bounds.\n\n"
           "> **Real world:** normal lane-hold driving."),
        code(sc1),
        md("---\n## Scenario 2 — Gradual Drift Off Road (Score → 0.0)\n\n"
           "Ego drifts laterally at 0.4 m/s. When the first corner crosses the road edge (|y| > 3.75 m), score flips to 0.0.\n\n"
           "> **Real world:** slowly driving off the road onto a grass verge or pavement.\n"
           "> Even one millimetre outside the drivable polygon triggers the penalty."),
        code(sc2),
        md("---\n## Scenario 3 — Diagonal Exit (Score → 0.0)\n\n"
           "Ego drives diagonally across the lane and exits the road boundary.\n\n"
           "> **Real world:** U-turn attempt that overshoots onto the verge, or cutting a corner across a median."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Off-road? | Score |\n|---|---|---|\n"
           "| Centered straight drive | No | 1.0 |\n"
           "| Gradual drift | Yes (lateral) | 0.0 |\n"
           "| Diagonal exit | Yes | 0.0 |\n\n"
           "> Drivable area is a hard constraint — **multiplicative**. A score of 0.0 here\n"
           "> zeroes the entire PDM score regardless of all other metrics."),
    ])


# ── Notebook 5: Driving Direction ────────────────────────────────────────────
def make_driving_dir_nb():
    helpers = """\
# Driving direction: accumulate wrong-way progress over a 10-step (1.0 s) sliding window.
# Correct direction: x increases (heading ~0). Wrong way: x decreases (heading ~pi).
# Thresholds: <2m → 1.0, 2-6m → 0.5, >6m → 0.0.

COMP_THRESH = 2.0
VIOL_THRESH = 6.0
HORIZON_STEPS = 10

def wrong_way_progress(ex, eh):
    # Wrong-way: moving in -x direction (cos(h) < 0) while in oncoming lane (y > 0)
    dx = np.diff(ex, prepend=ex[0])
    wrong = (dx < 0) & (True)  # simplified: any backward progress
    ww_prog = np.where(wrong, np.abs(dx), 0.)
    # sliding window sum
    window_sum = np.zeros(N)
    for i in range(N):
        window_sum[i] = ww_prog[max(0,i-HORIZON_STEPS):i+1].sum()
    return window_sum

def score_from_ww(ww):
    mx = ww.max()
    if mx < COMP_THRESH: return 1.0
    if mx < VIOL_THRESH: return 0.5
    return 0.0

def anim_ddir(ex,ey,eh,es, bttl='',mttl='',interval=100):
    ww=wrong_way_progress(ex,eh)
    sc=np.ones(N)
    reached_05=False; reached_00=False
    for i in range(N):
        if ww[i]>=VIOL_THRESH and not reached_00: sc[i:]=0.; reached_00=True; break
        elif ww[i]>=COMP_THRESH and not reached_05: sc[i:]=0.5; reached_05=True
    if not reached_00 and not reached_05: sc[:]=1.

    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    all_x=np.concatenate([ex,[ex.min()-5,ex.max()+5]])
    all_y=np.concatenate([ey,[-4,4]])
    cx,cy,side=blimits(ab,all_x,all_y,pad=5)
    draw_road(ab,cx,cy,side*2.5,W=7.5)
    # Direction arrows for correct lane
    for xa in np.linspace(cx-side+2,cx+side-2,6):
        ab.annotate('',xy=(xa+2,cy-1.875),xytext=(xa,cy-1.875),
                     arrowprops=dict(arrowstyle='->',color='#00BCD4',lw=1.5),zorder=2)
    # Oncoming lane arrows (opposite direction)
    for xa in np.linspace(cx+side-2,cx-side+2,6):
        ab.annotate('',xy=(xa-2,cy+1.875),xytext=(xa,cy+1.875),
                     arrowprops=dict(arrowstyle='->',color='orange',lw=1.5),zorder=2)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))

    am.set_xlim(time[0],time[-1]); am.set_ylim(-0.5,VIOL_THRESH*1.4)
    am.axhline(COMP_THRESH,color='orange',lw=1.5,ls='--',label=f'score 0.5 at {COMP_THRESH} m')
    am.axhline(VIOL_THRESH,color='red',lw=1.5,ls='--',label=f'score 0.0 at {VIOL_THRESH} m')
    wl,=am.plot([],[],color='#880E4F',lw=2,zorder=4,label='wrong-way dist (1 s window)')
    wd,=am.plot([],[],'o',color='#880E4F',ms=7,zorder=5)
    wv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=12,va='top',ha='right',fontweight='bold')
    am.set_xlabel('time [s]'); am.set_ylabel('wrong-way distance in window [m]')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); efd.set_data(*fpt(ex[f],ey[f],eh[f]))
        tt.set_text(f't = {time[f]:.1f} s')
        wl.set_data(time[:f+1],ww[:f+1]); wd.set_data([time[f]],[ww[f]])
        wv.set_xdata([time[f],time[f]]); s=sc[f]
        slbl.set_text(f'score={s:.1f}'); slbl.set_color('green' if s>=1. else 'orange' if s>=.5 else 'red')
        return [ep,efd,tt,wl,wd,wv,slbl]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Driving direction helpers loaded.")
"""
    sc1 = """\
# Correct direction: heading=0, y=-1.875 (right lane)
es1=np.full(N,40/3.6); ex1=ei(es1,v0=0.); ey1=np.full(N,-1.875); eh1=np.zeros(N)
anim=anim_ddir(ex1,ey1,eh1,es1,
    bttl='BEV | ego drives in correct direction (cyan arrows = correct lane)',
    mttl='Wrong-way progress = 0 → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Brief wrong-way: ego starts going backward for 1.5 s (< 2 m accumulated) then recovers
ex2=np.zeros(N)
v2=np.where(time<1.5, -8/3.6, 30/3.6)  # reverse briefly
for i in range(1,N): ex2[i]=ex2[i-1]+v2[i-1]*DT
ey2=np.zeros(N); eh2=np.where(time<1.5,np.pi,0.)
anim=anim_ddir(ex2,ey2,eh2,np.abs(v2),
    bttl='BEV | ego reverses briefly then drives forward — wrong-way distance < 2 m',
    mttl='Max wrong-way < 2 m → score = 1.0 (brief incursion forgiven)')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Severe wrong-way: ego drives backward at 30 km/h in oncoming lane for 3+ seconds
v3=np.where(time<4.,-30/3.6,30/3.6)
ex3=np.zeros(N)
for i in range(1,N): ex3[i]=ex3[i-1]+v3[i-1]*DT
ey3=np.full(N,1.875); eh3=np.where(time<4.,np.pi,0.)  # oncoming lane
anim=anim_ddir(ex3,ey3,eh3,np.abs(v3),
    bttl='BEV | ego drives in oncoming lane for 4 s at 30 km/h',
    mttl='Wrong-way > 6 m → score = 0.0 (full violation)')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Driving Direction Compliance\n\n**Source:** `pdm_scorer.py · _calculate_driving_direction_compliance()`\n\n"
           "**Role:** multiplicative factor\n\n"
           "| Wrong-way progress (1 s window) | Score |\n|---|---|\n"
           "| < 2.0 m | 1.0 |\n"
           "| 2.0 – 6.0 m | 0.5 |\n"
           "| > 6.0 m | 0.0 |\n\n"
           "**Key details:**\n"
           "- Progress accumulated over a **1-second sliding window** (10 steps)\n"
           "- Only counts progress while in **oncoming traffic** zone\n"
           "- **Intersections excluded** — entering the wrong side of an intersection is not penalised\n"
           "- Cyan arrows = correct direction; orange arrows = oncoming lane direction"),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Correct Direction (Score = 1.0)\n\nEgo drives forward in the right lane. No wrong-way progress.\n\n"
           "> **Real world:** normal forward driving."),
        code(sc1),
        md("---\n## Scenario 2 — Brief Reversal (Score = 1.0)\n\n"
           "Ego reverses briefly (1.5 s at 8 km/h) then drives forward. Wrong-way distance < 2 m → forgiven.\n\n"
           "> **Real world:** short back-up manoeuvre in a dead end, or brief reverse to correct position."),
        code(sc2),
        md("---\n## Scenario 3 — Wrong-Way Driving (Score → 0.0)\n\n"
           "Ego drives in the oncoming lane for 4 s at 30 km/h. Wrong-way distance accumulates past 6 m → score = 0.0.\n\n"
           "> **Real world:** driving on the wrong side of the road — ghost driver scenario.\n"
           "> The 6 m threshold (≈ 2 car lengths) distinguishes intent from brief incursion."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Wrong-way distance | Score |\n|---|---|---|\n"
           "| Correct direction | 0 m | 1.0 |\n"
           "| Brief reversal | < 2 m | 1.0 |\n"
           "| Wrong-way driving | > 6 m | 0.0 |"),
    ])


# ── Notebook 6: Traffic Light ────────────────────────────────────────────────
def make_traffic_light_nb():
    helpers = """\
# Traffic light: ego polygon overlaps with red light zone (stop box at intersection) → score = 0.0.
# Simulated as: stop_line at x=STOP_X; red zone extends from x=STOP_X to x=STOP_X+INTER_DEPTH.

STOP_X     = 15.0
INTER_DEPTH = 12.0
INTER_W    = 10.0

def in_red_zone(x,y,h):
    c=vcorn(x,y,h)
    return np.any((c[:,0]>STOP_X) & (c[:,0]<STOP_X+INTER_DEPTH) & (np.abs(c[:,1])<INTER_W/2))

def compute_tl_score(ex,ey,eh,light_is_red):
    sc=np.ones(N)
    for i in range(N):
        if light_is_red[i] and in_red_zone(ex[i],ey[i],eh[i]): sc[i:]=0.; break
    return sc

def anim_tl(ex,ey,eh,es,light_is_red,bttl='',mttl='',interval=100):
    sc=compute_tl_score(ex,ey,eh,light_is_red)
    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    cx,cy,side=blimits(ab,ex,ey,pad=10); draw_road(ab,cx,cy,side*2.5)
    # Intersection box
    inter=mpatches.FancyBboxPatch((STOP_X,-INTER_W/2),INTER_DEPTH,INTER_W,
                                   boxstyle='square',fc='#FFF9C4',ec='#F9A825',alpha=.5,lw=1.5,zorder=1)
    ab.add_patch(inter)
    # Stop line
    ab.plot([STOP_X,STOP_X],[-INTER_W/2,INTER_W/2],color='white',lw=2,zorder=2)
    ab.text(STOP_X+0.3,-INTER_W/2+0.5,'STOP',color='white',fontsize=7,zorder=3,rotation=90)
    # Traffic light indicator (circle)
    tl_circle=plt.Circle((STOP_X-1,4.5),0.8,color='green',zorder=5)
    ab.add_patch(tl_circle)
    ab.text(STOP_X-1,6.,'TL',ha='center',fontsize=7,color='white',zorder=6)
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8)
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=6)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=7)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))
    tl_txt=ab.text(.5,.06,'',transform=ab.transAxes,fontsize=11,ha='center',
                    fontweight='bold',va='bottom')
    am.set_xlim(time[0],time[-1]); am.set_ylim(-.15,1.25)
    am.axhline(1.,color='green',lw=1,ls=':',alpha=.5,label='pass=1.0')
    am.axhline(0.,color='red',lw=1,ls=':',alpha=.5,label='fail=0.0')
    sl,=am.plot([],[],color='#1565C0',lw=2.5,zorder=4)
    sd,=am.plot([],[],'o',color='#1565C0',ms=8,zorder=5)
    sv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=13,va='top',ha='right',fontweight='bold')
    am.set_xlabel('time [s]'); am.set_ylabel('traffic light score')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); efd.set_data(*fpt(ex[f],ey[f],eh[f]))
        tt.set_text(f't = {time[f]:.1f} s')
        is_red=bool(light_is_red[f])
        tl_circle.set_color('red' if is_red else 'green')
        tl_txt.set_text('RED LIGHT' if is_red else 'GREEN')
        tl_txt.set_color('red' if is_red else 'green')
        sl.set_data(time[:f+1],sc[:f+1]); sd.set_data([time[f]],[sc[f]])
        sv.set_xdata([time[f],time[f]]); s=sc[f]
        slbl.set_text('PASS' if s>=1. else 'FAIL'); slbl.set_color('green' if s>=1. else 'red')
        return [ep,efd,tt,tl_circle,tl_txt,sl,sd,sv,slbl]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Traffic light helpers loaded.")
"""
    sc1 = """\
# Green light: ego drives through at 40 km/h
es1=np.full(N,40/3.6); ex1=ei(es1,v0=0.); ey1=np.zeros(N); eh1=np.zeros(N)
red1=np.zeros(N,dtype=bool)  # always green
anim=anim_tl(ex1,ey1,eh1,es1,red1,
    bttl='BEV | ego (40 km/h) drives through green light',
    mttl='Light is green → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Red light, ego brakes and stops before stop line
es2=np.maximum(40/3.6 - np.arange(N)*DT*8., 0.)  # decelerate hard
ex2=ei(es2); ey2=np.zeros(N); eh2=np.zeros(N)
red2=np.ones(N,dtype=bool)  # always red
anim=anim_tl(ex2,ey2,eh2,es2,red2,
    bttl='BEV | ego brakes and stops before stop line — light is red',
    mttl='Ego stops before x=15 m → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Red light, ego runs it at 40 km/h
es3=np.full(N,40/3.6); ex3=ei(es3); ey3=np.zeros(N); eh3=np.zeros(N)
red3=np.ones(N,dtype=bool)  # always red
anim=anim_tl(ex3,ey3,eh3,es3,red3,
    bttl='BEV | ego (40 km/h) runs RED light — enters intersection box',
    mttl='Score: 1.0 → 0.0 when ego polygon enters red zone')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Traffic Light Compliance\n\n**Source:** `pdm_scorer.py · _calculate_traffic_light_compliance()`\n\n"
           "**Role:** multiplicative factor\n\n"
           "| Condition | Score |\n|---|---|\n"
           "| No intersection with red light zone | 1.0 |\n"
           "| Ego polygon overlaps red light token region | 0.0 |\n\n"
           "**How navsim implements it:**\n"
           "Red lights are stored as polygon tokens in the observation. If `ego_polygon` intersects\n"
           "any token whose name starts with `red_light_token`, the score drops to 0.0 immediately.\n\n"
           "Yellow box = intersection footprint. The traffic light circle (top) shows green/red state.\n"
           "White line = stop line. Ego must not enter the yellow box while the light is red."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Green Light (Score = 1.0)\n\nEgo drives through at 40 km/h; light is green. No penalty.\n\n"
           "> **Real world:** normal intersection crossing with right of way."),
        code(sc1),
        md("---\n## Scenario 2 — Stops at Red (Score = 1.0)\n\nEgo decelerates and stops before the stop line. Light is red, but ego never enters the intersection box.\n\n"
           "> **Real world:** correct behaviour at a red light — score stays 1.0 because the polygon\n"
           "> never crosses into the red zone."),
        code(sc2),
        md("---\n## Scenario 3 — Runs Red Light (Score → 0.0)\n\n"
           "Ego drives at 40 km/h through a red light, entering the intersection box.\n"
           "Score drops to 0.0 the moment any vehicle corner enters the red zone.\n\n"
           "> **Real world:** running a red light at an intersection — zero tolerance. This is a\n"
           "> multiplicative metric so a single red-light violation wipes out the entire PDM score."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Light | Enters intersection? | Score |\n|---|---|---|---|\n"
           "| Green light | Green | Yes (allowed) | 1.0 |\n"
           "| Stops at red | Red | No | 1.0 |\n"
           "| Runs red | Red | Yes | 0.0 |"),
    ])


# ── Notebook 7: Ego Progress ─────────────────────────────────────────────────
def make_progress_nb():
    helpers = """\
# Ego progress: forward displacement along route centerline (x-axis here).
# Normalised: score = min(1.0, progress / max_possible_progress).
# If max_possible_progress < 5.0 m → all proposals get 1.0 (static scene exemption).
# Only positive progress counts (np.clip(..., 0, None)).

PROGRESS_MIN_THRESHOLD = 5.0  # m — minimum to normalise against

def compute_progress_score(ex, max_progress):
    raw = max(0., ex[-1] - ex[0])
    if max_progress < PROGRESS_MIN_THRESHOLD: return 1.0, raw
    return min(1., raw / max_progress), raw

def anim_progress(ex,ey,eh,es, max_progress=None, bttl='', mttl='', interval=100):
    if max_progress is None: max_progress=max(ex[-1]-ex[0],PROGRESS_MIN_THRESHOLD)
    score,raw=compute_progress_score(ex,max_progress)
    prog_arr=np.maximum(ex-ex[0],0.)
    sc_arr=np.minimum(prog_arr/max(max_progress,1e-3),1.) if max_progress>=PROGRESS_MIN_THRESHOLD else np.ones(N)

    fig,(ab,am)=plt.subplots(1,2,figsize=(13,5))
    cx,cy,side=blimits(ab,ex,ey,pad=8); draw_road(ab,cx,cy,side*2.5)
    # Route start/end markers
    ab.axvline(ex[0],color='lime',lw=1.5,ls='--',alpha=.7,zorder=2,label='start')
    ab.axvline(ex[0]+max_progress,color='gold',lw=1.5,ls='--',alpha=.7,zorder=2,label='max progress')
    ab.plot(ex,ey,color='#1565C0',lw=1,alpha=.18,ls='--',zorder=2)
    ab.set_title(bttl,fontsize=8); ab.legend(fontsize=7,loc='upper left')
    ep=mpatch(ab,ex[0],ey[0],eh[0],'#1565C0',zorder=5)
    efd,=ab.plot(*fpt(ex[0],ey[0],eh[0]),'o',color='white',ms=4,zorder=6)
    prog_line,=ab.plot([],[],color='lime',lw=2,alpha=.8,zorder=3)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))

    am.set_xlim(time[0],time[-1]); am.set_ylim(-.05,1.15)
    am.axhline(1.,color='gold',lw=1.5,ls='--',label='max score = 1.0')
    pl,=am.plot([],[],color='#1565C0',lw=2,zorder=4,label='normalised progress score')
    pd,=am.plot([],[],'o',color='#1565C0',ms=7,zorder=5)
    pv=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    slbl=am.text(.97,.96,'',transform=am.transAxes,fontsize=11,va='top',ha='right',fontweight='bold')
    prog_txt=am.text(.5,.06,'',transform=am.transAxes,fontsize=9,ha='center',
                      bbox=dict(boxstyle='round',fc='#E3F2FD',alpha=.8))
    am.set_xlabel('time [s]'); am.set_ylabel('progress score (0–1)')
    am.set_title(mttl,fontsize=8); am.legend(fontsize=8); am.grid(True,alpha=.3)
    plt.tight_layout()
    def update(f):
        ep.set_xy(vcorn(ex[f],ey[f],eh[f])); efd.set_data(*fpt(ex[f],ey[f],eh[f]))
        prog_line.set_data([ex[0],ex[f]],[ey[0],ey[f]])
        tt.set_text(f't = {time[f]:.1f} s')
        pl.set_data(time[:f+1],sc_arr[:f+1]); pd.set_data([time[f]],[sc_arr[f]])
        pv.set_xdata([time[f],time[f]]); s=sc_arr[f]
        slbl.set_text(f'score={s:.2f}')
        slbl.set_color('#1565C0')
        prog_txt.set_text(f'progress={prog_arr[f]:.1f} m / {max_progress:.1f} m')
        return [ep,efd,prog_line,tt,pl,pd,pv,slbl,prog_txt]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Progress helpers loaded.")
"""
    sc1 = """\
# Stationary: no progress → score = 0.0
ex1=np.zeros(N); ey1=np.zeros(N); eh1=np.zeros(N); es1=np.zeros(N)
anim=anim_progress(ex1,ey1,eh1,es1,max_progress=30.,
    bttl='BEV | ego stationary — zero displacement',
    mttl='Progress = 0.0 m → score = 0.0 / 1.0 = 0.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Half progress: ego moves 15 m in 6 s (9 km/h) — max possible is 30 m
es2=np.full(N,15./T/1.); ex2=ei(es2); ey2=np.zeros(N); eh2=np.zeros(N)
anim=anim_progress(ex2,ey2,eh2,es2,max_progress=30.,
    bttl='BEV | ego moves 15 m out of 30 m max',
    mttl='Progress = 15 m / 30 m → score = 0.50')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Full progress: ego moves >= max possible (50 km/h for 6 s = 83 m)
es3=np.full(N,50/3.6); ex3=ei(es3); ey3=np.zeros(N); eh3=np.zeros(N)
# max_progress capped at ego's own displacement → score = 1.0
anim=anim_progress(ex3,ey3,eh3,es3,max_progress=float(ex3[-1]-ex3[0]),
    bttl='BEV | ego at full speed — achieves maximum possible progress',
    mttl='Progress = max_progress → score = 1.0')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Ego Progress\n\n**Source:** `pdm_scorer.py · _calculate_progress()`\n\n"
           "**Role:** weighted metric (weight = 5.0 — tied as highest weight)\n\n"
           "**Formula:**\n```\nprogress_raw = max(0, centerline_proj(end) - centerline_proj(start))\n"
           "score = min(1.0, progress_raw / max_progress_across_all_proposals)\n"
           "```\n"
           "If `max_progress < 5.0 m` (all proposals are nearly stationary) → all scores = 1.0\n\n"
           "**Key insight:** progress is measured along the **route centerline** (projected), not Euclidean.\n"
           "Driving circles adds no progress. Only forward displacement along the planned route counts.\n\n"
           "Green line = progress so far. Gold dashed line = max achievable progress."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Stationary (Score = 0.0)\n\nEgo sits still for 6 s. Progress = 0 m → score = 0.0.\n\n"
           "> **Real world:** ego refuses to move — e.g. overly conservative planner that freezes."),
        code(sc1),
        md("---\n## Scenario 2 — Half Progress (Score = 0.5)\n\n"
           "Ego moves at ~9 km/h for 6 s → 15 m. Max possible is 30 m → score = 0.5.\n\n"
           "> **Real world:** unnecessarily slow driving — the ego is being overly cautious.\n"
           "> Progress is relative to the best proposal, so a slow ego is penalised against a faster one."),
        code(sc2),
        md("---\n## Scenario 3 — Full Progress (Score = 1.0)\n\n"
           "Ego drives at 50 km/h — achieves the maximum possible forward displacement. Score = 1.0.\n\n"
           "> **Real world:** confident driving at the speed limit, covering ground efficiently."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Displacement | Max possible | Score |\n|---|---|---|---|\n"
           "| Stationary | 0 m | 30 m | 0.0 |\n"
           "| Half speed | 15 m | 30 m | 0.5 |\n"
           "| Full speed | 83 m | 83 m | 1.0 |\n\n"
           "> Progress has weight **5.0** — the same as TTC. A stationary ego loses roughly 35% of\n"
           "> the weighted sub-score (5 out of 14 total weight points)."),
    ])


# ── Notebook 8: Two-Frame Extended Comfort ────────────────────────────────────
def make_twoframe_nb():
    helpers = """\
# Two-frame extended comfort: measures consistency between two consecutive trajectory frames.
# For each of 4 features, compute RMS of (frame1 - frame2):
#   acceleration RMS <= 0.7 m/s²
#   jerk RMS        <= 0.5 m/s³
#   yaw_rate RMS    <= 0.1 rad/s
#   yaw_accel RMS   <= 0.1 rad/s²
# ALL must pass for score = 1.0.

ACC_THR  = 0.7   # m/s²
JERK_THR = 0.5   # m/s³
YR_THR   = 0.1   # rad/s
YA_THR   = 0.1   # rad/s²

def sg(arr,w=8,p=2,d=0): return savgol_filter(arr,window_length=min(w,N),polyorder=p,deriv=d,delta=DT)
def rms(arr): return np.sqrt(np.mean(arr**2))

def compute_twoframe(spd1,yr1,spd2,yr2):
    acc1=sg(spd1,w=8,d=1); acc2=sg(spd2,w=8,d=1)
    jrk1=sg(acc1,w=15,d=1); jrk2=sg(acc2,w=15,d=1)
    rms_a=rms(acc1-acc2); rms_j=rms(jrk1-jrk2)
    rms_yr=rms(yr1-yr2)
    rms_ya=rms(sg(yr1,w=15,d=1)-sg(yr2,w=15,d=1))
    passes=[rms_a<=ACC_THR, rms_j<=JERK_THR, rms_yr<=YR_THR, rms_ya<=YA_THR]
    return rms_a,rms_j,rms_yr,rms_ya,all(passes)

def anim_twoframe(
    ex1,ey1,eh1,spd1,yr1,
    ex2,ey2,eh2,spd2,yr2,
    bttl='',mttl='',interval=100
):
    rms_a,rms_j,rms_yr,rms_ya,passes=compute_twoframe(spd1,yr1,spd2,yr2)
    acc1=sg(spd1,w=8,d=1); acc2=sg(spd2,w=8,d=1)
    jrk1=sg(acc1,w=15,d=1); jrk2=sg(acc2,w=15,d=1)

    fig=plt.figure(figsize=(18,5))
    ab=fig.add_subplot(1,3,1)   # BEV
    am=fig.add_subplot(1,3,2)   # feature diff
    ar=fig.add_subplot(1,3,3)   # RMS summary bar chart

    # BEV
    all_x=np.concatenate([ex1,ex2]); all_y=np.concatenate([ey1,ey2])
    cx,cy,side=blimits(ab,all_x,all_y,pad=8); draw_road(ab,cx,cy,side*2.5)
    ab.plot(ex1,ey1,color='#1565C0',lw=1.5,alpha=.3,ls='--',zorder=2,label='frame 1')
    ab.plot(ex2,ey2,color='#CC0000',lw=1.5,alpha=.3,ls='--',zorder=2,label='frame 2')
    ab.legend(fontsize=7,loc='upper left'); ab.set_title(bttl,fontsize=8)
    ep1=mpatch(ab,ex1[0],ey1[0],eh1[0],'#1565C0',zorder=5)
    ep2=mpatch(ab,ex2[0],ey2[0],eh2[0],'#CC0000',alpha=.6,zorder=4)
    tt=ab.text(.03,.96,'t=0.0s',transform=ab.transAxes,fontsize=9,va='top',
               bbox=dict(boxstyle='round,pad=.25',fc='white',alpha=.75))

    # Feature difference panel
    diff_acc=acc1-acc2; diff_yr=yr1-yr2
    am.plot(time,diff_acc,color='#1565C0',lw=1,alpha=.18)
    am.plot(time,diff_yr,color='#880E4F',lw=1,alpha=.18)
    al,=am.plot([],[],color='#1565C0',lw=2,label='Δaccel [m/s²]')
    yl,=am.plot([],[],color='#880E4F',lw=2,ls='--',label='Δyaw_rate [rad/s]')
    av=am.axvline(time[0],color='gray',lw=1,ls=':',alpha=.8)
    am.set_xlim(time[0],time[-1])
    yl_hi=max(np.abs(diff_acc).max(),np.abs(diff_yr).max())*1.3+0.1
    am.set_ylim(-yl_hi,yl_hi); am.axhline(0,color='gray',lw=.5,alpha=.5)
    am.set_xlabel('time [s]'); am.set_ylabel('frame1 − frame2')
    am.set_title('Feature differences (frame 1 − frame 2)',fontsize=8)
    am.legend(fontsize=8); am.grid(True,alpha=.3)

    # RMS bar chart (static)
    vals=[rms_a,rms_j,rms_yr,rms_ya]
    thrs=[ACC_THR,JERK_THR,YR_THR,YA_THR]
    labels=['accel\\n(≤0.7)','jerk\\n(≤0.5)','yaw_rate\\n(≤0.1)','yaw_accel\\n(≤0.1)']
    colors=['green' if v<=t else 'red' for v,t in zip(vals,thrs)]
    bars=ar.bar(labels,vals,color=colors,alpha=.7,zorder=3)
    for t,x in zip(thrs,range(4)): ar.plot([x-.4,x+.4],[t,t],color='k',lw=2,ls='--',zorder=4)
    ar.set_ylabel('RMS difference'); ar.set_title(mttl,fontsize=8); ar.grid(True,alpha=.3,axis='y')
    score_str='PASS ✓' if passes else 'FAIL ✗'
    sc_color='green' if passes else 'red'
    ar.text(.5,.96,score_str,transform=ar.transAxes,fontsize=16,ha='center',va='top',
             fontweight='bold',color=sc_color)
    for bar,v in zip(bars,vals): ar.text(bar.get_x()+bar.get_width()/2,v+.01,f'{v:.3f}',
                                           ha='center',va='bottom',fontsize=8)
    plt.tight_layout()

    def update(f):
        ep1.set_xy(vcorn(ex1[f],ey1[f],eh1[f])); ep2.set_xy(vcorn(ex2[f],ey2[f],eh2[f]))
        tt.set_text(f't = {time[f]:.1f} s')
        al.set_data(time[:f+1],diff_acc[:f+1]); yl.set_data(time[:f+1],diff_yr[:f+1])
        av.set_xdata([time[f],time[f]])
        return [ep1,ep2,tt,al,yl,av]
    anim=FuncAnimation(fig,update,frames=N,interval=interval,blit=False)
    plt.close(fig); return anim

print("Two-frame comfort helpers loaded.")
"""
    sc1 = """\
# Consistent frames: both trajectories nearly identical (same speed profile)
spd1=np.full(N,30/3.6); yr1=np.full(N,0.2)  # gentle constant turn
ex1,ey1=ixy(spd1,ei(yr1,v0=0.))
eh1=ei(yr1,v0=0.)

spd2=np.full(N,30/3.6)+np.random.default_rng(42).normal(0,.05,N)  # tiny noise
yr2=np.full(N,0.2)+np.random.default_rng(42).normal(0,.005,N)
ex2,ey2=ixy(spd2,ei(yr2,v0=0.)); eh2=ei(yr2,v0=0.)

anim=anim_twoframe(ex1,ey1,eh1,spd1,yr1, ex2,ey2,eh2,spd2,yr2,
    bttl='BEV | frame 1 (blue) ≈ frame 2 (red) — nearly identical trajectories',
    mttl='All 4 RMS values below threshold → score = 1.0')
HTML(anim.to_jshtml())
"""
    sc2 = """\
# Acceleration inconsistency: frame1 brakes, frame2 accelerates → large Δaccel RMS
spd1b=np.maximum(40/3.6-np.arange(N)*DT*3.,5/3.6)  # decelerating
yr1b=np.zeros(N)
spd2b=np.minimum(5/3.6+np.arange(N)*DT*3.,40/3.6)  # accelerating
yr2b=np.zeros(N)
ex1b,ey1b=ixy(spd1b,np.zeros(N)); eh1b=np.zeros(N)
ex2b,ey2b=ixy(spd2b,np.zeros(N)); eh2b=np.zeros(N)

anim=anim_twoframe(ex1b,ey1b,eh1b,spd1b,yr1b, ex2b,ey2b,eh2b,spd2b,yr2b,
    bttl='BEV | frame 1 decelerates (blue), frame 2 accelerates (red)',
    mttl='Δaccel RMS >> 0.7 m/s² → FAIL — inconsistent longitudinal intent')
HTML(anim.to_jshtml())
"""
    sc3 = """\
# Yaw rate inconsistency: frame1 goes straight, frame2 turns hard
yr1c=np.zeros(N)
yr2c=np.where(time<3.,0.5,0.)  # frame2 turns then straightens
spd1c=np.full(N,20/3.6); spd2c=np.full(N,20/3.6)
ex1c,ey1c=ixy(spd1c,ei(yr1c)); eh1c=ei(yr1c)
ex2c,ey2c=ixy(spd2c,ei(yr2c)); eh2c=ei(yr2c)

anim=anim_twoframe(ex1c,ey1c,eh1c,spd1c,yr1c, ex2c,ey2c,eh2c,spd2c,yr2c,
    bttl='BEV | frame 1 straight (blue), frame 2 turns (red)',
    mttl='Δyaw_rate RMS >> 0.1 rad/s → FAIL — inconsistent lateral intent')
HTML(anim.to_jshtml())
"""
    return nb([
        md("# Two-Frame Extended Comfort\n\n**Source:** `pdm_comfort_metrics.py · ego_is_two_frame_extended_comfort()`\n\n"
           "**Role:** evaluated by the `SceneAggregator` between consecutive planning frames (not inside the per-scene PDM score).\n\n"
           "| Feature | Threshold (RMS of difference) |\n|---|---|\n"
           "| Acceleration magnitude | ≤ 0.7 m/s² |\n"
           "| Jerk magnitude | ≤ 0.5 m/s³ |\n"
           "| Yaw rate | ≤ 0.1 rad/s |\n"
           "| Yaw acceleration | ≤ 0.1 rad/s² |\n\n"
           "**What it checks:** when the planner replans at t+1, does the new trajectory agree with\n"
           "the old one at the **same time points**? Large RMS differences indicate the planner is\n"
           "inconsistent — oscillating, changing intent, or unstable.\n\n"
           "**BEV interpretation:** blue = frame 1 trajectory, red = frame 2 trajectory.\n"
           "If both curves overlap, the planner is consistent. Diverging paths = inconsistency.\n\n"
           "**Bar chart:** green bar = passes threshold, red bar = fails. All 4 must be green for score = 1.0."),
        code(IMPORTS),
        code(helpers),
        md("---\n## Scenario 1 — Consistent Replanning (Score = 1.0)\n\n"
           "Both frames compute nearly identical trajectories (tiny sensor noise only).\n"
           "All 4 RMS differences are tiny — well within thresholds.\n\n"
           "> **Real world:** stable, predictable planner that doesn't change its mind."),
        code(sc1),
        md("---\n## Scenario 2 — Acceleration Inconsistency (Score = 0.0)\n\n"
           "Frame 1 decelerates (braking), frame 2 accelerates (full throttle). The two frames\n"
           "have completely opposite longitudinal intent — RMS of Δaccel >> 0.7 m/s².\n\n"
           "> **Real world:** planner sees an obstacle in frame 1, then ignores it in frame 2.\n"
           "> This oscillation would cause uncomfortable jerk for occupants and is flagged as inconsistent."),
        code(sc2),
        md("---\n## Scenario 3 — Yaw Rate Inconsistency (Score = 0.0)\n\n"
           "Frame 1 drives straight; frame 2 turns hard left for 3 s then straightens.\n"
           "Δyaw_rate RMS >> 0.1 rad/s — the planner is changing lateral intent between frames.\n\n"
           "> **Real world:** unstable lane-change decision — planner initiates a turn then\n"
           "> changes its mind. Passengers feel a sudden steering wheel movement followed by correction."),
        code(sc3),
        md("---\n## Summary\n\n"
           "| Scenario | Δaccel RMS | Δyaw_rate RMS | Score |\n|---|---|---|---|\n"
           "| Consistent replanning | ≈ 0 | ≈ 0 | 1.0 |\n"
           "| Accel inconsistency | >> 0.7 | ≈ 0 | 0.0 |\n"
           "| Yaw rate inconsistency | ≈ 0 | >> 0.1 | 0.0 |\n\n"
           "> Unlike the other metrics, two-frame comfort evaluates **planner consistency over time**,\n"
           "> not single-trajectory quality. It catches planners that pass all per-frame metrics but\n"
           "> oscillate between frames — a behaviour invisible to single-frame evaluation."),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating metric notebooks...")
    save(make_collision_nb(),       "metric_collision.ipynb")
    save(make_ttc_nb(),             "metric_ttc.ipynb")
    save(make_lane_keeping_nb(),    "metric_lane_keeping.ipynb")
    save(make_drivable_area_nb(),   "metric_drivable_area.ipynb")
    save(make_driving_dir_nb(),     "metric_driving_direction.ipynb")
    save(make_traffic_light_nb(),   "metric_traffic_light.ipynb")
    save(make_progress_nb(),        "metric_ego_progress.ipynb")
    save(make_twoframe_nb(),        "metric_two_frame_comfort.ipynb")
    print("Done — 8 notebooks written to docs/")
