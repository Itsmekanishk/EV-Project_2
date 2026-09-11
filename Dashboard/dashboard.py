"""
Real-Time IDS Dashboard — EV Charging Network (DoS / DDoS detection)

Runs on Streamlit. Live charts refresh every second via a fragment; the engine
keeps capturing in the background, so the page never flickers or restarts.

    streamlit run dashboard.py

The sidebar chooses a traffic source:
    * Scenario demo — scripted Normal -> DoS -> DDoS loop (no root needed)
    * Live capture   — scapy sniff() on an interface (root needed on macOS)
    * Replay         — replay an offline pcap/pcapng file
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from ids_backend import get_engine, interface_choices

# ---------------------------------------------------------------------------
# constants / colours
# ---------------------------------------------------------------------------

C = {
    "Normal": "#16a34a",
    "DoS": "#d97706",
    "DDoS": "#dc2626",
    "Suspicious": "#7c3aed",
    "Malformed": "#64748b",
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "paper": "#ffffff",
}

HAS_FRAGMENT = hasattr(st, "fragment")
DEFAULT_SPEED = 1.0
DEFAULT_PCAP = str(Path(__file__).resolve().parents[1] / "traffic_capture.pcapng")

st.set_page_config(
    page_title="EV-Charging IDS — Real-Time Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# engine bootstrap (starts the scenario once per process)
# ---------------------------------------------------------------------------

ENGINE = get_engine()

if "ids_started" not in st.session_state:
    st.session_state.ids_started = True
    ENGINE.start("scenario", speed=DEFAULT_SPEED)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def fmt_bps(bps: float) -> str:
    if bps >= 1_000_000:
        return f"{bps / 1_048_576:.2f} MB/s"
    if bps >= 1_000:
        return f"{bps / 1024:.1f} KB/s"
    return f"{bps:.0f} B/s"


def status_segments(t_list, s_list):
    """Compress a per-second status series into (start, end, status)."""
    if not t_list:
        return []
    segs = []
    cur = s_list[0]
    start = t_list[0]
    for tt, ss in zip(t_list[1:], s_list[1:]):
        if ss != cur:
            segs.append((start, tt, cur))
            cur = ss
            start = tt
    segs.append((start, t_list[-1], cur))
    return segs


def base_layout(fig, height=280, ytitle=None):
    fig.update_layout(
        height=height,
        paper_bgcolor=C["paper"],
        plot_bgcolor=C["paper"],
        font=dict(family="-apple-system, Segoe UI, sans-serif",
                  color=C["ink"], size=12),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(gridcolor=C["grid"], zeroline=False,
                   title=dict(text="session (s)", font=dict(size=11))),
        yaxis=dict(gridcolor=C["grid"], zeroline=False,
                   title=dict(text=ytitle, font=dict(size=11))),
    )
    return fig


def pill(text, color, big=False):
    size = "20px" if big else "13px"
    pad = "10px 18px" if big else "3px 10px"
    return (f'<span style="background:{color};color:#fff;'
            f'font-weight:700;font-size:{size};padding:{pad};'
            f'border-radius:8px;white-space:nowrap;">{text}</span>')


def chip(text, color):
    return (f'<span style="border:1.5px solid {color};color:{color};'
            f'font-weight:600;font-size:12px;padding:2px 10px;'
            f'border-radius:999px;white-space:nowrap;">{text}</span>')


def sub(title, color=C["ink"]):
    st.markdown(
        f'<div style="font-weight:700;color:{color};font-size:15px;'
        f'margin:6px 0 2px 0;">{title}</div>',
        unsafe_allow_html=True,
    )


def plot_fig(fig):
    """plotly_chart that works across streamlit versions."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_traffic(snap):
    h = snap["history"]
    t, pps, sts = h["t"], h["pps"], h["regime"]
    import plotly.graph_objects as go

    fig = go.Figure()
    ymax = max(pps) if pps else 1
    for a, b, s in status_segments(t, sts):
        if s == "Normal":
            continue
        fig.add_shape(type="rect", x0=a, x1=b, y0=0, y1=ymax,
                      fillcolor=C[s], opacity=0.10, line_width=0, layer="below")
    if pps:
        fig.add_trace(go.Scatter(
            x=t, y=pps, mode="lines", name="packets/s",
            line=dict(color="#0284c7", width=2),
            fill="tozeroy", fillcolor="rgba(2,132,199,0.10)",
        ))
    base_layout(fig, height=300, ytitle="packets/s")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(showlegend=False)
    return fig


def fig_ips_bps(snap):
    h = snap["history"]
    import plotly.graph_objects as go
    fig = go.Figure()
    if h["t"]:
        fig.add_trace(go.Scatter(
            x=h["t"], y=h["ips"], mode="lines",
            name="unique IPs (1s)",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.12)",
        ))
    base_layout(fig, height=220, ytitle="unique IPs")
    return fig


def fig_ml(snap):
    h = snap["history"]
    import plotly.graph_objects as go
    fig = go.Figure()
    if h["t"]:
        colors = [C.get(lbl, C["Malformed"]) for lbl in h["ml"]]
        fig.add_trace(go.Scatter(
            x=h["t"], y=h["conf"], mode="lines+markers",
            name="confidence",
            line=dict(color="#94a3b8", width=1.5),
            marker=dict(size=6, color=colors, line=dict(width=1, color="#fff")),
        ))
        fig.add_hline(y=0.5, line_dash="dot", line_color=C["DoS"], opacity=0.6)
    base_layout(fig, height=220, ytitle="confidence")
    return fig


def fig_timeline(snap):
    h = snap["history"]
    import plotly.graph_objects as go
    fig = go.Figure()
    segs = status_segments(h["t"], h["regime"])
    if segs:
        for a, b, s in segs:
            fig.add_trace(go.Bar(
                x=[max(b - a, 0.05)], y=["traffic"],
                base=[a], width=0.55, orientation="h",
                marker=dict(color=C.get(s, "#94a3b8")),
                hovertemplate=f"{s} · %{{x[0]:.0f}}–%{{x[1]:.0f}} s<extra></extra>",
            ))
    base_layout(fig, height=120)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="", showgrid=True, zeroline=False),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
        margin=dict(l=10, r=10, t=10, b=6),
    )
    fig.update_xaxes(showticklabels=False)
    return fig


def fig_donut(snap):
    import plotly.graph_objects as go
    labels = ["Normal", "DoS", "DDoS", "Malformed"]
    vals = snap["ml_total"]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.62,
        marker=dict(colors=[C["Normal"], C["DoS"], C["DDoS"], C["Malformed"]],
                    line=dict(color="#fff", width=2)),
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        height=240, paper_bgcolor=C["paper"], showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text="LSTM<br>votes", showarrow=False,
                          font=dict(size=13, color=C["ink"]))],
    )
    return fig


def fig_sources(snap):
    import plotly.graph_objects as go
    fig = go.Figure()
    tops = snap["top_sources"][:8]
    if tops:
        names = [f"{ip}\n{n} pkts" for ip, n in tops]
        fig.add_trace(go.Bar(
            x=[n for _, n in tops],
            y=[ip for ip, _ in tops],
            orientation="h",
            marker=dict(color="#0ea5e9"),
        ))
    base_layout(fig, height=240, ytitle="")
    fig.update_yaxes(autorange="reversed")
    return fig


def render_alert(alert):
    colors = {"critical": "#dc2626", "warning": "#d97706", "info": "#16a34a"}
    bar = colors.get(alert["level"], "#64748b")
    st.markdown(
        f'<div style="border-left:4px solid {bar};background:#f8fafc;'
        f'padding:7px 10px;margin:4px 0;border-radius:4px;'
        f'font-size:13px;">'
        f'<span style="color:{bar};font-weight:700;">T+{alert["t"]:.0f}s</span>'
        f' &nbsp;{alert["msg"]}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# live body (runs as an auto-refreshing fragment)
# ---------------------------------------------------------------------------

def render_live():
    snap = ENGINE.snapshot()
    status = snap["status"]
    colour = C.get(status, C["Normal"])

    # ----- header banner --------------------------------------------------
    ml_extra = ""
    if snap["ml_label"] != "Normal":
        chip_html = chip("LSTM: " + snap["ml_label"] + " · " +
                         format(snap["ml_conf"], ".0%"),
                         C.get(snap["ml_label"], C["Malformed"]))
        ml_extra = " &nbsp; " + chip_html
    paused = " · ⏸ paused" if snap["paused"] else ""
    banner = (
        '<div style="background:linear-gradient(90deg,#0f172a,#1e3a8a);'
        'border-radius:14px;padding:18px 22px;margin-bottom:8px;'
        'display:flex;align-items:center;justify-content:space-between;">'
        '<div><span style="color:#fff;font-size:22px;font-weight:800;">'
        '🚗 EV-Charging Intrusion Detection</span><br>'
        '<span style="color:#cbd5e1;font-size:13px;">'
        'Hybrid IDS — LSTM + rules · {} · {}{}</span></div>'
        '<div style="text-align:right;">{}{}</div></div>'
    ).format(snap["info"]["source"], snap["info"]["ml"], paused,
             pill(status, colour, big=True), ml_extra)
    st.markdown(banner, unsafe_allow_html=True)

    if snap["info"].get("error"):
        st.error(snap["info"]["error"])

    # ----- metric cards ---------------------------------------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📈 Packet rate", f"{snap['pps']:.0f} pkt/s")
    m2.metric("🪪 Unique IPs", f"{snap['ips']}")
    m3.metric("💾 Throughput", fmt_bps(snap["bps"]))
    m4.metric("📦 Total packets", f"{snap['total_packets']:,}")
    m5.metric("🧠 Model verdict", snap["ml_label"])
    m6.metric("🎯 ML↔status agreement", f"{snap['agreement']}%")

    # ----- traffic + alerts ----------------------------------------------
    left, right = st.columns([3, 2])
    with left:
        sub("📊 Packet rate  <span style='font-weight:400;color:#64748b;"
            "font-size:12px;'>(coloured bands = detected attack episodes)"
            "</span>")
        if snap["history"]["t"]:
            plot_fig(fig_traffic(snap))
        else:
            st.info("Waiting for traffic…")
    with right:
        sub("🚨 Security alerts")
        alerts = snap["alerts"]
        if alerts:
            for alert in alerts[-14:][::-1]:
                render_alert(alert)
        else:
            st.info("No alerts yet — keep watching.")

    # ----- middle charts --------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        sub("🪪 Unique source IPs (1 s window)")
        if snap["history"]["t"]:
            plot_fig(fig_ips_bps(snap))
    with c2:
        sub("🧠 LSTM confidence over time  "
            "<span style='font-weight:400;color:#64748b;font-size:12px;'>"
            "(coloured by predicted class)</span>")
        if snap["history"]["t"]:
            plot_fig(fig_ml(snap))

    # ----- bottom row -----------------------------------------------------
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        sub("🕐 Attack timeline (session)")
        if snap["history"]["t"]:
            plot_fig(fig_timeline(snap))
    with b2:
        sub("🧮 LSTM class votes")
        if sum(snap["ml_total"]) > 0:
            plot_fig(fig_donut(snap))
    with b3:
        sub("🌐 Top source IPs")
        if snap["top_sources"]:
            plot_fig(fig_sources(snap))


if HAS_FRAGMENT:
    @st.fragment(run_every="1s")
    def _fragment_body():
        render_live()

    _fragment_body()
else:  # pragma: no cover - old streamlit fallback
    st.warning("Streamlit ≥ 1.37 is recommended for automatic live refresh.")

# ---------------------------------------------------------------------------
# sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎛️ Controls")
    source = st.radio(
        "Traffic source",
        ["Scenario demo", "Live capture", "Replay pcap"],
        index=0,
        help="Scenario = built-in scripted attacks (works everywhere). "
             "Live = sniff a real interface (sudo needed on macOS).",
    )
    speed = st.number_input("Demo speed (×)", min_value=0.5, max_value=10.0,
                            value=DEFAULT_SPEED, step=0.5)
    thr = st.slider("Attack threshold (pkt/s)", 5, 300,
                    int(ENGINE.threshold_pps), step=5,
                    help="Rule engine raises an alarm above this rate.")
    ENGINE.threshold_pps = float(thr)

    iface = None
    target = None
    replay_path = None
    if source == "Live capture":
        targets = interface_choices()
        if targets:
            names = ["auto"] + [f"{i}  ({ip})" for i, ip in targets]
            sel = st.selectbox("Interface", names)
            iface = None if sel == "auto" else targets[int(names.index(sel)) - 1][0]
        else:
            st.text_input("Interface (auto)", "", key="iface_manual")
            iface = st.session_state.get("iface_manual") or None
        target = st.text_input("Capture filter host IP", "127.0.0.1",
                               help="Set to your web-server IP (e.g. 127.0.0.1).")
        st.caption("⚠️ On macOS run Streamlit with `sudo` for live capture.")
    elif source == "Replay pcap":
        replay_path = st.text_input("pcap/pcapng path", DEFAULT_PCAP)

    if st.button("▶️ Apply source / restart"):
        params = {"speed": float(speed)}
        if source == "Scenario demo":
            ENGINE.start("scenario", **params)
        elif source == "Live capture":
            ENGINE.start("live", iface=iface, target=target)
        else:
            ENGINE.start("replay", path=replay_path, speed=float(speed))
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⏸️ Pause" if not ENGINE.paused else "▶️ Resume"):
            ENGINE.pause(not ENGINE.paused)
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear history"):
            ENGINE.reset()
            st.rerun()
    if st.button("⏹️ Stop engine"):
        ENGINE.stop()
        st.rerun()

    st.divider()
    st.markdown(
        "**Scenario cycle (~50 s)**\n"
        "Normal 8s → **DoS** 12s → Normal 8s → **DDoS** 12s → Normal 8s.\n"
        "Alerts appear in the feed and coloured bands mark the traffic chart.\n"
        "Increase **Demo speed** for a quicker cycle, then Apply."
    )

# ---------------------------------------------------------------------------
# classic auto-refresh fallback for streamlit < 1.37
# ---------------------------------------------------------------------------

if not HAS_FRAGMENT:  # pragma: no cover
    _placeholder = st.empty()
    while True:
        with _placeholder.container():
            render_live()
        time.sleep(1.0)
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:  # very old streamlit: single static pass
            break
