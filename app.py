"""
Glideluke – Beregningsverktøy for tappe- og stengeorgan
Faglig grunnlag: NVE Retningslinjer 1/2011
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ─────────────────────────────────────────────────────────
#  SIDEOPPSETT
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glideluke – NVE 1/2011",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  KONSTANTER  (NVE Retningslinjer 1/2011)
# ─────────────────────────────────────────────────────────
GAMMA_W = 9.81          # kN/m³

# Lastfaktorer – Tabell 2.1 og §2.5
LF = {
    "brudd": {"standard": 1.2, "tappeluke": 1.4},
    "ulykke": {"standard": 1.0, "tappeluke": 1.0},
    "bruks":  {"standard": 1.0, "tappeluke": 1.0},
}

# Materialfaktorer – Tabell 4.1
MF = {
    "brudd":  {"med": 1.25, "uten": 1.60},
    "ulykke": {"med": 1.10, "uten": 1.30},
    "bruks":  {"med": 1.00, "uten": 1.00},
}

# Friksjonsfaktorer – Tabell 5.1 (minimumsverdi)
MU = {"bronse": 0.60, "polymer": 0.40, "spesial": 0.15}

# Paslagsfaktorer §5.2
PSI = {"tappe": 1.0, "flom": 1.2, "inntak": 1.2}

# ─────────────────────────────────────────────────────────
#  BEREGNINGSFUNKSJONER
# ─────────────────────────────────────────────────────────
def beregn(B, H_L, z_UK, h1, h2, Re, G, funksjon, glideliste, grensetilstand):
    mu  = MU[glideliste]
    psi = PSI[funksjon]
    gamma_m = MF[grensetilstand]["med"]

    # Trykkfordeling
    p_UK1 = max(h1, 0) * GAMMA_W
    p_OK1 = max(h1 - H_L, 0) * GAMMA_W
    p_UK2 = max(h2, 0) * GAMMA_W
    p_OK2 = max(h2 - H_L, 0) * GAMMA_W
    p_net_UK = p_UK1 - p_UK2
    p_net_OK = p_OK1 - p_OK2

    # Resultantkraft (trapesfordeling)
    F_hyd = ((p_net_UK + p_net_OK) / 2) * B * H_L

    # Angrepspunkt over underkant (korrekt trapesformel)
    if abs(p_net_UK + p_net_OK) < 1e-9:
        e_kp = H_L / 2
    else:
        e_kp = H_L * (2 * p_net_OK + p_net_UK) / (3 * (p_net_UK + p_net_OK))

    e_abs  = z_UK + e_kp
    M_hyd  = F_hyd * e_kp
    p_avg  = F_hyd / (B * H_L) if B * H_L > 0 else 0

    # Lastfaktor
    if grensetilstand == "brudd":
        gamma_f = LF["brudd"]["tappeluke"] if funksjon in ("tappe", "flom") else LF["brudd"]["standard"]
    else:
        gamma_f = 1.0

    F_dim = gamma_f * F_hyd
    M_dim = gamma_f * M_hyd

    # Manøvreringskrefter
    F_frict    = mu * F_hyd
    F_lift     = G + F_frict
    F_lower    = G - F_frict
    F_actuator = psi * F_lift

    # Selvlukkende?
    self_closing = (F_lower > 0) and (F_lower > 0.25 * F_frict)

    # Tetningskraft
    tettelengde = 2 * (B + H_L)
    F_seal_req  = 5 * tettelengde

    # Materialkapasitet
    f_yd = (Re * 0.8) / gamma_m

    return dict(
        p_UK1=p_UK1, p_OK1=p_OK1, p_UK2=p_UK2, p_OK2=p_OK2,
        p_net_UK=p_net_UK, p_net_OK=p_net_OK, p_avg=p_avg,
        F_hyd=F_hyd, e_kp=e_kp, e_abs=e_abs, M_hyd=M_hyd,
        gamma_f=gamma_f, F_dim=F_dim, M_dim=M_dim,
        mu=mu, F_frict=F_frict, F_lift=F_lift, F_lower=F_lower,
        psi=psi, F_actuator=F_actuator, self_closing=self_closing,
        tettelengde=tettelengde, F_seal_req=F_seal_req,
        gamma_m=gamma_m, f_yd=f_yd,
        area=B * H_L,
    )


def valider(B, H_L, z_UK, h1, h2, Re, G):
    errors, warnings = [], []
    if B <= 0:     errors.append("Lukebredde B må være positiv.")
    if H_L <= 0:   errors.append("Lukehøyde H_L må være positiv.")
    if z_UK < 0:   errors.append("Underkant luke z_UK kan ikke være negativ.")
    if h1 < 0:     errors.append("Oppstrøms vannstand h₁ kan ikke være negativ.")
    if h2 < 0:     errors.append("Nedstrøms vannstand h₂ kan ikke være negativ.")
    if h2 > h1:    errors.append("Nedstrøms h₂ kan ikke overstige oppstrøms h₁.")
    if Re < 100:   errors.append("Flytegrense Re må være ≥ 100 MPa.")
    if G < 0:      errors.append("Egenvekt G kan ikke være negativ.")
    if Re > 700:   warnings.append("Flytegrense Re > 700 MPa – kontroller materialvalg.")
    if h1 < H_L:   warnings.append("h₁ < H_L: luken er delvis eksponert (delvis trykk).")
    if h1 > 40:    warnings.append("Trykkhøyde > 40 m: spesiell utforming nødvendig (NVE C.4). Kavitasjon og vibrasjon må vurderes.")
    elif h1 > 20:  warnings.append("Trykkhøyde > 20 m: lufting nedstrøms tappeluke må vurderes (NVE C.1).")
    if h2 > 0 and (h1 - h2) < 10:
        warnings.append(f"Trykkdifferanse Δh = {h1-h2:.2f} m < 10 m minstekrav (NVE §2.2).")
    if B > H_L:    warnings.append("B > H_L: styreruller eller styreskinner bør vurderes (NVE C.4).")
    return errors, warnings


# ─────────────────────────────────────────────────────────
#  FIGUR
# ─────────────────────────────────────────────────────────
def lag_figur(B, H_L, z_UK, h1, h2, r, dpi=130):
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # --- Koordinatsystem ---
    # Alt i meter, y=0 er underkant luke
    gate_cx = 0.0
    gw = B * 0.12   # lukens visuelle bredde i plotkoordinater (skalert)
    gw = max(min(gw, 0.5), 0.18)

    maxH = max(h1 * 1.15, H_L * 1.4, 1.0)

    # Hjelper: tegn pil
    def pil(x1, y1, x2, y2, color, lw=2, hw=0.04, hl=0.08):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=f"->,head_width={hw},head_length={hl}",
                                    color=color, lw=lw))

    def dobbelpil(x1, y1, x2, y2, color="#475569", lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=lw))

    # --- Bakgrunn / vann ---
    # Oppstrøms
    ax.fill_betweenx([0, h1],
                     [-2.5, -2.5], [gate_cx - gw/2, gate_cx - gw/2],
                     color="#2563a8", alpha=0.10)
    # Nedstrøms
    if h2 > 0.01:
        ax.fill_betweenx([0, h2],
                         [gate_cx + gw/2]*2, [2.5]*2,
                         color="#2563a8", alpha=0.07)

    # Kanal - bunn
    ax.fill_between([-2.5, 2.8], [-0.12, -0.12], [0, 0], color="#94a3b8", zorder=3)
    # Vegger
    for xw, side in [(-2.5, "left"), (2.5, "right")]:
        ax.plot([xw, xw], [0, maxH + 0.3], color="#334155", lw=2.5, zorder=4)
        hatch_step = maxH / 10
        for hy in np.arange(0, maxH + 0.3, hatch_step):
            dx = 0.12 if side == "left" else -0.12
            ax.plot([xw, xw - dx], [hy, hy + hatch_step * 0.6],
                    color="#94a3b8", lw=0.8)

    # --- Trykkdiagram (trapesfordeling) ---
    pmax = max(r["p_net_UK"], 0.001)
    pscale = 1.5 / pmax      # 1.5 m plotbredde = maks trykk
    pw_UK = r["p_net_UK"] * pscale
    pw_OK = r["p_net_OK"] * pscale
    px_gate = gate_cx - gw/2

    if r["F_hyd"] > 0.01:
        # Fylt trapesdiagram
        xs = [px_gate, px_gate - pw_UK, px_gate - pw_OK, px_gate]
        ys = [0,        0,               H_L,              H_L]
        ax.fill(xs, ys, color="#2563a8", alpha=0.15, zorder=2)
        ax.plot([px_gate - pw_UK, px_gate - pw_OK],
                [0, H_L], color="#1d4ed8", lw=2)
        ax.plot([px_gate, px_gate - pw_UK], [0, 0],        color="#1d4ed8", lw=1.5)
        ax.plot([px_gate, px_gate - pw_OK], [H_L, H_L],    color="#1d4ed8", lw=1.5)

        # Trykkpiler
        n_pil = max(4, min(8, int((H_L / maxH) * 10)))
        for i in range(n_pil + 1):
            frac = i / n_pil
            yp = frac * H_L
            p_here = r["p_net_OK"] + (r["p_net_UK"] - r["p_net_OK"]) * (1 - frac)
            pw_here = p_here * pscale
            pil(px_gate - pw_here, yp, px_gate - 0.02, yp,
                "#2563a8", lw=1.2, hw=0.025, hl=0.05)

        # Trykkverdi-etiketter
        ax.text(px_gate - pw_UK - 0.06, 0.0,
                f"{r['p_net_UK']:.1f} kN/m²",
                ha="right", va="bottom", fontsize=8.5,
                color="#1d4ed8", fontweight="bold")
        if r["p_net_OK"] > 0.1:
            ax.text(px_gate - pw_OK - 0.06, H_L,
                    f"{r['p_net_OK']:.1f} kN/m²",
                    ha="right", va="top", fontsize=8.5,
                    color="#1d4ed8", fontweight="bold")

        # Resultantpil (F_hyd)
        y_res = r["e_kp"]
        pil(px_gate - pw_UK - 0.3, y_res, px_gate, y_res,
            "#e07b00", lw=3, hw=0.06, hl=0.10)

        # Stiplet angrepspunkt-linje
        ax.plot([px_gate - pw_UK - 0.3, gate_cx + gw/2 + 1.2],
                [y_res, y_res], color="#e07b00", lw=1, ls="--", alpha=0.7)

        # Angrepspunkt-prikk
        ax.plot(px_gate, y_res, "o", color="#e07b00", ms=7, zorder=10)

        # F_hyd boks
        bx = gate_cx + gw/2 + 0.12
        by = y_res
        ax.annotate(
            f"F_hyd = {r['F_hyd']:.2f} kN\ne = {r['e_kp']:.3f} m over UK",
            xy=(px_gate, y_res), xytext=(bx, by),
            fontsize=9, color="#c2410c", fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed", ec="#e07b00", lw=1.5),
            arrowprops=dict(arrowstyle="-", color="#e07b00", lw=1),
        )

        # e-målsetting (venstre side av luke)
        ex = px_gate - 0.12
        dobbelpil(ex, 0, ex, y_res, "#e07b00")
        ax.text(ex - 0.06, y_res / 2,
                f"e={r['e_kp']:.3f}m",
                ha="right", va="center", fontsize=7.5, color="#e07b00", rotation=90)

    # --- Lukelegeme ---
    rect = plt.Rectangle((gate_cx - gw/2, 0), gw, H_L,
                          fc="#475569", ec="#0f172a", lw=2, zorder=5)
    ax.add_patch(rect)
    # Hatch
    for yh in np.arange(H_L * 0.1, H_L, H_L * 0.15):
        ax.plot([gate_cx - gw/2 + 0.02, gate_cx + gw/2 - 0.02],
                [yh, yh], color="white", lw=0.6, alpha=0.3, zorder=6)
    ax.text(gate_cx, H_L / 2, "GLIDE-\nLUKE",
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", zorder=7)

    # Glidelister
    for xg in [gate_cx - gw/2 - 0.05, gate_cx + gw/2]:
        gr = plt.Rectangle((xg, H_L * 0.1), 0.05, H_L * 0.8,
                            fc="#94a3b8", ec="#475569", lw=0.8, zorder=5)
        ax.add_patch(gr)

    # --- Vannoverflater ---
    ax.plot([-2.5, gate_cx - gw/2], [h1, h1],
            color="#1d4ed8", lw=2, ls=(0, (8, 4)), zorder=6)
    ax.text(-2.4, h1 + maxH * 0.02,
            f"h₁ = {h1:.2f} m  (HRV)",
            color="#1d4ed8", fontsize=9, fontweight="bold")

    if h2 > 0.01:
        ax.plot([gate_cx + gw/2, 2.5], [h2, h2],
                color="#0369a1", lw=2, ls=(0, (6, 4)), zorder=6)
        ax.text(gate_cx + gw/2 + 0.1, h2 + maxH * 0.02,
                f"h₂ = {h2:.2f} m",
                color="#0369a1", fontsize=9, fontweight="bold")
    else:
        ax.text(gate_cx + gw/2 + 0.12, H_L / 2,
                "Tørt\nnedstrøms", ha="left", va="center",
                fontsize=8, color="#94a3b8")

    # --- Egenvekt G ---
    if r.get("F_lift", 0) > 0:
        gy_top = H_L + 0.55
        pil(gate_cx, gy_top, gate_cx, H_L + 0.02,
            "#15803d", lw=2.5, hw=0.05, hl=0.08)
        ax.annotate(f"G = {r.get('G_val', 0):.1f} kN",
                    xy=(gate_cx, gy_top), xytext=(gate_cx + 0.1, gy_top + 0.1),
                    fontsize=8.5, color="#15803d", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#15803d", lw=1.2))

    # --- F_lift (lilla, opp – venstre) ---
    lx = gate_cx - gw/2 - 0.35
    pil(lx, H_L * 0.1, lx, H_L + 0.45,
        "#7c3aed", lw=2.5, hw=0.05, hl=0.08)
    ax.annotate(f"F_lift\n{r['F_lift']:.1f} kN",
                xy=(lx, H_L + 0.45), xytext=(lx - 0.5, H_L + 0.35),
                fontsize=8.5, color="#5b21b6", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f3ff", ec="#7c3aed", lw=1.2),
                arrowprops=dict(arrowstyle="-", color="#7c3aed", lw=0.8))

    # --- F_frict (rød, ned – høyre) ---
    fr_y_top = max(r["e_kp"] + 0.3, H_L * 0.6)
    fr_y_bot = max(fr_y_top - 0.4, 0.05)
    fx = gate_cx + gw/2 + 0.35
    pil(fx, fr_y_top, fx, fr_y_bot,
        "#be185d", lw=2.5, hw=0.05, hl=0.08)
    ax.annotate(f"F_frict\n{r['F_frict']:.1f} kN",
                xy=(fx, fr_y_bot), xytext=(fx + 0.5, fr_y_bot + 0.1),
                fontsize=8.5, color="#9d174d", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f8", ec="#be185d", lw=1.2),
                arrowprops=dict(arrowstyle="-", color="#be185d", lw=0.8))

    # --- Målsettinger ---
    # H_L (høyre)
    dim_x = gate_cx + gw/2 + 0.75
    dobbelpil(dim_x, 0, dim_x, H_L)
    ax.text(dim_x + 0.06, H_L / 2,
            f"H_L = {H_L:.2f} m",
            ha="left", va="center", fontsize=8, color="#334155", fontweight="bold")

    # B (under)
    dobbelpil(gate_cx - gw/2, -0.22, gate_cx + gw/2, -0.22)
    ax.text(gate_cx, -0.30,
            f"B = {B:.2f} m",
            ha="center", va="top", fontsize=8, color="#334155", fontweight="bold")

    # h1 (venstre)
    dobbelpil(-2.35, 0, -2.35, h1, "#1d4ed8")
    ax.text(-2.42, h1 / 2,
            f"{h1:.2f} m", ha="right", va="center",
            fontsize=7.5, color="#1d4ed8", rotation=90)

    # --- Dybdeskala (akse) ---
    ax_x = -2.55
    ax.plot([ax_x, ax_x], [0, maxH], color="#cbd5e1", lw=1)
    ts = 10 if maxH > 30 else 5 if maxH > 15 else 2 if maxH > 6 else 1
    for hh in np.arange(0, maxH + 0.01, ts):
        ax.plot([ax_x - 0.05, ax_x], [hh, hh], color="#94a3b8", lw=0.8)
        ax.text(ax_x - 0.08, hh, f"{hh:.0f}",
                ha="right", va="center", fontsize=7, color="#94a3b8")

    # --- Dimensjonerende laster (boks øvre høyre) ---
    info = (f"γ_f = {r['gamma_f']:.1f}  (NVE §2.5)\n"
            f"F_dim = {r['F_dim']:.2f} kN\n"
            f"F_akt = {r['F_actuator']:.2f} kN  (ψ={r['psi']})")
    ax.text(2.45, maxH * 0.95, info,
            ha="right", va="top", fontsize=8, color="#1a3a5c",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#2563a8", lw=1.5))

    # --- Tegnforklaring ---
    legend_items = [
        mpatches.Patch(color="#2563a8", alpha=0.5, label="Netto trykkfordeling"),
        mpatches.Patch(color="#e07b00", label="Resultant F_hyd"),
        mpatches.Patch(color="#15803d", label="Egenvekt G"),
        mpatches.Patch(color="#7c3aed", label="Løftekraft F_lift"),
        mpatches.Patch(color="#be185d", label="Friksjonskraft F_frict"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=7.5, framealpha=0.9, edgecolor="#e2e8f0",
              bbox_to_anchor=(-0.01, 1.0))

    # --- Tittel ---
    ax.set_title("Glideluke – Hydrostatisk trykkfordeling og krefter",
                 fontsize=13, fontweight="bold", color="#1a3a5c", pad=10)

    ax.set_xlim(-2.7, 2.8)
    ax.set_ylim(-0.45, maxH + 0.9)
    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────
#  PDF-GENERERING  (ReportLab)
# ─────────────────────────────────────────────────────────
def generer_pdf(inp, r, fig, prosjekt, autor, dokref):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title=f"Beregningsrapport – {prosjekt}",
        author=autor,
    )

    stiler = getSampleStyleSheet()
    N  = ParagraphStyle("N",  parent=stiler["Normal"],  fontSize=9.5, leading=14)
    H1 = ParagraphStyle("H1", parent=stiler["Heading1"], fontSize=16, textColor=colors.HexColor("#1a3a5c"), spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=stiler["Heading2"], fontSize=12, textColor=colors.HexColor("#1a3a5c"), spaceAfter=4, spaceBefore=14)
    H3 = ParagraphStyle("H3", parent=stiler["Heading3"], fontSize=10.5, textColor=colors.HexColor("#334155"), spaceAfter=3, spaceBefore=8)
    BL = ParagraphStyle("BL", parent=N, textColor=colors.HexColor("#64748b"), fontSize=9)
    NOTE = ParagraphStyle("NOTE", parent=N, leftIndent=12, rightIndent=6, fontSize=9,
                          borderPad=6, backColor=colors.HexColor("#f0f4f8"))
    MONO = ParagraphStyle("MONO", parent=N, fontName="Courier", fontSize=9, textColor=colors.HexColor("#2563a8"))

    BLK = colors.HexColor("#1a3a5c")
    LBL = colors.HexColor("#dbeafe")
    ALT = colors.HexColor("#f8fafc")
    HL  = colors.HexColor("#dbeafe")

    def tabell(data, col_widths, header=True):
        t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
        style = [
            ("FONTNAME",   (0,0), (-1,0 if header else -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("BACKGROUND", (0,0), (-1,0 if header else -1), BLK),
            ("TEXTCOLOR",  (0,0), (-1,0 if header else -1), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, ALT]),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ]
        if not header:
            style[0] = ("FONTNAME", (0,0), (-1,-1), "Helvetica")
            style[1] = ("FONTSIZE", (0,0), (-1,-1), 9)
            del style[2]; del style[2]
        t.setStyle(TableStyle(style))
        return t

    def hl_rad(t, row_indices):
        for ri in row_indices:
            t._argW  # noqa – access to force build
        return t

    dato = date.today().strftime("%d.%m.%Y")

    B, H_L, z_UK = inp["B"], inp["H_L"], inp["z_UK"]
    h1, h2       = inp["h1"], inp["h2"]
    Re, G        = inp["Re"], inp["G"]
    funksjon_label  = {"tappe":"Tappeluke / stengeorgan","flom":"Flomluke","inntak":"Inntaksluke"}[inp["funksjon"]]
    drift_label     = {"stengt":"Stengt – fullt hydrostatisk trykk","apning":"Åpning – trykk-belastet","mellom":"Mellomstilling – regulering"}[inp["driftstilstand"]]
    glide_label     = {"bronse":"Bronse mot rustfritt stål (μ=0,60)","polymer":"Polymer/polyamid (μ=0,40)","spesial":"Spesialglidelager (μ=0,15)"}[inp["glideliste"]]
    grense_label    = {"brudd":"Bruddgrensetilstand (ULS)","ulykke":"Ulykkesgrensetilstand (ALS)","bruks":"Bruksgrensetilstand (SLS)"}[inp["grensetilstand"]]

    story = []

    # ── Tittel ──────────────────────────────────────────
    story.append(Paragraph("Beregningsrapport – Glideluke (tappe- og stengeorgan)", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=BLK, spaceAfter=8))

    meta = [
        ["Prosjekt:",        prosjekt],
        ["Utarbeidet av:",   autor],
        ["Dokumentnr.:",     dokref],
        ["Dato:",            dato],
        ["Beregningsgrunnlag:", "NVE Retningslinjer 1/2011"],
        ["Beregningsverktøy:", "Glideluke-kalkulator v1.0"],
    ]
    mt = Table([[Paragraph(k, BL), Paragraph(v, N)] for k,v in meta],
               colWidths=[4.5*cm, 12*cm])
    mt.setStyle(TableStyle([
        ("FONTNAME",  (0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",  (0,0),(-1,-1), 9),
        ("TEXTCOLOR", (0,0),(0,-1), colors.HexColor("#64748b")),
        ("FONTNAME",  (1,0),(1,-1),"Helvetica-Bold"),
        ("TOPPADDING",(0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))

    note_txt = ("<b>Formål:</b> Beregning av hydrostatiske krefter, resultant og angrepspunkt, "
                "dimensjonerende laster og manøvreringskrefter for rektangulær glideluke. "
                "Beregnet i henhold til NVE Retningslinjer 1/2011.")
    story.append(Paragraph(note_txt, NOTE))
    story.append(Spacer(1, 8))

    # ── 1. Inndata ───────────────────────────────────────
    story.append(Paragraph("1. Inndata og forutsetninger", H2))
    story.append(Paragraph("1.1 Lukegeometri", H3))
    d = [["Parameter","Symbol","Verdi","Enhet"],
         ["Lukebredde","B", f"{B:.3f}", "m"],
         ["Lukehøyde","H_L", f"{H_L:.3f}", "m"],
         ["Underkant luke (over bunn)","z_UK", f"{z_UK:.3f}", "m"],
         ["Lukeareal","A", f"{r['area']:.3f}", "m²"]]
    story.append(tabell(d, [6*cm,2.5*cm,3*cm,3*cm]))
    story.append(Spacer(1,6))

    story.append(Paragraph("1.2 Vannstander", H3))
    d = [["Parameter","Symbol","Verdi","Enhet","Merknad"],
         ["Oppstrøms vannstand","h₁", f"{h1:.3f}", "m", "HRV"],
         ["Nedstrøms vannstand","h₂", f"{h2:.3f}", "m", "Tørt" if h2==0 else "Nedstrøms mottrykk"],
         ["Netto trykkhøyde","Δh", f"{h1-h2:.3f}", "m",""]]
    story.append(tabell(d, [5*cm,2*cm,2.5*cm,2*cm,5*cm]))
    story.append(Spacer(1,6))

    story.append(Paragraph("1.3 Driftsparametere", H3))
    d = [["Parameter","Valg"],
         ["Lukefunksjon", funksjon_label],
         ["Driftstilstand", drift_label],
         ["Konsekvensklasse", f"Klasse {inp['konsekvensklasse']}  (jf. damsikkerhetsforskriften §4-1)"],
         ["Grensetilstand", grense_label]]
    story.append(tabell(d, [5*cm,11.5*cm]))
    story.append(Spacer(1,6))

    story.append(Paragraph("1.4 Material- og friksjonsparametere", H3))
    d = [["Parameter","Symbol","Verdi","Enhet","Referanse"],
         ["Stålkvalitet (flytegrense)","Re", f"{Re:.0f}", "MPa", "NS-EN 10025"],
         ["Glidelisttype","–", glide_label, "–", "NVE 1/2011 Tabell 5.1"],
         ["Friksjonsfaktor (minimum)","μ", f"{r['mu']:.2f}", "–", "NVE 1/2011 §5.2"],
         ["Egenvekt lukekonstruksjon","G", f"{G:.2f}", "kN", "–"]]
    story.append(tabell(d, [5*cm,1.8*cm,4.5*cm,1.8*cm,4.5*cm]))

    # ── 2. Beregninger ───────────────────────────────────
    story.append(Paragraph("2. Beregninger", H2))

    story.append(Paragraph("2.1 Trykkfordeling", H3))
    story.append(Paragraph(
        "Vanntrykket beregnes som hydrostatisk trykk (lineær fordeling). "
        "Netto trykk er differansen mellom oppstrøms og nedstrøms trykk.", N))
    story.append(Spacer(1,4))
    d = [["Størrelse","Formel","Resultat","Enhet","Ref."],
         ["γ_w  (vannvekt)", "γ_w = ρ·g", f"{GAMMA_W:.2f}", "kN/m³", "–"],
         ["p_UK,1  (oppstr. UK)", "γ_w · h₁", f"{r['p_UK1']:.3f}", "kN/m²", "NVE §2.2"],
         ["p_OK,1  (oppstr. OK)", "γ_w · max(h₁–H_L, 0)", f"{r['p_OK1']:.3f}", "kN/m²", "NVE §2.2"],
         ["p_UK,2  (nedstr. UK)", "γ_w · h₂", f"{r['p_UK2']:.3f}", "kN/m²", "NVE §2.2"],
         ["p_OK,2  (nedstr. OK)", "γ_w · max(h₂–H_L, 0)", f"{r['p_OK2']:.3f}", "kN/m²", "NVE §2.2"],
         ["p_net,UK  (netto UK)", "p_UK,1 – p_UK,2", f"{r['p_net_UK']:.3f}", "kN/m²", "–"],
         ["p_net,OK  (netto OK)", "p_OK,1 – p_OK,2", f"{r['p_net_OK']:.3f}", "kN/m²", "–"]]
    t2 = tabell(d, [4.2*cm,5.5*cm,2.8*cm,2*cm,2.5*cm])
    # Uthev netto-rader
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,6), (-1,7), HL),
        ("FONTNAME",   (0,6), (-1,7), "Helvetica-Bold"),
    ]))
    story.append(t2)
    story.append(Spacer(1,6))

    story.append(Paragraph("2.2 Hydrostatisk resultantkraft og angrepspunkt", H3))
    d = [["Størrelse","Formel","Resultat","Enhet","Ref."],
         ["Resultantkraft F_hyd", "[(p_UK + p_OK)/2] · B · H_L", f"{r['F_hyd']:.3f}", "kN", "Hydrostatikk"],
         ["Middeltrykk p_avg", "F_hyd / (B · H_L)", f"{r['p_avg']:.3f}", "kN/m²", "–"],
         ["Angrepspunkt e", "H_L·(2·p_OK+p_UK) / [3·(p_UK+p_OK)]", f"{r['e_kp']:.4f}", "m", "Trapesformel"],
         ["Angrepspunkt (absolutt)", "z_UK + e", f"{r['e_abs']:.4f}", "m", "–"],
         ["Moment M_hyd", "F_hyd · e", f"{r['M_hyd']:.3f}", "kN·m", "–"]]
    t3 = tabell(d, [4.2*cm,6*cm,2.8*cm,2*cm,2*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,1),(-1,1), HL),
        ("BACKGROUND", (0,3),(-1,3), HL),
        ("FONTNAME",   (0,1),(-1,1), "Helvetica-Bold"),
        ("FONTNAME",   (0,3),(-1,3), "Helvetica-Bold"),
    ]))
    story.append(t3)
    story.append(Spacer(1,6))

    story.append(Paragraph("2.3 Dimensjonerende laster", H3))
    fn = "tappeluke/flomluke" if inp["funksjon"] in ("tappe","flom") else inp["funksjon"]
    story.append(Paragraph(
        f"Lastfaktor fastsatt iht. NVE 1/2011 Tabell 2.1 og §2.5. "
        f"{'<b>Spesialregel §2.5: γ_f = 1,4 for tappe-/flomorganer i bruddgrensetilstand.</b>' if inp['funksjon'] in ('tappe','flom') and inp['grensetilstand']=='brudd' else ''}", N))
    story.append(Spacer(1,4))
    d = [["Størrelse","Formel","Resultat","Enhet","Ref."],
         ["Lastfaktor γ_f", f"{'NVE §2.5 (tappeluke)' if inp['funksjon'] in ('tappe','flom') else 'NVE Tabell 2.1'}", f"{r['gamma_f']:.1f}", "–", "NVE 1/2011"],
         ["Dim. kraft F_dim", "γ_f · F_hyd", f"{r['F_dim']:.3f}", "kN", "NVE 1/2011 §2.5"],
         ["Dim. moment M_dim", "γ_f · M_hyd", f"{r['M_dim']:.3f}", "kN·m", "–"]]
    t4 = tabell(d, [4.2*cm,5.5*cm,2.8*cm,2*cm,2.5*cm])
    t4.setStyle(TableStyle([
        ("BACKGROUND",(0,2),(-1,2), HL),
        ("FONTNAME",  (0,2),(-1,2), "Helvetica-Bold"),
    ]))
    story.append(t4)
    story.append(Spacer(1,6))

    story.append(Paragraph("2.4 Manøvreringskrefter", H3))
    sc_txt = "Ja – netto lukkekraft > 25 % av motkrefter" if r["self_closing"] else "Nei – ekstern lukkekraft nødvendig"
    d = [["Størrelse","Formel","Resultat","Enhet","Ref."],
         ["Friksjonsfaktor μ", f"{inp['glideliste']} (NVE minimum)", f"{r['mu']:.2f}", "–", "NVE Tab. 5.1"],
         ["Friksjonskraft F_frict", "μ · F_hyd", f"{r['F_frict']:.3f}", "kN", "NVE §5.2"],
         ["Egenvekt G", "–", f"{G:.2f}", "kN", "–"],
         ["Løftekraft F_lift", "G + F_frict", f"{r['F_lift']:.3f}", "kN", "NVE §5.2"],
         ["Paslagsfaktor ψ", f"{fn}", f"{r['psi']:.1f}", "–", "NVE §5.2"],
         ["Aktuatorkapasitet F_akt", "ψ · F_lift", f"{r['F_actuator']:.3f}", "kN", "NVE §5.2"],
         ["Senkekraft F_lower", "G – F_frict", f"{r['F_lower']:.3f}", "kN", "–"],
         ["Selvlukkende?", "F_lower > 0 og > 25% F_frict", sc_txt, "–", "NVE §5.2"],
         ["Tettelengde L_t", "2·(B + H_L)", f"{r['tettelengde']:.2f}", "m", "–"],
         ["Min. tetningskraft", "5 kN/m · L_t", f"{r['F_seal_req']:.1f}", "kN", "NVE §5.2"]]
    t5 = tabell(d, [4.2*cm,5*cm,3.5*cm,1.8*cm,2.5*cm])
    t5.setStyle(TableStyle([
        ("BACKGROUND",(0,4),(-1,4), HL),
        ("BACKGROUND",(0,6),(-1,6), HL),
        ("FONTNAME",  (0,4),(-1,4), "Helvetica-Bold"),
        ("FONTNAME",  (0,6),(-1,6), "Helvetica-Bold"),
    ]))
    story.append(t5)
    story.append(Spacer(1,6))

    story.append(Paragraph("2.5 Materialkapasitet – stål", H3))
    d = [["Størrelse","Formel","Resultat","Enhet","Ref."],
         ["Karakteristisk flytegrense Re","–", f"{Re:.0f}", "MPa", "NS-EN 10025"],
         ["Materialfaktor γ_m (med plast.res.)","NVE Tabell 4.1", f"{r['gamma_m']:.2f}", "–", "NVE §4.2"],
         ["Dim. flytegrense f_yd","(Re · 0,8) / γ_m", f"{r['f_yd']:.1f}", "MPa", "NVE §4.2"],
         ["Korrosjonstillegg","1 mm per eksponert flate","1,0","mm","NVE §4.2"]]
    t6 = tabell(d, [5*cm,5*cm,2.5*cm,1.8*cm,2.7*cm])
    t6.setStyle(TableStyle([
        ("BACKGROUND",(0,3),(-1,3), HL),
        ("FONTNAME",  (0,3),(-1,3), "Helvetica-Bold"),
    ]))
    story.append(t6)

    # ── 3. Figur ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Skjematisk figur", H2))
    fig_buf = io.BytesIO()
    fig.savefig(fig_buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig_buf.seek(0)
    img = RLImage(fig_buf, width=16*cm, height=11*cm)
    story.append(img)
    story.append(Spacer(1,4))
    story.append(Paragraph(
        f"<i>Figur 1: Glideluke med netto hydrostatisk trykkfordeling og krefter. "
        f"F_hyd = {r['F_hyd']:.2f} kN  |  e = {r['e_kp']:.3f} m over UK  |  "
        f"F_dim = {r['F_dim']:.2f} kN (γ_f = {r['gamma_f']:.1f})  |  F_lift = {r['F_lift']:.2f} kN</i>",
        ParagraphStyle("FC", parent=N, fontSize=8.5, textColor=colors.HexColor("#64748b"),
                       alignment=TA_CENTER)))

    # ── 4. Kontroll ──────────────────────────────────────
    story.append(Paragraph("4. Kontroll og advarsler", H2))
    story.append(Paragraph("4.1 Gyldighetssjekker", H3))
    ok_c  = colors.HexColor("#dcfce7")
    wrn_c = colors.HexColor("#fef3c7")
    chk = [
        ("Trykkhøyde",
         f"h₁ = {h1:.2f} m",
         ("OK – ≤20m, kavitasjon uproblematisk" if h1<=20 else
          "Advarsel – 20–40m, lufting må vurderes" if h1<=40 else
          "Advarsel – >40m, spesiell utforming nødvendig"),
         h1 > 20, "NVE C.4"),
        ("Lastfaktor", f"γ_f = {r['gamma_f']:.1f}",
         f"OK – korrekt faktor for {funksjon_label}", False, "NVE §2.5"),
        ("Friksjonsfaktor", f"μ = {r['mu']:.2f}",
         "OK – NVE minimumsverdi brukt", False, "NVE Tab. 5.1"),
        ("Selvlukkende", sc_txt,
         "OK" if r["self_closing"] else "Advarsel – ekstern kraft nødvendig",
         not r["self_closing"], "NVE §5.2"),
        ("B vs. H_L", f"B={B:.2f} m, H_L={H_L:.2f} m",
         "OK" if B<=H_L else "Advarsel – styreruller bør vurderes",
         B>H_L, "NVE C.4"),
    ]
    cd = [["Kontroll","Verdi","Status","Ref."]]
    for k,v,s2,warn,ref in chk:
        cd.append([k, v, s2, ref])
    ct = tabell(cd, [4*cm,4*cm,7*cm,2*cm])
    ct_style = []
    for i, (_,_,_,warn,_) in enumerate(chk):
        ct_style.append(("BACKGROUND", (0,i+1),(-1,i+1), wrn_c if warn else ok_c))
    ct.setStyle(TableStyle(ct_style))
    story.append(ct)

    # ── 5. Forutsetninger ────────────────────────────────
    story.append(Paragraph("5. Forutsetninger og gyldighetsområde", H2))
    story.append(Paragraph(
        "Beregningsverktøyet dekker <b>hydrostatisk kraftberegning</b> og manøvreringskrefter "
        "for rektangulære glideluker med uniform bredde i stasjonær tilstand. "
        "Følgende er <b>ikke</b> inkludert og må beregnes separat:", N))
    story.append(Spacer(1,4))
    for punkt in [
        "Hydrodynamiske trykk og trykkstøt fra strømmende vann",
        "Vibrasjon og kavitasjon (vurder for h > 20 m, jf. NVE C.4)",
        "Luftingsbehov nedstrøms tappeluke (alltid vurderes, jf. NVE §5.1)",
        "Strukturdimensjonering av platekasse, bjelker og forbindelser",
        "Utmattingsanalyse (jf. NVE Vedlegg B)",
        "Islast, temperaturlast og jordskjelvlast",
    ]:
        story.append(Paragraph(f"• {punkt}", ParagraphStyle("BU", parent=N, leftIndent=14)))

    story.append(Spacer(1,8))
    story.append(Paragraph("Referanser", H3))
    refs = [
        ["[1]", "NVE Retningslinjer 1/2011", "Stenge- og tappeorganer, ror og tverrslagsporter"],
        ["[2]", "NVE Retningslinjer (2003/2018)", "Retningslinje for laster og dimensjonering"],
        ["[3]", "Damsikkerhetsforskriften", "Forskrift om sikkerhet ved vassdragsanlegg"],
        ["[4]", "NS 3472", "Prosjektering av stålkonstruksjoner"],
        ["[5]", "NS-EN 13445", "Trykkpåkjente konstruksjoner"],
    ]
    rt = Table([[Paragraph(a,BL), Paragraph(b,N), Paragraph(c,N)] for a,b,c in refs],
               colWidths=[1.5*cm, 6*cm, 9*cm])
    rt.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),2),
        ("BOTTOMPADDING",(0,0),(-1,-1),2),
    ]))
    story.append(rt)

    # ── Footer ───────────────────────────────────────────
    story.append(Spacer(1,16))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        f"Rapport generert {dato}  |  Glideluke-kalkulator NVE 1/2011  |  "
        "Alle beregninger er brukerens ansvar og bør verifiseres av kompetent ingeniør.",
        ParagraphStyle("FT", parent=N, fontSize=7.5, textColor=colors.HexColor("#94a3b8"),
                       alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1a3a5c; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stNumberInput label { color: #93c5fd !important; font-weight:600; }
  .metric-card { background:#f0f4f8; border-radius:8px; padding:12px 16px;
                 border-left:4px solid #2563a8; margin-bottom:8px; }
  .metric-card .val { font-size:22px; font-weight:800; color:#1a3a5c; font-family:monospace; }
  .metric-card .lbl { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.5px; }
  .metric-card .sub { font-size:11px; color:#94a3b8; margin-top:2px; font-family:monospace; }
  .metric-hl { border-left-color:#e07b00 !important; background:#fff7ed !important; }
  .metric-hl .val { color:#c2410c !important; }
  .warn-box { background:#fef3c7; border-left:4px solid #d97706;
              border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
  .err-box  { background:#fee2e2; border-left:4px solid #dc2626;
              border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
  .ok-box   { background:#dcfce7; border-left:4px solid #15803d;
              border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
  h2 { color:#1a3a5c !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar – inndata ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔩 Glideluke-kalkulator")
    st.markdown("**NVE Retningslinjer 1/2011**")
    st.markdown("---")

    st.markdown("#### Lukegeometri")
    B    = st.number_input("Lukebredde B (m)", min_value=0.05, max_value=30.0, value=1.50, step=0.05)
    H_L  = st.number_input("Lukehøyde H_L (m)", min_value=0.05, max_value=30.0, value=1.20, step=0.05)
    z_UK = st.number_input("Underkant luke z_UK (m)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

    st.markdown("#### Vannstander")
    h1 = st.number_input("Oppstrøms h₁ (m) – HRV", min_value=0.0, max_value=300.0, value=5.0, step=0.1)
    h2 = st.number_input("Nedstrøms h₂ (m)", min_value=0.0, max_value=300.0, value=0.0, step=0.1)

    st.markdown("#### Driftstilstand")
    funksjon = st.selectbox("Funksjon", ["tappe","flom","inntak"],
                             format_func=lambda x: {"tappe":"Tappeluke / stengeorgan","flom":"Flomluke","inntak":"Inntaksluke"}[x])
    driftstilstand = st.selectbox("Driftstilstand", ["stengt","apning","mellom"],
                                   format_func=lambda x: {"stengt":"Stengt – fullt trykk","apning":"Åpning – trykk-belastet","mellom":"Mellomstilling"}[x])
    konsekvensklasse = st.selectbox("Konsekvensklasse", [1,2,3], index=1)
    grensetilstand = st.selectbox("Grensetilstand", ["brudd","ulykke","bruks"],
                                    format_func=lambda x: {"brudd":"Bruddgrense (ULS)","ulykke":"Ulykkesgrensetilstand (ALS)","bruks":"Bruksgrensetilstand (SLS)"}[x])

    st.markdown("#### Material og friksjon")
    Re = st.number_input("Flytegrense Re (MPa)", min_value=100, max_value=700, value=355, step=5)
    glideliste = st.selectbox("Glidelisttype", ["bronse","polymer","spesial"],
                               index=1,
                               format_func=lambda x: {"bronse":"Bronse – μ=0,60","polymer":"Polymer – μ=0,40","spesial":"Spesial – μ=0,15"}[x])
    G = st.number_input("Egenvekt G (kN)", min_value=0.0, max_value=5000.0, value=5.0, step=0.5)

# ── Validering ────────────────────────────────────────────
errors, warnings_list = valider(B, H_L, z_UK, h1, h2, Re, G)

# ── Beregning ─────────────────────────────────────────────
r = None
if not errors:
    r = beregn(B, H_L, z_UK, h1, h2, Re, G, funksjon, glideliste, grensetilstand)
    r["G_val"] = G  # for figuren

# ── Meldinger ─────────────────────────────────────────────
for e in errors:
    st.markdown(f'<div class="err-box">❌ <strong>Feil:</strong> {e}</div>', unsafe_allow_html=True)
for w in warnings_list:
    st.markdown(f'<div class="warn-box">⚠️ <strong>Advarsel:</strong> {w}</div>', unsafe_allow_html=True)
if not errors and not warnings_list:
    st.markdown('<div class="ok-box">✅ Inndata validert – alle kontroller OK.</div>', unsafe_allow_html=True)

if errors:
    st.stop()

# ── Faner ─────────────────────────────────────────────────
tab_res, tab_fig, tab_trace, tab_check, tab_rapport = st.tabs([
    "📊 Resultater", "📐 Figur", "🧮 Beregningsspor", "✅ Kontroll", "📄 Rapport"
])

# ───────── TAB: RESULTATER ────────────────────────────────
with tab_res:
    st.subheader("Hydrostatiske krefter")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Trykk ved UK", f"{r['p_net_UK']:.2f} kN/m²", help="Netto trykk ved underkant luke")
        st.metric("Trykk ved OK", f"{r['p_net_OK']:.2f} kN/m²", help="Netto trykk ved overkant luke")
    with c2:
        st.metric("Resultantkraft F_hyd", f"{r['F_hyd']:.2f} kN", help="Hydrostatisk resultantkraft (trapesfordeling)")
        st.metric("Middeltrykk", f"{r['p_avg']:.2f} kN/m²")
    with c3:
        st.metric("Angrepspunkt e", f"{r['e_kp']:.4f} m", help="Høyde over underkant luke")
        st.metric("Moment om UK", f"{r['M_hyd']:.2f} kN·m")
    with c4:
        st.metric("Lukeareal A", f"{r['area']:.3f} m²")
        st.metric("Netto trykkhøyde Δh", f"{h1-h2:.2f} m")

    st.subheader("Dimensjonerende laster")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Lastfaktor γ_f", f"{r['gamma_f']:.1f}",
                        help="NVE §2.5: 1,4 for tappeluke, 1,2 ellers (ULS)")
    with c2: st.metric("Dim. kraft F_dim", f"{r['F_dim']:.2f} kN",
                        help="γ_f × F_hyd")
    with c3: st.metric("Dim. moment M_dim", f"{r['M_dim']:.2f} kN·m")

    st.subheader("Manøvreringskrefter")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Friksjonskraft F_frict", f"{r['F_frict']:.2f} kN", help=f"μ={r['mu']} × F_hyd")
        st.metric("Friksjonsfaktor μ", f"{r['mu']:.2f}", help="NVE Tabell 5.1 (minimumsverdi)")
    with c2:
        st.metric("Løftekraft F_lift", f"{r['F_lift']:.2f} kN", help="G + F_frict")
        st.metric("Aktuatorkapasitet F_akt", f"{r['F_actuator']:.2f} kN", help=f"ψ={r['psi']} × F_lift")
    with c3:
        st.metric("Senkekraft F_lower", f"{r['F_lower']:.2f} kN", help="G – F_frict")
        selfcls = "✅ Ja" if r["self_closing"] else "❌ Nei"
        st.metric("Selvlukkende?", selfcls, help="NVE §5.2: netto lukkekraft > 25% av motkrefter")
    with c4:
        st.metric("Tettelengde", f"{r['tettelengde']:.2f} m")
        st.metric("Min. tetningskraft", f"{r['F_seal_req']:.1f} kN", help="5 kN/m × tettelengde (NVE §5.2)")

    st.subheader("Materialkapasitet – stål")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Flytegrense Re", f"{Re} MPa")
    with c2: st.metric("Materialfaktor γ_m", f"{r['gamma_m']:.2f}", help="Med plastisitetsreserve, NVE Tabell 4.1")
    with c3: st.metric("Dim. flytegrense f_yd", f"{r['f_yd']:.1f} MPa", help="(Re × 0,8) / γ_m  –  NVE §4.2")

# ───────── TAB: FIGUR ─────────────────────────────────────
with tab_fig:
    fig = lag_figur(B, H_L, z_UK, h1, h2, r)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ───────── TAB: BEREGNINGSSPOR ────────────────────────────
with tab_trace:
    st.caption("Alle beregningstrinn med formler og referanser til NVE-veilederen.")
    trace = [
        {"#":"1","Navn":"Trykk ved underkant (oppstr.)","Formel":"p_UK,1 = γ_w · h₁",
         "Verdier":f"γ_w={GAMMA_W}, h₁={h1}","Resultat":f"{r['p_UK1']:.3f} kN/m²","Ref":"NVE §2.2"},
        {"#":"2","Navn":"Trykk ved overkant (oppstr.)","Formel":"p_OK,1 = γ_w · max(h₁–H_L, 0)",
         "Verdier":f"h₁={h1}, H_L={H_L}","Resultat":f"{r['p_OK1']:.3f} kN/m²","Ref":"NVE §2.2"},
        {"#":"3","Navn":"Trykk nedstrøms (UK og OK)","Formel":"p_UK,2 = γ_w·h₂ ; p_OK,2 = γ_w·max(h₂–H_L,0)",
         "Verdier":f"h₂={h2}","Resultat":f"{r['p_UK2']:.3f} / {r['p_OK2']:.3f} kN/m²","Ref":"NVE §2.2"},
        {"#":"4","Navn":"Netto trykkfordeling","Formel":"p_net = p_1 – p_2  (UK og OK)",
         "Verdier":"","Resultat":f"{r['p_net_UK']:.3f} / {r['p_net_OK']:.3f} kN/m²","Ref":""},
        {"#":"5","Navn":"Hydrostatisk resultantkraft","Formel":"F_hyd = [(p_UK + p_OK)/2] · B · H_L",
         "Verdier":f"B={B}, H_L={H_L}","Resultat":f"{r['F_hyd']:.4f} kN","Ref":"Hydrostatikk"},
        {"#":"6","Navn":"Angrepspunkt (over underkant)","Formel":"e = H_L·(2·p_OK+p_UK) / [3·(p_UK+p_OK)]",
         "Verdier":"","Resultat":f"{r['e_kp']:.4f} m","Ref":"Trapesformel"},
        {"#":"7","Navn":"Moment om underkant","Formel":"M_hyd = F_hyd · e",
         "Verdier":"","Resultat":f"{r['M_hyd']:.4f} kN·m","Ref":"Statikk"},
        {"#":"8","Navn":"Lastfaktor","Formel":"γ_f = 1,4 (tappeluke/ULS) / 1,2 (standard)",
         "Verdier":f"Funksjon={funksjon}, grensetilstand={grensetilstand}","Resultat":f"{r['gamma_f']:.1f}","Ref":"NVE 1/2011 §2.5"},
        {"#":"9","Navn":"Dimensjonerende kraft","Formel":"F_dim = γ_f · F_hyd",
         "Verdier":"","Resultat":f"{r['F_dim']:.4f} kN","Ref":"NVE §2.5"},
        {"#":"10","Navn":"Friksjonskraft","Formel":"F_frict = μ · F_hyd",
         "Verdier":f"μ={r['mu']}","Resultat":f"{r['F_frict']:.4f} kN","Ref":"NVE Tabell 5.1"},
        {"#":"11","Navn":"Løftekraft","Formel":"F_lift = G + F_frict",
         "Verdier":f"G={G}","Resultat":f"{r['F_lift']:.4f} kN","Ref":"NVE §5.2"},
        {"#":"12","Navn":"Aktuatorkapasitet","Formel":"F_akt = ψ · F_lift",
         "Verdier":f"ψ={r['psi']}","Resultat":f"{r['F_actuator']:.4f} kN","Ref":"NVE §5.2"},
        {"#":"13","Navn":"Dim. flytegrense stål","Formel":"f_yd = (Re·0,8) / γ_m",
         "Verdier":f"Re={Re}, γ_m={r['gamma_m']}","Resultat":f"{r['f_yd']:.2f} MPa","Ref":"NVE §4.2"},
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(trace).set_index("#"), use_container_width=True, height=500)

# ───────── TAB: KONTROLL ──────────────────────────────────
with tab_check:
    st.subheader("Gyldighetssjekker")
    checks = [
        (h1 <= 20, h1 <= 40,
         "Trykkhøyde",
         f"h₁ = {h1:.2f} m",
         "≤20 m: kavitasjon/vibrasjon uproblematisk" if h1<=20 else ("20–40 m: lufting nedstrøms må vurderes" if h1<=40 else ">40 m: SPESIELL UTFORMING NØDVENDIG"),
         "NVE C.4"),
        (True, False, "Lastfaktor", f"γ_f = {r['gamma_f']:.1f}", f"Korrekt for {funksjon}, {grensetilstand}", "NVE §2.5"),
        (True, False, "Friksjonsfaktor", f"μ = {r['mu']:.2f}", f"NVE minimumsverdi ({glideliste})", "NVE Tab. 5.1"),
        (r["self_closing"], not r["self_closing"], "Selvlukkende", "", "Netto lukkekraft > 25% av motkrefter" if r["self_closing"] else "Ekstern lukkekraft nødvendig", "NVE §5.2"),
        (B <= H_L, B > H_L, "B vs. H_L", f"B={B:.2f} m, H_L={H_L:.2f} m", "OK" if B<=H_L else "Styreruller/styreskinner bør vurderes", "NVE C.4"),
    ]
    for ok, warn, name, val, detail, ref in checks:
        icon = "✅" if ok and not warn else ("⚠️" if warn else "❌")
        cls  = "ok-box" if (ok and not warn) else "warn-box"
        st.markdown(
            f'<div class="{cls}">{icon} <strong>{name}</strong>'
            f'{" – " + val if val else ""}:  {detail} &nbsp;<code>{ref}</code></div>',
            unsafe_allow_html=True)

    st.subheader("Kapasitetsveiledning")
    st.info(
        f"**Dimensjonerende flytegrense:** f_yd = (Re × 0,8) / γ_m = "
        f"({Re} × 0,8) / {r['gamma_m']} = **{r['f_yd']:.1f} MPa** *(NVE §4.2)*\n\n"
        "Kapasitetskontroll av platekasse og bjelker gjøres mot "
        f"F_dim = {r['F_dim']:.2f} kN og M_dim = {r['M_dim']:.2f} kN·m. "
        "Trekk fra 1 mm korrosjonstillegg per eksponert flate (NVE §4.2).\n\n"
        "**Flatetrykk-grenser (NVE §5.2):**  "
        "Bronse mot rustfritt stål: maks 14,0 MPa  |  "
        "Polyamid mot rustfritt stål: maks 10,0 MPa  |  "
        "Innstøpt stål mot betong: skjær ≤ 0,5 MPa, trykk ≤ 15,0 MPa"
    )

    st.subheader("Gyldighetsområde")
    st.warning(
        "**Verktøyet dekker:** Hydrostatisk kraftberegning og manøvreringskrefter for rektangulære "
        "glideluker med uniform bredde i stasjonær tilstand.\n\n"
        "**Ikke inkludert – beregnes separat:** Hydrodynamiske trykk og trykkstøt · "
        "Vibrasjon og kavitasjon (h > 20 m) · Luftingsbehov nedstrøms · "
        "Strukturdimensjonering av platekasse og bjelker · "
        "Utmatting (NVE Vedlegg B) · Islast, temperatur og jordskjelv"
    )

# ───────── TAB: RAPPORT ───────────────────────────────────
with tab_rapport:
    st.subheader("Eksport – Beregningsrapport")
    c1,c2,c3 = st.columns([3,2,2])
    with c1: prosjekt = st.text_input("Prosjektnavn", placeholder="F.eks. Reguleringsdam Glomma")
    with c2: autor    = st.text_input("Utarbeidet av", placeholder="Navn / firma")
    with c3: dokref   = st.text_input("Dokumentnr.", placeholder="F.eks. STA-2025-001")

    st.markdown("---")

    inp = dict(B=B, H_L=H_L, z_UK=z_UK, h1=h1, h2=h2, Re=Re, G=G,
               funksjon=funksjon, driftstilstand=driftstilstand,
               konsekvensklasse=konsekvensklasse, glideliste=glideliste,
               grensetilstand=grensetilstand)

    col_pdf, col_info = st.columns([1, 2])
    with col_pdf:
        if st.button("⬇️ Generer og last ned PDF", type="primary", use_container_width=True):
            with st.spinner("Genererer rapport..."):
                fig_rpt = lag_figur(B, H_L, z_UK, h1, h2, r, dpi=130)
                pdf_buf = generer_pdf(inp, r, fig_rpt,
                                      prosjekt or "(ikke angitt)",
                                      autor    or "(ikke angitt)",
                                      dokref   or "–")
                plt.close(fig_rpt)
                dato_str = date.today().strftime("%Y-%m-%d")
                safe_name = (prosjekt or "Glideluke").replace(" ","_")
                st.download_button(
                    label="📄 Last ned PDF nå",
                    data=pdf_buf,
                    file_name=f"Beregningsrapport_{safe_name}_{dato_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
    with col_info:
        st.info(
            "**Rapporten inneholder:**\n"
            "- Prosjektinfo og beregningsgrunnlag\n"
            "- Inndata: geometri, vannstander, material\n"
            "- Fullstendige beregninger med formler og NVE-referanser\n"
            "- Skjematisk figur med krefter og trykkfordeling\n"
            "- Gyldighetssjekker og advarsler\n"
            "- Forutsetninger og referanseliste"
        )
