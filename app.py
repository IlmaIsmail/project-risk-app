"""
════════════════════════════════════════════════════════════════════════════
 Project Risk Prediction — Decision Support Interface
 Final Meta-Learned Stacking Ensemble (XGBoost + LightGBM + Random Forest
 combined through a Logistic Regression meta-learner)

 Author : Ahamed Ismail Fathima Ilma (2020/ICT/48)

 DESIGN NOTES
 ------------
 1. SCHEMA-DRIVEN: all column lists, encoders and the scaler are loaded from
    outputs/processed_data.pkl, so the app cannot drift from the notebook.

 2. PREPROCESSING ORDER IS MANDATORY and mirrors the notebook exactly:
        ordinal-encode -> engineer -> one-hot -> scale -> align
    The engineered features are computed FROM ordinal-encoded values
    (e.g. 3 - Org_Process_Maturity), so this order must not be changed.

 3. EXPLANATIONS ARE ANCHORED TO THE High+ CLASS. SHAP values are per-class;
    anchoring to High+ means a positive value ALWAYS means "pushes toward
    higher risk", regardless of which grade was predicted.

 4. RECOMMENDATIONS ARE ASSOCIATIVE, NOT CAUSAL. They are derived from the
    model's learned feature contributions and are decision support only.

 5. THEME: the light/dark toggle injects CSS overrides. Every rule sets both
    a background and a foreground colour so no element can become invisible.
    Widget chrome still follows .streamlit/config.toml.
════════════════════════════════════════════════════════════════════════════
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Project Risk Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = Path("outputs")
HIGH_CLS = 2                      # index of the High+ class
NOT_KNOWN = "— Not known —"

STEPS = ["Start", "Scale & Budget", "Team & Delivery",
         "Process Maturity", "Delivery Context", "Result"]

# Risk colours chosen to stay legible on BOTH backgrounds
RISK_COLORS = {
    "light": {"Low": "#1B7F45", "Medium": "#B36B00", "High+": "#B0281F"},
    "dark":  {"Low": "#5BD98A", "Medium": "#F5B942", "High+": "#FF7B6B"},
}


# ═══════════════════════════════ theming ════════════════════════════════
def inject_theme(mode: str):
    """Inject CSS for the selected mode. Every rule sets BOTH a background and
    a foreground colour, so no element can end up with invisible text."""
    if mode == "dark":
        bg, panel, fg, muted, border, accent = (
            "#0F1419", "#1A212B", "#E8EDF2", "#A8B3BF", "#2E3A47", "#5AA9E6")
    else:
        bg, panel, fg, muted, border, accent = (
            "#FBFCFD", "#EEF3F8", "#1F2933", "#5A6672", "#D4DEE8", "#2D6A9F")

    st.markdown(f"""
    <style>
      .stApp {{ background-color: {bg}; color: {fg}; }}

      /* headings, body text, list items, labels */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
      .stApp p, .stApp li, .stApp label, .stApp strong {{ color: {fg} !important; }}

      /* captions and helper text */
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stCaptionContainer"] p,
      .stApp small {{ color: {muted} !important; }}

      /* sidebar */
      section[data-testid="stSidebar"] {{ background-color: {panel}; }}
      section[data-testid="stSidebar"] * {{ color: {fg} !important; }}

      /* metrics */
      [data-testid="stMetricValue"] {{ color: {fg} !important; }}
      [data-testid="stMetricLabel"] {{ color: {muted} !important; }}

      /* dataframes */
      [data-testid="stDataFrame"] {{ background-color: {panel}; }}
      [data-testid="stDataFrame"] * {{ color: {fg} !important; }}

      /* expanders */
      .streamlit-expanderHeader, [data-testid="stExpander"] summary {{
          background-color: {panel} !important; color: {fg} !important; }}
      [data-testid="stExpander"] {{ border: 1px solid {border} !important;
          border-radius: 8px; }}
      [data-testid="stExpander"] div {{ color: {fg} !important; }}

      /* form controls */
      .stSelectbox div[data-baseweb="select"] > div,
      .stNumberInput input, .stTextInput input {{
          background-color: {panel} !important; color: {fg} !important;
          border-color: {border} !important; }}
      div[data-baseweb="popover"] li {{ color: {fg} !important;
          background-color: {panel} !important; }}

      /* checkbox and slider labels */
      .stCheckbox label p, .stSlider label p {{ color: {fg} !important; }}

      /* dividers and progress */
      hr {{ border-color: {border} !important; }}
      .stProgress > div > div > div {{ background-color: {accent} !important; }}

      /* BUTTONS — set background AND text colour explicitly, otherwise the
         label inherits the base theme's dark text and vanishes in dark mode */
      .stButton button,
      .stButton button[kind="secondary"],
      .stButton button[data-testid="baseButton-secondary"],
      .stDownloadButton button {{
          background-color: {panel} !important;
          color: {fg} !important;
          border: 1px solid {border} !important;
      }}
      .stButton button[kind="primary"],
      .stButton button[data-testid="baseButton-primary"] {{
          background-color: {accent} !important;
          color: #FFFFFF !important;
          border: 1px solid {accent} !important;
      }}
      /* the label is wrapped in <p>/<div>/<span>: force it to follow the button */
      .stButton button *, .stDownloadButton button * {{
          color: inherit !important; fill: inherit !important; }}

      /* hover / focus states keep contrast */
      .stButton button:hover,
      .stButton button[kind="secondary"]:hover {{
          background-color: {bg} !important;
          color: {accent} !important;
          border-color: {accent} !important; }}
      .stButton button[kind="primary"]:hover {{
          background-color: {accent} !important;
          color: #FFFFFF !important;
          filter: brightness(1.15); }}
      .stButton button:disabled, .stButton button:disabled * {{
          color: {muted} !important; opacity: 1 !important; }}

      /* alerts keep their semantic colours but need readable text */
      .stAlert p {{ color: inherit !important; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════ recommendation knowledge base ══════════════════
# feature -> (can the manager change it?, advice when it INCREASES risk)
RECOMMENDATIONS = {
    "Team_Experience_Level": (True, "Bring in at least one senior or expert engineer, or pair junior staff with experienced mentors on critical-path work."),
    "Requirement_Stability": (True, "Freeze the scope baseline and route all changes through formal change control before the next phase gate."),
    "Technology_Familiarity": (True, "Schedule a proof-of-concept on the unfamiliar technology, and budget training time before committing to delivery dates."),
    "Stakeholder_Engagement_Level": (True, "Establish a fixed cadence of stakeholder reviews with named decision-owners for each workstream."),
    "Executive_Sponsorship": (True, "Secure a named executive sponsor with budget authority and a standing escalation slot."),
    "Project_Manager_Experience": (True, "Assign a more experienced PM, or provide senior PM oversight at phase gates."),
    "Org_Process_Maturity": (True, "Introduce defined, repeatable delivery processes — even lightweight ones — before scaling the team."),
    "Key_Stakeholder_Availability": (True, "Secure written time commitments from key stakeholders and name backup decision-makers."),
    "Resource_Contention_Level": (True, "Negotiate dedicated allocation for critical roles rather than shared capacity across projects."),
    "Change_Control_Maturity": (True, "Formalise change control with a mandatory impact assessment before approval."),
    "Risk_Management_Maturity": (True, "Stand up a live risk register with named owners and a regular review cadence."),
    "Documentation_Quality": (True, "Allocate explicit time for documentation and set a minimum standard as a phase-gate criterion."),
    "Regulatory_Compliance_Level": (False, "Compliance burden is largely fixed — engage compliance specialists early and build approval lead-time into the schedule."),
    "Data_Security_Requirements": (False, "Security requirements are typically non-negotiable — plan security checkpoints into the timeline rather than treating them as late-stage gates."),
    "Industry_Volatility": (False, "Market volatility is outside project control — build schedule buffer and plan for re-baselining."),
    "Priority_Level": (False, "High priority raises exposure — ensure resourcing actually matches the stated priority."),

    "Team_Size": (True, "Large teams increase coordination overhead. Consider splitting into smaller autonomous sub-teams with clear interfaces."),
    "Estimated_Timeline_Months": (True, "Break delivery into shorter, independently valuable phases so risk surfaces earlier."),
    "Complexity_Score": (True, "Decompose the highest-complexity components and tackle them first, while contingency still exists."),
    "Stakeholder_Count": (True, "Consolidate decision-making through a smaller steering group to reduce approval latency."),
    "External_Dependencies_Count": (True, "Map every external dependency, agree written SLAs, and identify fallbacks for the critical ones."),
    "Change_Request_Frequency": (True, "Batch change requests into scheduled review cycles rather than accepting them continuously."),
    "Team_Turnover_Rate": (True, "Address retention directly, and ensure knowledge is documented so departures are survivable."),
    "Schedule_Pressure": (True, "Re-negotiate the deadline or reduce scope — sustained schedule pressure is among the strongest risk amplifiers."),
    "Budget_Utilization_Rate": (True, "Review burn rate against delivered value now; early over-consumption limits later contingency."),
    "Integration_Complexity": (True, "Prototype the riskiest integration points early rather than deferring them to a late integration phase."),
    "Geographical_Distribution": (True, "Establish overlapping working hours and a single source of truth for decisions across locations."),
    "Communication_Frequency": (True, "Increase structured communication cadence — low frequency correlates with late problem discovery."),
    "Historical_Risk_Incidents": (False, "Past incidents cannot be changed, but their root causes should directly inform this project's risk plan."),
    "Vendor_Reliability_Score": (True, "Reassess vendor selection, or add contractual milestones for the least reliable suppliers."),
    "Past_Similar_Projects": (False, "Limited comparable experience — invest in external expertise or a pilot phase before full commitment."),
    "Project_Budget_USD": (False, "Budget size drives scrutiny and exposure; ensure governance is proportionate to the amount at stake."),
    "Resource_Availability": (True, "Confirm resource commitments in writing before the next phase begins."),

    # engineered features
    "Resource_Health": (True, "The net resource position is weak — either increase availability or reduce competing demands on the same people."),
    "Turnover_x_Complexity": (True, "Staff churn is especially damaging on complex work. Prioritise retention and knowledge capture on the most complex components."),
    "Pressure_x_LowMaturity": (True, "Schedule pressure combined with immature process is a compounding risk. Relieve one or the other before proceeding."),
    "Compliance_x_Complexity": (False, "Compliance obligations scale with complexity — allow proportionally more approval time than a simpler project would need."),
}

GENERIC_ADVICE = ("This factor is contributing to elevated risk — review it explicitly "
                  "at the next project board and assign a named owner.")


def get_recommendation(fname: str):
    """Return (changeable, advice), matching one-hot columns to their base name."""
    if fname in RECOMMENDATIONS:
        return RECOMMENDATIONS[fname]
    for key, val in RECOMMENDATIONS.items():
        if fname.startswith(key + "_"):
            return val
    return (True, GENERIC_ADVICE)


# ═════════════════════════════ artifact loading ═════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    missing = [f for f in ("processed_data.pkl", "model_voting.pkl")
               if not (OUTPUTS / f).exists()]
    if missing:
        raise FileNotFoundError("Missing in ./outputs: " + ", ".join(missing))

    with open(OUTPUTS / "processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(OUTPUTS / "model_voting.pkl", "rb") as f:
        model = pickle.load(f)

    fn_path = OUTPUTS / "final_feature_names.pkl"
    if fn_path.exists():
        with open(fn_path, "rb") as f:
            feature_names = list(pickle.load(f)["feature_names"])
    else:
        feature_names = list(data["X_train"].columns)

    return data, model, feature_names


def require_keys(data, keys):
    absent = [k for k in keys if k not in data]
    if absent:
        st.error("**processed_data.pkl is missing keys the app needs:** "
                 + ", ".join(f"`{k}`" for k in absent)
                 + "\n\nApply the Cell 23 patch, re-run the notebook, then restart.")
        st.stop()


# ═════════════════════════════ preprocessing ════════════════════════════
def compute_engineered_display(raw: dict, data: dict) -> dict:
    """Recompute engineered features on ordinal-ENCODED values so explanations
    show a readable value and formula instead of a dash."""
    ordinal_cols = data["ordinal_cols"]
    out = {}
    try:
        df = pd.DataFrame([raw])
        if ordinal_cols:
            df[ordinal_cols] = data["ordinal_encoder"].transform(
                df[ordinal_cols].astype(object))

        def v(name):
            if name not in df:
                return None
            return float(pd.to_numeric(df[name], errors="coerce").iloc[0])

        specs = {
            "Turnover_x_Complexity": ("Team_Turnover_Rate", "Complexity_Score",
                                      "x", lambda a, b: a * b),
            "Pressure_x_LowMaturity": ("Schedule_Pressure", "Org_Process_Maturity",
                                       "lowmat", lambda a, b: a * (3 - b)),
            "Resource_Health": ("Resource_Availability", "Resource_Contention_Level",
                                "-", lambda a, b: a - b),
            "Compliance_x_Complexity": ("Regulatory_Compliance_Level", "Complexity_Score",
                                        "x", lambda a, b: a * b),
        }
        for feat, (c1, c2, op, fn) in specs.items():
            a, b = v(c1), v(c2)
            if a is None or b is None:
                continue
            r = fn(a, b)
            if op == "lowmat":
                out[feat] = f"{r:.2f}   ({c1}={a:g} × (3 − {c2}={b:g}))"
            else:
                out[feat] = f"{r:.2f}   ({c1}={a:g} {op} {c2}={b:g})"
    except Exception:
        pass
    return out


def build_feature_vector(raw: dict, data: dict, feature_names: list) -> pd.DataFrame:
    ordinal_cols = data["ordinal_cols"]
    nominal_cols = data["nominal_cols"]
    num_cols = data["num_cols"]
    engineered = data.get("engineered_features", [])

    df = pd.DataFrame([raw])

    # STEP 1 — ordinal encoding
    if ordinal_cols:
        df[ordinal_cols] = data["ordinal_encoder"].transform(
            df[ordinal_cols].astype(object))

    # STEP 2 — engineered features (require ordinal-encoded values)
    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df else 0.0

    if "Turnover_x_Complexity" in engineered:
        df["Turnover_x_Complexity"] = col("Team_Turnover_Rate") * col("Complexity_Score")
    if "Pressure_x_LowMaturity" in engineered:
        df["Pressure_x_LowMaturity"] = col("Schedule_Pressure") * (3 - col("Org_Process_Maturity"))
    if "Resource_Health" in engineered:
        df["Resource_Health"] = col("Resource_Availability") - col("Resource_Contention_Level")
    if "Compliance_x_Complexity" in engineered:
        df["Compliance_x_Complexity"] = col("Regulatory_Compliance_Level") * col("Complexity_Score")

    # STEP 3 — one-hot encoding
    if nominal_cols:
        arr = data["onehot_encoder"].transform(df[nominal_cols].astype(object))
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        ohe_df = pd.DataFrame(arr, columns=list(data["ohe_feature_names"]), index=df.index)
    else:
        ohe_df = pd.DataFrame(index=df.index)

    # STEP 4 — scaling (numeric columns only)
    num_block = df.reindex(columns=num_cols).astype(float).fillna(0.0)
    num_scaled = pd.DataFrame(
        data["scaler"].transform(num_block), columns=num_cols, index=df.index)

    # STEP 5 — assemble and align to the model's expected column order
    ord_block = df[ordinal_cols].astype(float) if ordinal_cols else pd.DataFrame(index=df.index)
    full = pd.concat([num_scaled, ord_block, ohe_df], axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    return full.reindex(columns=feature_names, fill_value=0.0)


def predict(X, model):
    stack_input = np.hstack([
        model["xgb_model"].predict_proba(X),
        model["lgb_model"].predict_proba(X),
        model["rf_model"].predict_proba(X),
    ])
    return model["meta_learner"].predict_proba(stack_input)


# ═════════════════════════ navigation helpers ═══════════════════════════
def init_state():
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("inputs", {})
    st.session_state.setdefault("imputed", [])
    st.session_state.setdefault("theme", "light")


def nav_buttons(last_label="Predict risk"):
    st.divider()
    c1, _, c3 = st.columns([1, 4, 1])
    with c1:
        if st.session_state.step > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    with c3:
        is_last_input = st.session_state.step == len(STEPS) - 2
        label = last_label if is_last_input else "Next →"
        kind = "primary" if is_last_input else "secondary"
        if st.button(label, use_container_width=True, type=kind):
            st.session_state.step += 1
            st.rerun()


# ═════════════════════════════════ UI ═══════════════════════════════════
def main():
    init_state()
    inject_theme(st.session_state.theme)
    colors = RISK_COLORS[st.session_state.theme]

    try:
        data, model, feature_names = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Copy the notebook's `outputs/` folder next to `app.py`, then reload.")
        st.stop()

    require_keys(data, ["industry_mode_lookup", "num_medians",
                        "industry_num_medians", "num_ranges",
                        "nominal_categories", "engineered_features"])

    RISK_LABELS = data["risk_labels"]
    INDUSTRIES = data["industries"]
    ORDINAL_ORDERS = data["ordinal_orders"]
    ordinal_cols = data["ordinal_cols"]
    nominal_cols = data["nominal_cols"]
    num_ranges = data["num_ranges"]
    nominal_categories = data["nominal_categories"]

    # ───────────────────── title + theme toggle ─────────────────────
    hcol, tcol = st.columns([9, 1])
    with hcol:
        st.title("Project Risk Prediction")
    with tcol:
        st.write("")   # nudge the button down to align with the title
        is_dark = st.session_state.theme == "dark"
        if st.button("🌙 Dark" if not is_dark else "☀️ Light",
                     use_container_width=True,
                     help="Switch between light and dark appearance"):
            st.session_state.theme = "dark" if not is_dark else "light"
            st.rerun()

    # ─────────────────────────── sidebar ────────────────────────────
    with st.sidebar:
        st.markdown("### About this tool")
        st.write(
            "This tool gives an **early warning** about how risky a project is "
            "likely to be "
        )
        st.write(
            "You describe the project by answering a few questions about it "
        )
        st.write(
            "The tool then compares estimates whether it is likely to run into "
            "**low**, **medium** or **high** trouble."
        )
        st.write(
            " You will always see which "
            "factors pushed the risk up or down, so you can decide what to fix."
        )
        st.info("This is a decision-support aid, not a guarantee. Always apply your "
                "own professional judgement.", icon="💡")

        st.divider()
        st.markdown("#### How sure must the tool be?")
        conf_threshold = st.slider(
            "Confidence threshold", 0.40, 0.90, 0.60, 0.05,
            label_visibility="collapsed")
        st.caption("Answers at or above this confidence are treated as reliable. "
                   "Anything less confident is flagged for a person to review.")

        with st.expander("What happens if I change this?"):
            st.markdown(
                "**Raise it** → the tool answers fewer projects on its own, but is "
                "right more often when it does. Safer, but more manual work.\n\n"
                "**Lower it** → it answers more projects, but gets more of them "
                "wrong. Faster, but less oversight.\n\n"
                "Measured on projects the tool had never seen:")
            st.dataframe(
                pd.DataFrame({
                    "Setting": ["0.50", "0.60", "0.70", "0.80"],
                    "Decides": ["99%", "85%", "69%", "44%"],
                    "Correct": ["78%", "82%", "85%", "93%"],
                }),
                hide_index=True, use_container_width=True)
            st.caption("*Decides* = share of projects it answers alone. "
                       "*Correct* = how often it is right on those. At 0.80 it is right "
                       "93% of the time but handles only 44% of projects.")

        st.divider()
        if st.button("Start over", use_container_width=True):
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.imputed = []
            st.rerun()

    step = st.session_state.step
    st.caption(f"Step {step + 1} of {len(STEPS)} — **{STEPS[step]}**")
    st.progress(step / (len(STEPS) - 1))

    inputs = st.session_state.inputs
    imputed = st.session_state.imputed

    num_names = list(num_ranges.keys())
    half = len(num_names) // 2
    other_nominals = [c for c in nominal_cols if c != "Project_Type"]

    def remember(name, value, was_imputed):
        inputs[name] = value
        if was_imputed and name not in imputed:
            imputed.append(name)
        if not was_imputed and name in imputed:
            imputed.remove(name)

    def numeric_widget(name, container, med_lookup):
        r = num_ranges[name]
        known = container.checkbox(f"Specify {name}", value=(name not in imputed),
                                   key=f"chk_{name}")
        if not known:
            fallback = float(med_lookup.get(name, r["median"]))
            remember(name, fallback, True)
            container.caption(f"Using typical value for this industry: {fallback:.2f}")
        else:
            v = container.number_input(
                name, min_value=float(r["min"]), max_value=float(r["max"]),
                value=float(inputs.get(name, r["median"])), key=f"num_{name}")
            remember(name, v, False)

    # ─────────────────────── STEP 0: start ──────────────────────────
    if step == 0:
        st.subheader("Which industry commissioned this project?")
        ind = st.selectbox(
            "Industry", INDUSTRIES,
            index=INDUSTRIES.index(inputs.get("Project_Type", INDUSTRIES[0])))
        inputs["Project_Type"] = ind

        st.markdown("#### Before you begin")
        st.markdown(
            "- If a question does not apply to your project, "            
            "**untick its checkbox** - or leave the dropdown on "
            f"*“{NOT_KNOWN}”*.\n"
            "- When you leave something out, the tool fills in the typical value for "
            "your industry instead, so it can still give you an answer.\n"
            "- Every field you skip is listed with your result. "
            "  The more you fill in, the more you can rely on "
            "the answer.\n"
            "- You can move **Back** and **Next** freely - nothing you enter is lost."
        )
        nav_buttons()

    else:
        industry = inputs.get("Project_Type", INDUSTRIES[0])
        mode_lookup = data["industry_mode_lookup"].get(industry, {})
        med_lookup = data["industry_num_medians"].get(industry, data["num_medians"])

        # ───────────────── STEPS 1 & 2: numeric inputs ───────────────
        if step in (1, 2):
            st.caption("Untick any item that does not apply to this project.")
            subset = num_names[:half] if step == 1 else num_names[half:]
            c1, c2 = st.columns(2)
            for i, name in enumerate(subset):
                numeric_widget(name, c1 if i % 2 == 0 else c2, med_lookup)
            nav_buttons()

        # ───────────────────── STEP 3: ordinals ──────────────────────
        elif step == 3:
            st.caption(f"Leave any item on “{NOT_KNOWN}” if it does not apply.")
            c1, c2 = st.columns(2)
            for i, name in enumerate(ordinal_cols):
                cont = c1 if i % 2 == 0 else c2
                opts = [NOT_KNOWN] + list(ORDINAL_ORDERS[name])
                prev = inputs.get(name)
                idx = opts.index(prev) if (prev in opts and name not in imputed) else 0
                choice = cont.selectbox(name, opts, index=idx, key=f"ord_{name}")
                if choice == NOT_KNOWN:
                    remember(name, mode_lookup.get(name, ORDINAL_ORDERS[name][0]), True)
                else:
                    remember(name, choice, False)
            nav_buttons()

        # ───────────────────── STEP 4: nominals ──────────────────────
        elif step == 4:
            st.caption(f"Leave any item on “{NOT_KNOWN}” if it does not apply.")
            c1, c2 = st.columns(2)
            for i, name in enumerate(other_nominals):
                cont = c1 if i % 2 == 0 else c2
                cats = list(nominal_categories.get(name, []))
                opts = [NOT_KNOWN] + cats
                prev = inputs.get(name)
                idx = opts.index(prev) if (prev in opts and name not in imputed) else 0
                choice = cont.selectbox(name, opts, index=idx, key=f"nom_{name}")
                if choice == NOT_KNOWN:
                    remember(name, mode_lookup.get(name, cats[0] if cats else ""), True)
                else:
                    remember(name, choice, False)
            nav_buttons(last_label="Predict risk")

        # ────────────────────── STEP 5: result ───────────────────────
        elif step == 5:
            total = len(num_names) + len(ordinal_cols) + len(other_nominals)
            completeness = 1 - (len(imputed) / max(total, 1))

            try:
                X = build_feature_vector(inputs, data, feature_names)
                proba = predict(X, model)[0]
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                if st.button("← Back to inputs"):
                    st.session_state.step = 4
                    st.rerun()
                st.stop()

            cls = int(np.argmax(proba))
            label = RISK_LABELS[cls]
            conf = float(proba[cls])
            p_at_risk = float(proba[HIGH_CLS])

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f"<h2 style='color:{colors.get(label,'#888')};margin:0'>{label}</h2>",
                unsafe_allow_html=True)
            c1.caption("Predicted risk grade")
            c2.metric("Confidence", f"{conf*100:.1f}%")
            c3.metric("At-risk probability", f"{p_at_risk*100:.1f}%")
            c4.metric("Input completeness", f"{completeness*100:.0f}%")

            if conf < conf_threshold:
                st.warning(f"Confidence ({conf*100:.1f}%) is below your "
                           f"{conf_threshold*100:.0f}% setting — have a person review "
                           "this project rather than acting automatically.", icon="⚠️")
            if completeness < 0.7:
                st.warning("More than 30% of the answers were filled in "
                           "automatically. Treat this result as indicative only.",
                           icon="⚠️")
            if imputed:
                with st.expander(f"{len(imputed)} field(s) filled in automatically"):
                    st.write(", ".join(imputed))

            st.subheader("How likely is each risk level?")
            st.bar_chart(pd.DataFrame({"probability": proba}, index=RISK_LABELS))

            display_vals = dict(inputs)
            display_vals.update(compute_engineered_display(inputs, data))

            try:
                import shap
                explainer = shap.TreeExplainer(model["xgb_model"])
                sv = np.array(explainer.shap_values(X))
                vals = sv[0, :, HIGH_CLS] if sv.ndim == 3 else sv[0]
                order = np.argsort(np.abs(vals))[::-1][:8]

                st.subheader("Why this prediction?")
                rows = []
                for i in order[:5]:
                    fn = feature_names[i]
                    rows.append({
                        "Factor": fn,
                        "This project's value": display_vals.get(fn, "not applicable"),
                        "Effect on risk": "↑ increases" if vals[i] > 0 else "↓ reduces",
                        "Strength": round(float(abs(vals[i])), 4),
                        "Filled in automatically": "yes" if fn in imputed else "no",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
                st.caption(
                    f"↑ always means higher risk, whichever grade was predicted "
                    f"(here: {label}). All factors are used in the prediction; the "
                    "five strongest are shown for readability.")

                # ---------------- recommendations ----------------
                st.subheader("Recommended actions")
                drivers = [(feature_names[i], float(vals[i]))
                           for i in order if vals[i] > 0][:5]

                if not drivers:
                    st.success("Nothing is pushing this project toward higher risk. "
                               "The main factors are all protective — keep current "
                               "practice and re-check if the project changes.")
                else:
                    st.caption("Ordered by how strongly each factor raises risk.")
                    for rank, (fn, v) in enumerate(drivers, 1):
                        changeable, advice = get_recommendation(fn)
                        tag = "🔧 You can change this" if changeable else "📌 Largely fixed"
                        flag = ("  ·  ⚠️ *based on an automatically filled value*"
                                if fn in imputed else "")
                        with st.expander(f"{rank}. {fn}  —  {tag}", expanded=(rank <= 2)):
                            st.markdown(
                                f"**Current value:** `{display_vals.get(fn, 'not applicable')}`{flag}")
                            st.markdown(advice)

                    st.info("These suggestions come from patterns the tool learned in "
                            "past project data. They point to what is *associated* with "
                            "higher risk, not what is guaranteed to cause it.", icon="ℹ️")

            except Exception as exc:
                st.info(f"Explanation unavailable: {exc}")

            st.divider()
            c1, c2 = st.columns(2)
            if c1.button("← Edit answers", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
            if c2.button("Assess another project", use_container_width=True,
                         type="primary"):
                st.session_state.step = 0
                st.session_state.inputs = {}
                st.session_state.imputed = []
                st.rerun()


if __name__ == "__main__":
    main()