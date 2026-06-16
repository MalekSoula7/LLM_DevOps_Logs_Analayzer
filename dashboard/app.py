"""Streamlit dashboard for the DevOps Log Analyzer.

4-page UI: Classify Log | Drift Monitor | Model Performance | Pipeline History.
All data is fetched from the FastAPI backend at localhost:8000.
"""

import json
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 15

LABEL_COLORS = {
    "build_failure": "#4A90D9",
    "test_failure": "#9B59B6",
    "timeout": "#E67E22",
    "config_error": "#F1C40F",
    "infra_failure": "#E74C3C",
}

LABEL_ICONS = {
    "build_failure": "🔨",
    "test_failure": "🧪",
    "timeout": "⏱",
    "config_error": "⚙️",
    "infra_failure": "🖥️",
}

st.set_page_config(
    page_title="DevOps Log Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _api_get(path: str) -> tuple[dict | None, str | None]:
    """Send a GET request to the API.

    Args:
        path: URL path starting with '/'.

    Returns:
        Tuple of (response_json, error_message). One will be None.
    """
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API — is the FastAPI server running on port 8000?"
    except requests.exceptions.Timeout:
        return None, f"Request to {path} timed out after {REQUEST_TIMEOUT}s."
    except requests.exceptions.HTTPError as exc:
        return None, f"API error {exc.response.status_code}: {exc.response.text[:200]}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    """Send a POST request to the API.

    Args:
        path: URL path starting with '/'.
        payload: JSON-serialisable dict to send as request body.

    Returns:
        Tuple of (response_json, error_message). One will be None.
    """
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API — is the FastAPI server running on port 8000?"
    except requests.exceptions.Timeout:
        return None, f"Request to {path} timed out after {REQUEST_TIMEOUT}s."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", exc.response.text[:200])
        except Exception:
            detail = exc.response.text[:200]
        return None, f"API error {exc.response.status_code}: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _sidebar_status() -> None:
    """Render the sidebar with project title and API status indicator."""
    with st.sidebar:
        st.markdown("## 🔍 DevOps Log Analyzer")
        st.markdown("*MLOps portfolio project by Malek Soula*")
        st.divider()

        health, err = _api_get("/health")
        if health:
            dot = "🟢" if health.get("status") == "ok" else "🔴"
            model_ok = "✅" if health.get("model_loaded") else "❌"
            mlflow_ok = "✅" if health.get("mlflow_connected") else "⚠️"
            st.markdown(f"{dot} **API** reachable")
            st.markdown(f"{model_ok} Model loaded")
            st.markdown(f"{mlflow_ok} MLflow connected")
        else:
            st.markdown("🔴 **API** unreachable")
            st.caption(err)

        st.divider()
        st.markdown("**Navigate:**")
        page = st.radio(
            "Page",
            ["Classify Log", "Drift Monitor", "Model Performance", "Pipeline History"],
            label_visibility="collapsed",
        )
    return page


def page_classify() -> None:
    """Page 1: Classify a raw CI/CD log snippet."""
    st.title("🔨 Classify Log")
    st.markdown("Paste a raw CI/CD pipeline log to get instant failure classification.")

    col1, col2 = st.columns([3, 1])
    with col1:
        log_text = st.text_area(
            "Pipeline log",
            height=250,
            placeholder=(
                "Paste your CI/CD log here…\n\n"
                "Example:\nnpm ERR! code ENOENT\nnpm ERR! path /app/package.json\n"
                "Build failed with exit code 1"
            ),
        )
    with col2:
        true_label = st.selectbox(
            "True label (optional)",
            ["(unknown)", "build_failure", "test_failure", "timeout", "config_error", "infra_failure"],
        )
        st.markdown("&nbsp;")
        analyze = st.button("Analyze", type="primary", use_container_width=True)

    if analyze:
        if not log_text or len(log_text.strip()) < 10:
            st.warning("Please enter at least 10 characters of log text.")
            return

        with st.spinner("Classifying…"):
            result, err = _api_post("/classify", {"log_text": log_text})

        if err:
            st.error(err)
            return

        pred = result["predicted_label"]
        conf = result["confidence"]
        expl = result["explanation"]
        ms = result["inference_time_ms"]
        ver = result["model_version"]

        color = LABEL_COLORS.get(pred, "#888888")
        icon = LABEL_ICONS.get(pred, "❓")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""<div style="background:{color};padding:12px 20px;border-radius:8px;text-align:center">
                <span style="color:white;font-size:1.4em;font-weight:bold">{icon} {pred.replace('_',' ').title()}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Confidence", f"{conf * 100:.1f}%")
            st.progress(conf)
        with c3:
            st.metric("Inference time", f"{ms:.0f} ms")
            st.caption(f"Model: `{ver}`")

        if true_label != "(unknown)":
            match = true_label == pred
            status = "✅ Correct" if match else f"❌ Mismatch (true: {true_label})"
            st.info(f"Label comparison: **{status}**")

        st.markdown("### Explanation")
        st.info(expl)


def page_drift() -> None:
    """Page 2: Drift monitoring dashboard."""
    st.title("📊 Drift Monitor")
    st.markdown("Monitor embedding distribution shift between training and inference data.")

    col_refresh, col_simulate = st.columns(2)
    with col_refresh:
        if st.button("🔄 Refresh drift status", use_container_width=True):
            st.rerun()
    with col_simulate:
        simulate = st.button("🧪 Simulate Drift", type="primary", use_container_width=True)

    if simulate:
        with st.spinner("Running inference on drift dataset and detecting drift…"):
            result, err = _api_post("/simulate-drift", {})
        if err:
            st.error(err)
        else:
            drift = result["drift_detected"]
            msg = "DRIFT DETECTED" if drift else "NO DRIFT"
            if drift:
                st.error(f"**{msg}** — {result['drift_share'] * 100:.1f}% of features drifted")
            else:
                st.success(f"**{msg}** — distribution is stable")
            st.json(result["inference_summary"])
            st.rerun()

    st.divider()

    data, err = _api_get("/drift-report")
    if err:
        st.warning(f"No drift report available yet. {err}")
        return

    drift = data["drift_detected"]
    share = data["drift_share"]
    drifted = data["drifted_features"]
    total = data["total_features"]
    ts = data["timestamp"]

    if drift:
        st.markdown(
            '<div style="background:#E74C3C;padding:20px;border-radius:8px;text-align:center">'
            '<h2 style="color:white;margin:0">⚠️ DRIFT DETECTED</h2></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#27AE60;padding:20px;border-radius:8px;text-align:center">'
            '<h2 style="color:white;margin:0">✅ NO DRIFT</h2></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drift share", f"{share * 100:.1f}%")
    m2.metric("Drifted features", drifted)
    m3.metric("Total features", total)
    m4.metric("Last check", ts[:16].replace("T", " "))

    st.divider()
    st.markdown("### Evidently Report")
    try:
        resp = requests.get(f"{API_BASE}/drift-report/html", timeout=15)
        if resp.status_code == 200:
            components.html(resp.text, height=600, scrolling=True)
        else:
            st.info("HTML report not available yet.")
    except Exception as exc:
        st.info(f"Could not load HTML report: {exc}")


def page_performance() -> None:
    """Page 3: Model performance metrics."""
    import pandas as pd

    st.title("📈 Model Performance")

    data, err = _api_get("/metrics")
    if err:
        st.error(err)
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", data["best_model"])
    m2.metric("Weighted F1", f"{data['weighted_f1']:.4f}")
    m3.metric(
        "Train / Test split",
        f"{data['n_train']} / {data['n_test']}",
    )
    st.caption(f"Trained: {data['timestamp'][:16].replace('T', ' ')}  |  "
               f"Run ID: `{data.get('run_id', 'N/A')}`")

    st.divider()
    st.markdown("### Per-class Metrics")

    per_class = data.get("per_class", {})
    if per_class:
        rows = []
        for label, m in sorted(per_class.items()):
            rows.append({
                "Label": label,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
            })
        df = pd.DataFrame(rows).set_index("Label")
        st.dataframe(df.style.highlight_max(axis=0, color="#d4edda").highlight_min(axis=0, color="#f8d7da"), use_container_width=True)

        st.markdown("### F1 Score by Class")
        f1_data = {row["Label"]: row["F1"] for row in rows}
        st.bar_chart(f1_data)

        st.markdown("### Precision vs Recall")
        chart_df = df[["Precision", "Recall"]]
        st.bar_chart(chart_df)
    else:
        st.info("No per-class metrics available.")


def page_history() -> None:
    """Page 4: Pipeline run history."""
    import pandas as pd

    st.title("📋 Pipeline History")
    st.markdown("Last 10 inference pipeline runs.")

    if st.button("🔄 Refresh"):
        st.rerun()

    data, err = _api_get("/pipeline-status")
    if err:
        st.error(err)
        return

    runs = data.get("runs", [])
    total = data.get("total_logs_processed", 0)

    st.metric("Total logs processed", total)
    st.divider()

    if not runs:
        st.info("No inference runs yet. Run the inference pipeline first.")
        return

    rows = []
    for run in runs:
        rows.append({
            "Timestamp": run["timestamp"][:16].replace("T", " ") if run["timestamp"] else "—",
            "Logs": run["log_count"],
            "Avg Confidence": f"{run['avg_confidence']:.2%}",
            "Drift": "⚠️ YES" if run["drift_detected"] else "✅ NO",
            "Output file": run["output_file"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if runs:
        st.markdown("### Logs per run")
        counts = {r["Timestamp"]: r["Logs"] for r in rows}
        st.bar_chart(counts)


def main() -> None:
    """Entry point: render sidebar and dispatch to the selected page."""
    page = _sidebar_status()

    if page == "Classify Log":
        page_classify()
    elif page == "Drift Monitor":
        page_drift()
    elif page == "Model Performance":
        page_performance()
    elif page == "Pipeline History":
        page_history()


if __name__ == "__main__":
    main()
