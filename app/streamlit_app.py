"""SmokeAgent demo UI (Streamlit).  Runs against the real backend in this repo.

    source ~/.smoke-agent.env && cd ~/work/smoke-agent && .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RUNS = ROOT / "runs"
THRESH = ROOT / "configs" / "thresholds_fr_default.yaml"

st.set_page_config(page_title="SmokeAgent", page_icon="🔥", layout="wide")

# ----------------------------------------------------------------------------- helpers

def list_runs() -> list[Path]:
    if not RUNS.is_dir():
        return []
    runs = [p for p in RUNS.iterdir() if (p / "spec.json").is_file()]
    return sorted(runs, key=lambda p: (not (p / "assumptions.json").is_file(), -p.stat().st_mtime))


def read_spec(run: Path) -> dict:
    return json.loads((run / "spec.json").read_text())


def read_progress(run: Path) -> dict:
    p = run / "progress.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def read_csv_fds(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1)


def status_of(run: Path) -> str:
    spec = read_spec(run)
    out = run / f"{spec['chid']}.out"
    if (run / "criteria.json").is_file():
        return "jugé"
    if out.is_file() and "STOP: FDS completed successfully" in out.read_text(errors="replace")[-3000:]:
        return "terminé"
    if (run / "progress.json").is_file():
        return "en cours"
    if (run / f"{spec['chid']}.fds").is_file():
        return "généré"
    return "planifié"


def render_smokeview(run: Path, chid: str, heights: list[float]) -> list[Path]:
    """Headless Smokeview rendering of visibility/temperature slices at mid-time."""
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    smv_vars = Path.home() / "FDS" / "FDS6" / "bin" / "SMV6VARS.sh"
    spec = read_spec(run)
    t_mid = spec["sim"]["t_end_s"] / 2
    lines = ["RENDERDIR", " figs"]
    for h in heights:
        for label, slug in (("SOOT VISIBILITY", "vis"), ("TEMPERATURE", "temp")):
            lines += ["UNLOADALL", "LOADSLICE", f" {label}", f" 3 {h}", "SETTIMEVAL", f" {t_mid}", "RENDERONCE", f" {slug}_z{h:.1f}_t{int(t_mid)}"]
    (run / f"{chid}.ssf").write_text("\n".join(lines) + "\n")
    cmd = f"source {smv_vars} && xvfb-run -a -s '-screen 0 1280x800x24' smokeview -runscript {chid}"
    subprocess.run(["bash", "-lc", cmd], cwd=run, capture_output=True, timeout=300)
    return sorted(figs.glob("*.png"))


def launch_workflow(requirement: str, case_dir: Path, log: list[str]) -> None:
    """Run the full agent graph in a thread; messages go to `log`."""
    from config import Config
    from main import create_foam_agent_graph, initialize_state

    cfg = Config()
    cfg.case_dir = str(case_dir)
    cfg.overwrite_case_dir = True
    cfg.max_loop = 5
    cfg.fds_omp_threads = 2
    try:
        app = create_foam_agent_graph().compile()
        state = initialize_state(requirement, cfg)
        log.append("planification…")
        result = app.invoke(state, config={"recursion_limit": cfg.recursion_limit})
        log.append(f"terminé : {result.get('workflow_status')} ({result.get('termination_reason')})")
    except Exception as exc:  # noqa: BLE001
        log.append(f"erreur : {exc}")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🔥 SmokeAgent")
st.sidebar.caption("Copilote d'ingénierie du désenfumage · FDS 6.11 · exécution locale")
page = st.sidebar.radio("", ["1 · Nouvelle étude", "2 · Suivi des calculs", "3 · Critères ASET / RSET", "4 · Rapport"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(f"**Modèle** : `{os.environ.get('FOAMAGENT_MODEL_VERSION', 'non configuré')}`  \n**Solveur** : FDS 6.11.1  \n**Études** : {len(list_runs())}")

# ----------------------------------------------------------------------------- page 1
if page.startswith("1"):
    st.header("Nouvelle étude")
    st.markdown("Décrivez le bâtiment, le foyer, les ouvertures et le chemin d'évacuation. L'assistant produit une **spécification structurée** que vous validez avant tout calcul.")
    default = ("Trois pieces en enfilade de 4 m x 4 m x 2.6 m (domaine 12 x 4 x 2.6 m), cloisons de 0.2 m avec une porte "
               "de 1 m x 2 m au centre. Feu de 500 kW a croissance rapide sur un foyer de 1 m x 0.5 m au sol dans la premiere "
               "piece. Ouverture de fuite 2 m x 0.5 m en bas du mur x = 12 m. Chemin d'evacuation a 1.8 m du centre de la piece 2 "
               "au centre de la piece 3, RSET 120 s. Maillage 0.2 m, duree 300 s. Identifiant demo_trois_pieces.")
    req = st.text_area("Description du scénario", value=default, height=160)
    col1, col2 = st.columns([1, 3])
    with col1:
        plan_btn = st.button("Produire la spécification", type="primary")
    if plan_btn:
        with st.spinner("L'assistant lit la description et consulte la base de cas NIST…"):
            from config import Config
            from fds import agent as A
            from utils import LLMService
            cfg = Config()
            state = {"config": cfg, "user_requirement": req, "llm_service": LLMService(cfg)}
            out = A.plan(state)
        if out.get("workflow_status") == "failed":
            st.error("Spécification invalide : " + "; ".join(out.get("error_logs", [])))
        else:
            st.session_state["plan"] = out
            st.session_state["req"] = req
    if "plan" in st.session_state:
        out = st.session_state["plan"]
        spec = out["scenario_spec"]
        st.success(f"Spécification `{spec['chid']}` validée (domaine, géométrie, foyer, chemins contrôlés).")
        a, b = st.columns(2)
        with a:
            st.subheader("Foyer")
            st.table(pd.DataFrame({"paramètre": ["HRR max (kW)", "croissance", "emprise (m)", "suie (g/g)", "CO (g/g)"],
                                   "valeur": [spec["fire"]["hrr_peak_kw"], spec["fire"]["growth"], str(spec["fire"]["location_xb"]),
                                              spec["fire"]["fuel"]["soot_yield"], spec["fire"]["fuel"]["co_yield"]]}))
            st.subheader("Chemins d'évacuation")
            st.table(pd.DataFrame([{"id": p["id"], "points": len(p["polyline"]), "RSET (s)": p.get("rset_s")} for p in spec["escape_paths"]]))
        with b:
            st.subheader("Géométrie")
            st.table(pd.DataFrame([{"id": o["id"], "XB": str(o["xb"]), "portes": len(o.get("holes", []))} for o in spec["building"]["obstructions"]]))
            st.subheader("Ouvertures et désenfumage")
            st.table(pd.DataFrame([{"id": o["id"], "XB": str(o["xb"]), "type": "ouverture"} for o in spec["building"]["openings"]]
                                  + [{"id": e["id"], "XB": str(e["xb"]), "type": f"extraction {e['volume_flow_m3s']} m3/s"} for e in spec["smoke_control"]["exhaust"]]))
        with st.expander("Hypothèses prises par l'assistant (à vérifier par l'ingénieur)", expanded=True):
            ass = json.loads(Path(out["case_dir"], "assumptions.json").read_text()) if Path(out["case_dir"], "assumptions.json").is_file() else []
            for s_ in ass:
                st.markdown(f"- {s_}")
        with st.expander("Spécification complète (JSON)"):
            st.json(spec)
        if st.button("Générer le cas FDS et lancer le calcul", type="primary"):
            log: list[str] = []
            st.session_state["log"] = log
            th = threading.Thread(target=launch_workflow, args=(st.session_state["req"], Path(out["case_dir"]), log), daemon=True)
            th.start()
            st.info("Calcul lancé en arrière-plan. Suivez-le dans la page 2.")

# ----------------------------------------------------------------------------- page 2
elif page.startswith("2"):
    st.header("Suivi des calculs")
    runs = list_runs()
    if not runs:
        st.info("Aucune étude.")
    rows = []
    for r in runs:
        spec = read_spec(r)
        prog = read_progress(r)
        stt = status_of(r)
        t_sim = spec["sim"]["t_end_s"] if stt in ("terminé", "jugé") else prog.get("sim_time_s", 0)
        rows.append({"étude": r.name, "statut": stt, "t simulé (s)": round(t_sim),
                     "durée (s)": spec["sim"]["t_end_s"], "HRR (kW)": spec["fire"]["hrr_peak_kw"], "chemins": len(spec["escape_paths"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        pick = st.selectbox("Étude", [r.name for r in runs])
        run = RUNS / pick
        spec = read_spec(run)
        prog = read_progress(run)
        t_sim = spec["sim"]["t_end_s"] if status_of(run) in ("terminé", "jugé") else prog.get("sim_time_s", 0)
        st.progress(min(1.0, t_sim / spec["sim"]["t_end_s"]), text=f"{round(t_sim)} s / {spec['sim']['t_end_s']} s")
        hrr = run / f"{spec['chid']}_hrr.csv"
        if hrr.is_file():
            df = read_csv_fds(hrr)
            st.subheader("Puissance du foyer (HRR)")
            st.line_chart(df.set_index("Time")[["HRR"]])
        out = run / f"{spec['chid']}.out"
        if out.is_file():
            with st.expander("Journal FDS (fin)"):
                st.code(out.read_text(errors="replace")[-2500:])
        if st.button("Actualiser"):
            st.rerun()

# ----------------------------------------------------------------------------- page 3
elif page.startswith("3"):
    st.header("Critères de tenabilité : ASET / RSET")
    runs = [r for r in list_runs() if (r / f"{read_spec(r)['chid']}_devc.csv").is_file()]
    if not runs:
        st.info("Aucun calcul terminé.")
    else:
        pick = st.selectbox("Étude", [r.name for r in runs])
        run = RUNS / pick
        from fds import criteria as C
        from fds.spec import ScenarioSpec
        spec = ScenarioSpec.model_validate(read_spec(run))
        th = C.load_thresholds(THRESH)
        res = C.evaluate(run, spec, th)
        st.caption(f"Profil de seuils `{th['profile']}` · hauteur d'évaluation {th['eval_height_m']} m · lissage {th['smoothing_s']} s")
        rows = []
        for p in res.paths:
            rows.append({"chemin": p.id, "RSET (s)": p.rset_s, "ASET (s)": round(p.aset_s), "marge ASET/RSET": None if p.margin is None else round(p.margin, 2),
                         "verdict": "✅ conforme" if p.passed else ("❌ non conforme" if p.passed is False else "— (RSET absent)"),
                         **{f"{q} (s)": (None if r.first_exceed_s is None else round(r.first_exceed_s)) for q, r in p.quantities.items()}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**Seuils appliqués**")
        st.dataframe(pd.DataFrame([{"quantité": q, "critère": f"{c['op']} {c['value']} {c['unit']}", "source": c["source"]} for q, c in th["quantities"].items()]), use_container_width=True, hide_index=True)
        hrr = res.hrr_check
        st.markdown(f"Contrôle du foyer : HRR simulé **{hrr.peak_kw_sim:.0f} kW** pour **{hrr.peak_kw_set:.0f} kW** attendu → {'ok' if hrr.ok else 'écart'}")
        # curves
        t, data = C.read_devc(run / f"{spec.chid}_devc.csv")
        for p in spec.escape_paths:
            st.subheader(f"Chemin {p.id} : évolution le long du parcours")
            pts = C._samples(p.polyline)
            cols = st.columns(4)
            for col, (q, suffix, unit) in zip(cols, [("visibility", "VIS", "m"), ("temperature", "T", "°C"), ("co", "CO", "ppm"), ("layer_height", "LH", "m")]):
                series = {}
                for i in range(len(pts)):
                    cid = C.device_id(p.id, i, suffix)
                    if cid in data:
                        series[f"pt {i}"] = [v * 1e6 if q == "co" else v for v in data[cid]]
                if series:
                    df = pd.DataFrame(series, index=t)
                    col.caption(f"{q} ({unit}) · seuil {th['quantities'][q]['op']} {th['quantities'][q]['value']}")
                    col.line_chart(df)
        if st.button("Rendre les coupes Smokeview"):
            with st.spinner("Smokeview…"):
                figs = render_smokeview(run, spec.chid, sorted({round(p.polyline[0][2], 1) for p in spec.escape_paths}))
            for f in figs:
                st.image(str(f), caption=f.name)
        else:
            figs = sorted((run / "figs").glob("*.png")) if (run / "figs").is_dir() else []
            for f in figs:
                st.image(str(f), caption=f.name, width=600)

# ----------------------------------------------------------------------------- page 4
else:
    st.header("Rapport")
    runs = [r for r in list_runs() if (r / "criteria.json").is_file() or (r / f"{read_spec(r)['chid']}_devc.csv").is_file()]
    if not runs:
        st.info("Aucune étude jugée.")
    else:
        pick = st.selectbox("Étude", [r.name for r in runs])
        run = RUNS / pick
        from fds import criteria as C
        from fds.spec import ScenarioSpec
        spec = ScenarioSpec.model_validate(read_spec(run))
        th = C.load_thresholds(THRESH)
        res = C.evaluate(run, spec, th)
        ass = json.loads((run / "assumptions.json").read_text()) if (run / "assumptions.json").is_file() else []
        md = [f"# Étude d'ingénierie du désenfumage — {spec.chid}", "", "## 1. Scénarios d'incendie", "",
              f"Foyer de {spec.fire.hrr_peak_kw:.0f} kW, croissance {spec.fire.growth}, emprise {spec.fire.location_xb}.", "",
              "Hypothèses :", *[f"- {a}" for a in ass], "", "## 2. Modèle et hypothèses de calcul", "",
              f"FDS 6.11.1, domaine {spec.building.domain.xb}, durée {spec.sim.t_end_s:.0f} s, maille {spec.mesh.cell_size} m.", "",
              "## 3. Critères d'évaluation", "", "| quantité | critère | source |", "| --- | --- | --- |",
              *[f"| {q} | {c['op']} {c['value']} {c['unit']} | {c['source']} |" for q, c in th["quantities"].items()], "",
              "## 4. Résultats", "", "| chemin | RSET (s) | ASET (s) | marge | verdict |", "| --- | --- | --- | --- | --- |",
              *[f"| {p.id} | {p.rset_s} | {p.aset_s:.0f} | {'' if p.margin is None else round(p.margin, 2)} | {'conforme' if p.passed else 'non conforme'} |" for p in res.paths], "",
              "## 5. Conclusion", "", "(rédigée par l'ingénieur à partir du tableau ci-dessus)", ""]
        text = "\n".join(md)
        st.markdown(text)
        st.download_button("Télécharger le rapport (Markdown)", text, file_name=f"{spec.chid}_rapport.md")
        if st.button("Exporter en DOCX (pandoc)"):
            md_path = run / f"{spec.chid}_rapport.md"
            md_path.write_text(text, encoding="utf-8")
            docx = run / f"{spec.chid}_rapport.docx"
            subprocess.run(["pandoc", str(md_path), "-o", str(docx)], check=False)
            if docx.is_file():
                st.download_button("Télécharger le DOCX", docx.read_bytes(), file_name=docx.name)
