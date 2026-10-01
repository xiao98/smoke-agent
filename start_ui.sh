#!/bin/bash
# Start the SmokeAgent demo UI: http://localhost:8501
source ~/.smoke-agent.env
source "$HOME/FDS/FDS6/bin/FDS6VARS.sh"; source "$HOME/FDS/FDS6/bin/SMV6VARS.sh"
cd ~/work/smoke-agent && source .venv/bin/activate
export PYTHONPATH=src
exec .venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
