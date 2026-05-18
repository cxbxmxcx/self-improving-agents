"""Helix dashboard.

A Streamlit-based explorer for the archive, trajectories, and round logs
produced by Improver-driven runs. Mines the SQLite archive for artifacts,
measurements, lineage, and cached trajectories; renders them as an
interactive multi-page app so readers can see what a self-improving agent
actually did, without grepping the logs.

Run via:
    streamlit run helix/dashboard/app.py

The sidebar points at a `helix_archive.sqlite` file; switching paths lets
you compare runs from different chapter scripts side by side.
"""
