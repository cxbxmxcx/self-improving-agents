"""Helix Observatory: a web dashboard that consumes helix as a library.

The framework emits events and records archives; this app renders them. It
lives outside the helix package on purpose, per SPEC section 12's line that
the framework ships no UI and consumers do the rendering.
"""
