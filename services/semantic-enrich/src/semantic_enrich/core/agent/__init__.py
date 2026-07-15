"""The v2 turn pipeline: an explicit-phase orchestrator.

Five phases — triage, memory, research, verify, answer — with typed
interfaces. Triage/memory/verify ship as identity stubs here; the
research phase is the v1 tool loop extracted and cleaned. Flag-gated
(`agent_loop_impl`) so both orchestrators run side by side; neither
imports the other (enforced by an import-linter contract).
"""
from semantic_enrich.core.agent.phases import PipelineDeps as PipelineDeps
from semantic_enrich.core.agent.phases import TurnContext as TurnContext
from semantic_enrich.core.agent.pipeline import run_turn as run_turn
from semantic_enrich.core.agent.pipeline import (
    run_turn_collected as run_turn_collected,
)
