"""MapleQuery agent HTTP surface.

Wraps the semantic-enrich 5.1 loop as a FastAPI app. Almost no business
logic lives here — the routes marshal HTTP↔loop and add auth, CORS,
and SSE framing.
"""
