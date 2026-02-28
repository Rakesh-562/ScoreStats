# app/websockets/__init__.py
"""
WebSocket event handlers for real-time cricket updates.

Registers all Socket.IO event listeners onto the shared
socketio instance from app/extensions.py.

Called from app/__init__.py (the app factory) after socketio.init_app(app):

    from app.websockets import register_socket_events
    register_socket_events(socketio)
"""

from .match_socket import register_match_events


def register_socket_events(socketio):
    """
    Entry point — wire up all WebSocket namespaces/handlers.

    Add new modules here as your app grows:
        from .tournament_socket import register_tournament_events
        register_tournament_events(socketio)
    """
    register_match_events(socketio)