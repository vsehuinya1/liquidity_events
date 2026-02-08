"""
Binance User Data Stream WebSocket Handler
v2.3: WebSocket-authoritative exit detection
"""

import websocket
import json
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from bot_config import BINANCE_API_KEY, BINANCE_API_SECRET

RECONNECT_DELAY_INITIAL_SEC = 5
RECONNECT_DELAY_MAX_SEC = 300
LISTENKEY_REFRESH_INTERVAL_SEC = 3000  # 50 minutes (expires at 60)


@dataclass
class UserDataStreamState:
    connected: bool = False
    running: bool = False
    reconnect_attempts: int = 0
    current_reconnect_delay: float = RECONNECT_DELAY_INITIAL_SEC
    listen_key: Optional[str] = None
    last_refresh_time: float = 0.0


class BinanceUserDataStream:
    """
    Manages Binance Futures userDataStream WebSocket connection.
    Routes ORDER_TRADE_UPDATE events to executor callback.
    """
    
    def __init__(self, api_key: str, api_secret: str, order_update_callback: Callable, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.order_update_callback = order_update_callback
        self.testnet = testnet
        
        # REST client for listenKey management
        from binance.client import Client
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        self.state = UserDataStreamState()
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_lock = threading.Lock()
        
        # Threads
        self.ws_thread: Optional[threading.Thread] = None
        self.refresh_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger('UserDataStream')
    
    def _get_listen_key(self) -> str:
        """Generate a new listenKey via REST API."""
        try:
            response = self.client.futures_stream_get_listen_key()
            listen_key = response.get('listenKey')
            self.logger.info(f"Generated listenKey: {listen_key[:20]}...")
            return listen_key
        except Exception as e:
            self.logger.error(f"Failed to get listenKey: {e}")
            raise
    
    def _refresh_listen_key(self) -> bool:
        """Refresh the existing listenKey to prevent expiry."""
        if not self.state.listen_key:
            return False
        try:
            self.client.futures_stream_keepalive(self.state.listen_key)
            self.state.last_refresh_time = time.time()
            self.logger.debug("listenKey refreshed")
            return True
        except Exception as e:
            self.logger.warning(f"listenKey refresh failed: {e}")
            return False
    
    def _build_stream_url(self) -> str:
        """Build WebSocket URL with listenKey."""
        base = "wss://fstream.binance.com/ws" if not self.testnet else "wss://fstream.binance.com/ws"
        return f"{base}/{self.state.listen_key}"
    
    def _on_open(self, ws) -> None:
        """WebSocket connection opened."""
        self.logger.info("UserDataStream connected")
        self.state.connected = True
        self.state.current_reconnect_delay = RECONNECT_DELAY_INITIAL_SEC
        self.state.reconnect_attempts = 0
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket connection closed."""
        self.logger.warning(f"UserDataStream closed: {close_status_code} - {close_msg}")
        self.state.connected = False
        if self.state.running:
            self._schedule_reconnect()
    
    def _on_error(self, ws, error) -> None:
        """WebSocket error occurred."""
        self.logger.error(f"UserDataStream error: {error}")
        self.state.connected = False
    
    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            payload = json.loads(message)
            
            # Handle subscription confirmation
            if 'result' in payload:
                self.logger.info(f"Subscription confirmed: {payload}")
                return
            
            # Handle ORDER_TRADE_UPDATE
            if payload.get('e') == 'ORDER_TRADE_UPDATE':
                order_data = payload.get('o', {})
                self.logger.debug(f"ORDER_TRADE_UPDATE received: {order_data.get('s')} {order_data.get('X')}")
                
                # Route to executor callback
                if self.order_update_callback:
                    try:
                        self.order_update_callback(order_data)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}", exc_info=True)
            
            # Handle ACCOUNT_UPDATE (optional, for position tracking)
            elif payload.get('e') == 'ACCOUNT_UPDATE':
                self.logger.debug("ACCOUNT_UPDATE received")
            
        except json.JSONDecodeError:
            self.logger.error(f"JSON decode error: {message[:200]}")
        except Exception as e:
            self.logger.error(f"Message handling error: {e}", exc_info=True)
    
    def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        delay = self.state.current_reconnect_delay
        self.logger.info(f"Reconnecting in {delay}s...")
        time.sleep(delay)
        self.state.current_reconnect_delay = min(
            self.state.current_reconnect_delay * 2,
            RECONNECT_DELAY_MAX_SEC
        )
        self.state.reconnect_attempts += 1
        self.start()
    
    def _refresh_loop(self) -> None:
        """Background thread to refresh listenKey every 50 minutes."""
        while self.state.running:
            time.sleep(LISTENKEY_REFRESH_INTERVAL_SEC)
            if not self.state.running:
                break
            
            if not self._refresh_listen_key():
                self.logger.warning("listenKey refresh failed, reconnecting...")
                # Force reconnection to get new listenKey
                with self.ws_lock:
                    if self.ws:
                        try:
                            self.ws.close()
                        except:
                            pass
    
    def _run_websocket(self) -> None:
        """Run WebSocket connection in thread."""
        while self.state.running:
            try:
                with self.ws_lock:
                    if not self.state.listen_key:
                        self.state.listen_key = self._get_listen_key()
                        self.state.last_refresh_time = time.time()
                    
                    url = self._build_stream_url()
                    self.ws = websocket.WebSocketApp(
                        url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close
                    )
                
                self.logger.info(f"Connecting to {url[:50]}...")
                self.ws.run_forever(ping_interval=60, ping_timeout=10)
                
            except Exception as e:
                self.logger.error(f"WebSocket run error: {e}")
            
            if not self.state.running:
                break
            
            # Reconnect logic
            self._schedule_reconnect()
    
    def start(self) -> None:
        """Start the user data stream."""
        with self.ws_lock:
            if self.state.running:
                self.logger.warning("UserDataStream already running")
                return
            
            self.state.running = True
            
            # Get initial listenKey
            self.state.listen_key = self._get_listen_key()
            self.state.last_refresh_time = time.time()
            
            # Start WebSocket thread
            self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.ws_thread.start()
            
            # Start refresh thread
            self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self.refresh_thread.start()
            
            self.logger.info("UserDataStream started")
    
    def stop(self) -> None:
        """Stop the user data stream."""
        self.logger.info("Stopping UserDataStream...")
        self.state.running = False
        self.state.connected = False
        
        with self.ws_lock:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
        
        self.logger.info("UserDataStream stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            'connected': self.state.connected,
            'running': self.state.running,
            'reconnect_attempts': self.state.reconnect_attempts,
            'listen_key_age_sec': time.time() - self.state.last_refresh_time if self.state.last_refresh_time else 0
        }
