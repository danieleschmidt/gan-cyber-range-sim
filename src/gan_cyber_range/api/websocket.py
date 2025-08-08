"""
Real-time WebSocket collaboration features for GAN Cyber Range.

Provides real-time updates for:
- Simulation progress and metrics
- Agent actions and responses  
- Security events and incidents
- Collaborative analysis sessions
- Live chat and annotations
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # Simulation events
    SIMULATION_UPDATE = "simulation_update"
    AGENT_ACTION = "agent_action"
    SECURITY_EVENT = "security_event"
    METRICS_UPDATE = "metrics_update"
    
    # Collaboration events
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CHAT_MESSAGE = "chat_message"
    ANNOTATION = "annotation"
    
    # Control events
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    simulation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "simulation_id": self.simulation_id
        }


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Simulation subscriptions: simulation_id -> set of user_ids
        self.simulation_subscribers: Dict[str, Set[str]] = {}
        
        # User metadata
        self.user_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Chat rooms: simulation_id -> list of messages
        self.chat_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Annotations: simulation_id -> list of annotations
        self.annotations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Connection stats
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0
        }
    
    async def connect(self, websocket: WebSocket, user_id: str, user_metadata: Dict[str, Any] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        self.active_connections[user_id] = websocket
        self.user_metadata[user_id] = user_metadata or {}
        
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] = len(self.active_connections)
        
        logger.info(f"User {user_id} connected. Active connections: {len(self.active_connections)}")
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            type=MessageType.USER_JOINED,
            data={
                "user_id": user_id,
                "message": "Connected to GAN Cyber Range real-time collaboration",
                "stats": self.connection_stats
            },
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        await self.send_personal_message(welcome_message, user_id)
    
    def disconnect(self, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            
        if user_id in self.user_metadata:
            del self.user_metadata[user_id]
        
        # Remove from all simulation subscriptions
        for simulation_id in list(self.simulation_subscribers.keys()):
            self.simulation_subscribers[simulation_id].discard(user_id)
            if not self.simulation_subscribers[simulation_id]:
                del self.simulation_subscribers[simulation_id]
        
        self.connection_stats["active_connections"] = len(self.active_connections)
        
        logger.info(f"User {user_id} disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: WebSocketMessage, user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message.to_dict()))
                self.connection_stats["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast_to_simulation(self, message: WebSocketMessage, simulation_id: str):
        """Broadcast a message to all users subscribed to a simulation."""
        if simulation_id in self.simulation_subscribers:
            subscribers = self.simulation_subscribers[simulation_id].copy()
            
            for user_id in subscribers:
                await self.send_personal_message(message, user_id)
    
    async def broadcast_to_all(self, message: WebSocketMessage):
        """Broadcast a message to all connected users."""
        for user_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, user_id)
    
    def subscribe_to_simulation(self, user_id: str, simulation_id: str):
        """Subscribe a user to simulation updates."""
        if simulation_id not in self.simulation_subscribers:
            self.simulation_subscribers[simulation_id] = set()
        
        self.simulation_subscribers[simulation_id].add(user_id)
        
        logger.info(f"User {user_id} subscribed to simulation {simulation_id}")
    
    def unsubscribe_from_simulation(self, user_id: str, simulation_id: str):
        """Unsubscribe a user from simulation updates."""
        if simulation_id in self.simulation_subscribers:
            self.simulation_subscribers[simulation_id].discard(user_id)
            
            if not self.simulation_subscribers[simulation_id]:
                del self.simulation_subscribers[simulation_id]
        
        logger.info(f"User {user_id} unsubscribed from simulation {simulation_id}")
    
    async def handle_chat_message(self, user_id: str, simulation_id: str, message: str):
        """Handle chat message and broadcast to simulation subscribers."""
        chat_message = {
            "user_id": user_id,
            "username": self.user_metadata.get(user_id, {}).get("username", user_id),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in chat history
        if simulation_id not in self.chat_history:
            self.chat_history[simulation_id] = []
        
        self.chat_history[simulation_id].append(chat_message)
        
        # Keep only last 100 messages
        if len(self.chat_history[simulation_id]) > 100:
            self.chat_history[simulation_id] = self.chat_history[simulation_id][-100:]
        
        # Broadcast to simulation subscribers
        ws_message = WebSocketMessage(
            type=MessageType.CHAT_MESSAGE,
            data=chat_message,
            timestamp=datetime.now(),
            user_id=user_id,
            simulation_id=simulation_id
        )
        
        await self.broadcast_to_simulation(ws_message, simulation_id)
    
    async def handle_annotation(self, user_id: str, simulation_id: str, annotation: Dict[str, Any]):
        """Handle annotation and broadcast to simulation subscribers."""
        annotation_data = {
            "id": f"annotation_{datetime.now().timestamp()}",
            "user_id": user_id,
            "username": self.user_metadata.get(user_id, {}).get("username", user_id),
            "timestamp": datetime.now().isoformat(),
            **annotation
        }
        
        # Store annotation
        if simulation_id not in self.annotations:
            self.annotations[simulation_id] = []
        
        self.annotations[simulation_id].append(annotation_data)
        
        # Broadcast annotation
        ws_message = WebSocketMessage(
            type=MessageType.ANNOTATION,
            data=annotation_data,
            timestamp=datetime.now(),
            user_id=user_id,
            simulation_id=simulation_id
        )
        
        await self.broadcast_to_simulation(ws_message, simulation_id)
    
    def get_chat_history(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a simulation."""
        return self.chat_history.get(simulation_id, [])
    
    def get_annotations(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Get annotations for a simulation."""
        return self.annotations.get(simulation_id, [])
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.connection_stats,
            "simulations_active": len(self.simulation_subscribers),
            "users_by_simulation": {
                sim_id: len(users) 
                for sim_id, users in self.simulation_subscribers.items()
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()


class WebSocketHandler:
    """Handles WebSocket message processing."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
    
    async def handle_message(self, websocket: WebSocket, user_id: str, raw_message: str):
        """Process incoming WebSocket message."""
        try:
            message_data = json.loads(raw_message)
            message_type = MessageType(message_data.get("type"))
            
            self.manager.connection_stats["messages_received"] += 1
            
            if message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(user_id, message_data)
            
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(user_id, message_data)
            
            elif message_type == MessageType.CHAT_MESSAGE:
                await self._handle_chat_message(user_id, message_data)
            
            elif message_type == MessageType.ANNOTATION:
                await self._handle_annotation(user_id, message_data)
            
            elif message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(user_id)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                data={
                    "error": str(e),
                    "original_message": raw_message[:100]  # Truncate for safety
                },
                timestamp=datetime.now(),
                user_id=user_id
            )
            
            await self.manager.send_personal_message(error_message, user_id)
    
    async def _handle_subscribe(self, user_id: str, message_data: Dict[str, Any]):
        """Handle subscription request."""
        simulation_id = message_data.get("simulation_id")
        
        if simulation_id:
            self.manager.subscribe_to_simulation(user_id, simulation_id)
            
            # Send chat history and annotations
            response_message = WebSocketMessage(
                type=MessageType.SIMULATION_UPDATE,
                data={
                    "action": "subscribed",
                    "simulation_id": simulation_id,
                    "chat_history": self.manager.get_chat_history(simulation_id),
                    "annotations": self.manager.get_annotations(simulation_id)
                },
                timestamp=datetime.now(),
                user_id=user_id,
                simulation_id=simulation_id
            )
            
            await self.manager.send_personal_message(response_message, user_id)
    
    async def _handle_unsubscribe(self, user_id: str, message_data: Dict[str, Any]):
        """Handle unsubscription request."""
        simulation_id = message_data.get("simulation_id")
        
        if simulation_id:
            self.manager.unsubscribe_from_simulation(user_id, simulation_id)
            
            response_message = WebSocketMessage(
                type=MessageType.SIMULATION_UPDATE,
                data={
                    "action": "unsubscribed",
                    "simulation_id": simulation_id
                },
                timestamp=datetime.now(),
                user_id=user_id,
                simulation_id=simulation_id
            )
            
            await self.manager.send_personal_message(response_message, user_id)
    
    async def _handle_chat_message(self, user_id: str, message_data: Dict[str, Any]):
        """Handle chat message."""
        simulation_id = message_data.get("simulation_id")
        message = message_data.get("message", "")
        
        if simulation_id and message:
            await self.manager.handle_chat_message(user_id, simulation_id, message)
    
    async def _handle_annotation(self, user_id: str, message_data: Dict[str, Any]):
        """Handle annotation."""
        simulation_id = message_data.get("simulation_id")
        annotation = message_data.get("annotation", {})
        
        if simulation_id and annotation:
            await self.manager.handle_annotation(user_id, simulation_id, annotation)
    
    async def _handle_heartbeat(self, user_id: str):
        """Handle heartbeat message."""
        response_message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={
                "status": "alive",
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        await self.manager.send_personal_message(response_message, user_id)


# Global WebSocket handler
websocket_handler = WebSocketHandler(connection_manager)


class SimulationBroadcaster:
    """Broadcasts simulation events to connected users."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
    
    async def broadcast_simulation_update(self, simulation_id: str, update_data: Dict[str, Any]):
        """Broadcast simulation status update."""
        message = WebSocketMessage(
            type=MessageType.SIMULATION_UPDATE,
            data=update_data,
            timestamp=datetime.now(),
            simulation_id=simulation_id
        )
        
        await self.manager.broadcast_to_simulation(message, simulation_id)
    
    async def broadcast_agent_action(self, simulation_id: str, agent_type: str, action_data: Dict[str, Any]):
        """Broadcast agent action."""
        message = WebSocketMessage(
            type=MessageType.AGENT_ACTION,
            data={
                "agent_type": agent_type,
                "action": action_data
            },
            timestamp=datetime.now(),
            simulation_id=simulation_id
        )
        
        await self.manager.broadcast_to_simulation(message, simulation_id)
    
    async def broadcast_security_event(self, simulation_id: str, event_data: Dict[str, Any]):
        """Broadcast security event."""
        message = WebSocketMessage(
            type=MessageType.SECURITY_EVENT,
            data=event_data,
            timestamp=datetime.now(),
            simulation_id=simulation_id
        )
        
        await self.manager.broadcast_to_simulation(message, simulation_id)
    
    async def broadcast_metrics_update(self, simulation_id: str, metrics_data: Dict[str, Any]):
        """Broadcast metrics update."""
        message = WebSocketMessage(
            type=MessageType.METRICS_UPDATE,
            data=metrics_data,
            timestamp=datetime.now(),
            simulation_id=simulation_id
        )
        
        await self.manager.broadcast_to_simulation(message, simulation_id)


# Global simulation broadcaster
simulation_broadcaster = SimulationBroadcaster(connection_manager)


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Main WebSocket endpoint handler."""
    try:
        # Extract user metadata from query parameters or headers
        user_metadata = {
            "username": websocket.query_params.get("username", user_id),
            "role": websocket.query_params.get("role", "observer"),
            "connected_at": datetime.now().isoformat()
        }
        
        await connection_manager.connect(websocket, user_id, user_metadata)
        
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                await websocket_handler.handle_message(websocket, user_id, data)
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                heartbeat_message = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"status": "heartbeat"},
                    timestamp=datetime.now(),
                    user_id=user_id
                )
                await connection_manager.send_personal_message(heartbeat_message, user_id)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(user_id)