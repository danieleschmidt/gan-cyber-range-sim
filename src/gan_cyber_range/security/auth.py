"""Authentication and authorization management."""

import hashlib
import hmac
import secrets
import time
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import jwt


class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    STUDENT = "student"
    OBSERVER = "observer"


class Permission(Enum):
    """System permissions."""
    # Simulation permissions
    CREATE_SIMULATION = "create_simulation"
    VIEW_SIMULATION = "view_simulation"
    STOP_SIMULATION = "stop_simulation"
    DELETE_SIMULATION = "delete_simulation"
    
    # Deployment permissions
    DEPLOY_SERVICES = "deploy_services"
    SCALE_SERVICES = "scale_services"
    DELETE_SERVICES = "delete_services"
    
    # Configuration permissions
    MODIFY_CONFIG = "modify_config"
    VIEW_CONFIG = "view_config"
    
    # Monitoring permissions
    VIEW_METRICS = "view_metrics"
    VIEW_LOGS = "view_logs"
    EXPORT_DATA = "export_data"
    
    # Administrative permissions
    MANAGE_USERS = "manage_users"
    MANAGE_POLICIES = "manage_policies"
    SYSTEM_ADMIN = "system_admin"


@dataclass
class User:
    """User account."""
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "active": self.active,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """User session."""
    session_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str = ""
    active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return self.active and not self.is_expired()


class AuthenticationManager:
    """Authentication and authorization manager."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        session_timeout_hours: int = 24,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 15
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        
        # In-memory storage (would be replaced with persistent storage)
        self.users: Dict[str, User] = {}
        self.password_hashes: Dict[str, str] = {}
        self.sessions: Dict[str, Session] = {}
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        
        # Setup default role permissions
        self._setup_default_permissions()
        
        # Create default admin user
        self._create_default_admin()
    
    def _setup_default_permissions(self) -> None:
        """Setup default permissions for each role."""
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.CREATE_SIMULATION,
                Permission.VIEW_SIMULATION,
                Permission.STOP_SIMULATION,
                Permission.DELETE_SIMULATION,
                Permission.DEPLOY_SERVICES,
                Permission.SCALE_SERVICES,
                Permission.DELETE_SERVICES,
                Permission.MODIFY_CONFIG,
                Permission.VIEW_CONFIG,
                Permission.VIEW_METRICS,
                Permission.VIEW_LOGS,
                Permission.EXPORT_DATA,
                Permission.MANAGE_USERS,
                Permission.MANAGE_POLICIES,
                Permission.SYSTEM_ADMIN
            },
            UserRole.RESEARCHER: {
                Permission.CREATE_SIMULATION,
                Permission.VIEW_SIMULATION,
                Permission.STOP_SIMULATION,
                Permission.DEPLOY_SERVICES,
                Permission.SCALE_SERVICES,
                Permission.VIEW_CONFIG,
                Permission.VIEW_METRICS,
                Permission.VIEW_LOGS,
                Permission.EXPORT_DATA
            },
            UserRole.STUDENT: {
                Permission.CREATE_SIMULATION,
                Permission.VIEW_SIMULATION,
                Permission.STOP_SIMULATION,
                Permission.VIEW_CONFIG,
                Permission.VIEW_METRICS
            },
            UserRole.OBSERVER: {
                Permission.VIEW_SIMULATION,
                Permission.VIEW_CONFIG,
                Permission.VIEW_METRICS
            }
        }
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_username = "admin"
        admin_password = os.environ.get("GAN_ADMIN_PASSWORD", "change_me_immediately")
        
        if admin_username not in self.users:
            self.create_user(
                username=admin_username,
                email="admin@gan-cyber-range.local",
                password=admin_password,
                role=UserRole.ADMIN
            )
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole,
        additional_permissions: Optional[Set[Permission]] = None
    ) -> bool:
        """Create a new user."""
        if username in self.users:
            return False
        
        # Get role permissions
        permissions = self.role_permissions.get(role, set()).copy()
        
        # Add additional permissions if provided
        if additional_permissions:
            permissions.update(additional_permissions)
        
        # Create user
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=permissions
        )
        
        # Hash and store password
        password_hash = self._hash_password(password)
        
        self.users[username] = user
        self.password_hashes[username] = password_hash
        
        return True
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str = ""
    ) -> Optional[str]:
        """Authenticate user and create session."""
        # Check if account is locked
        if self._is_account_locked(username):
            return None
        
        # Check credentials
        if not self._verify_password(username, password):
            self._record_failed_login(username)
            return None
        
        # Check if user exists and is active
        user = self.users.get(username)
        if not user or not user.active:
            return None
        
        # Create session
        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            username=username,
            created_at=datetime.now(),
            expires_at=datetime.now() + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Update user last login
        user.last_login = datetime.now()
        
        # Clear failed login attempts
        if username in self.login_attempts:
            del self.login_attempts[username]
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user."""
        session = self.sessions.get(session_id)
        if not session or not session.is_valid():
            if session:
                # Clean up expired session
                del self.sessions[session_id]
            return None
        
        user = self.users.get(session.username)
        if not user or not user.active:
            return None
        
        return user
    
    def logout(self, session_id: str) -> bool:
        """Logout user by invalidating session."""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            del self.sessions[session_id]
            return True
        return False
    
    def check_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if user has permission."""
        user = self.validate_session(session_id)
        if not user:
            return False
        
        return user.has_permission(permission)
    
    def require_permission(self, session_id: str, permission: Permission) -> bool:
        """Require permission or raise exception."""
        if not self.check_permission(session_id, permission):
            raise PermissionError(f"Permission required: {permission.value}")
        return True
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """Change user password."""
        if not self._verify_password(username, old_password):
            return False
        
        if username not in self.users:
            return False
        
        # Hash new password
        new_password_hash = self._hash_password(new_password)
        self.password_hashes[username] = new_password_hash
        
        # Invalidate all user sessions
        self._invalidate_user_sessions(username)
        
        return True
    
    def reset_password(self, username: str, new_password: str) -> bool:
        """Reset user password (admin operation)."""
        if username not in self.users:
            return False
        
        # Hash new password
        new_password_hash = self._hash_password(new_password)
        self.password_hashes[username] = new_password_hash
        
        # Invalidate all user sessions
        self._invalidate_user_sessions(username)
        
        return True
    
    def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Update user role and permissions."""
        user = self.users.get(username)
        if not user:
            return False
        
        user.role = new_role
        user.permissions = self.role_permissions.get(new_role, set()).copy()
        
        # Invalidate user sessions to force re-authentication
        self._invalidate_user_sessions(username)
        
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account."""
        user = self.users.get(username)
        if not user:
            return False
        
        user.active = False
        
        # Invalidate all user sessions
        self._invalidate_user_sessions(username)
        
        return True
    
    def activate_user(self, username: str) -> bool:
        """Activate user account."""
        user = self.users.get(username)
        if not user:
            return False
        
        user.active = True
        
        # Remove from locked accounts if present
        if username in self.locked_accounts:
            del self.locked_accounts[username]
        
        return True
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        user = self.users.get(username)
        if not user:
            return None
        
        return user.to_dict()
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        return [user.to_dict() for user in self.users.values()]
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        sessions = []
        current_time = datetime.now()
        
        for session_id, session in list(self.sessions.items()):
            if session.is_expired():
                # Clean up expired session
                del self.sessions[session_id]
                continue
            
            user = self.users.get(session.username)
            sessions.append({
                "session_id": session_id,
                "username": session.username,
                "user_role": user.role.value if user else "unknown",
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "active": session.active
            })
        
        return sessions
    
    def generate_jwt_token(self, username: str, additional_claims: Optional[Dict] = None) -> str:
        """Generate JWT token for API access."""
        user = self.users.get(username)
        if not user:
            raise ValueError("User not found")
        
        payload = {
            "username": username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.session_timeout
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if user still exists and is active
            username = payload.get("username")
            user = self.users.get(username)
            if not user or not user.active:
                return None
            
            return payload
        
        except jwt.InvalidTokenError:
            return None
    
    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password against stored hash."""
        stored_hash = self.password_hashes.get(username)
        if not stored_hash:
            return False
        
        try:
            salt, hash_hex = stored_hash.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(password_hash.hex(), hash_hex)
        except ValueError:
            return False
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)
    
    def _record_failed_login(self, username: str) -> None:
        """Record failed login attempt."""
        current_time = datetime.now()
        
        if username not in self.login_attempts:
            self.login_attempts[username] = []
        
        self.login_attempts[username].append(current_time)
        
        # Clean up old attempts (older than lockout duration)
        cutoff_time = current_time - self.lockout_duration
        self.login_attempts[username] = [
            attempt for attempt in self.login_attempts[username]
            if attempt > cutoff_time
        ]
        
        # Check if account should be locked
        if len(self.login_attempts[username]) >= self.max_login_attempts:
            self.locked_accounts[username] = current_time + self.lockout_duration
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked."""
        lock_until = self.locked_accounts.get(username)
        if not lock_until:
            return False
        
        if datetime.now() > lock_until:
            # Lock expired, remove it
            del self.locked_accounts[username]
            if username in self.login_attempts:
                del self.login_attempts[username]
            return False
        
        return True
    
    def _invalidate_user_sessions(self, username: str) -> None:
        """Invalidate all sessions for a user."""
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.username == username:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count."""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)