import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from config.settings import settings
import os

class SQLiteClient:
    """SQLite client for managing chat history with paired messages"""
    
    def __init__(self):
        self.db_path = settings.DATABASE_PATH
        self.message_limit = settings.MAX_HISTORY_MESSAGES
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create conversation pairs table (human + AI messages paired)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    human_message TEXT NOT NULL,
                    ai_message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_session_timestamp 
                ON conversation_pairs (session_id, timestamp DESC)
            ''')
            
            # Create index for session lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_session 
                ON conversation_pairs (session_id)
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure a session exists in the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
            conn.commit()
    
    def add_conversation_pair(self, session_id: str, human_message: str, ai_message: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a conversation pair (human + AI response) to the database
        
        Args:
            session_id: Unique session identifier
            human_message: The user's query
            ai_message: The AI's response
            metadata: Optional metadata (classification, etc.)
        """
        # Ensure session exists
        self._ensure_session_exists(session_id)
        
        # Add conversation pair
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_pairs 
                (session_id, human_message, ai_message, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id, 
                    human_message, 
                    ai_message, 
                    datetime.now().isoformat(),
                    json.dumps(metadata) if metadata else None
                )
            )
            
            # Update session's updated_at
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            
            conn.commit()
        
        # Trim old pairs if exceeding limit
        self._trim_conversation_pairs(session_id)
    
    def _trim_conversation_pairs(self, session_id: str) -> None:
        """Trim conversation pairs to maintain limit"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get count of pairs for this session
            cursor.execute(
                "SELECT COUNT(*) as count FROM conversation_pairs WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            count = result['count'] if result else 0
            
            # If exceeding limit, delete oldest pairs
            if count > self.message_limit:
                # Get IDs of pairs to keep (newest)
                cursor.execute(
                    """
                    SELECT id FROM conversation_pairs 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                    """,
                    (session_id, self.message_limit)
                )
                keep_ids = [row['id'] for row in cursor.fetchall()]
                
                if keep_ids:
                    # Delete pairs not in keep_ids
                    placeholders = ','.join(['?'] * len(keep_ids))
                    cursor.execute(
                        f"""
                        DELETE FROM conversation_pairs 
                        WHERE session_id = ? AND id NOT IN ({placeholders})
                        """,
                        (session_id, *keep_ids)
                    )
                    conn.commit()
    
    def get_recent_pairs(self, session_id: str, num_pairs: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation pairs for a session
        
        Args:
            session_id: Unique session identifier
            num_pairs: Number of recent pairs to retrieve
        
        Returns:
            List of conversation pair dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    id,
                    session_id,
                    human_message,
                    ai_message,
                    timestamp,
                    metadata
                FROM conversation_pairs 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (session_id, num_pairs)
            )
            pairs = cursor.fetchall()
            # Reverse to get chronological order (oldest to newest)
            return [dict(row) for row in reversed(pairs)]
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get chat history as a list of messages (for backwards compatibility)
        
        Args:
            session_id: Unique session identifier
            limit: Number of recent messages to retrieve
        
        Returns:
            List of message dictionaries with 'type' and 'content'
        """
        pairs = self.get_recent_pairs(session_id, limit if limit else 999)
        messages = []
        for pair in pairs:
            messages.append({
                'type': 'human',
                'content': pair['human_message'],
                'timestamp': pair['timestamp']
            })
            messages.append({
                'type': 'ai',
                'content': pair['ai_message'],
                'timestamp': pair['timestamp']
            })
        return messages
    
    def get_recent_context(self, session_id: str, num_pairs: int = 3) -> str:
        """
        Get formatted conversation context for the general agent
        
        Args:
            session_id: Unique session identifier
            num_pairs: Number of recent conversation pairs to include
        
        Returns:
            Formatted string of recent conversation context
        """
        pairs = self.get_recent_pairs(session_id, num_pairs)
        if not pairs:
            return "No previous conversation history."
        
        context_parts = ["Recent conversation:"]
        for i, pair in enumerate(pairs, 1):
            context_parts.append(f"User: {pair['human_message']}")
            context_parts.append(f"Assistant: {pair['ai_message']}")
            context_parts.append("")  # Empty line between pairs
        
        return "\n".join(context_parts)
    
    def get_recent_pairs_raw(self, session_id: str, num_pairs: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation pairs as raw dictionaries"""
        return self.get_recent_pairs(session_id, num_pairs)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about a session
        
        Returns:
            Dictionary with session statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute(
                """
                SELECT created_at, updated_at 
                FROM sessions 
                WHERE session_id = ?
                """,
                (session_id,)
            )
            session_info = cursor.fetchone()
            
            if not session_info:
                return {
                    "session_id": session_id,
                    "total_pairs": 0,
                    "first_message_time": None,
                    "last_message_time": None,
                    "created_at": None,
                    "updated_at": None
                }
            
            # Get pair counts
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    MIN(timestamp) as first_time,
                    MAX(timestamp) as last_time
                FROM conversation_pairs 
                WHERE session_id = ?
                """,
                (session_id,)
            )
            stats = cursor.fetchone()
            
            return {
                "session_id": session_id,
                "total_pairs": stats['total'] if stats else 0,
                "first_message_time": stats['first_time'] if stats else None,
                "last_message_time": stats['last_time'] if stats else None,
                "created_at": session_info['created_at'],
                "updated_at": session_info['updated_at']
            }
    
    def clear_session(self, session_id: str) -> None:
        """Clear all conversation pairs for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversation_pairs WHERE session_id = ?",
                (session_id,)
            )
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its conversation pairs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
    
    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export all data for a session"""
        pairs = self.get_recent_pairs(session_id, 999)
        stats = self.get_session_stats(session_id)
        
        return {
            "session_id": session_id,
            "statistics": stats,
            "conversation_pairs": pairs
        }
    
    def import_session(self, session_data: Dict[str, Any]) -> None:
        """Import a session from exported data"""
        session_id = session_data.get('session_id')
        if not session_id:
            raise ValueError("Session ID is required for import")
        
        pairs = session_data.get('conversation_pairs', [])
        if not pairs:
            return
        
        # Clear existing pairs for this session
        self.clear_session(session_id)
        self._ensure_session_exists(session_id)
        
        # Add all pairs
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for pair in pairs:
                cursor.execute(
                    """
                    INSERT INTO conversation_pairs 
                    (session_id, human_message, ai_message, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        pair['human_message'],
                        pair['ai_message'],
                        pair['timestamp'],
                        pair.get('metadata')
                    )
                )
            conn.commit()

# Singleton instance
sqlite_client = SQLiteClient()