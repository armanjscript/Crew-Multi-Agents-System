"""
Database migration script for managing schema changes
"""
import sqlite3
from config.settings import settings
import os

def run_migrations():
    """Run database migrations"""
    db_path = settings.DATABASE_PATH
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if sessions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        sessions_exists = cursor.fetchone() is not None
        
        if not sessions_exists:
            # Create tables
            cursor.execute('''
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE conversation_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    human_message TEXT NOT NULL,
                    ai_message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX idx_conversation_session_timestamp 
                ON conversation_pairs (session_id, timestamp DESC)
            ''')
            
            cursor.execute('''
                CREATE INDEX idx_conversation_session 
                ON conversation_pairs (session_id)
            ''')
            
            conn.commit()
            print("✅ Database tables created successfully")
        else:
            # Check if conversation_pairs table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_pairs'"
            )
            pairs_exists = cursor.fetchone() is not None
            
            if not pairs_exists:
                # Create conversation_pairs table
                cursor.execute('''
                    CREATE TABLE conversation_pairs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        human_message TEXT NOT NULL,
                        ai_message TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    )
                ''')
                
                # Create indexes
                cursor.execute('''
                    CREATE INDEX idx_conversation_session_timestamp 
                    ON conversation_pairs (session_id, timestamp DESC)
                ''')
                
                cursor.execute('''
                    CREATE INDEX idx_conversation_session 
                    ON conversation_pairs (session_id)
                ''')
                
                conn.commit()
                print("✅ Conversation pairs table created successfully")
            else:
                print("✅ Database already has all tables")

if __name__ == "__main__":
    run_migrations()