import json
import sys
from src.crew.main_crew import MainCrew
from config.settings import settings

def display_chat_history(main_crew):
    """Display chat history for the session"""
    pairs = main_crew.get_conversation_pairs()
    if not pairs:
        print("\n📋 No chat history yet.")
        return
    
    print("\n📋 Chat History (Conversation Pairs):")
    print("-" * 50)
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. 👤 You: {pair['human_message']}")
        print(f"   🤖 AI: {pair['ai_message']}")
        print(f"   🕐 {pair['timestamp']}")
        if pair.get('metadata'):
            print(f"   📊 Metadata: {pair['metadata']}")
        print("-" * 50)

def display_session_stats(main_crew):
    """Display session statistics"""
    stats = main_crew.get_session_stats()
    print("\n📊 Session Statistics:")
    print("-" * 50)
    print(f"Session ID: {stats['session_id']}")
    print(f"Total Conversation Pairs: {stats['total_pairs']}")
    if stats['first_message_time']:
        print(f"Started: {stats['first_message_time']}")
    if stats['last_message_time']:
        print(f"Last Message: {stats['last_message_time']}")
    if stats.get('created_at'):
        print(f"Session Created: {stats['created_at']}")
    if stats.get('updated_at'):
        print(f"Session Updated: {stats['updated_at']}")
    print("-" * 50)
    
def display_context(main_crew):
    """Display current conversation context"""
    context = main_crew.get_context()
    print("\n🔄 Current Conversation Context:")
    print("-" * 50)
    print(context)
    print("-" * 50)

def export_session(main_crew):
    """Export session data"""
    try:
        export_data = main_crew.export_session()
        filename = f"session_export_{main_crew.session_id}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Session exported to {filename}")
    except Exception as e:
        print(f"❌ Error exporting session: {str(e)}")

def main():
    """Main entry point for the Multi-Agent System"""
    try:
        # Validate settings
        settings.validate_settings()
        
        # Initialize the main crew with a new session
        session_id = input("Enter session ID (press Enter for new session): ").strip()
        if not session_id:
            session_id = None
        
        main_crew = MainCrew(session_id)
        
        print("\n🤖 Multi-Agent System Initialized")
        print("=" * 50)
        print("This system can handle:")
        print("  🌤️  Weather queries")
        print("  🔍  Search queries")
        print("  💬  General queries")
        print("=" * 50)
        print("\nCommands:")
        print("  /history   - Show chat history")
        print("  /stats     - Show session statistics")
        print("  /context   - Show conversation context")
        print("  /clear     - Clear chat history")
        print("  /new       - Start a new session")
        print("  /delete    - Delete current session")
        print("  /export    - Export session data")
        print("  /help      - Show this help message")
        print("  exit/quit  - Exit the program")
        print("=" * 50)
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye! Have a great day!")
                break
            
            if user_input.lower() == '/help':
                print("\n📚 Available Commands:")
                print("  /history   - Show chat history")
                print("  /stats     - Show session statistics")
                print("  /context   - Show conversation context")
                print("  /clear     - Clear chat history")
                print("  /new       - Start a new session")
                print("  /delete    - Delete current session")
                print("  /export    - Export session data to JSON")
                print("  exit/quit  - Exit the program")
                continue
            
            if user_input.lower() == '/history':
                display_chat_history(main_crew)
                continue
            
            if user_input.lower() == '/stats':
                display_session_stats(main_crew)
                continue
            
            if user_input.lower() == '/context':
                display_context(main_crew)
                continue
            
            if user_input.lower() == '/clear':
                main_crew.clear_session()
                print("✅ Chat history cleared!")
                continue
            
            if user_input.lower() == '/delete':
                confirm = input("Are you sure you want to delete this session? (y/n): ")
                if confirm.lower() == 'y':
                    main_crew.delete_session()
                    print("✅ Session deleted!")
                    main_crew = MainCrew()
                continue
            
            if user_input.lower() == '/export':
                export_session(main_crew)
                continue
            
            if user_input.lower() == '/new':
                old_session_id = main_crew.session_id
                main_crew = MainCrew()
                print(f"✅ New session started! Session ID: {main_crew.session_id}")
                print(f"   Old session ID: {old_session_id} (still available in database)")
                continue
            
            print("\n🔄 Processing your query...")
            print("-" * 50)
            
            try:
                # Process the query
                result = main_crew.process_query(user_input)
                
                print("\n📊 Classification Result:")
                print(json.dumps(result["classification"], indent=2))
                print("\n💬 Response:")
                print(result["result"])
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                print("Please try again with a different query.")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        print("Please check your API keys and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()