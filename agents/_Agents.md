# Agent Development Guide for Artemis AI

## 🎯 Overview

Artemis uses a sophisticated multi-agent architecture where specialized agents run in parallel to enhance conversations with contextual information. Each agent is triggered by different conditions and contributes unique capabilities to the system. This guide will help you understand and create new agents that seamlessly integrate with the Artemis ecosystem.

## 🏗️ Architecture Deep Dive

### Core Components

```
ArtemisCore
├── Agent Initialization (parallel)
├── Agent Processing (concurrent via ThreadPoolExecutor)
├── Context Aggregation (formatted injection)
└── Response Generation (streaming with metadata)
```

### Agent Lifecycle

1. **Initialization**: Agents are created with user context and LLM interface
2. **Parallel Processing**: All agents process input simultaneously via `asyncio.to_thread`
3. **Context Injection**: Agent outputs are formatted and injected into LLM context
4. **Metadata Collection**: Agent metadata is collected for UI display

### Base Agent Class

```python
class Agent:
    def __init__(self, name: Optional[str] = None, user: Optional[str] = None) -> None:
        self.name = name
        self.user = user
        self.llm = LLMInterface(_config.agent_llm)  # Dedicated LLM for agents
        self.metadata: Dict[str, Any] = {}
    
    def process(self, user_input: str, last_assistant_response: Optional[str] = None) -> Optional[str]:
        """Core method - MUST be overridden by all agents"""
        self.metadata = {}  # Always reset at start
        raise NotImplementedError("Subclasses must implement the 'process' method")
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
```

## 🚀 Creating Your First Agent

### 1. Basic Agent Template

```python
import logging
from typing import Optional, Dict, Any
from .Agent import Agent

logger = logging.getLogger(__name__)

class YourNewAgent(Agent):
    """Agent for [describe specific purpose].
    
    Triggered by: [describe trigger conditions]
    Provides: [describe what information/capabilities it adds]
    """
    
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        # Initialize any additional state here
        
    def process(self, user_input: str, last_assistant_response: Optional[str] = None) -> Optional[str]:
        """Process user input and return contextual information."""
        # ALWAYS reset metadata first
        self.metadata = {}
        
        # Determine if this agent should respond
        if not self._should_activate(user_input):
            return None
            
        try:
            # Core agent logic here
            result = self._perform_core_function(user_input)
            
            # Update metadata with processing info
            self.metadata = {
                "processed": True,
                "result_count": len(result) if result else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return None
    
    def _should_activate(self, user_input: str) -> bool:
        """Determine if agent should process this input."""
        # Implement your activation logic
        return True
        
    def _perform_core_function(self, user_input: str) -> str:
        """Implement your agent's core functionality."""
        # Your agent logic here
        return "Your agent output"
```

### 2. Add to Core System

In `core.py`, add your agent to the initialization:

```python
agent_configs = [
    (PersonalInfoAgent, "Personal Info", self.user),
    (DetectLanguageAgent, "Language Detection", self.user),
    (OnlineSearchAgent, "Online Research", self.user),
    (URLReaderAgent, "URL Reader", self.user),
    (FileReaderAgent, "File Reader", self.user),
    (DailyStoriesAgent, "Daily News", self.user),
    (YourNewAgent, "Your Agent Name", self.user),  # Add here
]
```

## 🎯 Agent Activation Patterns

Understanding how existing agents activate will help you design effective triggers:

### 1. Command-Based Activation
```python
# Example: DailyStoriesAgent
def process(self, user_input: str, last_assistant_response: Optional[str] = None) -> Optional[str]:
    if '!news' in user_input.lower():
        return self._fetch_news()
    elif '!games' in user_input.lower():
        return self._fetch_gaming_news()
    return None
```

### 2. Pattern Detection
```python
# Example: FileReaderAgent
def extract_filenames(self, text: str) -> List[str]:
    # Match filenames that start with / or ~/
    filename_pattern = r'(?<![:/])(?:\'|")?((?:/|~/)(?:[^\'"\s/:]+/?)+)(?:\'|")?'
    matches = re.findall(filename_pattern, text)
    return [match for match in matches if os.path.exists(os.path.expanduser(match.strip("'\"")))]
```

### 3. LLM-Powered Decision Making
```python
# Example: OnlineSearchAgent
def should_perform_search(self, user_input: str, last_assistant_response: Optional[str] = None) -> bool:
    prompt = f"""
    Determine if this user input REQUIRES retrieving information from the internet.
    USER INPUT: "{user_input}"
    
    RESPOND "YES" if the query requests current facts, news, or verifiable information.
    RESPOND "NO" if it's conversational or can be answered with general knowledge.
    
    Respond with ONLY "YES" or "NO".
    """
    
    response = self.llm.generate_single_response(prompt, max_tokens=50).strip().upper()
    return "YES" in response
```

### 4. Always-On Processing
```python
# Example: PersonalInfoAgent, DetectLanguageAgent
def process(self, user_input: str, last_assistant_response: Optional[str] = None) -> str:
    # These agents process every input but may return minimal context
    self.metadata = {}
    
    # Always perform some processing
    result = self._analyze_input(user_input)
    
    return result  # Never returns None
```

## 🔧 Advanced Agent Patterns

### 1. Concurrent Processing with ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ConcurrentAgent(Agent):
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def process_multiple_items(self, items: List[str]) -> List[Dict]:
        futures = [self.executor.submit(self._process_item, item) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
        
        return results
```

### 2. Caching for Performance

```python
import time
from typing import Tuple, Any

class CachedAgent(Agent):
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.cache_duration = 3600  # 1 hour
    
    def get_cached_or_fetch(self, key: str) -> Any:
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
        
        # Fetch fresh data
        data = self._fetch_data(key)
        self.cache[key] = (time.time(), data)
        return data
```

### 3. User-Specific Data Persistence

```python
import json
import hashlib
from pathlib import Path

class DataPersistentAgent(Agent):
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        
        # Create user-specific file path
        if user:
            hashed_user = hashlib.sha1(str(user).encode()).hexdigest()
            self.data_file = Path.cwd() / "data" / f".{self.name.lower()}_{hashed_user}.json"
        else:
            self.data_file = Path.cwd() / "data" / f".{self.name.lower()}.json"
            
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        self.load_data()
    
    def save_data(self) -> None:
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.user_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def load_data(self) -> None:
        try:
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self.user_data = json.load(f)
            else:
                self.user_data = {}
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.user_data = {}
```

### 4. External API Integration

```python
import requests
from requests.exceptions import RequestException, Timeout

class APIAgent(Agent):
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.base_url = "https://api.example.com"
    
    def make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except (RequestException, Timeout) as e:
            logger.error(f"API request failed: {str(e)}")
            return None
```

## 📊 Metadata Best Practices

Metadata provides valuable information to the UI and other system components:

```python
def process(self, user_input: str, last_assistant_response: Optional[str] = None) -> Optional[str]:
    self.metadata = {}  # Always reset
    
    # Process your logic here
    results = self._do_processing(user_input)
    
    # Set comprehensive metadata
    self.metadata = {
        "processed": True,
        "timestamp": datetime.now().isoformat(),
        "input_length": len(user_input),
        "results_count": len(results) if results else 0,
        "processing_time_ms": processing_time,
        "sources": [result.get('source') for result in results],
        "urls": [result.get('url') for result in results if result.get('url')],
        "confidence": 0.95,  # If applicable
        "cached": False,  # If using caching
    }
    
    return formatted_results
```

## 🔄 Integration with LLM Interface

All agents have access to a shared LLM interface optimized for agent tasks:

```python
# Available methods in self.llm:

# Generate single response
response = self.llm.generate_single_response(
    prompt="Your prompt here",
    max_tokens=500,
    temperature=0.3  # Optional
)

# Summarize content
summary = self.llm.summarize(
    text="Long content to summarize",
    max_words=200
)

# Async versions (if needed)
response = await self.llm.generate_single_response_async(prompt)
summary = await self.llm.summarize_async(text)
```

## 🎨 Output Formatting

Format your agent output for optimal integration:

```python
def create_context(self, results: List[Dict]) -> str:
    """Format results for context injection."""
    if not results:
        return ""
    
    context_parts = []
    
    for result in results:
        context_parts.append(f"Title: {result['title']}")
        context_parts.append(f"Source: {result['source']}")
        context_parts.append(f"Content: {result['content']}")
        context_parts.append("")  # Empty line separator
    
    return "\n".join(context_parts).strip()
```

## 🎯 Agent Types and Examples

### Information Retrieval Agents
- **OnlineSearchAgent**: Web search with DuckDuckGo
- **URLReaderAgent**: Extract content from URLs
- **FileReaderAgent**: Read various file formats
- **NewsAgent**: RSS feed aggregation

### Context Enhancement Agents
- **PersonalInfoAgent**: User preference tracking
- **DetectLanguageAgent**: Language detection and enforcement

### Action-Oriented Agents
- **HueLightsAgent**: Smart home device control

### Your Agent Ideas
Consider these types for new agents:
- **Email/Calendar Integration**: Read emails, calendar events
- **Code Analysis**: Analyze code repositories, suggest improvements
- **Document Processing**: Extract structured data from documents
- **Social Media**: Monitor specific hashtags or accounts
- **Weather/Location**: Location-based services
- **Translation**: Advanced translation with context
- **Task Management**: Integration with task management systems

## 🧪 Testing Your Agent

Create a simple test to verify your agent:

```python
# test_your_agent.py
import sys
sys.path.append('..')

from agents.YourNewAgent import YourNewAgent

def test_agent():
    agent = YourNewAgent("Test Agent", "test_user")
    
    # Test activation
    result = agent.process("test input that should trigger your agent")
    print(f"Result: {result}")
    print(f"Metadata: {agent.get_metadata()}")
    
    # Test non-activation
    result = agent.process("random text that shouldn't trigger")
    print(f"Non-trigger result: {result}")

if __name__ == "__main__":
    test_agent()
```

## 🚀 Performance Considerations

1. **Parallel Processing**: Agents run concurrently, so avoid blocking operations
2. **Timeout Handling**: Set reasonable timeouts for external calls
3. **Error Handling**: Always handle exceptions gracefully
4. **Caching**: Cache expensive operations when possible
5. **Resource Cleanup**: Clean up resources in destructors if needed

## 📝 Code Quality Guidelines

1. **Type Hints**: Use comprehensive type hints
2. **Docstrings**: Document all methods with Google-style docstrings
3. **Logging**: Use structured logging with appropriate levels
4. **Error Messages**: Provide helpful error messages for debugging
5. **Constants**: Define constants at module level
6. **Validation**: Validate inputs and outputs

## 🎯 Trigger Phrase Examples

When documenting your agent, specify clear trigger examples:

### Command-Based
- `!news` - Fetch latest news
- `!weather munich` - Get weather for Munich
- `!translate hello to spanish` - Translation request

### Content-Based
- `https://example.com` - URL detection
- `/home/user/file.txt` - File path detection
- `john.doe@email.com` - Email detection

### Semantic
- "What's happening in AI today?" - Current events query
- "I like jazz music" - Personal preference extraction
- "Turn on the living room lights" - Smart home command

## 🔗 Integration Checklist

Before deploying your agent:

- [ ] Inherits from `Agent` base class
- [ ] Implements `process()` method that resets metadata
- [ ] Returns `None` when not applicable
- [ ] Returns formatted string when applicable
- [ ] Sets comprehensive metadata
- [ ] Handles exceptions gracefully
- [ ] Added to `core.py` agent configuration
- [ ] Tested with various inputs
- [ ] Documented trigger conditions
- [ ] Performance optimized

## 📚 Reference Implementations

Study these agents for specific patterns:

- **PersonalInfoAgent**: Complex LLM-powered information extraction and encryption
- **OnlineSearchAgent**: Multi-source concurrent web search with content processing
- **FileReaderAgent**: File format detection and content extraction
- **DailyStoriesAgent**: RSS feed aggregation with concurrent processing
- **HueLightsAgent**: External device control with JSON parsing
- **DetectLanguageAgent**: Simple always-on language processing

Start with simpler patterns and evolve toward more complex implementations as needed.

---

*Ready to build something amazing? The Artemis agent ecosystem is designed to be endlessly extensible. Each new agent makes the system smarter and more capable.*