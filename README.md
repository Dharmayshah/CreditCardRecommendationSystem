# Credit Card Recommendation Chatbot

## Project Overview

This conversational chatbot helps users find the best credit card based on their specific needs through natural language interaction. The system uses LangChain for LLM orchestration and a JSON dataset of credit cards to provide personalized recommendations.

### Problem It Solves
- **Information Overload**: Simplifies the complex process of choosing from dozens of credit cards
- **Personalized Matching**: Matches cards based on individual spending patterns, income, and preferences
- **Transparent Recommendations**: Provides clear explanations for why a specific card is recommended
- **Interactive Conversations**: Handles follow-up questions and alternative recommendations

## How It Works

### High-Level Flow
```
1. User Preference Collection
   ├── Employment type, income, spending categories
   ├── Preferences (fees, rewards, features)
   └── Bank preferences, credit score

2. Card Filtering & Scoring
   ├── Eligibility filtering (income, credit score, bank requirements)
   ├── Preference-based scoring with weighted algorithm
   └── Ranking by match quality

3. LLM-Powered Recommendation
   ├── Intelligent card selection from top candidates
   ├── Personalized explanation generation
   └── Context-aware presentation

4. Interactive Conversation Loop
   ├── Follow-up question handling
   ├── Alternative card suggestions
   ├── Web content integration for current offers
   └── Conversational state management
```

### Conversation Flow
1. **Preference Collection**: Structured questionnaire for user requirements
2. **Card Analysis**: Multi-stage filtering and weighted scoring
3. **Recommendation**: LLM-powered card selection with detailed explanation
4. **Interactive Q&A**: Conversational interface for questions and alternatives
5. **Alternative Suggestions**: Smart recommendations based on user feedback

## Tech Stack

- **Language**: Python 3.8+
- **LLM Framework**: LangChain
- **LLM Provider**: Cohere (command-r-08-2024 model)
- **Data Storage**: JSON file (in-memory processing)
- **Web Scraping**: BeautifulSoup4 + Requests
- **Environment Management**: python-dotenv

### Key Dependencies
```
langchain==0.1.0
langchain-cohere==0.1.0
langchain-community==0.0.10
cohere==4.37
beautifulsoup4==4.12.2
requests==2.31.0
python-dotenv==1.0.0
pydantic==2.5.0
lxml==4.9.3
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
COHERE_API_KEY=your_cohere_api_key_here
MODEL_NAME=command-r-08-2024
LOG_LEVEL=ERROR
REQUEST_TIMEOUT=10
MAX_WEB_CONTENT_LENGTH=3000
```

### 3. Run the Chatbot
```bash
python main.py
```

### 4. Interact with the System
- Answer the structured questions about your preferences
- Review the recommended card with detailed explanation
- Ask follow-up questions in natural language
- Request alternative recommendations
- Get current information from official card links
- Type 'exit' to end the conversation

## Design Decisions

### 1. Hybrid Approach
**Decision**: Combine rule-based filtering with LLM intelligence
**Implementation**:
- Structured preference collection for reliability
- Rule-based eligibility filtering and scoring
- LLM for intelligent recommendation and conversation
- Session state management for context awareness

### 2. Enhanced Preference Collection Strategy
**Decision**: Comprehensive questionnaire based on dataset analysis
**Rationale**:
- Covers all available card categories from the dataset
- Allows multiple preference selection for better matching
- Includes specialized categories like co-branded, secured, business cards
- Provides granular preference options for precise recommendations

**Implementation**:
```python
# Expanded categories based on dataset analysis
categories_map = {
    '1': 'Travel', '2': 'Shopping', '3': 'Dining', '4': 'Fuel', 
    '5': 'Entertainment', '6': 'Online', '7': 'Premium', '8': 'Rewards',
    '9': 'Lifestyle', '10': 'Co-branded', '11': 'Movies', '12': 'Business',
    '13': 'Secured', '14': 'Cashback', '15': 'Lounge Access', '16': 'Railway'
}
```

### 3. Enhanced Card Filtering and Ranking Algorithm
**Decision**: Multi-stage filtering with intelligent weighted scoring
**Stages**:
1. **Eligibility Filter**: Income requirements, credit score, bank customer status
2. **Enhanced Preference Scoring**: Exact and partial category matching, comprehensive feature alignment
3. **Income-based Scoring**: Premium card recommendations for higher income users
4. **Final Ranking**: Weighted combination with preference priorities

**Enhanced Scoring Logic**:
```python
score = (exact_category_matches * 15) +    # Exact spending category alignment
        (partial_matches * 8) +            # Partial category matches
        (no_annual_fee * 20) +             # No annual fee preference
        (lounge_access * 18) +             # Lounge access benefits
        (cashback_benefits * 18) +         # Cashback rewards
        (travel_rewards * 16) +            # Travel rewards and miles
        (fuel_benefits * 15) +             # Fuel surcharge waivers
        (bank_preference * 25) +           # Preferred bank bonus
        (income_tier_bonus * 10)           # Premium cards for high income
```

### 4. Enhanced Web Integration Strategy
**Decision**: Intelligent web browsing with link prioritization
**Triggers**: Keywords like "latest", "current", "offers", "application", "website", "official", "bank"
**Implementation**: 
- Prioritizes official bank links over third-party sources
- Uses only verified links from card JSON data
- Intelligent link selection based on bank name matching
- BeautifulSoup4 for content extraction with improved parsing
- Multiple fallback levels for robust error handling

### 5. Conversation State Management
**Decision**: Comprehensive session state tracking
**Components**:
- Recommended cards history (top 5 alternatives)
- Excluded institutions (for alternative suggestions)
- LLM call counter (usage monitoring)
- Current conversation context and history
- User preferences and current card selection

### 6. Error Handling and Fallbacks
**Decision**: Graceful degradation with multiple fallback levels
**Hierarchy**:
1. LLM with official web content from card links
2. LLM with JSON data only
3. Direct JSON data extraction and display
4. User-friendly error messages with guidance

### 7. Conversational Intelligence
**Decision**: LLM-powered natural language interaction
**Features**:
- Context-aware responses using conversation history
- Intelligent alternative suggestions based on feedback
- Dynamic card switching during conversation
- Personalized explanations and recommendations

## Current Implementation Features

### Core Capabilities

1. **Enhanced Preference Collection**
   - Comprehensive questionnaire covering 16 card categories
   - Multiple preference selection (12 different preference types)
   - Dataset-driven category mapping for complete coverage
   - Support for specialized cards (secured, business, co-branded)

2. **Advanced Card Matching Algorithm**
   - Multi-stage eligibility filtering with income and credit score validation
   - Intelligent scoring with exact and partial category matching
   - Income-tier based recommendations (premium cards for high earners)
   - Comprehensive preference scoring (fees, rewards, benefits)
   - LLM-powered final recommendation with detailed explanations

3. **Conversational Interface**
   - Natural language follow-up questions with context awareness
   - Dynamic card switching during conversation
   - Alternative card suggestions with exclusion tracking
   - Personalized explanations based on user profile

4. **Enhanced Web Integration**
   - Intelligent link prioritization (official bank sites first)
   - Multi-source content fetching with fallback mechanisms
   - Real-time information from verified card links
   - Improved content extraction and parsing

5. **Advanced Session Management**
   - Comprehensive state tracking (cards, preferences, history)
   - LLM usage optimization and monitoring
   - Conversation context preservation
   - Excluded institution tracking for better alternatives

### Current Limitations

1. **Dataset Scope**: Limited to cards in JSON file
   - **Mitigation**: Web integration for current offers

2. **Web Scraping**: Basic content extraction
   - **Impact**: May not capture all structured information

3. **No Persistence**: Session data not saved between runs
   - **Impact**: Cannot learn from user interactions over time

### Potential Improvements

#### Short-term Enhancements
- **Enhanced Web Parsing**: Better structured data extraction from bank websites
- **Improved Scoring**: Machine learning-based preference weighting
- **Caching System**: Cache web content and LLM responses
- **Input Validation**: Enhanced validation with suggestions

#### Medium-term Features
- **User Profiles**: Persistent user preferences and history
- **Comparison Mode**: Side-by-side card feature comparisons
- **Real-time Data**: Integration with bank APIs for live offers
- **Multi-language Support**: Regional language support

#### Long-term Vision
- **Machine Learning**: Collaborative filtering and recommendation learning
- **Advanced NLP**: Better intent recognition and entity extraction
- **Application Integration**: Direct card application process
- **Mobile Interface**: Web/mobile app interface

### Performance Considerations

1. **LLM Usage Optimization**: Tracks and minimizes API calls
2. **Web Request Management**: Timeout handling and rate limiting
3. **Memory Efficiency**: In-memory JSON processing
4. **Error Recovery**: Multiple fallback strategies

## Architecture Overview

### Current Architecture
- **Single-user CLI application** with conversational interface
- **Modular design** with separate concerns:
  - `CreditCardChatbot`: Main orchestration class
  - `prompts.py`: LLM prompt templates
  - JSON data loading and processing
  - Web scraping utilities

### Key Components

1. **Preference Collection**: Interactive questionnaire system
2. **Card Processing**: Filtering, scoring, and ranking algorithms
3. **LLM Integration**: Cohere API with LangChain framework
4. **Web Tools**: BeautifulSoup4-based content extraction
5. **Session Management**: State tracking and conversation context

### Configuration Management
- **Environment Variables**: API keys and settings in `.env`
- **Logging**: Configurable logging levels
- **Error Handling**: Comprehensive exception management

### Security Features
- **API Key Protection**: Environment variable storage
- **Input Validation**: User input sanitization
- **Web Scraping**: Respectful scraping with timeouts
- **Error Isolation**: Graceful degradation on failures

## File Structure

```
CreditCardRecommendationSystem/
├── main.py                 # Main chatbot application
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── Data for project.json  # Credit card dataset
└── README.md             # Project documentation
```

### Key Files

- **main.py**: Contains the `CreditCardChatbot` class with all core functionality
- **prompts.py**: Structured prompts for different LLM interactions
- **Data for project.json**: Credit card database with structured information
- **.env**: Configuration file for API keys and settings

## Usage Examples

### Enhanced Recommendation Flow
```
System: Are you salaried or self-employed? (salaried/self-employed):
User: salaried
System: What's your annual income in lakhs? (e.g., 5, 10, 15):
User: 8
System: What are your main spending categories? (Select multiple...)
        1. Travel  2. Shopping  3. Dining  4. Fuel  5. Entertainment
        6. Online  7. Premium  8. Rewards  9. Lifestyle  10. Co-branded
        11. Movies  12. Business  13. Secured  14. Cashback  15. Lounge Access  16. Railway
User: 1,3,4,15  # Travel, Dining, Fuel, Lounge Access
System: What's most important to you? (Select multiple...)
        1. Cashback  2. Travel Rewards  3. Low Fees  4. Lounge Access
        5. Fuel Surcharge Waiver  6. Movie Benefits  7. Dining Discounts
        8. Railway Benefits  9. Insurance Coverage  10. Milestone Rewards
        11. Welcome Benefits  12. No Annual Fee
User: 2,4,5  # Travel Rewards, Lounge Access, Fuel Surcharge Waiver

System: Analyzing credit cards for your profile...
System: RECOMMENDED CREDIT CARD
        Card: Axis Bank SELECT Credit Card
        Bank: Axis Bank
        This premium card perfectly matches your travel preferences with unlimited lounge access,
        excellent travel rewards, and fuel surcharge waivers...
```

### Enhanced Interactive Conversation
```
User: Does this card have lounge access?
System: Yes, the Axis Bank SELECT offers 6 complimentary international lounge visits annually
        plus 2 complimentary domestic lounge visits per quarter...

User: What about railway booking benefits?
System: This card doesn't offer specific railway benefits. For IRCTC bookings and train travel,
        I'd recommend the Kotak League Platinum which offers railway surcharge waivers...

User: Show me cards with no annual fee
System: [Fetching current information from official sources...]
        Based on current offers, here are no annual fee alternatives:
        ALTERNATIVE RECOMMENDATION
        Card: Kotak 811 Dream Different Credit Card
        This lifetime free card offers 4X rewards on online spends and fuel surcharge waivers.
```

### Enhanced Web-Integrated Responses
```
User: What are the current offers on this card?
System: Fetching current information from official sources...
        From axisbank.com: Current welcome offer includes 10,000 EDGE reward points
        worth ₹2,000 on first transaction + complimentary Priority Pass membership.
        From paisabazaar.com: Additional offer - Waived joining fee for online applications
        through select channels until December 2024.
        
User: How do I apply for this card?
System: [Prioritizing official bank sources...]
        From Axis Bank official website: You can apply online at axisbank.com/credit-cards
        or visit any Axis Bank branch. Required documents include PAN card, Aadhaar,
        income proof, and address proof. Processing time is typically 7-10 working days.
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## License

This project is for educational and demonstration purposes. Credit card data is sourced from public information and should be verified with respective banks before making decisions.