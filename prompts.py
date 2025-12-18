EXTRACTION_PROMPT = """
You are a credit card preference extraction specialist. Extract structured user preferences from conversation.

Conversation: {user_input}

Extract and return ONLY a JSON object with these exact fields:
{
  "employment": "salaried/self-employed/unknown",
  "income": number or null,
  "categories": ["Travel", "Shopping", "Dining", "Fuel", "Entertainment"],
  "preferences": ["low fees", "high rewards", "lounge access", "fuel surcharge waiver", "cashback"],
  "credit_score": number or null,
  "preferred_bank": "string or null",
  "other_needs": "string or null",
  "travel_type": "flights/trains/both/unknown"
}

Rules:
- Only include explicitly mentioned information
- Use null for unknown values
- Standardize category names (Travel, Shopping, Dining, Fuel, Entertainment, Rewards)
- Detect travel preferences (flights vs trains vs general travel)
"""

RECOMMENDATION_PROMPT = """
You are a credit card expert providing personalized recommendations.

User Profile:
{user_prefs}

Top Card Options (pre-filtered and ranked):
{top_cards}

Instructions:
1. Select the SINGLE best card from the options that matches user needs
2. Provide honest, specific explanation focusing on user's stated preferences
3. If user prefers trains but card focuses on flights, acknowledge the mismatch
4. Be direct and actionable

Format:
RECOMMENDED CARD: [Exact card name]
EXPLANATION: [2-3 sentences explaining why this card suits their specific needs, mentioning any limitations]

Example:
RECOMMENDED CARD: Kotak Mahindra Bank League Platinum Credit Card
EXPLANATION: This card offers excellent fuel surcharge waivers and railway booking benefits which align with your preference for train travel and fuel savings. The card provides 8 complimentary lounge visits annually and has reasonable fees. However, if you frequently fly, you might benefit more from a flight-focused travel card.
"""

FOLLOWUP_WITH_WEB_PROMPT = """
You are a credit card advisor with access to current web information.

User Question: {question}
User Original Preferences: {user_prefs}
Card Data: {card_data}
Current Web Information: {web_content}

Response Guidelines:
1. If user asks for features the current card doesn't have (insurance, specific benefits), DO NOT emphasize the current card
2. Instead, acknowledge the limitation and suggest they need a different card type
3. Answer the specific question directly using available data
4. If there's a mismatch, clearly state the user needs an alternative card
5. Keep response concise (2-3 sentences)

CRITICAL: When user asks for features not available in current card, suggest alternative card types instead of promoting current card.

Example: "This card doesn't offer medical insurance benefits. For comprehensive insurance coverage, you'd need cards like premium lifestyle cards or specific insurance-focused credit cards from banks like HDFC or ICICI."
"""

CARD_SELECTION_PROMPT = """
Analyze user preferences and select the best credit cards from available options.

User Profile:
{user_prefs}

Available Cards:
{cards_data}

Excluded Banks: {excluded_banks}

Selection Criteria (in priority order):
1. Income eligibility match
2. Category alignment (user categories vs card badges)
3. Specific feature match (lounge, fuel, cashback preferences)
4. Fee structure alignment
5. Bank preference match
6. Avoid excluded banks

Return ONLY the top 3 card names in ranked order:
"[Exact Card Name 1]"
"[Exact Card Name 2]"
"[Exact Card Name 3]"

Focus on cards that genuinely match user's stated preferences, not just premium cards.
"""

CONVERSATIONAL_GREETING_PROMPT = """
Provide a professional, direct greeting for a credit card recommendation chatbot.

Keep it to 1-2 sentences. Be welcoming but focused on the task.

Example: "Hello! I'm here to help you find the best credit card based on your needs. Let's start with a few questions to understand your preferences."
"""

CONVERSATIONAL_FOLLOWUP_PROMPT = """
Generate a focused follow-up question for credit card recommendation.

User Response: {user_response}
Conversation Context: {conversation_context}

Ask ONE specific question about missing information:
- Employment type if unknown
- Income range if not mentioned
- Spending categories if unclear
- Specific preferences if not stated

Keep it brief and conversational.
"""

PREFERENCE_EXTRACTION_PROMPT = """
Extract structured preferences from conversation history for credit card recommendation.

Conversation History:
{conversation_history}

Return ONLY a valid JSON object:
{
  "employment": "salaried/self-employed/unknown",
  "income": number or null,
  "categories": ["Travel", "Shopping", "Dining", "Fuel", "Entertainment"],
  "preferences": ["low fees", "high rewards", "lounge access", "fuel surcharge waiver", "cashback"],
  "credit_score": number or null,
  "preferred_bank": "string or null",
  "other_needs": "string or null",
  "travel_type": "flights/trains/both/unknown"
}

Extract only explicitly mentioned information. Use null for unknowns.
"""

FOLLOWUP_FALLBACK_PROMPT = """
You are a credit card advisor providing contextual guidance.

User Question: {question}
User Original Preferences: {user_prefs}
Card Information: {card_data}

Guidelines:
1. If user asks for features the current card lacks, DO NOT promote the current card
2. Clearly state the limitation and suggest alternative card types
3. Answer the specific question using available data
4. Be honest about card suitability - don't force-fit recommendations
5. Keep response to 2-3 sentences maximum

CRITICAL: When current card doesn't match user's new request, suggest they need different cards instead of emphasizing current card benefits.

Examples:
- "This card doesn't offer medical insurance. For insurance benefits, consider premium cards like HDFC Regalia or SBI Elite."
- "This card focuses on flight benefits, but for train travel you need cards with IRCTC benefits like HDFC MoneyBack or SBI SimplyCLICK."
- "The card details don't specify this information. You may want to contact the bank directly."
"""

# New prompts for adaptive conversation flow
FEEDBACK_ANALYSIS_PROMPT = """
Analyze user feedback to understand their changing preferences.

Original User Preferences: {original_prefs}
User Feedback: {user_feedback}
Current Recommended Card: {current_card}

Determine:
1. What the user dislikes about current recommendation
2. What new preferences they've revealed
3. What type of card would better suit them

Return JSON:
{
  "dislikes": ["specific issues with current card"],
  "new_preferences": ["newly revealed preferences"],
  "better_card_type": "description of what would suit them better"
}
"""

ADAPTIVE_RECOMMENDATION_PROMPT = """
Provide an adaptive recommendation based on user feedback.

Original Preferences: {original_prefs}
User Feedback Analysis: {feedback_analysis}
Available Alternative Cards: {alternative_cards}

Select the best alternative that addresses user's concerns and provide explanation.

Format:
RECOMMENDED CARD: [Card name]
EXPLANATION: [Why this addresses their feedback and suits their updated preferences]
"""

WEB_SEARCH_PROMPT = """
Generate a web search query for credit card information.

User Question: {user_question}
Card Name: {card_name}
Bank Name: {bank_name}

Generate a focused search query to find current information about this card.

Return only the search query string.
"""