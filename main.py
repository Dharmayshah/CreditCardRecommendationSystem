import json
import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from prompts import EXTRACTION_PROMPT, RECOMMENDATION_PROMPT, FOLLOWUP_WITH_WEB_PROMPT, CARD_SELECTION_PROMPT, FOLLOWUP_FALLBACK_PROMPT, CONVERSATIONAL_GREETING_PROMPT, CONVERSATIONAL_FOLLOWUP_PROMPT, PREFERENCE_EXTRACTION_PROMPT

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

load_dotenv()

class CreditCardChatbot:
    def __init__(self, json_path: str):
        self.cards = self._load_cards(json_path)
        self.llm = ChatCohere(
            model=os.getenv("MODEL_NAME"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            temperature=0.3
        )
        self.user_prefs = {}
        self.conversation_history = []
        self.session_state = {
            'recommended_cards': [],
            'current_card': None,
            'excluded_institutions': [],
            'llm_calls_count': 0
        }
        
        # Initialize web browsing tools
        self.web_tools = self._setup_web_tools()
        
    def _load_cards(self, path: str) -> List[Dict]:
        """Load and validate card data"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cards = json.load(f)
            
            # Validate and clean card data
            valid_cards = []
            for card in cards:
                if (card.get('name') and 
                    card.get('Institution') and 
                    card.get('Institution') != 'None' and
                    card.get('Institution') is not None):
                    valid_cards.append(card)
            
            logger.info(f"Loaded {len(valid_cards)} valid cards from {len(cards)} total cards")
            return valid_cards
        except Exception as e:
            logger.error(f"Error loading cards: {e}")
            return []
    
    def _setup_web_tools(self) -> List[Tool]:
        """Setup LangChain tools for web browsing"""
        
        def browse_page(url: str) -> str:
            """Browse a specific page and extract content"""
            return self._fetch_web_content(url, max_chars=3000)
        
        def web_search(query: str) -> str:
            """Search for credit card information"""
            # Simple implementation - in production, use proper search API
            search_urls = [
                f"https://www.google.com/search?q={query.replace(' ', '+')}+credit+card+india",
            ]
            
            results = []
            for url in search_urls[:1]:  # Limit to prevent too many requests
                content = self._fetch_web_content(url, max_chars=1500)
                if content:
                    results.append(content)
            
            return "\n\n".join(results) if results else "No search results found"
        
        return [
            Tool(
                name="browse_page",
                description="Browse a specific webpage to get detailed information about credit cards",
                func=browse_page
            ),
            Tool(
                name="web_search", 
                description="Search the web for credit card information and current offers",
                func=web_search
            )
        ]
    
    def collect_user_preferences(self):
        """Collect user preferences without LLM"""
        print("CREDIT CARD ADVISOR - FIND YOUR PERFECT CREDIT CARD\n")
        print("Let me ask you a few quick questions to find your perfect credit card.\n")
        
        # Employment type
        while True:
            employment = input("Are you salaried or self-employed? (salaried/self-employed): ").strip().lower()
            if employment in ['salaried', 'self-employed']:
                self.user_prefs['employment'] = employment
                break
            print("Please enter 'salaried' or 'self-employed'")
        
        # Income
        while True:
            try:
                income_input = input("What's your annual income in lakhs? (e.g., 5, 10, 15): ").strip()
                income = float(income_input) * 100000
                self.user_prefs['income'] = int(income)
                break
            except ValueError:
                print("Please enter a valid number (e.g., 5 for 5 lakhs)")
        
        # Spending categories - expanded based on dataset analysis
        print("\nWhat are your main spending categories? (Select multiple by entering numbers separated by commas)")
        categories_map = {
            '1': 'Travel', '2': 'Shopping', '3': 'Dining', '4': 'Fuel', 
            '5': 'Entertainment', '6': 'Online', '7': 'Premium', '8': 'Rewards',
            '9': 'Lifestyle', '10': 'Co-branded', '11': 'Movies', '12': 'Business',
            '13': 'Secured', '14': 'Cashback', '15': 'Lounge Access', '16': 'Railway'
        }
        
        # Display in tabular format
        print("\n┌─────┬─────────────────┬─────┬─────────────────┬─────┬─────────────────┬─────┬─────────────────┐")
        print("│ No. │    Category     │ No. │    Category     │ No. │    Category     │ No. │    Category     │")
        print("├─────┼─────────────────┼─────┼─────────────────┼─────┼─────────────────┼─────┼─────────────────┤")
        
        categories_list = list(categories_map.items())
        for i in range(0, len(categories_list), 4):
            row = categories_list[i:i+4]
            line = "│"
            for j in range(4):
                if j < len(row):
                    num, cat = row[j]
                    line += f" {num:>2}. │ {cat:<15} │"
                else:
                    line += "     │                 │"
            print(line)
        
        print("└─────┴─────────────────┴─────┴─────────────────┴─────┴─────────────────┴─────┴─────────────────┘")
        
        while True:
            cat_input = input("Enter your choices (e.g., 1,3,4): ").strip()
            try:
                selected_nums = [num.strip() for num in cat_input.split(',')]
                selected_categories = [categories_map[num] for num in selected_nums if num in categories_map]
                if selected_categories:
                    self.user_prefs['categories'] = selected_categories
                    break
                else:
                    print("Please select at least one valid category")
            except:
                print("Please enter valid numbers separated by commas (e.g., 1,3,4)")
        
        # Preferences - expanded based on dataset analysis
        print("\nWhat's most important to you? (Select multiple by entering numbers separated by commas)")
        prefs_map = {
            '1': 'cashback', '2': 'travel rewards', '3': 'low fees', 
            '4': 'lounge access', '5': 'fuel surcharge waiver', '6': 'movie benefits',
            '7': 'dining discounts', '8': 'railway benefits', '9': 'insurance coverage',
            '10': 'milestone rewards', '11': 'welcome benefits', '12': 'no annual fee'
        }
        
        # Display in tabular format
        print("\n┌─────┬─────────────────────┬─────┬─────────────────────┬─────┬─────────────────────┐")
        print("│ No. │     Preference      │ No. │     Preference      │ No. │     Preference      │")
        print("├─────┼─────────────────────┼─────┼─────────────────────┼─────┼─────────────────────┤")
        
        prefs_list = list(prefs_map.items())
        for i in range(0, len(prefs_list), 3):
            row = prefs_list[i:i+3]
            line = "│"
            for j in range(3):
                if j < len(row):
                    num, pref = row[j]
                    line += f" {num:>2}. │ {pref.title():<19} │"
                else:
                    line += "     │                     │"
            print(line)
        
        print("└─────┴─────────────────────┴─────┴─────────────────────┴─────┴─────────────────────┘")
        
        while True:
            pref_input = input("Enter your choices (e.g., 1,3,5): ").strip()
            try:
                selected_nums = [num.strip() for num in pref_input.split(',')]
                selected_prefs = [prefs_map[num] for num in selected_nums if num in prefs_map]
                if selected_prefs:
                    self.user_prefs['preferences'] = selected_prefs
                    break
                else:
                    print("Please select at least one valid preference")
            except:
                print("Please enter valid numbers separated by commas (e.g., 1,3,5)")
        
        # Optional: Preferred bank
        bank = input("\nDo you have a preferred bank? (optional, press Enter to skip): ").strip()
        if bank:
            self.user_prefs['preferred_bank'] = bank
        
        # Optional: Credit score
        credit_input = input("What's your approximate credit score? (optional, press Enter to skip): ").strip()
        if credit_input:
            try:
                self.user_prefs['credit_score'] = int(credit_input)
            except ValueError:
                pass
        
        print("\n" + "="*60)
        print("Thank you! Let me find the best credit card for you...")
        print("="*60 + "\n")
    

    

    
    def filter_cards_by_eligibility(self) -> List[Dict]:
        """Filter cards based on basic eligibility without LLM"""
        eligible_cards = []
        user_income = self.user_prefs.get('income', 0)
        user_employment = self.user_prefs.get('employment')
        user_credit_score = self.user_prefs.get('credit_score')
        
        for card in self.cards:
            # Skip excluded institutions
            if card.get('Institution') in self.session_state['excluded_institutions']:
                continue
            
            # Check income eligibility
            income_req = card.get('eligibility_income_min', {})
            income_eligible = True
            
            if user_employment == 'salaried' and income_req.get('salaried'):
                income_eligible = user_income >= income_req['salaried']
            elif user_employment == 'self-employed' and income_req.get('self_employed'):
                income_eligible = user_income >= income_req['self_employed']
            elif income_req.get('Any'):
                income_eligible = user_income >= income_req['Any']
            
            # Check credit score if specified
            credit_eligible = True
            if user_credit_score and card.get('minimum_credit_score'):
                credit_eligible = user_credit_score >= card['minimum_credit_score']
            
            # Check bank customer requirement
            bank_eligible = True
            if card.get('is_bank_customer_only') and self.user_prefs.get('preferred_bank'):
                bank_eligible = self.user_prefs['preferred_bank'].upper() in card.get('Institution', '').upper()
            elif card.get('is_bank_customer_only') and not self.user_prefs.get('preferred_bank'):
                bank_eligible = False  # User doesn't have preferred bank but card requires it
            
            if income_eligible and credit_eligible and bank_eligible:
                eligible_cards.append(card)
        
        return eligible_cards
    
    def score_cards(self, eligible_cards: List[Dict]) -> List[Dict]:
        """Score and rank cards based on user preferences without LLM"""
        scored_cards = []
        
        for card in eligible_cards:
            score = 0
            
            # Category matching - improved with exact and partial matching
            card_badges = [badge.lower() for badge in card.get('badge', [])]
            user_categories = [cat.lower() for cat in self.user_prefs.get('categories', [])]
            
            # Exact category matches
            exact_matches = sum(1 for cat in user_categories if cat in card_badges)
            score += exact_matches * 15
            
            # Partial category matches (e.g., 'travel' matches 'co-branded' travel cards)
            partial_matches = 0
            for user_cat in user_categories:
                for card_badge in card_badges:
                    if user_cat in card_badge or card_badge in user_cat:
                        partial_matches += 1
                        break
            score += (partial_matches - exact_matches) * 8  # Avoid double counting
            
            # Preference matching - enhanced scoring
            user_prefs = self.user_prefs.get('preferences', [])
            
            # Get rewards and fee data safely
            rewards_data = card.get('rewards', [])
            if isinstance(rewards_data, dict):
                rewards = rewards_data.get('rewards', [])
            else:
                rewards = rewards_data if isinstance(rewards_data, list) else []
            
            fee_data = card.get('fee_breakdown', [])
            if isinstance(fee_data, dict):
                fee_breakdown = fee_data.get('fee_breakdown', [])
            else:
                fee_breakdown = fee_data if isinstance(fee_data, list) else []
            
            # Low fees / No annual fee preference
            if 'low fees' in user_prefs or 'no annual fee' in user_prefs:
                joining_fee_low = False
                annual_fee_low = False
                
                for fee in fee_breakdown:
                    if isinstance(fee, dict):
                        if fee.get('type') == 'joining_fee':
                            fee_details = ' '.join(fee.get('details', [])).lower()
                            if 'nil' in fee_details or 'waived' in fee_details or '0' in fee_details or 'free' in fee_details:
                                joining_fee_low = True
                        elif fee.get('type') == 'annual_fee':
                            fee_details = ' '.join(fee.get('details', [])).lower()
                            if 'nil' in fee_details or 'waived' in fee_details or '0' in fee_details or 'free' in fee_details or 'lifetime' in fee_details:
                                annual_fee_low = True
                
                if joining_fee_low and annual_fee_low:
                    score += 20
                elif annual_fee_low:  # Annual fee waiver is more important
                    score += 15
                elif joining_fee_low:
                    score += 10
            
            # Lounge access preference
            if 'lounge access' in user_prefs:
                has_lounge = any('lounge' in str(reward).lower() for reward in rewards)
                if has_lounge:
                    score += 18
            
            # Fuel surcharge waiver preference
            if 'fuel surcharge waiver' in user_prefs:
                has_fuel_benefit = (
                    any('fuel' in str(reward).lower() for reward in rewards) or
                    any('fuel' in str(fee).lower() for fee in fee_breakdown)
                )
                if has_fuel_benefit:
                    score += 15
            
            # Cashback preference
            if 'cashback' in user_prefs:
                has_cashback = any('cashback' in str(reward).lower() for reward in rewards)
                if has_cashback:
                    score += 18
            
            # Travel rewards preference
            if 'travel rewards' in user_prefs:
                has_travel_rewards = any(
                    any(keyword in str(reward).lower() for keyword in ['travel', 'miles', 'points', 'air']) 
                    for reward in rewards
                )
                if has_travel_rewards:
                    score += 16
            
            # Movie benefits preference
            if 'movie benefits' in user_prefs:
                has_movie_benefits = any('movie' in str(reward).lower() or 'pvr' in str(reward).lower() for reward in rewards)
                if has_movie_benefits:
                    score += 12
            
            # Dining discounts preference
            if 'dining discounts' in user_prefs:
                has_dining_benefits = any('dining' in str(reward).lower() or 'restaurant' in str(reward).lower() for reward in rewards)
                if has_dining_benefits:
                    score += 12
            
            # Railway benefits preference
            if 'railway benefits' in user_prefs:
                has_railway_benefits = any('railway' in str(reward).lower() or 'irctc' in str(reward).lower() for reward in rewards)
                if has_railway_benefits:
                    score += 12
            
            # Welcome benefits preference
            if 'welcome benefits' in user_prefs:
                has_welcome_benefits = any('welcome' in str(reward).lower() for reward in rewards)
                if has_welcome_benefits:
                    score += 8
            
            # Milestone rewards preference
            if 'milestone rewards' in user_prefs:
                has_milestone_benefits = any('milestone' in str(reward).lower() for reward in rewards)
                if has_milestone_benefits:
                    score += 10
            
            # Insurance coverage preference
            if 'insurance coverage' in user_prefs:
                has_insurance = any('insurance' in str(reward).lower() or 'cover' in str(reward).lower() for reward in rewards)
                if has_insurance:
                    score += 8
            
            # Preferred bank bonus
            if (self.user_prefs.get('preferred_bank') and 
                self.user_prefs['preferred_bank'].upper() in card.get('Institution', '').upper()):
                score += 25
            
            # Income-based scoring (higher income gets premium cards)
            user_income = self.user_prefs.get('income', 0)
            if user_income >= 1000000:  # 10+ lakhs
                if 'premium' in card_badges or 'signature' in card.get('name', '').lower():
                    score += 10
            elif user_income >= 500000:  # 5+ lakhs
                if any(badge in card_badges for badge in ['lifestyle', 'rewards', 'travel']):
                    score += 5
            
            scored_cards.append((card, score))
        
        # Sort by score (descending)
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        return [card for card, score in scored_cards]
    
    def llm_recommend_and_explain(self, top_cards: List[Dict]) -> Optional[Dict]:
        """Use LLM for intelligent card analysis and recommendation"""
        if not top_cards:
            return None
        
        # Increment LLM call counter
        self.session_state['llm_calls_count'] += 1
        
        # Prepare comprehensive card data for LLM analysis
        cards_for_llm = []
        for card in top_cards[:5]:  # Analyze top 5 instead of 3
            card_summary = {
                'name': card.get('name'),
                'institution': card.get('Institution'),
                'categories': card.get('badge', []),
                'rewards': card.get('rewards', {}),
                'fees': card.get('fee_breakdown', {}),
                'eligibility': card.get('eligibility_income_min', {}),
                'interest_rate': card.get('interest_rate'),
                'bank_requirement': card.get('is_bank_customer_only'),
                'key_features': self._extract_key_features(card)
            }
            cards_for_llm.append(card_summary)
        
        prompt = PromptTemplate(
            input_variables=["user_prefs", "top_cards"],
            template=RECOMMENDATION_PROMPT
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(
                user_prefs=json.dumps(self.user_prefs, indent=2),
                top_cards=json.dumps(cards_for_llm, indent=2)
            )
            
            # Extract recommended card name
            lines = response.split('\n')
            recommended_name = None
            explanation = ""
            
            for line in lines:
                if line.startswith('RECOMMENDED CARD:'):
                    recommended_name = line.replace('RECOMMENDED CARD:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
            
            # Find the recommended card
            recommended_card = None
            if recommended_name:
                for card in top_cards:
                    if recommended_name.lower() in card.get('name', '').lower():
                        recommended_card = card
                        break
            
            # Fallback to first card if not found
            if not recommended_card:
                recommended_card = top_cards[0]
                explanation = f"Based on your preferences, I recommend the {recommended_card.get('name')} as it best matches your requirements."
            
            # Display recommendation
            print("\n" + "="*60)
            print("YOUR RECOMMENDED CREDIT CARD")
            print("="*60)
            print(f"\nCard: {recommended_card.get('name')}")
            print(f"Issuer: {recommended_card.get('Institution')}")
            print(f"\n{explanation}\n")
            
            # Add key details
            self._display_card_details(recommended_card)
            
            return recommended_card
            
        except Exception as e:
            logger.error(f"LLM recommendation error: {e}")
            # Fallback to first card
            card = top_cards[0]
            print(f"\nRecommended Card: {card.get('name')}")
            print(f"Issuer: {card.get('Institution')}")
            return card
    
    def _display_card_details(self, card: Dict):
        """Display key card details in a friendly way"""
        print("KEY DETAILS:")
        print("-" * 25)
        
        # Fee structure
        fee_data = card.get('fee_breakdown', [])
        if isinstance(fee_data, dict):
            fee_info = fee_data.get('fee_breakdown', [])
        else:
            fee_info = fee_data if isinstance(fee_data, list) else []
            
        joining_fee = "Not specified"
        annual_fee = "Not specified"
        
        for fee in fee_info:
            if isinstance(fee, dict):
                if fee.get('type') == 'joining_fee':
                    joining_fee = ', '.join(fee.get('details', []))
                elif fee.get('type') == 'annual_fee':
                    annual_fee = ', '.join(fee.get('details', []))
        
        print(f"Joining Fee: {joining_fee}")
        print(f"Annual Fee: {annual_fee}")
        
        # Interest rate
        if card.get('interest_rate'):
            print(f"Interest Rate: {card.get('interest_rate')}")
        
        # Income requirement
        income_req = card.get('eligibility_income_min', {})
        employment = self.user_prefs.get('employment', '')
        if employment == 'salaried' and income_req.get('salaried'):
            print(f"Min Income: ₹{income_req['salaried']:,} (Salaried)")
        elif employment == 'self-employed' and income_req.get('self_employed'):
            print(f"Min Income: ₹{income_req['self_employed']:,} (Self-employed)")
        elif income_req.get('Any'):
            print(f"Min Income: ₹{income_req['Any']:,}")
        
        # Bank customer requirement
        if card.get('is_bank_customer_only'):
            print(f"Bank Account: Required ({card.get('Institution')} customer)")
        else:
            print(f"Bank Account: Not required")
        
        print()
    
    def recommend(self):
        """Recommendation process using collected preferences as context"""
        print("Analyzing credit cards for your profile...")
        
        # Display collected preferences
        print(f"Your Profile: {self.user_prefs.get('employment', 'Unknown')} with ₹{self.user_prefs.get('income', 0):,} income")
        print(f"Preferred Categories: {', '.join(self.user_prefs.get('categories', []))}")
        print(f"Priority: {', '.join(self.user_prefs.get('preferences', []))}")
        
        # Step 1: Filter by eligibility
        eligible_cards = self.filter_cards_by_eligibility()
        
        if not eligible_cards:
            print("\nNo exact matches found. Expanding search criteria...")
            eligible_cards = self.cards[:20]
        
        print(f"\nFound {len(eligible_cards)} eligible cards.")
        
        # Step 2: Score and rank cards
        ranked_cards = self.score_cards(eligible_cards)
        
        if not ranked_cards:
            print("\nNo matches found. Showing popular alternatives...")
            ranked_cards = eligible_cards[:5]
        
        print("\nRanking cards based on your preferences...")
        
        # Step 3: LLM recommendation with collected preferences as context
        recommended_card = self.conversational_recommendation(ranked_cards)
        
        if recommended_card:
            self.session_state['current_card'] = recommended_card
            self.session_state['recommended_cards'] = ranked_cards[:5]
        
        return recommended_card
    
    def conversational_recommendation(self, top_cards: List[Dict]) -> Optional[Dict]:
        """Present recommendation in a conversational way"""
        if not top_cards:
            return None
        
        self.session_state['llm_calls_count'] += 1
        
        # Prepare card data
        cards_for_llm = []
        for card in top_cards[:3]:
            cards_for_llm.append({
                'name': card.get('name'),
                'institution': card.get('Institution'),
                'categories': card.get('badge', []),
                'key_features': self._extract_key_features(card),
                'rewards': card.get('rewards', {}),
                'fees': card.get('fee_breakdown', {})
            })
        
        prompt = PromptTemplate(
            input_variables=["user_prefs", "top_cards"],
            template=RECOMMENDATION_PROMPT
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(
                user_prefs=json.dumps(self.user_prefs, indent=2),
                top_cards=json.dumps(cards_for_llm, indent=2)
            )
            
            # Parse response
            lines = response.split('\n')
            recommended_name = None
            presentation = ""
            
            for line in lines:
                if line.startswith('RECOMMENDED_CARD:'):
                    recommended_name = line.replace('RECOMMENDED_CARD:', '').strip()
                elif line.startswith('PRESENTATION:'):
                    presentation = line.replace('PRESENTATION:', '').strip()
            
            # Find the card
            recommended_card = None
            if recommended_name:
                for card in top_cards:
                    if recommended_name.lower() in card.get('name', '').lower():
                        recommended_card = card
                        break
            
            if not recommended_card:
                recommended_card = top_cards[0]
                presentation = f"I found the perfect card for you! The {recommended_card.get('name')} matches your preferences beautifully."
            
            # Present the recommendation conversationally
            print("\n" + "=" * 50)
            print("RECOMMENDED CREDIT CARD")
            print("=" * 50)
            print(f"\nCard: {recommended_card.get('name')}")
            print(f"Bank: {recommended_card.get('Institution')}")
            print(f"\n{presentation}\n")
            
            # Show key details
            self._display_card_details(recommended_card)
            
            return recommended_card
            
        except Exception as e:
            logger.error(f"Conversational recommendation error: {e}")
            card = top_cards[0]
            print(f"\nRecommended: {card.get('name')} from {card.get('Institution')}")
            print("\nThis card matches your preferences and offers good value.\n")
            self._display_card_details(card)
            return card
    
    def _fetch_web_content(self, url: str, max_chars: int = 2000) -> str:
        """Fetch and extract text content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())  # Clean whitespace
            
            return text[:max_chars] if len(text) > max_chars else text
            
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""
    
    def handle_followup_with_web(self, question: str, card: Dict):
        """Handle follow-up questions using card links from JSON only"""
        
        # Check if question needs current info from card links
        web_keywords = ['latest', 'current', 'offers', 'application', 'apply', 'website', 'official', 'bank']
        needs_current_info = any(keyword in question.lower() for keyword in web_keywords)
        
        if needs_current_info and card.get('links'):
            print("\nFetching current information from official sources...")
            
            # Use card's official links from JSON - prioritize bank official sites
            web_content = ""
            bank_name = card.get('Institution', '').lower()
            
            # Prioritize official bank links
            official_links = []
            other_links = []
            
            for link in card.get('links', []):
                if link.get('uri'):
                    link_title = link.get('title', '').lower()
                    if any(bank_word in link_title for bank_word in [bank_name.split()[0], 'bank.com', '.com']) and 'bank' in link_title:
                        official_links.append(link)
                    else:
                        other_links.append(link)
            
            # Try official links first, then others
            links_to_try = (official_links + other_links)[:3]  # Limit to 3 links
            
            for link in links_to_try:
                content = self._fetch_web_content(link['uri'], max_chars=1500)
                if content:
                    web_content += f"\nFrom {link.get('title', 'official source')}: {content}\n"
            
            if web_content:
                self.session_state['llm_calls_count'] += 1
                
                prompt = PromptTemplate(
                    input_variables=["question", "card_data", "web_content", "user_prefs"],
                    template=FOLLOWUP_WITH_WEB_PROMPT
                )
                
                chain = LLMChain(llm=self.llm, prompt=prompt)
                
                try:
                    answer = chain.run(
                        question=question,
                        card_data=json.dumps(card, indent=2),
                        web_content=web_content,
                        user_prefs=json.dumps(self.user_prefs, indent=2)
                    )
                    print(f"\n{answer}\n")
                    return
                except Exception as e:
                    logger.error(f"LLM error with web content: {e}")
        
        # Fallback to JSON-only response
        self.handle_followup_json_only(question, card)
    
    def handle_followup_json_only(self, question: str, card: Dict):
        """Handle follow-up questions using ONLY JSON data"""
        self.session_state['llm_calls_count'] += 1
        
        # Enhanced prompt with JSON-only constraints
        prompt = PromptTemplate(
            input_variables=["question", "user_prefs", "card_data", "all_cards_data"],
            template="""You are a credit card advisor. Answer using ONLY the provided JSON data.

CRITICAL CONSTRAINTS:
- Use ONLY information from the JSON card data provided
- NEVER recommend cards not in the JSON dataset
- If asked about features not in JSON, clearly state "This information is not available in our database"
- Only suggest alternatives from the provided cards data

User Question: {question}

User Preferences: {user_prefs}

Current Card (full JSON): {card_data}

Available Alternative Cards (full JSON): {all_cards_data}

Instructions:
1. Answer the question using ONLY JSON data
2. If current card lacks requested feature, suggest alternatives from JSON that have it
3. Be honest about data limitations
4. Provide specific JSON-based information
5. Never invent or assume information not in JSON

Response:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # Get full JSON data for alternatives
            alternatives = self.session_state.get('recommended_cards', [])[:10]
            alternative_cards = [alt_card for alt_card in alternatives if alt_card != card]
            
            answer = chain.run(
                question=question,
                user_prefs=json.dumps(self.user_prefs, indent=2),
                card_data=json.dumps(card, indent=2),
                all_cards_data=json.dumps(alternative_cards, indent=2)
            )
            print(f"\n{answer}\n")
            
        except Exception as e:
            logger.error(f"LLM followup error: {e}")
            print("\nI can only provide information from our credit card database. Please ask about specific card features or request alternatives.\n")
    

    
    def _extract_key_features(self, card: Dict) -> List[str]:
        """Extract key features of a card for summary"""
        features = []
        
        # Check for key benefits - handle both dict and list formats
        rewards_data = card.get('rewards', [])
        if isinstance(rewards_data, dict):
            rewards = rewards_data.get('rewards', [])
        else:
            rewards = rewards_data if isinstance(rewards_data, list) else []
            
        for reward in rewards:
            if isinstance(reward, dict):
                reward_type = reward.get('type', '')
                if reward_type and len(features) < 5:
                    features.append(reward_type)
            elif isinstance(reward, str) and len(features) < 5:
                features.append(reward[:50])  # Truncate long strings
        
        # Add fee info
        fee_data = card.get('fee_breakdown', [])
        if isinstance(fee_data, dict):
            fee_breakdown = fee_data.get('fee_breakdown', [])
        else:
            fee_breakdown = fee_data if isinstance(fee_data, list) else []
            
        for fee in fee_breakdown:
            if isinstance(fee, dict) and fee.get('type') == 'joining_fee':
                details = ' '.join(fee.get('details', [])).lower()
                if 'nil' in details or 'waived' in details:
                    features.append('No joining fee')
                    break
        
        # Add categories
        badges = card.get('badge', [])
        if badges:
            features.extend(badges[:3])
        
        return features[:5]
    
    def _extract_relevant_data(self, question: str, card: Dict) -> str:
        """Extract relevant card data based on the question"""
        relevant_data = {
            'name': card.get('name'),
            'institution': card.get('Institution'),
            'key_features': self._extract_key_features(card)
        }
        
        # Add specific data based on question keywords
        question_lower = question.lower()
        if 'fee' in question_lower:
            relevant_data['fees'] = card.get('fee_breakdown', {})
        if 'reward' in question_lower or 'benefit' in question_lower:
            relevant_data['rewards'] = card.get('rewards', {})
        if 'eligib' in question_lower or 'income' in question_lower:
            relevant_data['eligibility'] = card.get('eligibility_income_min', {})
        
        return json.dumps(relevant_data, indent=2)
    

    
    def suggest_alternative(self, user_feedback: str = ""):
        """Intelligently suggest alternative card based on user feedback"""
        self.session_state['llm_calls_count'] += 1
        
        recommended_cards = self.session_state.get('recommended_cards', [])
        current_card = self.session_state.get('current_card')
        
        # Prepare alternatives excluding current card and excluded institutions
        alternatives = []
        for card in recommended_cards:
            if (card != current_card and 
                card.get('Institution') not in self.session_state['excluded_institutions']):
                alternatives.append({
                    'name': card.get('name'),
                    'institution': card.get('Institution'),
                    'categories': card.get('badge', []),
                    'key_features': self._extract_key_features(card),
                    'full_data': card
                })
        
        if not alternatives:
            print("\nI've shown you the best available alternatives. Would you like me to explain more about any of the recommended cards?")
            return None
        
        # Use CARD_SELECTION_PROMPT for better alternative selection
        prompt = PromptTemplate(
            input_variables=["user_prefs", "cards_data", "excluded_banks"],
            template=CARD_SELECTION_PROMPT
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(
                user_prefs=json.dumps(self.user_prefs, indent=2),
                cards_data=json.dumps(alternatives[:5], indent=2),
                excluded_banks=json.dumps(self.session_state['excluded_institutions'])
            )
            
            # Extract recommended alternative
            lines = response.split('\n')
            alternative_name = None
            explanation = ""
            
            for line in lines:
                if line.startswith('ALTERNATIVE CARD:'):
                    alternative_name = line.replace('ALTERNATIVE CARD:', '').strip()
                elif line.startswith('WHY THIS IS BETTER:'):
                    explanation = line.replace('WHY THIS IS BETTER:', '').strip()
            
            # Find the recommended alternative
            selected_card = None
            if alternative_name:
                for alt in alternatives:
                    if alternative_name.lower() in alt['name'].lower():
                        selected_card = alt['full_data']
                        break
            
            if not selected_card and alternatives:
                selected_card = alternatives[0]['full_data']
                explanation = "Here's another excellent option that might better suit your needs."
            
            if selected_card:
                self.session_state['current_card'] = selected_card
                self.session_state['excluded_institutions'].append(selected_card.get('Institution'))
                
                print("\n" + "="*50)
                print("ALTERNATIVE RECOMMENDATION")
                print("="*50)
                print(f"\nCard: {selected_card.get('name')}")
                print(f"Issuer: {selected_card.get('Institution')}")
                print(f"\n{explanation}\n")
                
                self._display_card_details(selected_card)
                return selected_card
            
        except Exception as e:
            logger.error(f"Alternative suggestion error: {e}")
            # Fallback to simple alternative
            if alternatives:
                card = alternatives[0]['full_data']
                self.session_state['current_card'] = card
                print(f"\nAlternative: {card.get('name')} from {card.get('Institution')}")
                return card
        
        return None
    
    def conversational_interaction_loop(self, recommended_card):
        """LLM-powered conversational interaction with full context awareness"""
        print("\nI'm here to answer any questions about your recommended card or help you explore alternatives.")
        print("Feel free to ask about features, fees, benefits, or request different cards. Type 'exit' to finish.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'done']:
                self._conversational_goodbye()
                break
            
            if not user_input:
                print("\nWhat would you like to know about your card or alternatives?")
                continue
            
            # Add to conversation history
            self.conversation_history.append(f"User: {user_input}")
            
            # LLM handles everything from here
            response = self._llm_conversational_handler(user_input, recommended_card)
            
            # Add response to history
            if response:
                self.conversation_history.append(f"Assistant: {response}")
                print(f"\n{response}\n")
    
    def _llm_conversational_handler(self, user_input: str, current_card: Dict) -> str:
        """LLM handles conversations using ONLY JSON data and card links"""
        self.session_state['llm_calls_count'] += 1
        
        # Prepare JSON-only context
        alternatives = self.session_state.get('recommended_cards', [])
        alternatives_data = []
        for card in alternatives[:10]:
            if card != current_card:
                alternatives_data.append(card)  # Full card data from JSON
        
        # Get conversation context
        recent_history = '\n'.join(self.conversation_history[-6:]) if self.conversation_history else "No previous conversation"
        
        prompt = PromptTemplate(
            input_variables=["user_query", "current_card_data", "user_prefs", "alternatives_data", "conversation_history"],
            template="""You are a credit card advisor. You can ONLY use the provided JSON card data and fetch additional info from card links.

CRITICAL CONSTRAINTS:
- ONLY recommend cards from the provided JSON data
- NEVER suggest cards not in the JSON dataset
- NEVER provide information not available in JSON or card links
- If asked about cards not in JSON, clearly state they're not in your database
- Only fetch additional data from 'links' field in card JSON

User query: "{user_query}"

Current card (full JSON data): {current_card_data}

User preferences: {user_prefs}

Available alternative cards (full JSON data): {alternatives_data}

Conversation history: {conversation_history}

Your responses must:
1. Answer questions using ONLY JSON card data
2. Recommend alternatives ONLY from provided cards
3. Compare cards using ONLY JSON information
4. Detect preference changes and suggest JSON cards that match
5. Use card links for additional current information if needed

Commands:
- "SWITCH_TO: [exact_card_name_from_json]" - Switch to alternative from JSON
- "FETCH_LINK: [card_link_url]" - Get current info from card's official link

Be helpful but stay strictly within JSON data boundaries:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(
                user_query=user_input,
                current_card_data=json.dumps(current_card, indent=2),
                user_prefs=json.dumps(self.user_prefs, indent=2),
                alternatives_data=json.dumps(alternatives_data, indent=2),
                conversation_history=recent_history
            )
            
            # Handle LLM commands
            if "SWITCH_TO:" in response:
                card_name = response.split("SWITCH_TO:")[1].split("\n")[0].strip()
                new_card = self._switch_to_alternative(card_name)
                if new_card:
                    self.session_state['current_card'] = new_card
                    response = response.replace(f"SWITCH_TO: {card_name}", "").strip()
                    response += f"\n\n[Switched to {new_card.get('name')}]"
            
            elif "FETCH_LINK:" in response:
                link_url = response.split("FETCH_LINK:")[1].split("\n")[0].strip()
                link_content = self._fetch_card_link_content(link_url, current_card)
                if link_content:
                    response = response.replace(f"FETCH_LINK: {link_url}", "").strip()
                    response += f"\n\nCurrent information from official source: {link_content[:500]}..."
            
            return response
            
        except Exception as e:
            logger.error(f"LLM conversational error: {e}")
            return "I'm having trouble processing that. Could you rephrase your question or ask about specific card features?"
    
    def _switch_to_alternative(self, alt_name: str) -> Optional[Dict]:
        """Switch to suggested alternative card"""
        alternatives = self.session_state.get('recommended_cards', [])
        for card in alternatives:
            if alt_name.lower() in card.get('name', '').lower():
                return card
        return None
    
    def _fetch_card_link_content(self, url: str, card: Dict) -> str:
        """Fetch content only from card's official links in JSON"""
        # Verify URL is from card's links
        card_links = card.get('links', [])
        valid_url = False
        for link in card_links:
            if isinstance(link, dict) and link.get('uri') == url:
                valid_url = True
                break
        
        if not valid_url:
            return "This link is not associated with the current card in our database."
        
        try:
            return self._fetch_web_content(url, max_chars=2000)
        except Exception as e:
            logger.error(f"Link fetch error: {e}")
            return "Unable to fetch information from this link at the moment."
    
    def _conversational_goodbye(self):
        """Context-aware goodbye message using LLM"""
        self.session_state['llm_calls_count'] += 1
        
        # Prepare conversation summary
        conversation_summary = '\n'.join(self.conversation_history[-4:]) if self.conversation_history else "Brief interaction"
        
        prompt = PromptTemplate(
            input_variables=["recommended_card", "conversation_summary", "user_prefs"],
            template="""Create a personalized goodbye message based on the conversation.

Final recommended card: {recommended_card}
User preferences: {user_prefs}
Conversation highlights: {conversation_summary}

Make it:
1. Personalized based on their preferences and conversation
2. Encouraging about their final choice
3. Include a relevant tip based on their card/preferences
4. Warm but professional
5. Brief (2-3 sentences max)

Be natural and helpful."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            current_card = self.session_state.get('current_card', {})
            goodbye = chain.run(
                recommended_card=current_card.get('name', 'your chosen card'),
                conversation_summary=conversation_summary,
                user_prefs=json.dumps(self.user_prefs, indent=2)
            )
            print(f"\n{goodbye}\n")
        except:
            print("\nGreat choice! Use your new credit card responsibly and enjoy the benefits. Have a wonderful day!\n")
    
    def run(self):
        """Main chatbot execution"""
        # Collect preferences without LLM
        self.collect_user_preferences()
        
        # Get recommendation using collected preferences as context
        recommended_card = self.recommend()
        
        if not recommended_card:
            print("\nCouldn't find suitable recommendations. Please try with different criteria.")
            return
        
        # Start conversational interaction
        self.conversational_interaction_loop(recommended_card)

if __name__ == "__main__":
    chatbot = CreditCardChatbot("Data for project.json")
    chatbot.run()