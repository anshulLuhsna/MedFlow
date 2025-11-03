"""
Groq API Client for LLM-based preference analysis
Uses Llama 3.3 70B for deep semantic understanding of user preferences

Based on official Groq documentation:
https://console.groq.com/docs/quickstart
"""

import os
import time
import json
from typing import Dict, List, Optional
from groq import Groq


class GroqPreferenceAnalyzer:
    """
    Wrapper for Groq API with Llama 3.3 70B
    Analyzes user interaction patterns and generates preference insights
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)

        Raises:
            ValueError: If API key not provided
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize Groq client (official SDK)
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

        # Rate limiting (Groq is fast but still has limits)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests/sec max

        print(f"✓ Groq client initialized with model: {self.model}")

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        now = time.time()
        elapsed = now - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def analyze_interaction_pattern(
        self,
        interactions: List[Dict],
        max_interactions: int = 10
    ) -> Dict:
        """
        Analyze user interaction patterns using Llama 3.3 70B

        Args:
            interactions: List of recent user interactions
            max_interactions: Maximum interactions to analyze (avoid token limits)

        Returns:
            Dict with preference insights:
            {
                'preference_type': str,
                'confidence': float,
                'key_patterns': List[str],
                'reasoning': str,
                'objective_weights': Dict[str, float],
                'tokens_used': int
            }
        """
        # Limit to recent interactions
        recent = interactions[-max_interactions:]

        if len(recent) == 0:
            return self._fallback_analysis([])

        # Build prompt
        prompt = self._build_pattern_analysis_prompt(recent)

        # Call Groq API
        self._rate_limit()

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": PREFERENCE_ANALYSIS_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,  # Lower for consistency
                max_tokens=1000,
                response_format={"type": "json_object"}  # Force JSON output
            )

            # Parse response
            content = chat_completion.choices[0].message.content
            result = json.loads(content)

            return {
                'preference_type': result.get('preference_type', 'balanced'),
                'confidence': result.get('confidence', 0.5),
                'key_patterns': result.get('key_patterns', []),
                'reasoning': result.get('reasoning', ''),
                'objective_weights': result.get('objective_weights', {
                    'minimize_cost': 0.25,
                    'maximize_coverage': 0.25,
                    'minimize_shortage': 0.25,
                    'fairness': 0.25
                }),
                'tokens_used': chat_completion.usage.total_tokens
            }

        except json.JSONDecodeError as e:
            print(f"⚠ Groq JSON parsing error: {e}")
            return self._fallback_analysis(recent)
        except Exception as e:
            print(f"⚠ Groq API error: {e}")
            return self._fallback_analysis(recent)

    def explain_recommendation_ranking(
        self,
        user_profile: Dict,
        recommendation: Dict
    ) -> str:
        """
        Generate natural language explanation for why recommendation matches user

        Args:
            user_profile: User's preference profile
            recommendation: The recommended allocation strategy

        Returns:
            Natural language explanation (1-2 sentences)
        """
        prompt = self._build_explanation_prompt(user_profile, recommendation)

        self._rate_limit()

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": EXPLANATION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=200
            )

            explanation = chat_completion.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            print(f"⚠ Groq API error: {e}")
            return self._fallback_explanation(recommendation)

    def _build_pattern_analysis_prompt(self, interactions: List[Dict]) -> str:
        """Build prompt for pattern analysis"""

        # Summarize interactions
        summary = []
        for idx, interaction in enumerate(interactions):
            selected = interaction.get('selected_recommendation_index', 0)
            recs = interaction.get('recommendations', [])

            if selected < len(recs):
                chosen = recs[selected]
                summary.append({
                    'interaction': idx + 1,
                    'chosen_strategy': chosen.get('strategy_name', 'Unknown'),
                    'cost': chosen.get('summary', {}).get('total_cost', 0),
                    'hospitals_helped': chosen.get('summary', {}).get('hospitals_helped', 0),
                    'transfer_count': chosen.get('summary', {}).get('total_transfers', 0),
                    'shortage_reduction': chosen.get('summary', {}).get('shortage_reduction_percent', 0)
                })

        prompt = f"""Analyze the following user interaction history for medical resource allocation decisions.

User selected these strategies in past {len(summary)} interactions:

{json.dumps(summary, indent=2)}

Based on this pattern, determine:
1. What type of decision-maker is this user? (cost-conscious, coverage-focused, balanced, urgency-driven)
2. What are the key patterns in their choices?
3. What objective weights would best represent their preferences?
4. How confident are you in this assessment?

Respond in JSON format with these exact fields:
{{
  "preference_type": "cost-conscious|coverage-focused|balanced|urgency-driven",
  "confidence": 0.0-1.0,
  "key_patterns": ["pattern1", "pattern2", "pattern3"],
  "reasoning": "Brief explanation in one sentence",
  "objective_weights": {{
    "minimize_cost": 0.0-1.0,
    "maximize_coverage": 0.0-1.0,
    "minimize_shortage": 0.0-1.0,
    "fairness": 0.0-1.0
  }}
}}"""

        return prompt

    def _build_explanation_prompt(
        self,
        user_profile: Dict,
        recommendation: Dict
    ) -> str:
        """Build prompt for recommendation explanation"""

        prompt = f"""A user with the following preference profile:
- Preference type: {user_profile.get('preference_type', 'unknown')}
- Key patterns: {', '.join(user_profile.get('key_patterns', []))}

Was shown this allocation recommendation:
- Strategy: {recommendation.get('strategy_name', 'Unknown')}
- Cost: ${recommendation.get('summary', {}).get('total_cost', 0):,.2f}
- Hospitals helped: {recommendation.get('summary', {}).get('hospitals_helped', 0)}
- Transfers: {recommendation.get('summary', {}).get('total_transfers', 0)}
- Shortage reduction: {recommendation.get('summary', {}).get('shortage_reduction_percent', 0):.1f}%

In 1-2 sentences, explain why this recommendation matches their preferences."""

        return prompt

    def _fallback_analysis(self, interactions: List[Dict]) -> Dict:
        """Fallback analysis if API fails"""
        return {
            'preference_type': 'balanced',
            'confidence': 0.3,
            'key_patterns': ['Insufficient data for LLM analysis'],
            'reasoning': 'Using fallback heuristic analysis due to API unavailability',
            'objective_weights': {
                'minimize_cost': 0.25,
                'maximize_coverage': 0.25,
                'minimize_shortage': 0.25,
                'fairness': 0.25
            },
            'tokens_used': 0
        }

    def _fallback_explanation(self, recommendation: Dict) -> str:
        """Fallback explanation if API fails"""
        strategy = recommendation.get('strategy_name', 'This strategy')
        return f"{strategy} balances cost, coverage, and urgency effectively."


# System prompts
PREFERENCE_ANALYSIS_SYSTEM_PROMPT = """You are an expert data analyst specializing in healthcare resource allocation decision patterns.

Your task is to analyze user interaction histories and identify their decision-making preferences.

Key considerations:
- Users may prioritize cost-efficiency, broad coverage, urgent needs, or balanced approaches
- Look for consistent patterns across multiple decisions
- Higher costs with better coverage suggests coverage-focused
- Lower costs with fewer hospitals suggests cost-conscious
- High shortage reduction suggests urgency-driven
- Balanced metrics suggests balanced approach

Always respond in valid JSON format with the exact structure requested."""

EXPLANATION_SYSTEM_PROMPT = """You are a helpful assistant explaining medical resource allocation recommendations.

Your explanations should:
- Be concise (1-2 sentences maximum)
- Highlight why the recommendation fits the user's preferences
- Use concrete numbers when relevant
- Avoid technical jargon
- Be conversational and clear

Example: "This recommendation aligns with your cost-conscious approach, achieving 85% shortage reduction at 30% lower cost than alternatives."
"""
