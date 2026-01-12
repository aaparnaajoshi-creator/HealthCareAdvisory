#added this file for v 0.6
# =============================================================================
# AUTO-CATEGORIZER SERVICE
# =============================================================================
# LLM-based categorization for KB articles using CI list as ontology

import json
from typing import Optional, Dict, Any
from openai import OpenAI
from config import Config

class AutoCategorizer:
    """
    Automatically categorize KB articles using LLM with CI ontology.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.CATEGORIZATION_MODEL
        self.confidence_threshold = Config.CATEGORIZATION_CONFIDENCE_THRESHOLD
        self.ci_list = []
    
    def load_ci_list(self, ci_categories: list):
        """
        Load CI list from database records.
        
        Args:
            ci_categories: List of CICategory objects or dicts with 'ci_name' key
        """
        self.ci_list = []
        for ci in ci_categories:
            if hasattr(ci, 'ci_name'):
                self.ci_list.append(ci.ci_name)
            elif isinstance(ci, dict) and 'ci_name' in ci:
                self.ci_list.append(ci['ci_name'])
            elif isinstance(ci, str):
                self.ci_list.append(ci)
    
    def load_ci_list_from_db(self, db_session):
        """
        Load CI list directly from database.
        
        Args:
            db_session: SQLAlchemy database session
        """
        from models import CICategory
        categories = db_session.query(CICategory).filter_by(is_active=True).all()
        self.ci_list = [cat.ci_name for cat in categories]
    
    def _build_ontology_string(self) -> str:
        """Build numbered CI list for prompt."""
        if not self.ci_list:
            return "No CI categories available."
        
        lines = []
        for i, ci in enumerate(self.ci_list, 1):
            lines.append(f"{i}. {ci}")
        return "\n".join(lines)
    
    def _build_prompt(self, title: str, short_description: str, article_body: str) -> str:
        """
        Build the categorization prompt.
        
        Args:
            title: KB article title
            short_description: KB article short description
            article_body: KB article full content (will be truncated if too long)
        """
        # Truncate article body to avoid token limits
        max_body_length = 2000
        truncated_body = article_body[:max_body_length] if article_body else ""
        if len(article_body or "") > max_body_length:
            truncated_body += "... [truncated]"
        
        ontology = self._build_ontology_string()
        
        prompt = f"""You are a categorization assistant for IT Service Management. 
Your task is to assign a KB article to the most appropriate CI (Configuration Item) / Service from the provided list.

## AVAILABLE CI/SERVICES (Ontology):
{ontology}

## KB ARTICLE TO CATEGORIZE:

**Title:** {title}

**Short Description:** {short_description}

**Content Preview:**
{truncated_body}

## INSTRUCTIONS:
1. Analyze the KB article content carefully
2. Match it to the MOST relevant CI/Service from the list above
3. If the article could belong to multiple CIs, choose the PRIMARY one
4. If no CI matches well (confidence < 0.5), use "Uncategorized"

## RESPONSE FORMAT:
Respond with ONLY a valid JSON object (no markdown, no explanation):
{{
    "ci_name": "exact CI name from the list above",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief 1-2 sentence explanation"
}}

Examples of confidence levels:
- 0.9-1.0: Perfect match, article is clearly about this CI
- 0.7-0.9: Strong match, article is primarily about this CI
- 0.5-0.7: Moderate match, article relates to this CI
- Below 0.5: Weak match, use "Uncategorized"
"""
        return prompt
    
    def categorize(
        self, 
        title: str, 
        short_description: str = "", 
        article_body: str = ""
    ) -> Dict[str, Any]:
        """
        Categorize a KB article using LLM.
        
        Args:
            title: KB article title
            short_description: KB article short description
            article_body: KB article full content
            
        Returns:
            Dict with keys: ci_name, confidence, reasoning, needs_review, error
        """
        # Default response for errors
        default_response = {
            "ci_name": "Uncategorized",
            "confidence": 0.0,
            "reasoning": "Categorization failed",
            "needs_review": True,
            "error": None
        }
        
        # Check if CI list is loaded
        if not self.ci_list:
            default_response["error"] = "CI list not loaded. Call load_ci_list() first."
            return default_response
        
        # Check if we have content to categorize
        if not title and not short_description and not article_body:
            default_response["error"] = "No content provided for categorization."
            return default_response
        
        try:
            # Build prompt
            prompt = self._build_prompt(title, short_description, article_body)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise categorization assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            # Parse JSON
            result = json.loads(result_text)
            
            # Validate CI name exists in list
            ci_name = result.get("ci_name", "Uncategorized")
            if ci_name != "Uncategorized" and ci_name not in self.ci_list:
                # Try to find close match (case-insensitive)
                ci_lower = ci_name.lower()
                found = False
                for ci in self.ci_list:
                    if ci.lower() == ci_lower:
                        ci_name = ci  # Use exact case from list
                        found = True
                        break
                if not found:
                    ci_name = "Uncategorized"
                    result["reasoning"] = f"LLM suggested '{result.get('ci_name')}' which is not in CI list."
                    result["confidence"] = 0.3
            
            # Determine if needs review
            confidence = float(result.get("confidence", 0.0))
            needs_review = confidence < self.confidence_threshold
            
            return {
                "ci_name": ci_name,
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
                "needs_review": needs_review,
                "error": None
            }
            
        except json.JSONDecodeError as e:
            default_response["error"] = f"Failed to parse LLM response: {str(e)}"
            return default_response
        except Exception as e:
            default_response["error"] = f"Categorization error: {str(e)}"
            return default_response
    
    def categorize_batch(
        self, 
        articles: list,
        progress_callback: Optional[callable] = None
    ) -> list:
        """
        Categorize multiple KB articles.
        
        Args:
            articles: List of dicts with keys: title, short_description, article_body
            progress_callback: Optional function called with (current, total) for progress
            
        Returns:
            List of categorization results
        """
        results = []
        total = len(articles)
        
        for i, article in enumerate(articles):
            result = self.categorize(
                title=article.get("title", ""),
                short_description=article.get("short_description", ""),
                article_body=article.get("article_body", "")
            )
            result["source_id"] = article.get("article_id", f"article_{i}")
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results


# =============================================================================
# HELPER FUNCTION FOR EASY USAGE
# =============================================================================

def categorize_kb_article(
    title: str,
    short_description: str = "",
    article_body: str = "",
    ci_list: list = None,
    db_session = None
) -> Dict[str, Any]:
    """
    Convenience function to categorize a single KB article.
    
    Args:
        title: KB article title
        short_description: KB article short description
        article_body: KB article full content
        ci_list: List of CI names (optional if db_session provided)
        db_session: SQLAlchemy session to load CI list from DB
        
    Returns:
        Dict with categorization result
    """
    categorizer = AutoCategorizer()
    
    if db_session:
        categorizer.load_ci_list_from_db(db_session)
    elif ci_list:
        categorizer.load_ci_list(ci_list)
    else:
        return {
            "ci_name": "Uncategorized",
            "confidence": 0.0,
            "reasoning": "No CI list provided",
            "needs_review": True,
            "error": "Must provide either ci_list or db_session"
        }
    
    return categorizer.categorize(title, short_description, article_body)
