import json
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class PersonalityAssessmentScorer:
    """AI-powered personality assessment scoring using Gemini AI"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def score_mbti_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score MBTI assessment using AI analysis of responses

        Args:
            assessment_data: Dictionary containing:
                - responses: List of question responses with question_id and answer
                - questions: List of question objects with id and text
                - user_info: Optional user information

        Returns:
            Dictionary with MBTI type, detailed analysis, and recommendations
        """
        try:
            # Extract responses and questions
            responses = assessment_data.get('responses', [])
            questions = assessment_data.get('questions', [])
            user_info = assessment_data.get('user_info', {})

            # Format responses for AI analysis
            response_text = self._format_mbti_responses(responses, questions)

            # Create AI prompt for MBTI analysis
            prompt = f"""
            You are a professional personality psychologist specializing in MBTI assessment.

            Analyze the following assessment responses and determine the person's MBTI personality type.
            Provide detailed insights about their personality, strengths, challenges, and career recommendations.

            Assessment Responses:
            {response_text}

            Please provide the analysis in the following JSON format:
            {{
                "mbti_type": "ENTJ",
                "type_name": "The Commander",
                "description": "Detailed description of the personality type",
                "scores": {{
                    "E": 75,
                    "I": 25,
                    "S": 40,
                    "N": 60,
                    "T": 80,
                    "F": 20,
                    "J": 90,
                    "P": 10
                }},
                "strengths": ["List of key strengths"],
                "challenges": ["List of potential challenges"],
                "career_suggestions": ["List of suitable careers"],
                "relationship_insights": "Insights about relationships and communication",
                "workplace_insights": "Insights about work behavior and environment",
                "detailed_analysis": "Comprehensive personality analysis"
            }}

            Base your analysis on the patterns in their responses and provide thoughtful, personalized insights.
            """

            # Get AI response
            response = self.model.generate_content(prompt)

            # Parse the AI response
            try:
                # Extract JSON from response
                response_text = response.text
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)

                    # Add metadata
                    result['assessment_type'] = 'MBTI'
                    result['total_questions'] = len(questions)
                    result['completed_at'] = assessment_data.get('completed_at', '')

                    return result
                else:
                    raise ValueError("Could not extract JSON from AI response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                # Return fallback analysis
                return self._get_fallback_mbti_result(assessment_data)

        except Exception as e:
            logger.error(f"Error scoring MBTI assessment: {e}")
            return self._get_fallback_mbti_result(assessment_data)

    def score_big_five_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Big Five assessment using AI analysis

        Args:
            assessment_data: Dictionary containing responses and questions

        Returns:
            Dictionary with Big Five scores and detailed analysis
        """
        try:
            responses = assessment_data.get('responses', [])
            questions = assessment_data.get('questions', [])

            # Format responses for AI analysis
            response_text = self._format_big_five_responses(responses, questions)

            prompt = f"""
            You are a personality psychology expert specializing in the Big Five (OCEAN) model.

            Analyze the following assessment responses and provide detailed Big Five personality analysis.

            Assessment Responses:
            {response_text}

            Please provide the analysis in the following JSON format:
            {{
                "ocean_scores": {{
                    "Openness": {{"score": 85, "description": "High openness to experience"}},
                    "Conscientiousness": {{"score": 70, "description": "Well-organized and disciplined"}},
                    "Extraversion": {{"score": 60, "description": "Moderately extraverted"}},
                    "Agreeableness": {{"score": 75, "description": "Cooperative and compassionate"}},
                    "Neuroticism": {{"score": 30, "description": "Emotionally stable"}}
                }},
                "personality_summary": "Overall personality summary",
                "strengths": ["List of key personality strengths"],
                "growth_areas": ["Areas for personal development"],
                "career_recommendations": ["List of suitable careers"],
                "relationship_style": "Description of relationship tendencies",
                "work_style": "Description of work behavior and preferences",
                "detailed_analysis": "Comprehensive personality breakdown"
            }}

            Scores should be on a scale of 0-100 where higher scores indicate more of the trait.
            """

            response = self.model.generate_content(prompt)

            try:
                response_text = response.text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)

                    # Add metadata
                    result['assessment_type'] = 'Big Five'
                    result['total_questions'] = len(questions)
                    result['completed_at'] = assessment_data.get('completed_at', '')

                    return result
                else:
                    raise ValueError("Could not extract JSON from AI response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                return self._get_fallback_big_five_result(assessment_data)

        except Exception as e:
            logger.error(f"Error scoring Big Five assessment: {e}")
            return self._get_fallback_big_five_result(assessment_data)

    def score_disc_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score DISC assessment using AI analysis

        Args:
            assessment_data: Dictionary containing responses and questions

        Returns:
            Dictionary with DISC profile and workplace behavior analysis
        """
        try:
            responses = assessment_data.get('responses', [])
            questions = assessment_data.get('questions', [])

            response_text = self._format_disc_responses(responses, questions)

            prompt = f"""
            You are an organizational psychology expert specializing in DISC workplace behavior assessment.

            Analyze the following assessment responses and provide detailed DISC profile analysis.

            Assessment Responses:
            {response_text}

            Please provide the analysis in the following JSON format:
            {{
                "disc_profile": "DC",
                "profile_name": "Challenger",
                "description": "Description of the DISC profile type",
                "trait_composition": [
                    {{"letter": "D", "percentage": 45, "description": "Dominance"}},
                    {{"letter": "I", "percentage": 15, "description": "Influence"}},
                    {{"letter": "S", "percentage": 20, "description": "Steadiness"}},
                    {{"letter": "C", "percentage": 20, "description": "Conscientiousness"}}
                ],
                "workplace_insights": [
                    {{
                        "title": "Communication Style",
                        "icon": "fas fa-comments",
                        "description": "How they communicate in the workplace"
                    }},
                    {{
                        "title": "Decision Making",
                        "icon": "fas fa-brain",
                        "description": "Their approach to making decisions"
                    }}
                ],
                "strengths": ["List of workplace strengths"],
                "challenges": ["Potential workplace challenges"],
                "careers": ["List of suitable careers"],
                "quote": "Inspirational quote representing this profile"
            }}

            Ensure the percentages add up to 100%.
            """

            response = self.model.generate_content(prompt)

            try:
                response_text = response.text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)

                    # Add metadata
                    result['assessment_type'] = 'DISC'
                    result['total_questions'] = len(questions)
                    result['completed_at'] = assessment_data.get('completed_at', '')

                    return result
                else:
                    raise ValueError("Could not extract JSON from AI response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                return self._get_fallback_disc_result(assessment_data)

        except Exception as e:
            logger.error(f"Error scoring DISC assessment: {e}")
            return self._get_fallback_disc_result(assessment_data)

    def _format_mbti_responses(self, responses: List[Dict], questions: List[Dict]) -> str:
        """Format MBTI responses for AI analysis"""
        formatted = []
        for response in responses:
            q_id = response.get('question_id')
            answer = response.get('answer', '')

            # Find the question text
            question_text = ""
            for q in questions:
                if str(q.get('id')) == str(q_id):
                    question_text = q.get('question', '')
                    break

            if question_text:
                formatted.append(f"Q: {question_text}\nA: {answer}\n")

        return "\n".join(formatted)

    def _format_big_five_responses(self, responses: List[Dict], questions: List[Dict]) -> str:
        """Format Big Five responses for AI analysis"""
        formatted = []
        for response in responses:
            q_id = response.get('question_id')
            answer = response.get('answer', '')

            question_text = ""
            for q in questions:
                if str(q.get('id')) == str(q_id):
                    question_text = q.get('question', '')
                    break

            if question_text:
                formatted.append(f"Q: {question_text}\nA: {answer}\n")

        return "\n".join(formatted)

    def _format_disc_responses(self, responses: List[Dict], questions: List[Dict]) -> str:
        """Format DISC responses for AI analysis"""
        formatted = []
        for response in responses:
            q_id = response.get('question_id')
            answer = response.get('answer', '')

            question_text = ""
            for q in questions:
                if str(q.get('id')) == str(q_id):
                    question_text = q.get('question', '')
                    break

            if question_text:
                formatted.append(f"Q: {question_text}\nA: {answer}\n")

        return "\n".join(formatted)

    def _get_fallback_mbti_result(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback MBTI result if AI scoring fails"""
        return {
            "assessment_type": "MBTI",
            "mbti_type": "ENFP",
            "type_name": "The Champion",
            "description": "You are an enthusiastic, creative, and sociable person who loves to explore new possibilities.",
            "scores": {"E": 70, "I": 30, "S": 40, "N": 60, "T": 50, "F": 50, "J": 40, "P": 60},
            "strengths": ["Creative", "Enthusiastic", "Sociable", "Innovative"],
            "challenges": ["May struggle with routine", "Can be disorganized"],
            "career_suggestions": ["Creative Director", "Marketing", "Entrepreneur", "Writer"],
            "relationship_insights": "You value deep connections and authentic communication.",
            "workplace_insights": "You thrive in creative environments with autonomy.",
            "detailed_analysis": "AI analysis temporarily unavailable. This is a general assessment."
        }

    def _get_fallback_big_five_result(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback Big Five result if AI scoring fails"""
        return {
            "assessment_type": "Big Five",
            "ocean_scores": {
                "Openness": {"score": 70, "description": "Open to new experiences"},
                "Conscientiousness": {"score": 60, "description": "Fairly organized"},
                "Extraversion": {"score": 65, "description": "Moderately extraverted"},
                "Agreeableness": {"score": 75, "description": "Generally agreeable"},
                "Neuroticism": {"score": 35, "description": "Emotionally stable"}
            },
            "personality_summary": "You are a well-balanced individual with openness to experience.",
            "strengths": ["Adaptable", "Emotionally stable", "Cooperative"],
            "growth_areas": ["Further develop organizational skills"],
            "career_recommendations": ["Project Manager", "Team Lead", "Consultant"],
            "relationship_style": "You value harmony in relationships.",
            "work_style": "You work well in team environments.",
            "detailed_analysis": "AI analysis temporarily unavailable. This is a general assessment."
        }

    def _get_fallback_disc_result(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback DISC result if AI scoring fails"""
        return {
            "assessment_type": "DISC",
            "disc_profile": "SI",
            "profile_name": "Specialist",
            "description": "You are a supportive, stable individual who values relationships and quality.",
            "trait_composition": [
                {"letter": "D", "percentage": 15, "description": "Dominance"},
                {"letter": "I", "percentage": 35, "description": "Influence"},
                {"letter": "S", "percentage": 35, "description": "Steadiness"},
                {"letter": "C", "percentage": 15, "description": "Conscientiousness"}
            ],
            "workplace_insights": [
                {"title": "Team Collaboration", "icon": "fas fa-users", "description": "Excellent team player"},
                {"title": "Consistency", "icon": "fas fa-chart-line", "description": "Reliable and dependable"}
            ],
            "strengths": ["Supportive", "Reliable", "Patient", "Cooperative"],
            "challenges": ["May avoid conflict", "Resistant to sudden change"],
            "careers": ["Customer Service", "HR Specialist", "Teacher", "Healthcare"],
            "quote": "Alone we can do so little; together we can do so much."
        }

def score_personality_assessment(assessment_data: Dict[str, Any], model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """
    Main function to score personality assessments

    Args:
        assessment_data: Dictionary containing assessment data
        model_name: Name of the AI model to use

    Returns:
        Dictionary with assessment results
    """
    scorer = PersonalityAssessmentScorer(model_name)
    assessment_type = assessment_data.get('assessment_type', '').lower()

    if assessment_type == 'mbti':
        return scorer.score_mbti_assessment(assessment_data)
    elif assessment_type == 'big-five':
        return scorer.score_big_five_assessment(assessment_data)
    elif assessment_type == 'disc':
        return scorer.score_disc_assessment(assessment_data)
    else:
        # Try to determine type from data
        if any('mbti' in str(k).lower() for k in assessment_data.keys()):
            return scorer.score_mbti_assessment(assessment_data)
        elif any('big-five' in str(k).lower() or 'ocean' in str(k).lower() for k in assessment_data.keys()):
            return scorer.score_big_five_assessment(assessment_data)
        elif any('disc' in str(k).lower() for k in assessment_data.keys()):
            return scorer.score_disc_assessment(assessment_data)
        else:
            # Default fallback
            return scorer.score_mbti_assessment(assessment_data)