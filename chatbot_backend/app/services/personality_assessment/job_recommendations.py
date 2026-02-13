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

class PersonalityJobRecommender:
    """AI-powered job recommendations based on personality assessment results"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def get_job_recommendations(self, personality_data: Dict[str, Any], user_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate job recommendations based on personality assessment results

        Args:
            personality_data: Dictionary containing personality assessment results
            user_profile: Optional user profile information

        Returns:
            Dictionary with job recommendations and career insights
        """
        try:
            assessment_type = personality_data.get('assessment_type', 'MBTI').upper()

            # Create personality summary for AI analysis
            personality_summary = self._create_personality_summary(personality_data)

            # Create comprehensive prompt for job recommendations
            prompt = f"""
            You are an expert career counselor and HR professional specializing in career guidance based on personality assessments.

            Based on the following personality assessment results, provide comprehensive job recommendations and career insights:

            Personality Assessment Results ({assessment_type}):
            {personality_summary}

            User Profile (if available):
            {json.dumps(user_profile, indent=2) if user_profile else "No additional user information provided"}

            Please provide detailed job recommendations in the following JSON format:
            {{
                "primary_career_paths": [
                    {{
                        "category": "Technology & Innovation",
                        "job_titles": ["Software Developer", "Data Scientist", "UX Designer"],
                        "match_percentage": 85,
                        "reasoning": "Detailed explanation why this career path suits the personality",
                        "growth_potential": "High growth opportunities in tech industry",
                        "salary_range": "$70,000 - $120,000",
                        "work_environment": "Innovative, collaborative, results-oriented"
                    }}
                ],
                "secondary_career_paths": [
                    {{
                        "category": "Creative Arts",
                        "job_titles": ["Graphic Designer", "Content Creator", "Marketing Specialist"],
                        "match_percentage": 70,
                        "reasoning": "Secondary fit based on creative aspects of personality",
                        "growth_potential": "Moderate to high growth opportunities"
                    }}
                ],
                "skills_to_develop": [
                    {{
                        "skill": "Leadership",
                        "importance": "High",
                        "development_tips": "Take on leadership roles in projects, pursue management training",
                        "time_to_develop": "1-2 years"
                    }}
                ],
                "work_environment_preferences": {{
                    "ideal_environment": "Collaborative, innovative, flexible",
                    "avoid": "Highly structured, routine-based environments",
                    "team_size": "Small to medium teams (5-20 people)",
                    "management_style": "Supportive and empowering leadership"
                }},
                "entrepreneurial_suitability": {{
                    "score": 75,
                    "analysis": "Strong entrepreneurial potential with creativity and risk-taking abilities",
                    "recommended_startup_types": ["Tech startup", "Creative agency", "Consulting"]
                }},
                "industry_recommendations": [
                    "Technology",
                    "Creative industries",
                    "Healthcare",
                    "Education"
                ],
                "career_development_roadmap": {{
                    "entry_level": {{
                        "position": "Junior Developer / Analyst",
                        "timeframe": "0-2 years",
                        "focus": "Technical skills and industry knowledge"
                    }},
                    "mid_level": {{
                        "position": "Senior Developer / Team Lead",
                        "timeframe": "2-5 years",
                        "focus": "Specialization and leadership skills"
                    }},
                    "senior_level": {{
                        "position": "Manager / Director",
                        "timeframe": "5+ years",
                        "focus": "Strategic planning and organizational leadership"
                    }}
                }},
                "additional_insights": {{
                    "networking_tips": "Focus on industry conferences and professional communities",
                    "interview_strengths": "Creative problem-solving and enthusiasm",
                    "potential_challenges": "May need to develop more structured planning habits"
                }}
            }}

            Ensure the recommendations are practical, actionable, and tailored to the specific personality profile.
            Include realistic salary ranges and growth potential based on current market trends.
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
                    result['assessment_type'] = assessment_type
                    result['generated_at'] = personality_data.get('completed_at', '')
                    result['confidence_score'] = 0.9

                    return result
                else:
                    raise ValueError("Could not extract JSON from AI response")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                return self._get_fallback_job_recommendations(personality_data)

        except Exception as e:
            logger.error(f"Error generating job recommendations: {e}")
            return self._get_fallback_job_recommendations(personality_data)

    def _create_personality_summary(self, personality_data: Dict[str, Any]) -> str:
        """Create a comprehensive summary of personality assessment results"""
        assessment_type = personality_data.get('assessment_type', 'MBTI')

        if assessment_type.upper() == 'MBTI':
            mbti_type = personality_data.get('mbti_type', 'ENFP')
            scores = personality_data.get('scores', {})
            strengths = personality_data.get('strengths', [])
            return f"""
            MBTI Type: {mbti_type}
            Cognitive Function Scores: E={scores.get('E', 0)}%, I={scores.get('I', 0)}%, S={scores.get('S', 0)}%, N={scores.get('N', 0)}%, T={scores.get('T', 0)}%, F={scores.get('F', 0)}%, J={scores.get('J', 0)}%, P={scores.get('P', 0)}%
            Key Strengths: {', '.join(strengths[:5])}
            Personality Description: {personality_data.get('description', '')}
            """

        elif assessment_type.upper() == 'BIG FIVE':
            ocean_scores = personality_data.get('ocean_scores', {})
            traits_summary = []
            for trait, data in ocean_scores.items():
                traits_summary.append(f"{trait}: {data.get('score', 0)}/100 - {data.get('description', '')}")

            return f"""
            Big Five (OCEAN) Profile:
            {chr(10).join(traits_summary)}
            Personality Summary: {personality_data.get('personality_summary', '')}
            Key Strengths: {', '.join(personality_data.get('strengths', [])[:5])}
            """

        elif assessment_type.upper() == 'DISC':
            disc_profile = personality_data.get('disc_profile', 'SI')
            profile_name = personality_data.get('profile_name', 'Specialist')
            traits = personality_data.get('trait_composition', [])

            traits_summary = []
            for trait in traits:
                traits_summary.append(f"{trait.get('letter', '')}: {trait.get('percentage', 0)}%")

            return f"""
            DISC Profile: {disc_profile} - {profile_name}
            Trait Composition: {', '.join(traits_summary)}
            Description: {personality_data.get('description', '')}
            Key Strengths: {', '.join(personality_data.get('strengths', [])[:5])}
            """

        else:
            return "Generic personality assessment results available"

    def _get_fallback_job_recommendations(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback job recommendations if AI service fails"""
        assessment_type = personality_data.get('assessment_type', 'MBTI').upper()

        if assessment_type == 'MBTI':
            return self._get_mbti_fallback_recommendations(personality_data)
        elif assessment_type == 'BIG FIVE':
            return self._get_bigfive_fallback_recommendations(personality_data)
        else:
            return self._get_disc_fallback_recommendations(personality_data)

    def _get_mbti_fallback_recommendations(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback MBTI job recommendations"""
        mbti_type = personality_data.get('mbti_type', 'ENFP')

        # Define career recommendations for different MBTI types
        career_map = {
            'ENTJ': {
                'primary': [
                    {
                        'category': 'Leadership & Management',
                        'job_titles': ['CEO', 'Management Consultant', 'Executive Director', 'Strategy Manager'],
                        'match_percentage': 90,
                        'reasoning': 'Natural leadership abilities with strategic thinking and result-orientation',
                        'growth_potential': 'Executive leadership opportunities',
                        'salary_range': '$120,000 - $250,000+',
                        'work_environment': 'Strategic, results-focused, high-autonomy'
                    }
                ],
                'secondary': [
                    {
                        'category': 'Technology & Innovation',
                        'job_titles': ['CTO', 'Startup Founder', 'Product Manager'],
                        'match_percentage': 75,
                        'growth_potential': 'Technology leadership paths'
                    }
                ]
            },
            'ENFP': {
                'primary': [
                    {
                        'category': 'Creative & Communications',
                        'job_titles': ['Marketing Manager', 'Content Creator', 'UX Researcher', 'Public Relations Specialist'],
                        'match_percentage': 85,
                        'reasoning': 'Strong creativity, empathy, and communication skills',
                        'growth_potential': 'Creative leadership and brand development',
                        'salary_range': '$60,000 - $120,000',
                        'work_environment': 'Creative, collaborative, flexible'
                    }
                ],
                'secondary': [
                    {
                        'category': 'Human Services',
                        'job_titles': ['Counselor', 'Teacher', 'Social Worker'],
                        'match_percentage': 80,
                        'growth_potential': 'Helping professions with advancement'
                    }
                ]
            },
            'INTJ': {
                'primary': [
                    {
                        'category': 'Strategy & Analysis',
                        'job_titles': ['Data Scientist', 'Management Consultant', 'Research Director', 'Systems Architect'],
                        'match_percentage': 90,
                        'reasoning': 'Strategic thinking with analytical capabilities and long-range planning',
                        'growth_potential': 'Executive and expert-level positions',
                        'salary_range': '$90,000 - $200,000',
                        'work_environment': 'Intellectual, autonomous, challenging'
                    }
                ]
            }
        }

        # Default to ENFP recommendations if type not found
        recommendations = career_map.get(mbti_type, career_map['ENFP'])

        return {
            'assessment_type': 'MBTI',
            'primary_career_paths': recommendations['primary'],
            'secondary_career_paths': recommendations.get('secondary', []),
            'skills_to_develop': [
                {
                    'skill': 'Leadership',
                    'importance': 'Medium',
                    'development_tips': 'Take on leadership opportunities in projects',
                    'time_to_develop': '2-3 years'
                }
            ],
            'work_environment_preferences': {
                'ideal_environment': 'Collaborative and innovative',
                'avoid': 'Highly structured and routine-based',
                'team_size': 'Small to medium teams',
                'management_style': 'Supportive and empowering'
            },
            'entrepreneurial_suitability': {
                'score': 65,
                'analysis': 'Moderate entrepreneurial potential',
                'recommended_startup_types': ['Creative services', 'Consulting']
            },
            'industry_recommendations': ['Technology', 'Creative industries', 'Healthcare', 'Education'],
            'generated_at': personality_data.get('completed_at', ''),
            'confidence_score': 0.7
        }

    def _get_bigfive_fallback_recommendations(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback Big Five job recommendations"""
        ocean_scores = personality_data.get('ocean_scores', {})

        # Determine career paths based on high-scoring traits
        recommendations = []

        if ocean_scores.get('Openness', {}).get('score', 0) > 70:
            recommendations.append({
                'category': 'Creative & Innovative',
                'job_titles': ['UX Designer', 'Marketing Specialist', 'Content Developer'],
                'match_percentage': 85,
                'reasoning': 'High openness to experience and creativity',
                'growth_potential': 'Creative and innovation roles'
            })

        if ocean_scores.get('Conscientiousness', {}).get('score', 0) > 70:
            recommendations.append({
                'category': 'Professional Services',
                'job_titles': ['Project Manager', 'Financial Analyst', 'Operations Manager'],
                'match_percentage': 80,
                'reasoning': 'Strong organization and discipline skills',
                'growth_potential': 'Management and leadership paths'
            })

        if ocean_scores.get('Extraversion', {}).get('score', 0) > 70:
            recommendations.append({
                'category': 'Sales & Customer Relations',
                'job_titles': ['Sales Manager', 'Customer Success Manager', 'Public Relations'],
                'match_percentage': 75,
                'reasoning': 'Strong interpersonal and communication skills',
                'growth_potential': 'Customer-facing leadership roles'
            })

        return {
            'assessment_type': 'Big Five',
            'primary_career_paths': recommendations[:2] if recommendations else [{
                'category': 'Professional Services',
                'job_titles': ['Business Analyst', 'Project Coordinator'],
                'match_percentage': 70,
                'growth_potential': 'Professional development opportunities'
            }],
            'secondary_career_paths': [],
            'skills_to_develop': [
                {
                    'skill': 'Communication',
                    'importance': 'High',
                    'development_tips': 'Join professional networks and practice public speaking',
                    'time_to_develop': '6-12 months'
                }
            ],
            'work_environment_preferences': {
                'ideal_environment': 'Collaborative and growth-oriented',
                'avoid': 'Highly restrictive environments',
                'team_size': 'Medium-sized teams',
                'management_style': 'Supportive leadership'
            },
            'entrepreneurial_suitability': {
                'score': 60,
                'analysis': 'Moderate entrepreneurial potential',
                'recommended_startup_types': ['Professional services', 'Consulting']
            },
            'industry_recommendations': ['Technology', 'Healthcare', 'Education', 'Finance'],
            'generated_at': personality_data.get('completed_at', ''),
            'confidence_score': 0.7
        }

    def _get_disc_fallback_recommendations(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback DISC job recommendations"""
        disc_profile = personality_data.get('disc_profile', 'SI')

        disc_careers = {
            'D': {
                'primary': [
                    {
                        'category': 'Leadership & Management',
                        'job_titles': ['CEO', 'Director', 'Sales Manager', 'Entrepreneur'],
                        'match_percentage': 90,
                        'reasoning': 'Natural leadership and decision-making abilities',
                        'growth_potential': 'Executive leadership opportunities'
                    }
                ]
            },
            'I': {
                'primary': [
                    {
                        'category': 'Sales & Marketing',
                        'job_titles': ['Sales Manager', 'Marketing Director', 'Public Relations', 'Communications'],
                        'match_percentage': 85,
                        'reasoning': 'Strong persuasion and communication skills',
                        'growth_potential': 'Customer-facing leadership roles'
                    }
                ]
            },
            'S': {
                'primary': [
                    {
                        'category': 'Support Services',
                        'job_titles': ['HR Manager', 'Customer Service', 'Teacher', 'Healthcare Provider'],
                        'match_percentage': 80,
                        'reasoning': 'Strong support and relationship-building skills',
                        'growth_potential': 'Service management positions'
                    }
                ]
            },
            'C': {
                'primary': [
                    {
                        'category': 'Analytical & Technical',
                        'job_titles': ['Data Analyst', 'Quality Assurance', 'Financial Analyst', 'Engineer'],
                        'match_percentage': 85,
                        'reasoning': 'Strong analytical and quality-focused mindset',
                        'growth_potential': 'Technical expert and specialist roles'
                    }
                ]
            }
        }

        # Handle combined profiles (e.g., DI, SC)
        primary_trait = disc_profile[0] if disc_profile else 'S'
        recommendations = disc_careers.get(primary_trait, disc_careers['S'])

        return {
            'assessment_type': 'DISC',
            'primary_career_paths': recommendations['primary'],
            'secondary_career_paths': [],
            'skills_to_develop': [
                {
                    'skill': 'Communication',
                    'importance': 'High',
                    'development_tips': 'Focus on both verbal and written communication skills',
                    'time_to_develop': '6-12 months'
                }
            ],
            'work_environment_preferences': {
                'ideal_environment': 'Collaborative and results-focused',
                'avoid': 'Highly chaotic or overly restrictive environments',
                'team_size': 'Small to medium teams',
                'management_style': 'Supportive and goal-oriented'
            },
            'entrepreneurial_suitability': {
                'score': 70,
                'analysis': 'Good entrepreneurial potential with adaptability',
                'recommended_startup_types': ['Service businesses', 'Consulting']
            },
            'industry_recommendations': ['Technology', 'Sales', 'Healthcare', 'Finance'],
            'generated_at': personality_data.get('completed_at', ''),
            'confidence_score': 0.7
        }

def get_personality_job_recommendations(personality_data: Dict[str, Any],
                                         user_profile: Optional[Dict] = None,
                                         model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """
    Main function to get job recommendations based on personality assessment

    Args:
        personality_data: Dictionary containing personality assessment results
        user_profile: Optional user profile information
        model_name: Name of the AI model to use

    Returns:
        Dictionary with job recommendations and career insights
    """
    recommender = PersonalityJobRecommender(model_name)
    return recommender.get_job_recommendations(personality_data, user_profile)