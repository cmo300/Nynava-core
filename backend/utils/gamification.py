"""
Nynava - Gamification System
User engagement through badges, points, and leaderboards
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import structlog

logger = structlog.get_logger()

class BadgeType(Enum):
    """Types of badges users can earn"""
    FIRST_UPLOAD = "first_upload"
    DATA_CONTRIBUTOR = "data_contributor"
    PRIVACY_CHAMPION = "privacy_champion"
    AI_PIONEER = "ai_pioneer"
    COMMUNITY_HELPER = "community_helper"
    RESEARCHER = "researcher"
    MILESTONE_10 = "milestone_10"
    MILESTONE_50 = "milestone_50"
    MILESTONE_100 = "milestone_100"
    EARLY_ADOPTER = "early_adopter"
    FEEDBACK_PROVIDER = "feedback_provider"

class ActivityType(Enum):
    """Types of activities that earn points"""
    CONSENT_SUBMISSION = "consent_submission"
    DATA_UPLOAD = "data_upload"
    MODEL_USAGE = "model_usage"
    DATASET_CONTRIBUTION = "dataset_contribution"
    MODEL_IMPROVEMENT = "model_improvement"
    FEEDBACK_SUBMISSION = "feedback_submission"
    REFERRAL = "referral"
    COMMUNITY_PARTICIPATION = "community_participation"

class GamificationManager:
    """Manages user gamification features"""
    
    def __init__(self):
        # In-memory storage for demo (use database in production)
        self.user_profiles = {}
        self.leaderboard_cache = None
        self.cache_expiry = None
        
        # Badge definitions
        self.badge_definitions = {
            BadgeType.FIRST_UPLOAD: {
                'name': 'First Steps',
                'description': 'Uploaded your first medical data',
                'icon': '🏥',
                'points': 100,
                'criteria': {'data_uploads': 1}
            },
            BadgeType.DATA_CONTRIBUTOR: {
                'name': 'Data Contributor',
                'description': 'Contributed 5 or more datasets',
                'icon': '📊',
                'points': 250,
                'criteria': {'data_uploads': 5}
            },
            BadgeType.PRIVACY_CHAMPION: {
                'name': 'Privacy Champion',
                'description': 'Completed advanced privacy settings',
                'icon': '🔒',
                'points': 150,
                'criteria': {'privacy_actions': 3}
            },
            BadgeType.AI_PIONEER: {
                'name': 'AI Pioneer',
                'description': 'Used 3 different AI models',
                'icon': '🤖',
                'points': 200,
                'criteria': {'unique_models_used': 3}
            },
            BadgeType.COMMUNITY_HELPER: {
                'name': 'Community Helper',
                'description': 'Helped improve 2 AI models',
                'icon': '🤝',
                'points': 300,
                'criteria': {'model_improvements': 2}
            },
            BadgeType.RESEARCHER: {
                'name': 'Researcher',
                'description': 'Accessed 10 different datasets',
                'icon': '🔬',
                'points': 400,
                'criteria': {'datasets_accessed': 10}
            },
            BadgeType.MILESTONE_10: {
                'name': '10 Contributions',
                'description': 'Made 10 total contributions',
                'icon': '🎯',
                'points': 500,
                'criteria': {'total_contributions': 10}
            },
            BadgeType.MILESTONE_50: {
                'name': '50 Contributions',
                'description': 'Made 50 total contributions',
                'icon': '⭐',
                'points': 1000,
                'criteria': {'total_contributions': 50}
            },
            BadgeType.MILESTONE_100: {
                'name': '100 Contributions',
                'description': 'Made 100 total contributions',
                'icon': '👑',
                'points': 2000,
                'criteria': {'total_contributions': 100}
            },
            BadgeType.EARLY_ADOPTER: {
                'name': 'Early Adopter',
                'description': 'Joined Nynava in the first month',
                'icon': '🚀',
                'points': 500,
                'criteria': {'early_user': True}
            },
            BadgeType.FEEDBACK_PROVIDER: {
                'name': 'Feedback Provider',
                'description': 'Provided valuable feedback on 5 occasions',
                'icon': '💬',
                'points': 200,
                'criteria': {'feedback_submissions': 5}
            }
        }
        
        # Point values for activities
        self.activity_points = {
            ActivityType.CONSENT_SUBMISSION: 50,
            ActivityType.DATA_UPLOAD: 100,
            ActivityType.MODEL_USAGE: 25,
            ActivityType.DATASET_CONTRIBUTION: 200,
            ActivityType.MODEL_IMPROVEMENT: 150,
            ActivityType.FEEDBACK_SUBMISSION: 30,
            ActivityType.REFERRAL: 100,
            ActivityType.COMMUNITY_PARTICIPATION: 50
        }
        
        logger.info("GamificationManager initialized")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's gamification profile"""
        try:
            if user_id not in self.user_profiles:
                self._initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Calculate level based on total points
            level_info = self._calculate_level(profile['total_points'])
            
            # Get recent activities
            recent_activities = profile['activities'][-10:] if profile['activities'] else []
            
            return {
                'user_id': user_id,
                'total_points': profile['total_points'],
                'level': level_info['level'],
                'level_name': level_info['name'],
                'progress_to_next': level_info['progress_to_next'],
                'points_to_next': level_info['points_to_next'],
                'badges': profile['badges'],
                'badge_count': len(profile['badges']),
                'statistics': profile['statistics'],
                'recent_activities': recent_activities,
                'joined_date': profile['joined_date'],
                'last_activity': profile['last_activity']
            }
            
        except Exception as e:
            logger.error("Failed to get user profile", user_id=user_id, error=str(e))
            return self._get_empty_profile(user_id)
    
    def award_points(self, user_id: str, activity_type: str, custom_points: Optional[int] = None) -> Dict[str, Any]:
        """Award points to user for activity"""
        try:
            if user_id not in self.user_profiles:
                self._initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Determine points to award
            if custom_points is not None:
                points = custom_points
            else:
                activity_enum = ActivityType(activity_type)
                points = self.activity_points.get(activity_enum, 0)
            
            # Award points
            profile['total_points'] += points
            profile['last_activity'] = datetime.utcnow().isoformat()
            
            # Record activity
            activity_record = {
                'type': activity_type,
                'points': points,
                'timestamp': datetime.utcnow().isoformat()
            }
            profile['activities'].append(activity_record)
            
            # Update statistics
            self._update_statistics(user_id, activity_type)
            
            # Check for new badges
            new_badges = self._check_badge_eligibility(user_id)
            
            # Clear leaderboard cache
            self.leaderboard_cache = None
            
            result = {
                'points_awarded': points,
                'total_points': profile['total_points'],
                'new_badges': new_badges,
                'activity_type': activity_type
            }
            
            logger.info("Points awarded", 
                       user_id=user_id, 
                       activity=activity_type,
                       points=points,
                       total=profile['total_points'])
            
            return result
            
        except Exception as e:
            logger.error("Failed to award points", user_id=user_id, error=str(e))
            return {'points_awarded': 0, 'total_points': 0, 'new_badges': []}
    
    def award_badge(self, user_id: str, badge_type: BadgeType) -> bool:
        """Award specific badge to user"""
        try:
            if user_id not in self.user_profiles:
                self._initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Check if user already has this badge
            if any(badge['type'] == badge_type.value for badge in profile['badges']):
                return False
            
            badge_def = self.badge_definitions[badge_type]
            
            # Create badge record
            badge_record = {
                'type': badge_type.value,
                'name': badge_def['name'],
                'description': badge_def['description'],
                'icon': badge_def['icon'],
                'points': badge_def['points'],
                'earned_date': datetime.utcnow().isoformat()
            }
            
            # Add badge to profile
            profile['badges'].append(badge_record)
            profile['total_points'] += badge_def['points']
            
            logger.info("Badge awarded", 
                       user_id=user_id, 
                       badge=badge_def['name'],
                       points=badge_def['points'])
            
            return True
            
        except Exception as e:
            logger.error("Failed to award badge", user_id=user_id, badge=badge_type, error=str(e))
            return False
    
    def get_leaderboard(self, limit: int = 50, category: str = 'overall') -> List[Dict[str, Any]]:
        """Get leaderboard rankings"""
        try:
            # Check cache
            if (self.leaderboard_cache and self.cache_expiry and 
                datetime.utcnow() < self.cache_expiry):
                return self.leaderboard_cache[:limit]
            
            # Build leaderboard
            leaderboard = []
            
            for user_id, profile in self.user_profiles.items():
                entry = {
                    'user_id': user_id,
                    'username': profile.get('username', f'User_{user_id[:8]}'),
                    'total_points': profile['total_points'],
                    'badge_count': len(profile['badges']),
                    'level': self._calculate_level(profile['total_points'])['level'],
                    'contributions': profile['statistics'].get('total_contributions', 0),
                    'last_activity': profile['last_activity']
                }
                
                # Category-specific scoring
                if category == 'data_contributors':
                    entry['score'] = profile['statistics'].get('data_uploads', 0)
                elif category == 'ai_users':
                    entry['score'] = profile['statistics'].get('model_usage_count', 0)
                elif category == 'community_helpers':
                    entry['score'] = profile['statistics'].get('model_improvements', 0)
                else:  # overall
                    entry['score'] = profile['total_points']
                
                leaderboard.append(entry)
            
            # Sort by score (descending)
            leaderboard.sort(key=lambda x: x['score'], reverse=True)
            
            # Add rankings
            for i, entry in enumerate(leaderboard):
                entry['rank'] = i + 1
            
            # Cache results
            self.leaderboard_cache = leaderboard
            self.cache_expiry = datetime.utcnow() + timedelta(minutes=15)
            
            return leaderboard[:limit]
            
        except Exception as e:
            logger.error("Failed to get leaderboard", error=str(e))
            return []
    
    def get_achievements(self, user_id: str) -> Dict[str, Any]:
        """Get user's achievements and progress"""
        try:
            if user_id not in self.user_profiles:
                return {'badges': [], 'progress': {}}
            
            profile = self.user_profiles[user_id]
            stats = profile['statistics']
            
            # Calculate progress towards unearned badges
            progress = {}
            
            for badge_type, badge_def in self.badge_definitions.items():
                # Check if user already has this badge
                has_badge = any(badge['type'] == badge_type.value for badge in profile['badges'])
                
                if not has_badge:
                    criteria = badge_def['criteria']
                    progress_info = {}
                    
                    for criterion, required_value in criteria.items():
                        current_value = stats.get(criterion, 0)
                        if isinstance(required_value, bool):
                            progress_info[criterion] = {
                                'current': current_value,
                                'required': required_value,
                                'completed': current_value == required_value
                            }
                        else:
                            progress_info[criterion] = {
                                'current': current_value,
                                'required': required_value,
                                'completed': current_value >= required_value,
                                'progress_percent': min(100, (current_value / required_value) * 100)
                            }
                    
                    progress[badge_type.value] = {
                        'badge_info': badge_def,
                        'progress': progress_info,
                        'overall_completed': all(p.get('completed', False) for p in progress_info.values())
                    }
            
            return {
                'earned_badges': profile['badges'],
                'available_badges': progress,
                'total_badges_available': len(self.badge_definitions),
                'badges_earned': len(profile['badges']),
                'completion_percentage': (len(profile['badges']) / len(self.badge_definitions)) * 100
            }
            
        except Exception as e:
            logger.error("Failed to get achievements", user_id=user_id, error=str(e))
            return {'badges': [], 'progress': {}}
    
    def _initialize_user_profile(self, user_id: str):
        """Initialize new user profile"""
        self.user_profiles[user_id] = {
            'user_id': user_id,
            'username': f'User_{user_id[:8]}',
            'total_points': 0,
            'badges': [],
            'activities': [],
            'statistics': {
                'data_uploads': 0,
                'model_usage_count': 0,
                'unique_models_used': 0,
                'datasets_accessed': 0,
                'model_improvements': 0,
                'feedback_submissions': 0,
                'privacy_actions': 0,
                'total_contributions': 0,
                'referrals': 0,
                'early_user': self._is_early_user()
            },
            'joined_date': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat()
        }
    
    def _update_statistics(self, user_id: str, activity_type: str):
        """Update user statistics based on activity"""
        profile = self.user_profiles[user_id]
        stats = profile['statistics']
        
        # Update specific statistics
        if activity_type == ActivityType.DATA_UPLOAD.value:
            stats['data_uploads'] += 1
            stats['total_contributions'] += 1
        elif activity_type == ActivityType.MODEL_USAGE.value:
            stats['model_usage_count'] += 1
        elif activity_type == ActivityType.DATASET_CONTRIBUTION.value:
            stats['total_contributions'] += 1
        elif activity_type == ActivityType.MODEL_IMPROVEMENT.value:
            stats['model_improvements'] += 1
            stats['total_contributions'] += 1
        elif activity_type == ActivityType.FEEDBACK_SUBMISSION.value:
            stats['feedback_submissions'] += 1
        elif activity_type == ActivityType.REFERRAL.value:
            stats['referrals'] += 1
        elif activity_type == ActivityType.COMMUNITY_PARTICIPATION.value:
            stats['total_contributions'] += 1
    
    def _check_badge_eligibility(self, user_id: str) -> List[Dict[str, Any]]:
        """Check if user is eligible for new badges"""
        new_badges = []
        profile = self.user_profiles[user_id]
        stats = profile['statistics']
        
        for badge_type, badge_def in self.badge_definitions.items():
            # Check if user already has this badge
            has_badge = any(badge['type'] == badge_type.value for badge in profile['badges'])
            
            if not has_badge:
                # Check if user meets criteria
                criteria_met = True
                for criterion, required_value in badge_def['criteria'].items():
                    current_value = stats.get(criterion, 0)
                    
                    if isinstance(required_value, bool):
                        if current_value != required_value:
                            criteria_met = False
                            break
                    else:
                        if current_value < required_value:
                            criteria_met = False
                            break
                
                if criteria_met:
                    # Award the badge
                    if self.award_badge(user_id, badge_type):
                        new_badges.append({
                            'type': badge_type.value,
                            'name': badge_def['name'],
                            'description': badge_def['description'],
                            'icon': badge_def['icon'],
                            'points': badge_def['points']
                        })
        
        return new_badges
    
    def _calculate_level(self, total_points: int) -> Dict[str, Any]:
        """Calculate user level based on points"""
        # Level thresholds
        levels = [
            (0, "Newcomer", "🌱"),
            (100, "Contributor", "📊"),
            (500, "Researcher", "🔬"),
            (1000, "Expert", "🎓"),
            (2500, "Champion", "🏆"),
            (5000, "Master", "⭐"),
            (10000, "Legend", "👑")
        ]
        
        current_level = levels[0]
        next_level = None
        
        for i, (threshold, name, icon) in enumerate(levels):
            if total_points >= threshold:
                current_level = (threshold, name, icon)
                if i + 1 < len(levels):
                    next_level = levels[i + 1]
            else:
                break
        
        # Calculate progress to next level
        if next_level:
            points_in_current_level = total_points - current_level[0]
            points_needed_for_next = next_level[0] - current_level[0]
            progress_percent = (points_in_current_level / points_needed_for_next) * 100
            points_to_next = next_level[0] - total_points
        else:
            progress_percent = 100
            points_to_next = 0
        
        return {
            'level': levels.index(current_level) + 1,
            'name': current_level[1],
            'icon': current_level[2],
            'threshold': current_level[0],
            'progress_to_next': progress_percent,
            'points_to_next': points_to_next,
            'next_level': next_level[1] if next_level else None
        }
    
    def _is_early_user(self) -> bool:
        """Check if user is an early adopter (first month)"""
        # For demo purposes, consider all users as early adopters
        # In production, check against actual launch date
        return True
    
    def _get_empty_profile(self, user_id: str) -> Dict[str, Any]:
        """Return empty profile structure"""
        return {
            'user_id': user_id,
            'total_points': 0,
            'level': 1,
            'level_name': 'Newcomer',
            'progress_to_next': 0,
            'points_to_next': 100,
            'badges': [],
            'badge_count': 0,
            'statistics': {},
            'recent_activities': [],
            'joined_date': datetime.utcnow().isoformat(),
            'last_activity': None
        }