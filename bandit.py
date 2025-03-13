import numpy as np
import random
import math
import datetime
from typing import List, Dict, Tuple, Optional
from models import Question, QuestionPosition, TestSession, QuestionInteraction

# Multi-armed bandit algorithm for optimizing test questions and their positions
class MultiBanditTest:
    """Multi-armed bandit algorithm for optimizing test questions and their positions"""
    
    def __init__(self, session, exploration_param=1.0, min_impressions=10):
        """
        Initialize the multi-armed bandit algorithm
        
        Args:
            session: SQLAlchemy session
            exploration_param: Controls exploration vs exploitation balance (higher = more exploration)
            min_impressions: Minimum number of impressions before using UCB formula
        """
        self.session = session
        self.exploration_param = exploration_param
        self.min_impressions = min_impressions
    
    def get_questions(self) -> List[Question]:
        """Get all active questions from the database"""
        return self.session.query(Question).filter(Question.is_active == True).all()
    
    def get_question_positions(self) -> Dict[int, List[QuestionPosition]]:
        """Get all question positions grouped by position index"""
        positions = self.session.query(QuestionPosition).all()
        position_dict = {}
        
        for pos in positions:
            if pos.position not in position_dict:
                position_dict[pos.position] = []
            position_dict[pos.position].append(pos)
        
        return position_dict
    
    def calculate_ucb(self, position: QuestionPosition) -> float:
        """Calculate Upper Confidence Bound for a question position"""
        if position.impressions < self.min_impressions:
            # Return a high value to encourage exploration of new questions
            return float('inf')
        
        # Calculate conversion rate
        conversion_rate = position.conversions / position.impressions if position.impressions > 0 else 0
        
        # Calculate exploration bonus
        exploration_bonus = self.exploration_param * math.sqrt(2 * math.log(self.total_impressions) / position.impressions)
        
        # UCB score combines exploitation (conversion rate) and exploration
        return conversion_rate + exploration_bonus
    
    def select_question_for_position(self, position_index: int) -> Question:
        """Select the best question for a given position using UCB algorithm"""
        # Get all question positions for this position index
        position_dict = self.get_question_positions()
        positions = position_dict.get(position_index, [])
        
        # If no positions exist for this index, select a random question
        if not positions:
            questions = self.get_questions()
            return random.choice(questions) if questions else None
        
        # Calculate total impressions for normalization
        self.total_impressions = sum(pos.impressions for pos in positions) or 1
        
        # Calculate UCB for each question position
        ucb_scores = [(pos, self.calculate_ucb(pos)) for pos in positions]
        
        # Select the question with the highest UCB score
        best_position = max(ucb_scores, key=lambda x: x[1])[0]
        return best_position.question
    
    def generate_test_sequence(self, num_questions: int = 30) -> List[Question]:
        """Generate a sequence of questions for a test using the bandit algorithm"""
        questions = []
        
        for position in range(num_questions):
            question = self.select_question_for_position(position)
            if question:
                questions.append(question)
        
        return questions
    
    def update_question_metrics(self, session_id: str, converted: bool) -> None:
        """Update question metrics based on test session outcome"""
        # Get the test session
        test_session = self.session.query(TestSession).filter(TestSession.session_id == session_id).first()
        if not test_session:
            return
        
        # Update conversion status
        test_session.converted = converted
        test_session.completed = True
        test_session.completed_at = datetime.datetime.utcnow()
        
        # Update metrics for each question position
        for interaction in test_session.interactions:
            # Find the corresponding question position
            question_position = self.session.query(QuestionPosition).filter(
                QuestionPosition.question_id == interaction.question_id,
                QuestionPosition.position == interaction.position
            ).first()
            
            if not question_position:
                # Create a new question position if it doesn't exist
                question_position = QuestionPosition(
                    question_id=interaction.question_id,
                    position=interaction.position,
                    impressions=0,
                    conversions=0,
                    reward=0.0
                )
                self.session.add(question_position)
            
            # Update metrics
            question_position.impressions += 1
            if converted:
                question_position.conversions += 1
            
            # Update reward (conversion rate)
            question_position.reward = question_position.conversions / question_position.impressions
        
        self.session.commit()
    
    def evaluate_new_question(self, question_id: int, num_trials: int = 100) -> Dict[int, float]:
        """Evaluate a new question by simulating its performance at different positions"""
        # Get the question
        question = self.session.query(Question).get(question_id)
        if not question:
            return {}
        
        # Get current number of positions in test
        position_dict = self.get_question_positions()
        num_positions = max(position_dict.keys()) + 1 if position_dict else 30
        
        # Initialize results dictionary
        position_scores = {pos: 0.0 for pos in range(num_positions)}
        
        # Simulate the question at each position
        for position in range(num_positions):
            # Create a temporary question position for simulation
            temp_position = QuestionPosition(
                question_id=question_id,
                position=position,
                impressions=0,
                conversions=0,
                reward=0.0
            )
            
            # Run simulations
            for _ in range(num_trials):
                # Simulate user interaction and conversion
                # This is a simplified simulation - in reality, you'd need more complex modeling
                converted = random.random() < 0.25  # Assuming 25% baseline conversion rate
                
                temp_position.impressions += 1
                if converted:
                    temp_position.conversions += 1
            
            # Calculate final reward
            position_scores[position] = temp_position.conversions / temp_position.impressions
        
        return position_scores
    
    def recommend_question_position(self, question_id: int) -> int:
        """Recommend the best position for a new question"""
        position_scores = self.evaluate_new_question(question_id)
        if not position_scores:
            return 0
        
        # Find position with highest expected reward
        best_position = max(position_scores.items(), key=lambda x: x[1])[0]
        return best_position