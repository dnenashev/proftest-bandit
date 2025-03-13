import sys
import os
from models import init_db, Question
from bandit import MultiBanditTest

# Initialize database session
session = init_db()

# Initialize multi-armed bandit algorithm
bandit = MultiBanditTest(session)

def add_new_question(text, answers):
    """Add a new question to the test pool and get recommended position"""
    # Create new question
    question = Question(
        text=text,
        answers=answers,
        is_active=True
    )
    session.add(question)
    session.commit()
    
    # Recommend best position for this question
    best_position = bandit.recommend_question_position(question.id)
    
    print(f"Added question: '{text}'")
    print(f"Recommended position: {best_position}")
    
    return question.id, best_position

if __name__ == "__main__":
    # Example usage
    new_question_text = "Какой подход к решению проблем вам ближе?"
    new_question_answers = [
        "Систематический анализ всех факторов", 
        "Поиск креативных и нестандартных решений", 
        "Консультация с экспертами и коллегами"
    ]
    
    question_id, position = add_new_question(new_question_text, new_question_answers)
    
    print("\nTo add this question to the test with the recommended position, run:")
    print(f"""from models import init_db, QuestionPosition

session = init_db()
position = QuestionPosition(
    question_id={question_id},
    position={position},
    impressions=5,  # Start with some impressions to avoid cold start
    conversions=1,  # Assume 20% conversion rate initially
    reward=0.2  # Initial reward estimate
)
session.add(position)
session.commit()
""")