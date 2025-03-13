import os
import json
import sys
from models import init_db, Question, QuestionPosition

# Add parent directory to path to import from career-test-react
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def import_questions_from_react():
    """Import questions from the React frontend"""
    try:
        # Path to the React app's App.jsx file
        app_jsx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'career-test-react', 'src', 'App.jsx'
        )
        
        # Read the file content
        with open(app_jsx_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract questions array
        questions_start = content.find('const questions = [')
        questions_end = content.find('];', questions_start)
        questions_block = content[questions_start:questions_end+2]
        
        # Extract answers array
        answers_start = content.find('const questionAnswers = [')
        answers_end = content.find('];', answers_start)
        answers_block = content[answers_start:answers_end+2]
        
        # Parse questions
        questions_lines = questions_block.split('\n')[1:-1]  # Skip first and last lines
        questions = []
        for line in questions_lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('",'):
                questions.append(line[1:-2])  # Remove quotes and comma
            elif line.startswith('"') and line.endswith('"'):
                questions.append(line[1:-1])  # Remove quotes
        
        # Parse answers
        answers = []
        current_answers = []
        in_array = False
        
        for line in answers_block.split('\n')[1:-1]:  # Skip first and last lines
            line = line.strip()
            
            if line.startswith('['):
                in_array = True
                current_answers = []
                # If the array is on a single line
                if line.endswith('],') or line.endswith(']'):
                    answer_line = line[1:-2] if line.endswith('],') else line[1:-1]
                    current_answers = [a.strip('"') for a in answer_line.split('", "')]
                    answers.append(current_answers)
                    in_array = False
            elif in_array and (line.endswith('",') or line.endswith('"')):
                current_answers.append(line.strip('"').strip('",'))
                if line.endswith('"]') or line.endswith('"],'):
                    answers.append(current_answers)
                    current_answers = []
                    in_array = False
        
        return list(zip(questions, answers))
    except Exception as e:
        print(f"Error importing questions from React: {e}")
        return []

def initialize_database():
    """Initialize the database with questions from the React frontend"""
    # Initialize database session
    session = init_db()
    
    # Import questions from React
    question_data = import_questions_from_react()
    
    if not question_data:
        print("No questions found in React frontend. Using default questions.")
        # Default questions if import fails
        question_data = [
            ("Я предпочитаю", ["Работать самостоятельно", "Работать в команде", "Организовывать и контролировать процесс работы"]),
            ("В школе мне больше нравились", ["Точные науки (информатика, математика, химия)", "Гуманитарные науки (языки, литература)", "Уроки творчества (рисование, музыка)"])
        ]
    
    # Add questions to database
    for i, (text, answers) in enumerate(question_data):
        # Check if question already exists
        existing = session.query(Question).filter(Question.text == text).first()
        if existing:
            print(f"Question already exists: {text}")
            continue
        
        # Create new question
        question = Question(
            text=text,
            answers=answers,
            is_active=True
        )
        session.add(question)
        session.flush()  # Flush to get the ID
        
        # Create initial position entry
        position = QuestionPosition(
            question_id=question.id,
            position=i,  # Initial position based on current order
            impressions=10,  # Start with some impressions to avoid cold start
            conversions=2,  # Assume 20% conversion rate initially
            reward=0.2  # Initial reward estimate
        )
        session.add(position)
    
    session.commit()
    print(f"Added {len(question_data)} questions to the database")

if __name__ == "__main__":
    initialize_database()