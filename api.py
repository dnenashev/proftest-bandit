import os
import json
import uuid
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import init_db, Question, QuestionPosition, TestSession, QuestionInteraction
from bandit import MultiBanditTest

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize database session
session = init_db()

# Initialize multi-armed bandit algorithm
bandit = MultiBanditTest(session)

@app.route('/api/start-test', methods=['POST'])
def start_test():
    """Start a new test session with optimized question sequence"""
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Generate optimized question sequence using bandit algorithm
    questions = bandit.generate_test_sequence()
    
    if not questions:
        return jsonify({'error': 'No questions available'}), 500
    
    # Create a new test session
    question_ids = [q.id for q in questions]
    test_session = TestSession(
        session_id=session_id,
        question_order=question_ids,
        completed=False,
        converted=False
    )
    session.add(test_session)
    session.commit()
    
    # Format questions for frontend
    formatted_questions = []
    for i, q in enumerate(questions):
        formatted_questions.append({
            'id': q.id,
            'text': q.text,
            'answers': q.answers,
            'position': i
        })
    
    return jsonify({
        'session_id': session_id,
        'questions': formatted_questions
    })

@app.route('/api/answer', methods=['POST'])
def record_answer():
    """Record a user's answer to a question"""
    data = request.json
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    position = data.get('position')
    answer_index = data.get('answer_index')
    time_spent = data.get('time_spent')  # Optional
    
    # Validate required fields
    if not all([session_id, question_id, position is not None, answer_index is not None]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Get the test session
    test_session = session.query(TestSession).filter(TestSession.session_id == session_id).first()
    if not test_session:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    # Record the interaction
    interaction = QuestionInteraction(
        session_id=test_session.id,
        question_id=question_id,
        position=position,
        answer_index=answer_index,
        time_spent=time_spent
    )
    session.add(interaction)
    session.commit()
    
    return jsonify({'success': True})

@app.route('/api/complete', methods=['POST'])
def complete_test():
    """Mark a test as completed and record conversion status"""
    data = request.json
    session_id = data.get('session_id')
    converted = data.get('converted', False)  # Whether user submitted the form
    
    if not session_id:
        return jsonify({'error': 'Missing session ID'}), 400
    
    # Update question metrics based on test outcome
    bandit.update_question_metrics(session_id, converted)
    
    return jsonify({'success': True})

@app.route('/api/add-question', methods=['POST'])
def add_question():
    """Add a new question to the test pool"""
    data = request.json
    text = data.get('text')
    answers = data.get('answers')
    
    if not text or not answers or not isinstance(answers, list):
        return jsonify({'error': 'Invalid question data'}), 400
    
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
    
    # Create initial position entry
    position = QuestionPosition(
        question_id=question.id,
        position=best_position,
        impressions=5,  # Start with some impressions to avoid cold start
        conversions=1,  # Assume 20% conversion rate initially
        reward=0.2  # Initial reward estimate
    )
    session.add(position)
    session.commit()
    
    return jsonify({
        'success': True,
        'question_id': question.id,
        'recommended_position': best_position
    })

@app.route('/api/questions', methods=['GET'])
def get_questions():
    """Get all questions with their performance metrics"""
    questions = session.query(Question).filter(Question.is_active == True).all()
    
    result = []
    for q in questions:
        # Get positions for this question
        positions = session.query(QuestionPosition).filter(QuestionPosition.question_id == q.id).all()
        
        # Calculate overall metrics
        total_impressions = sum(p.impressions for p in positions)
        total_conversions = sum(p.conversions for p in positions)
        overall_conversion_rate = total_conversions / total_impressions if total_impressions > 0 else 0
        
        # Format position data
        position_data = [{
            'position': p.position,
            'impressions': p.impressions,
            'conversions': p.conversions,
            'conversion_rate': p.conversions / p.impressions if p.impressions > 0 else 0
        } for p in positions]
        
        result.append({
            'id': q.id,
            'text': q.text,
            'answers': q.answers,
            'overall_conversion_rate': overall_conversion_rate,
            'total_impressions': total_impressions,
            'positions': position_data
        })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)