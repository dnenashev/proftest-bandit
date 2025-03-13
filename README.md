# Multi-Armed Bandit Test Optimization System

This system optimizes career test questions and their order to maximize conversion rates using a multi-armed bandit algorithm. It dynamically adjusts question sequences based on user interactions and conversion data.

## Features

- **Dynamic Question Ordering**: Automatically determines the optimal order of questions to maximize conversion rates
- **Question Evaluation**: Evaluates new questions to determine their effectiveness and optimal placement
- **Performance Tracking**: Monitors question performance at different positions
- **API Integration**: Seamlessly integrates with the existing React frontend

## Components

- `models.py`: Database models for questions, positions, sessions, and interactions
- `bandit.py`: Core multi-armed bandit algorithm implementation
- `api.py`: Flask API server for frontend integration
- `init_db.py`: Database initialization script

## How It Works

1. The system uses the Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation
2. Each question is evaluated at different positions to determine where it performs best
3. As users complete tests, the system learns which questions and positions lead to higher conversion rates
4. New questions are automatically evaluated and placed in optimal positions

## Setup and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Initialize the Database

```bash
python init_db.py
```

### Start the API Server

```bash
python api.py
```

### API Endpoints

- `POST /api/start-test`: Start a new test session with optimized question sequence
- `POST /api/answer`: Record a user's answer to a question
- `POST /api/complete`: Mark a test as completed and record conversion status
- `POST /api/add-question`: Add a new question to the test pool
- `GET /api/questions`: Get all questions with their performance metrics

## Integration with React Frontend

To integrate with the existing React frontend, modify the App.jsx file to fetch questions from the API instead of using the hardcoded questions array.

## Adding New Questions

New questions can be added through the API, and the system will automatically evaluate them and determine their optimal position in the test sequence.