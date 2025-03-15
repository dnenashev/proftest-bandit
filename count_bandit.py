import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Создаем директорию для выходных файлов
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "count_bandit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Question:
    """Represents a single test question with unique ID and content."""
    def __init__(self, id: int, content: str):
        self.id = id
        self.content = content

class CountArm:
    """Represents an arm for a specific count of questions."""
    def __init__(self, count: int):
        self.count = count  # number of questions to include
        # Beta distribution parameters for Thompson Sampling
        self.alpha = 1  # successes + 1 (1 is the prior)
        self.beta = 1   # failures + 1 (1 is the prior)
        self.pulls = 0  # number of times this arm has been pulled
        self.conversions = 0  # number of successful conversions
    
    def get_conversion_rate(self) -> float:
        """Get the current estimated conversion rate."""
        if self.pulls == 0:
            return 0
        return self.conversions / self.pulls
    
    def sample_value(self) -> float:
        """Sample a value from the Beta distribution for this arm."""
        return np.random.beta(self.alpha, self.beta)

class CountBandit:
    """Multi-armed bandit for optimizing the number of questions for maximum conversion."""
    def __init__(self, 
                 questions: List[Question], 
                 top_questions: List[int] = None,
                 min_count: int = 20,
                 max_count: int = 40,
                 start_count: int = 30,
                 exploration_factor: float = 0.2,
                 data_file: str = "count_bandit_data.json"):
        self.questions = questions
        self.question_dict = {q.id: q for q in questions}
        
        # Если список лучших вопросов не предоставлен, используем первые 40
        self.top_questions = top_questions or [q.id for q in questions[:40]]
        
        self.min_count = min_count  # Минимальное количество вопросов
        self.max_count = max_count  # Максимальное количество вопросов
        self.start_count = start_count  # Начальное количество вопросов
        self.exploration_factor = exploration_factor
        self.data_file = os.path.join(OUTPUT_DIR, data_file)
        
        # Банки "рук" - каждая рука представляет собой количество вопросов
        self.arms: Dict[int, CountArm] = {}  # Maps count to CountArm
        
        self.current_session: Optional[Tuple[str, int]] = None
        
        # Tracking metrics over time
        self.conversion_history = []
        self.rolling_conversion_rate = []
        self.total_conversions = 0
        self.total_sessions = 0
        
        # Initialize arms
        self._initialize_arms()
        
        # Load existing data if available
        self._load_data()
    
    def _initialize_arms(self) -> None:
        """Initialize arms for different question counts."""
        # Create an arm for each possible count within range
        for count in range(self.min_count, self.max_count + 1):
            self.arms[count] = CountArm(count)
    
    def _load_data(self) -> None:
        """Load existing bandit data from file if available."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load metrics
                self.total_conversions = data.get('total_conversions', 0)
                self.total_sessions = data.get('total_sessions', 0)
                
                # Load arms
                for count_str, arm_data in data.get('arms', {}).items():
                    count = int(count_str)
                    
                    # Create or update the arm for this count
                    if count not in self.arms:
                        self.arms[count] = CountArm(count)
                    
                    arm = self.arms[count]
                    arm.alpha = arm_data.get('alpha', 1)
                    arm.beta = arm_data.get('beta', 1)
                    arm.pulls = arm_data.get('pulls', 0)
                    arm.conversions = arm_data.get('conversions', 0)
                
                # Load conversion history if available
                self.conversion_history = data.get('conversion_history', [])
                self.rolling_conversion_rate = data.get('rolling_conversion_rate', [])
                
                print(f"Loaded data for {len(self.arms)} count arms from existing data")
            except Exception as e:
                print(f"Error loading data: {e}")
                # Initialize fresh if loading fails
                self._initialize_arms()
    
    def _save_data(self) -> None:
        """Save current bandit data to file."""
        data = {
            'total_conversions': self.total_conversions,
            'total_sessions': self.total_sessions,
            'arms': {},
            'conversion_history': self.conversion_history,
            'rolling_conversion_rate': self.rolling_conversion_rate
        }
        
        # Save each arm's data
        for count, arm in self.arms.items():
            data['arms'][str(count)] = {
                'alpha': arm.alpha,
                'beta': arm.beta,
                'pulls': arm.pulls,
                'conversions': arm.conversions
            }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_test_questions(self, user_id: str) -> List[Question]:
        """Get test questions for a user based on current bandit knowledge."""
        # Decide whether to explore or exploit
        if random.random() < self.exploration_factor or max(arm.pulls for arm in self.arms.values()) < 10:
            # Exploration - try different counts with some bias towards unexplored counts
            counts_and_pulls = [(count, arm.pulls) for count, arm in self.arms.items()]
            # Prefer counts that have been pulled less often
            weights = [1 / (pulls + 1) for _, pulls in counts_and_pulls]
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted selection of count
            selected_count = random.choices(
                [count for count, _ in counts_and_pulls],
                weights=normalized_weights,
                k=1
            )[0]
        else:
            # Exploitation - use Thompson Sampling to select best count
            selected_count = self._select_count_thompson_sampling()
        
        # Record the current session
        self.current_session = (user_id, selected_count)
        
        # Get the selected number of top questions
        question_ids = self.top_questions[:selected_count]
        
        # Return the actual questions
        return [self.question_dict[qid] for qid in question_ids]
    
    def _select_count_thompson_sampling(self) -> int:
        """Use Thompson Sampling to select the best arm (question count)."""
        best_value = -1
        best_count = self.start_count  # Start with default count
        
        # Sample from each arm and select the one with highest value
        for count, arm in self.arms.items():
            value = arm.sample_value()
            if value > best_value:
                best_value = value
                best_count = count
        
        return best_count
    
    def update_with_result(self, user_id: str, conversion: bool) -> None:
        """Update the bandit with the result of a test session."""
        if self.current_session is None or self.current_session[0] != user_id:
            print(f"Warning: No active session for user {user_id}")
            return
        
        _, count = self.current_session
        
        # Update global conversion tracking
        self.total_sessions += 1
        if conversion:
            self.total_conversions += 1
        
        # Track this individual conversion result
        self.conversion_history.append(1 if conversion else 0)
        
        # Update rolling average
        self._update_rolling_average()
        
        # Update the arm that was pulled
        arm = self.arms[count]
        arm.pulls += 1
        
        if conversion:
            arm.conversions += 1
            arm.alpha += 1
        else:
            arm.beta += 1
        
        # Save the updated data
        self._save_data()
    
    def _update_rolling_average(self) -> None:
        """Update rolling average conversion rate."""
        window_size = min(200, len(self.conversion_history))
        if window_size > 0 and (self.total_sessions % 10 == 0 or self.total_sessions <= 200):
            recent_rate = sum(self.conversion_history[-window_size:]) / window_size
            self.rolling_conversion_rate.append((self.total_sessions, recent_rate))
    
    def get_best_counts(self, top_n: int = 10) -> List[Tuple[int, float, int]]:
        """Get the top N performing question counts."""
        # Only include counts with minimum pulls
        min_pulls = 10
        valid_arms = [(arm.count, arm.get_conversion_rate(), arm.pulls) 
                      for arm in self.arms.values() if arm.pulls >= min_pulls]
        
        # Sort by conversion rate (descending)
        valid_arms.sort(key=lambda x: x[1], reverse=True)
        
        return valid_arms[:top_n]
    
    def plot_conversion_over_time(self, save_path: str = "count_conversion_over_time.png") -> None:
        """Generate and save a plot showing conversion rate improvement over time."""
        if not self.rolling_conversion_rate:
            print("Not enough data to generate conversion rate plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract session numbers and rates from rolling_conversion_rate
        sessions = [entry[0] for entry in self.rolling_conversion_rate]
        rates = [entry[1] for entry in self.rolling_conversion_rate]
        
        # Plot rolling average conversion rate as a smooth line
        plt.plot(sessions, rates, 
                 'b-', 
                 linewidth=2,
                 label='Rolling Average Conversion Rate (200 samples)')
        
        # Вычисляем и строим кумулятивную скользящую среднюю
        cumulative_conversions = [sum(self.conversion_history[:i+1]) for i in range(len(self.conversion_history))]
        cumulative_rates = [(cumulative_conversions[i]/(i+1)) for i in range(0, len(self.conversion_history), 100)]
        cumulative_sessions = list(range(0, len(self.conversion_history), 100))
        if cumulative_sessions and cumulative_rates:
            plt.plot(cumulative_sessions, cumulative_rates, 
                    'g-', 
                    linewidth=2,
                    label='Cumulative Average Conversion Rate')
        
        # Add overall average line
        if self.total_sessions > 0:
            overall_rate = self.total_conversions / self.total_sessions
            plt.axhline(y=overall_rate, color='r', linestyle='--', 
                       label=f'Overall Average: {overall_rate:.2f}')
        
        # Определяем ожидаемую максимальную вероятность конверсии для тест-кейса
        expected_base_rate = 0.15  # значение по умолчанию
        expected_max_rate = 0.25   # значение по умолчанию
        
        if "fewer_questions" in self.data_file:
            expected_base_rate = 0.15  # базовая конверсия с 30 вопросами
            expected_max_rate = 0.25  # ожидаемая конверсия с оптимальным количеством (меньше 30)
        elif "more_questions" in self.data_file:
            expected_base_rate = 0.15  # базовая конверсия с 30 вопросами
            expected_max_rate = 0.22  # ожидаемая конверсия с оптимальным количеством (больше 30)
        elif "balanced_questions" in self.data_file:
            expected_base_rate = 0.15  # базовая конверсия с 30 вопросами
            expected_max_rate = 0.3   # ожидаемая конверсия с оптимальным количеством (около 25)
        
        # Add benchmarks for expected conversion rates
        plt.axhline(y=expected_base_rate, color='orange', linestyle='-.', alpha=0.5,
                   label=f'Base Conversion Rate ({expected_base_rate*100:.0f}%)')
        plt.axhline(y=expected_max_rate, color='green', linestyle='-.', alpha=0.5,
                   label=f'Target Conversion Rate ({expected_max_rate*100:.0f}%)')
        
        # Add labels and title
        plt.xlabel('Number of Sessions')
        plt.ylabel('Conversion Rate')
        plt.title('Conversion Rate Improvement Over Time (Question Count Optimization)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Установка динамического диапазона оси Y вместо фиксированного
        # Найдем максимальное значение среди всех данных на графике
        max_value = max(max(rates) if rates else 0, 
                        max(cumulative_rates) if cumulative_rates else 0,
                        overall_rate if self.total_sessions > 0 else 0,
                        expected_max_rate)
        
        # Добавим запас 10% сверху и установим диапазон от 0 до max_value + запас
        plt.ylim(0.0, max_value * 1.1)
        
        # Add text annotation for final conversion rate
        if rates:
            final_rate = rates[-1]
            plt.annotate(f'Final Rolling Rate: {final_rate:.2f}', 
                         xy=(sessions[-1], final_rate),
                         xytext=(sessions[-1] - 1000, final_rate + 0.02),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=10)
        
        # Save the plot
        if save_path:
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.join(OUTPUT_DIR, os.path.basename(save_path))), exist_ok=True)
            full_path = os.path.join(OUTPUT_DIR, os.path.basename(save_path))
            plt.tight_layout()
            plt.savefig(full_path)
            print(f"Conversion rate plot saved to {full_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_count_performance(self, save_path: str = "count_performance.png") -> None:
        """Plot performance of different question counts."""
        plt.figure(figsize=(10, 6))
        
        # Get all counts that have been pulled at least a few times
        min_pulls = 10
        valid_counts = [count for count, arm in self.arms.items() if arm.pulls >= min_pulls]
        valid_counts.sort()
        
        # Get conversion rates for each count
        conversion_rates = [self.arms[count].get_conversion_rate() for count in valid_counts]
        pulls = [self.arms[count].pulls for count in valid_counts]
        
        # Plot conversion rates by count
        plt.bar(valid_counts, conversion_rates, alpha=0.7, color='skyblue')
        
        # Add data points and pull counts
        for i, (count, rate, pull) in enumerate(zip(valid_counts, conversion_rates, pulls)):
            plt.text(count, rate + 0.01, f"{pull}", 
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Add best count highlight
        best_counts = self.get_best_counts(1)
        if best_counts:
            best_count, best_rate, _ = best_counts[0]
            plt.bar([best_count], [best_rate], color='green', alpha=0.7)
            plt.text(best_count, best_rate + 0.02, f"Best: {best_count} ({best_rate:.3f})", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Add labels and title
        plt.xlabel('Number of Questions')
        plt.ylabel('Conversion Rate')
        plt.title('Conversion Rate by Number of Questions')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Make x-axis only show integer values
        plt.xticks(valid_counts)
        
        # Ensure y-axis starts at 0
        plt.ylim(bottom=0)
        
        # Save the plot
        if save_path:
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.join(OUTPUT_DIR, os.path.basename(save_path))), exist_ok=True)
            full_path = os.path.join(OUTPUT_DIR, os.path.basename(save_path))
            plt.tight_layout()
            plt.savefig(full_path)
            print(f"Count performance plot saved to {full_path}")
        else:
            plt.show()
        
        plt.close()

# Load questions
def load_questions() -> List[Question]:
    """Load questions from a database or file."""
    questions = []
    for i in range(1, 51):
        questions.append(Question(i, f"Question content for question #{i}"))
    return questions

# Test case scenarios for count optimization
def run_count_test_case(test_name: str, conversion_rule: Callable, num_sessions: int = 1000,
                       top_questions: List[int] = None, min_count: int = 20, max_count: int = 40) -> None:
    """Run a test case for question count optimization with a specific conversion rule."""
    print(f"\n========== Running Test Case: {test_name} ==========\n")
    
    # Create a test-specific data file
    data_file = f"count_bandit_data_{test_name}.json"
    
    # Remove existing data file to start fresh
    full_path = os.path.join(OUTPUT_DIR, data_file)
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Load questions
    questions = load_questions()
    
    # Create the bandit
    bandit = CountBandit(
        questions=questions,
        top_questions=top_questions,
        min_count=min_count,
        max_count=max_count,
        start_count=30,
        exploration_factor=0.2,
        data_file=data_file
    )
    
    # Simulate user sessions
    for i in range(1, num_sessions + 1):
        user_id = f"user_{i}"
        
        # Get questions for this user (with the specific count)
        test_questions = bandit.get_test_questions(user_id)
        
        # Apply the conversion rule for this test case
        conversion = conversion_rule(test_questions)
        
        # Update the bandit with the result
        bandit.update_with_result(user_id, conversion)
    
    # Plot conversion over time and count performance
    bandit.plot_conversion_over_time(f"count_conversion_{test_name}.png")
    bandit.plot_count_performance(f"count_performance_{test_name}.png")
    
    # Итоги тестирования
    print(f"\n===== COUNT TEST RESULTS: '{test_name}' =====")
    print(f"Total sessions: {bandit.total_sessions}")
    print(f"Overall conversion rate: {bandit.total_conversions / bandit.total_sessions:.3f}")
    
    # Вывод лучших количеств вопросов
    best_counts = bandit.get_best_counts(top_n=10)
    print("\nTOP PERFORMING QUESTION COUNTS:")
    for i, (count, conv_rate, pulls) in enumerate(best_counts, 1):
        print(f"  {i}. {count} questions: {conv_rate:.3f} conversion rate (tested {pulls} times)")

    print(f"\nConversion rate plot saved to count_conversion_{test_name}.png")
    print(f"Count performance plot saved to count_performance_{test_name}.png")

# Функции конверсии для сценариев с оптимизацией количества вопросов

def fewer_questions_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Меньшее количество вопросов (25) дает лучшую конверсию, чем стандартные 30.
    Конверсия пропорционально снижается с увеличением количества вопросов выше оптимального.
    """
    num_questions = len(questions)
    
    # Оптимальное количество: 25 вопросов
    optimal_count = 25
    
    # Базовая вероятность конверсии при 30 вопросах
    base_prob = 0.15
    
    # Максимальное улучшение при оптимальном количестве
    max_improvement = 0.1
    
    # Вычисляем фактическую вероятность в зависимости от отклонения от оптимального значения
    if num_questions <= optimal_count:
        # Если меньше или равно оптимальному, конверсия растет линейно к оптимуму
        # При 20 вопросах: base_prob, при 25: base_prob + max_improvement
        factor = 1 - ((optimal_count - num_questions) / (optimal_count - 20)) if num_questions >= 20 else 0
        conversion_prob = base_prob + max_improvement * factor
    else:
        # Если больше оптимального, конверсия падает с ростом количества вопросов
        # При 25 вопросах: base_prob + max_improvement, при 40: base_prob
        factor = 1 - ((num_questions - optimal_count) / (40 - optimal_count)) if num_questions <= 40 else 0
        conversion_prob = base_prob + max_improvement * factor
    
    return random.random() < conversion_prob

def more_questions_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Большее количество вопросов (35) дает лучшую конверсию, чем стандартные 30.
    Однако слишком много вопросов (>38) снова снижает конверсию.
    """
    num_questions = len(questions)
    
    # Оптимальное количество: 35 вопросов
    optimal_count = 35
    
    # Базовая вероятность конверсии при 30 вопросах
    base_prob = 0.15
    
    # Максимальное улучшение при оптимальном количестве
    max_improvement = 0.07
    
    # Вычисляем фактическую вероятность в зависимости от отклонения от оптимального значения    
    if num_questions <= optimal_count:
        # Если меньше или равно оптимальному, конверсия растет с увеличением количества
        # При 20 вопросах: base_prob * 0.7, при 35: base_prob + max_improvement
        if num_questions < 20:
            conversion_prob = base_prob * 0.5  # Очень мало вопросов - низкая конверсия
        else:
            factor = (num_questions - 20) / (optimal_count - 20)
            conversion_prob = base_prob * 0.7 + (base_prob + max_improvement - base_prob * 0.7) * factor
    else:
        # Если больше оптимального, конверсия снова падает
        # При 35 вопросах: base_prob + max_improvement, при 40: base_prob
        factor = 1 - ((num_questions - optimal_count) / (40 - optimal_count)) if num_questions <= 40 else 0
        conversion_prob = base_prob + max_improvement * factor
    
    return random.random() < conversion_prob

def balanced_questions_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Существует четкий оптимум количества вопросов около 25-26,
    с быстрым снижением конверсии как при меньшем, так и при большем количестве.
    """
    num_questions = len(questions)
    
    # Идеальные количества: 25-26 вопросов
    optimal_min = 25
    optimal_max = 26
    
    # Базовая вероятность конверсии при 30 вопросах
    base_prob = 0.15
    
    # Максимальное улучшение при оптимальном количестве
    max_improvement = 0.15  # Значительное улучшение при оптимальном значении
    
    # Рассчитываем фактическую вероятность на основе гауссоподобной функции
    # с пиком при оптимальном количестве
    if optimal_min <= num_questions <= optimal_max:
        # В оптимальном диапазоне - максимальная конверсия
        conversion_prob = base_prob + max_improvement
    else:
        # Вне оптимального диапазона - конверсия снижается по мере удаления
        distance = min(abs(num_questions - optimal_min), abs(num_questions - optimal_max))
        # Быстрое снижение - примерно -3% за каждый вопрос отклонения
        conversion_prob = (base_prob + max_improvement) * (1.0 - 0.12 * distance)
        # Но не меньше 40% от базовой конверсии
        conversion_prob = max(conversion_prob, 0.4 * base_prob)
    
    return random.random() < conversion_prob

# Main function to run count test cases
def run_count_bandit_test_cases():
    """Run test cases for the count bandit."""
    print(f"Saving results to directory: {OUTPUT_DIR}")
    
    # Define a set of top questions to use (e.g., questions 1-40)
    top_questions = list(range(1, 41))
    
    # Тест-кейс 1: Оптимизация к меньшему количеству вопросов (около 25)
    run_count_test_case("fewer_questions", fewer_questions_rule, 
                       num_sessions=1000, top_questions=top_questions,
                       min_count=20, max_count=40)
    
    # Тест-кейс 2: Оптимизация к большему количеству вопросов (около 35)
    run_count_test_case("more_questions", more_questions_rule, 
                       num_sessions=1000, top_questions=top_questions,
                       min_count=20, max_count=40)
    
    # Тест-кейс 3: Оптимизация к очень конкретному количеству вопросов (25-26)
    run_count_test_case("balanced_questions", balanced_questions_rule, 
                       num_sessions=1000, top_questions=top_questions,
                       min_count=20, max_count=40)

if __name__ == "__main__":
    run_count_bandit_test_cases()  # Run test cases for count optimization 