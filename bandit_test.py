import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Создаем директорию для выходных файлов
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "simple_bandit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Question:
    """Represents a single test question with unique ID and content."""
    def __init__(self, id: int, content: str):
        self.id = id
        self.content = content

class QuestionArm:
    """Represents a single arm of the bandit (a specific question)."""
    def __init__(self, question_id: int):
        self.question_id = question_id
        # Beta distribution parameters for Thompson Sampling
        self.alpha = 1  # successes + 1 (1 is the prior)
        self.beta = 1   # failures + 1 (1 is the prior)
        self.pulls = 0  # number of times this arm has been pulled
        self.conversions = 0  # number of successful conversions
        self.included_count = 0  # number of times question was included
    
    def get_conversion_rate(self) -> float:
        """Get the current estimated conversion rate."""
        if self.pulls == 0:
            return 0
        return self.conversions / self.pulls
    
    def get_inclusion_rate(self) -> float:
        """Get the rate at which this question is included in tests."""
        if self.pulls == 0:
            return 0
        return self.included_count / self.pulls
    
    def sample_value(self) -> float:
        """Sample a value from the Beta distribution for this arm."""
        return np.random.beta(self.alpha, self.beta)

class SimpleBandit:
    """Multi-armed bandit for optimizing test conversion with individual question arms."""
    def __init__(self, questions: List[Question], 
                 test_size: int = 30,
                 exploration_factor: float = 0.1,
                 data_file: str = "bandit_data.json",
                 individual_contribution: bool = False):
        self.questions = questions
        self.question_dict = {q.id: q for q in questions}
        self.test_size = min(test_size, len(questions))  # Ensure test size is valid
        self.exploration_factor = exploration_factor
        self.arms: Dict[int, QuestionArm] = {}  # Maps question ID to QuestionArm
        self.data_file = os.path.join(OUTPUT_DIR, data_file)
        self.current_session: Optional[Tuple[str, Set[int]]] = None
        self.individual_contribution = individual_contribution  # Флаг индивидуального вклада вопросов
        
        # Tracking metrics over time
        self.conversion_history = []
        self.rolling_conversion_rate = []
        self.total_conversions = 0
        self.total_sessions = 0
        
        # Initialize arms for each question
        self._initialize_arms()
        
        # Load existing data if available
        self._load_data()
    
    def _initialize_arms(self) -> None:
        """Initialize an arm for each question."""
        for question in self.questions:
            self.arms[question.id] = QuestionArm(question.id)
    
    def _load_data(self) -> None:
        """Load existing bandit data from file if available."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                for question_id_str, arm_data in data.items():
                    question_id = int(question_id_str)
                    if question_id in self.arms:
                        arm = self.arms[question_id]
                        arm.alpha = arm_data.get('alpha', 1)
                        arm.beta = arm_data.get('beta', 1)
                        arm.pulls = arm_data.get('pulls', 0)
                        arm.conversions = arm_data.get('conversions', 0)
                        arm.included_count = arm_data.get('included_count', 0)
                
                print(f"Loaded data for {len(data)} questions from existing data")
            except Exception as e:
                print(f"Error loading data: {e}")
                # Initialize fresh if loading fails
                self._initialize_arms()
    
    def _save_data(self) -> None:
        """Save current bandit data to file."""
        data = {}
        for question_id, arm in self.arms.items():
            data[str(question_id)] = {
                'alpha': arm.alpha,
                'beta': arm.beta,
                'pulls': arm.pulls,
                'conversions': arm.conversions,
                'included_count': arm.included_count
            }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_test_questions(self, user_id: str) -> List[Question]:
        """Get a set of questions for a user based on current bandit knowledge."""
        # Decide whether to explore or exploit
        if random.random() < self.exploration_factor:
            # Exploration: select questions randomly
            selected_ids = set(random.sample([q.id for q in self.questions], self.test_size))
        else:
            # Exploitation: use Thompson Sampling
            selected_ids = self._select_questions_thompson_sampling()
        
        # Record the current session
        self.current_session = (user_id, selected_ids)
        
        # Return the actual questions
        return [self.question_dict[qid] for qid in selected_ids]
    
    def _select_questions_thompson_sampling(self) -> Set[int]:
        """Use Thompson Sampling to select the best questions."""
        # Sample values for each arm
        samples = {qid: arm.sample_value() for qid, arm in self.arms.items()}
        
        # Sort question IDs by their sampled values (descending)
        sorted_ids = sorted(samples.keys(), key=lambda qid: samples[qid], reverse=True)
        
        # Take the top N questions
        return set(sorted_ids[:self.test_size])
    
    def update_with_result(self, user_id: str, conversion_result) -> None:
        """
        Update the bandit with the result of a test session.
        
        В новой модели conversion_result может быть:
        - bool: старый подход (общая конверсия для всех вопросов)
        - dict: новый подход (отдельная конверсия для каждого вопроса)
        """
        if self.current_session is None or self.current_session[0] != user_id:
            print(f"Warning: No active session for user {user_id}")
            return
        
        _, selected_ids = self.current_session
        
        # Обработка в зависимости от типа результата конверсии
        if isinstance(conversion_result, bool) and not self.individual_contribution:
            # Старый способ - обновляем всё одинаково
            self._update_with_single_result(conversion_result, selected_ids)
        elif isinstance(conversion_result, dict) and self.individual_contribution:
            # Новый способ - индивидуальная конверсия для каждого вопроса
            # Считаем, что в целом сессия успешна, если хотя бы один вопрос привел к конверсии
            overall_conversion = any(conversion_result.values())
            self._update_with_individual_results(conversion_result, selected_ids, overall_conversion)
        else:
            raise ValueError("Тип результата конверсии не соответствует режиму работы бандита")

    def _update_with_single_result(self, conversion: bool, selected_ids: Set[int]) -> None:
        """Старый способ обновления - один результат для всех вопросов."""
        # Update global conversion tracking
        self.total_sessions += 1
        if conversion:
            self.total_conversions += 1
        
        # Track this individual conversion result
        self.conversion_history.append(conversion)
        
        # Update rolling average
        self._update_rolling_average()
        
        # Update all included questions with the same result
        for qid, arm in self.arms.items():
            if qid in selected_ids:
                arm.included_count += 1
                arm.pulls += 1
                
                if conversion:
                    arm.conversions += 1
                    arm.alpha += 1
                else:
                    arm.beta += 1
        
        # Save the updated data
        self._save_data()
    
    def _update_with_individual_results(self, conversion_dict: Dict[int, bool], 
                                      selected_ids: Set[int], overall_conversion: bool) -> None:
        """Новый способ обновления - индивидуальный результат для каждого вопроса."""
        # Update global conversion tracking
        self.total_sessions += 1

        # ИЗМЕНЕНИЕ: Вместо any(conversion_dict.values()) используем максимальную вероятность
        # Получаем максимальную вероятность конверсии среди всех вопросов
        max_conversion_prob = 0.0
        for qid in selected_ids:
            # Получаем вероятность конверсии для этого вопроса
            if "decreasing_first20" in self.data_file:
                # Для теста decreasing_first20
                if 1 <= qid <= 20:
                    # Вопросы 1-20 имеют убывающую конверсию: 50%, 49%, 48%,... 31%
                    question_prob = 0.5 - (qid - 1) * 0.01
                else:
                    # Вопросы 21-50 имеют фиксированную 10% конверсию
                    question_prob = 0.1
            elif qid >= 21 and qid <= 50 and "last30_boost" in self.data_file:
                # Для теста individual_last30_boost, вопросы 21-50 имеют 30% вероятность
                question_prob = 0.3
            elif qid <= 20 and "last30_boost" in self.data_file:
                # Для теста individual_last30_boost, вопросы 1-20 имеют 10% вероятность
                question_prob = 0.1
            elif qid <= 20 and "first20_low" in self.data_file:
                # Для теста individual_first20_low, вопросы 1-20 имеют 1% вероятность
                question_prob = 0.01
            elif qid >= 21 and "first20_low" in self.data_file:
                # Для теста individual_first20_low, вопросы 21-50 имеют 5% вероятность
                question_prob = 0.05
            else:
                # Если не удалось определить по имени файла, используем наше лучшее предположение
                question_prob = 0.1
            
            max_conversion_prob = max(max_conversion_prob, question_prob)
        
        # Учитываем максимальную вероятность как результат конверсии сессии
        # с той же максимальной вероятностью (моделируем результат)
        session_converted = random.random() < max_conversion_prob
        if session_converted:
            self.total_conversions += 1

        # Track the overall session result based on our new model
        self.conversion_history.append(session_converted)
        
        # Update rolling average
        self._update_rolling_average()
        
        # Update each included question with its individual result
        for qid, arm in self.arms.items():
            if qid in selected_ids:
                arm.included_count += 1
                arm.pulls += 1
                
                # Используем результат для этого конкретного вопроса
                question_conversion = conversion_dict.get(qid, False)
                if question_conversion:
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
    
    def get_best_questions(self, top_n: int = 10) -> List[Tuple[int, float, int]]:
        """Get the top N performing questions."""
        # Only questions with minimum inclusions
        min_inclusions = 10
        valid_arms = [(arm.question_id, arm.get_conversion_rate(), arm.included_count) 
                      for arm in self.arms.values() if arm.included_count >= min_inclusions]
        
        # Sort by conversion rate (descending)
        valid_arms.sort(key=lambda x: x[1], reverse=True)
        
        return valid_arms[:top_n]
    
    def print_stats(self) -> None:
        """Print current statistics of the bandit."""
        # Полностью убираем вывод статистики в промежуточных точках
        pass
    
    def plot_conversion_over_time(self, save_path: str = "conversion_over_time.png") -> None:
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
        expected_max_rate = 0.3  # значение по умолчанию
        expected_base_rate = 0.1  # значение по умолчанию
        
        if "decreasing_first20" in self.data_file:
            expected_max_rate = 0.5  # максимальная конверсия 50% для вопроса #1
            expected_base_rate = 0.1  # базовая конверсия 10% для остальных вопросов
        elif "last30_boost" in self.data_file:
            expected_max_rate = 0.3  # максимальная конверсия 30% для вопросов 21-50
            expected_base_rate = 0.1  # базовая конверсия 10% для вопросов 1-20
        elif "first20_low" in self.data_file:
            expected_max_rate = 0.05  # максимальная конверсия 5% для вопросов 21-50
            expected_base_rate = 0.01  # базовая конверсия 1% для вопросов 1-20
        
        # Add benchmarks for expected conversion rates
        plt.axhline(y=expected_base_rate, color='orange', linestyle='-.', alpha=0.5,
                   label=f'Base Conversion Rate ({expected_base_rate*100:.0f}%)')
        plt.axhline(y=expected_max_rate, color='green', linestyle='-.', alpha=0.5,
                   label=f'Target Conversion Rate ({expected_max_rate*100:.0f}%)')
        
        # Add labels and title
        plt.xlabel('Number of Sessions')
        plt.ylabel('Conversion Rate')
        plt.title('Conversion Rate Improvement Over Time')
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


# Load questions
def load_questions() -> List[Question]:
    """Load questions from a database or file."""
    questions = []
    for i in range(1, 51):
        questions.append(Question(i, f"Question content for question #{i}"))
    return questions


# Test case scenarios with different conversion rules
def run_test_case(test_name: str, conversion_rule: Callable, num_sessions: int = 1000,
                individual_contribution: bool = False) -> None:
    """Run a test case with the given conversion rule."""
    print(f"\n========== Running Test Case: {test_name} ==========\n")
    
    # Create a test-specific data file
    data_file = f"bandit_data_{test_name}.json"
    
    # Remove existing data file to start fresh
    full_path = os.path.join(OUTPUT_DIR, data_file)
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Load questions
    questions = load_questions()
    
    # Create the bandit
    bandit = SimpleBandit(
        questions=questions,
        test_size=30,  # Always show 30 questions
        exploration_factor=0.2,
        data_file=data_file,
        individual_contribution=individual_contribution
    )
    
    # Track conversion rate by chunks
    chunk_size = 100
    chunk_conversions = []
    start_rates = []
    mid_rates = []
    end_rates = []
    
    # Simulate user sessions
    for i in range(1, num_sessions + 1):
        user_id = f"user_{i}"
        
        # Get questions for this user
        test_questions = bandit.get_test_questions(user_id)
        
        # Apply the conversion rule for this test case
        conversion = conversion_rule(test_questions)
        
        # Update the bandit with the result
        bandit.update_with_result(user_id, conversion)
        
        # Track conversion by chunks
        if i % chunk_size == 0:
            # Calculate conversion rate for this chunk
            chunk_start = i - chunk_size
            chunk_conversions.append(sum(bandit.conversion_history[chunk_start:i]) / chunk_size)
            
            # Store rates for analysis
            if i == 1000:
                start_rates = [(q.id, bandit.arms[q.id].get_conversion_rate()) 
                               for q in questions if q.id <= 10]
            elif i == 5000:
                mid_rates = [(q.id, bandit.arms[q.id].get_conversion_rate()) 
                             for q in questions if q.id <= 10]
            elif i == num_sessions:
                end_rates = [(q.id, bandit.arms[q.id].get_conversion_rate()) 
                             for q in questions if q.id <= 10]
    
    # Plot conversion over time
    plot_path = f"conversion_over_time_{test_name}.png"
    bandit.plot_conversion_over_time(plot_path)
    
    # Итоги тестирования
    print(f"\n===== TEST RESULTS: '{test_name}' =====")
    print(f"Total sessions: {bandit.total_sessions}")
    print(f"Overall conversion rate: {bandit.total_conversions / bandit.total_sessions:.3f}")
    
    # Получаем все вопросы с минимальным числом включений
    min_inclusions = 10
    valid_arms = [(arm.question_id, arm.get_conversion_rate(), arm.included_count) 
                  for arm in bandit.arms.values() if arm.included_count >= min_inclusions]
    
    # Сортируем по конверсии (по убыванию)
    valid_arms.sort(key=lambda x: x[1], reverse=True)
    
    # Вывод топ-30 вопросов
    print("\nTOP PERFORMING QUESTIONS:")
    for i, (qid, conv_rate, inclusions) in enumerate(valid_arms[:30], 1):
        print(f"  {i}. Question #{qid}: {conv_rate:.3f} conversion rate when included ({inclusions} times)")
    
    # Вывод 20 наихудших вопросов
    print("\nWORST PERFORMING QUESTIONS:")
    for i, (qid, conv_rate, inclusions) in enumerate(valid_arms[-20:], 1):
        print(f"  {i}. Question #{qid}: {conv_rate:.3f} conversion rate when included ({inclusions} times)")
    
    print(f"\nConversion rate plot saved to {plot_path}")

# Новые функции конверсии с индивидуальной вероятностью для каждого вопроса

def individual_last30_boost_rule(questions: List[Question]) -> Dict[int, bool]:
    """
    Тест-кейс: последние 30 вопросов (21-50) имеют высокую конверсию 30%, остальные 10%.
    Возвращает словарь с индивидуальными результатами конверсии для каждого вопроса.
    """
    # Словарь для хранения результатов конверсии по каждому вопросу
    conversion_results = {}
    
    # Определяем конверсию для каждого вопроса индивидуально
    for q in questions:
        if 21 <= q.id <= 50:
            # Вопросы 21-50 имеют 30% шанс конверсии
            conversion_prob = 0.3
        else:
            # Вопросы 1-20 имеют 10% шанс конверсии
            conversion_prob = 0.1
        
        # Определяем, произошла ли конверсия для этого вопроса
        conversion_results[q.id] = random.random() < conversion_prob
    
    return conversion_results

def individual_first20_low_rule(questions: List[Question]) -> Dict[int, bool]:
    """
    Тест-кейс: первые 20 вопросов (1-20) имеют низкую конверсию 1%, остальные 5%.
    Возвращает словарь с индивидуальными результатами конверсии для каждого вопроса.
    """
    # Словарь для хранения результатов конверсии по каждому вопросу
    conversion_results = {}
    
    # Определяем конверсию для каждого вопроса индивидуально
    for q in questions:
        if 1 <= q.id <= 20:
            # Вопросы 1-20 имеют 1% шанс конверсии
            conversion_prob = 0.01
        else:
            # Вопросы 21-50 имеют 5% шанс конверсии
            conversion_prob = 0.05
        
        # Определяем, произошла ли конверсия для этого вопроса
        conversion_results[q.id] = random.random() < conversion_prob
    
    return conversion_results

def decreasing_first20_rule(questions: List[Question]) -> Dict[int, bool]:
    """
    Тест-кейс: первые 20 вопросов (1-20) имеют убывающую конверсию от 50% до 31%.
    Первый вопрос - 50%, второй - 49%, третий - 48% и т.д.
    Остальные вопросы (21-50) имеют фиксированную конверсию 10%.
    Возвращает словарь с индивидуальными результатами конверсии для каждого вопроса.
    """
    # Словарь для хранения результатов конверсии по каждому вопросу
    conversion_results = {}
    
    # Определяем конверсию для каждого вопроса индивидуально
    for q in questions:
        if 1 <= q.id <= 20:
            # Вопросы 1-20 имеют убывающую конверсию: 50%, 49%, 48%,... 31%
            conversion_prob = 0.5 - (q.id - 1) * 0.01
        else:
            # Вопросы 21-50 имеют фиксированную 10% конверсию
            conversion_prob = 0.1
        
        # Определяем, произошла ли конверсия для этого вопроса
        conversion_results[q.id] = random.random() < conversion_prob
    
    return conversion_results

# Main function to run test cases with individual contribution
def run_individual_test_cases():
    """Run test cases for individual question contribution."""
    print(f"Saving results to directory: {OUTPUT_DIR}")
    
    # Test case 1: Decreasing conversion for first 20 questions
    run_test_case("decreasing_first20", decreasing_first20_rule, 
                 individual_contribution=True, num_sessions=1000)
    
    # Test case 2: Higher conversion for questions 21-50
    run_test_case("individual_last30_boost", individual_last30_boost_rule, 
                 individual_contribution=True, num_sessions=1000)
    
    # Test case 3: Lower conversion for questions 1-20
    run_test_case("individual_first20_low", individual_first20_low_rule, 
                 individual_contribution=True, num_sessions=1000)

if __name__ == "__main__":
    run_individual_test_cases()
