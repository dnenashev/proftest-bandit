import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

# Создаем директорию для выходных файлов
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "order_bandit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Question:
    """Represents a single test question with unique ID and content."""
    def __init__(self, id: int, content: str):
        self.id = id
        self.content = content

class OrderArm:
    """Represents an ordering of questions (a specific permutation of top questions)."""
    def __init__(self, order_id: int, question_order: List[int]):
        self.order_id = order_id  # Уникальный ID для этого порядка
        self.question_order = question_order  # Список ID вопросов в определенном порядке
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

class OrderBandit:
    """Multi-armed bandit for optimizing the order of questions for maximum conversion."""
    def __init__(self, 
                 questions: List[Question], 
                 top_questions: List[int] = None,
                 max_arms: int = 100,
                 exploration_factor: float = 0.2,
                 data_file: str = "order_bandit_data.json"):
        self.questions = questions
        self.question_dict = {q.id: q for q in questions}
        
        # Если список лучших вопросов не предоставлен, используем первые 30
        self.top_questions = top_questions or [q.id for q in questions[:30]]
        
        self.n_questions = len(self.top_questions)
        self.max_arms = max_arms  # Максимальное количество различных порядков, которые мы храним
        self.exploration_factor = exploration_factor
        self.data_file = os.path.join(OUTPUT_DIR, data_file)
        
        # Банки "рук" - каждая рука представляет собой определенный порядок вопросов
        self.arms: Dict[int, OrderArm] = {}  # Maps order ID to OrderArm
        self.arms_by_order: Dict[str, int] = {}  # Maps order hash to order ID
        self.next_arm_id = 1  # Counter for generating unique order IDs
        
        self.current_session: Optional[Tuple[str, int]] = None
        
        # Tracking metrics over time
        self.conversion_history = []
        self.rolling_conversion_rate = []
        self.total_conversions = 0
        self.total_sessions = 0
        
        # Initialize arms with some random orders
        self._initialize_arms()
        
        # Load existing data if available
        self._load_data()
    
    def _initialize_arms(self) -> None:
        """Initialize arms with some random permutations of the top questions."""
        # Create first arm with the original order
        original_order = list(self.top_questions)
        self._add_arm(original_order)
        
        # Create arms with random permutations
        for _ in range(min(20, self.max_arms - 1)):  # Start with up to 20 random orders
            # Create a random permutation
            random_order = list(self.top_questions)
            random.shuffle(random_order)
            self._add_arm(random_order)
    
    def _add_arm(self, question_order: List[int]) -> int:
        """Add a new arm with the given question order if it doesn't exist."""
        # Create a string hash of the order for comparison
        order_hash = ','.join(map(str, question_order))
        
        # Check if this order already exists
        if order_hash in self.arms_by_order:
            return self.arms_by_order[order_hash]
        
        # Check if we've reached the maximum number of arms
        if len(self.arms) >= self.max_arms:
            # If we have too many arms, we don't create a new one
            return -1
        
        # Create a new arm with this order
        arm_id = self.next_arm_id
        self.next_arm_id += 1
        
        self.arms[arm_id] = OrderArm(arm_id, question_order)
        self.arms_by_order[order_hash] = arm_id
        
        return arm_id
    
    def _mutate_order(self, original_order: List[int], mutation_strength: float = 0.3) -> List[int]:
        """Create a slightly modified version of an order by swapping some elements."""
        # Create a copy of the original order
        new_order = original_order.copy()
        
        # Determine how many swaps to make based on mutation strength
        n_swaps = max(1, int(mutation_strength * len(new_order)))
        
        # Perform the swaps
        for _ in range(n_swaps):
            # Select two random positions
            i, j = random.sample(range(len(new_order)), 2)
            # Swap the elements
            new_order[i], new_order[j] = new_order[j], new_order[i]
        
        return new_order
    
    def _load_data(self) -> None:
        """Load existing bandit data from file if available."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load metrics
                self.total_conversions = data.get('total_conversions', 0)
                self.total_sessions = data.get('total_sessions', 0)
                self.next_arm_id = data.get('next_arm_id', 1)
                
                # Load arms
                for arm_id_str, arm_data in data.get('arms', {}).items():
                    arm_id = int(arm_id_str)
                    question_order = arm_data.get('question_order', [])
                    
                    # Create the arm
                    arm = OrderArm(arm_id, question_order)
                    arm.alpha = arm_data.get('alpha', 1)
                    arm.beta = arm_data.get('beta', 1)
                    arm.pulls = arm_data.get('pulls', 0)
                    arm.conversions = arm_data.get('conversions', 0)
                    
                    # Add to our collection
                    self.arms[arm_id] = arm
                    self.arms_by_order[','.join(map(str, question_order))] = arm_id
                
                # Load conversion history if available
                self.conversion_history = data.get('conversion_history', [])
                self.rolling_conversion_rate = data.get('rolling_conversion_rate', [])
                
                print(f"Loaded data for {len(self.arms)} order arms from existing data")
            except Exception as e:
                print(f"Error loading data: {e}")
                # Initialize fresh if loading fails
                self._initialize_arms()
    
    def _save_data(self) -> None:
        """Save current bandit data to file."""
        data = {
            'total_conversions': self.total_conversions,
            'total_sessions': self.total_sessions,
            'next_arm_id': self.next_arm_id,
            'arms': {},
            'conversion_history': self.conversion_history,
            'rolling_conversion_rate': self.rolling_conversion_rate
        }
        
        # Save each arm's data
        for arm_id, arm in self.arms.items():
            data['arms'][str(arm_id)] = {
                'question_order': arm.question_order,
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
        if random.random() < self.exploration_factor:
            # Exploration phase - consider trying a new order
            if len(self.arms) < self.max_arms and random.random() < 0.5:
                # Sometimes create a completely new random order
                random_order = list(self.top_questions)
                random.shuffle(random_order)
                arm_id = self._add_arm(random_order)
            else:
                # Otherwise mutate one of the existing good orders
                # Get all arms sorted by performance (descending)
                sorted_arms = sorted(self.arms.values(), 
                                    key=lambda arm: arm.sample_value(), 
                                    reverse=True)
                
                # If we have enough information, prefer to mutate a good performer
                if any(arm.pulls > 5 for arm in sorted_arms):
                    # Filter to only include arms that have been pulled enough times
                    candidate_arms = [arm for arm in sorted_arms if arm.pulls > 5]
                    # Take one of the top performing arms
                    source_arm = random.choice(candidate_arms[:max(3, len(candidate_arms) // 3)])
                else:
                    # Not enough data yet, select randomly
                    source_arm = random.choice(sorted_arms)
                
                # Mutate the order of the selected arm
                mutation_strength = random.uniform(0.1, 0.5)  # More variety in mutation strength
                new_order = self._mutate_order(source_arm.question_order, mutation_strength)
                
                # Add the new arm or get ID of existing one
                arm_id = self._add_arm(new_order)
        else:
            # Exploitation phase - use Thompson Sampling to select best order
            arm_id = self._select_order_thompson_sampling()
        
        # If we failed to get a valid arm (e.g., max arms reached), select using Thompson Sampling
        if arm_id == -1:
            arm_id = self._select_order_thompson_sampling()
        
        # Record the current session
        self.current_session = (user_id, arm_id)
        
        # Return the actual questions in the selected order
        selected_order = self.arms[arm_id].question_order
        return [self.question_dict[qid] for qid in selected_order]
    
    def _select_order_thompson_sampling(self) -> int:
        """Use Thompson Sampling to select the best arm (question order)."""
        best_value = -1
        best_arm_id = -1
        
        # Sample from each arm and select the one with highest value
        for arm_id, arm in self.arms.items():
            value = arm.sample_value()
            if value > best_value:
                best_value = value
                best_arm_id = arm_id
        
        return best_arm_id
    
    def update_with_result(self, user_id: str, conversion: bool) -> None:
        """Update the bandit with the result of a test session."""
        if self.current_session is None or self.current_session[0] != user_id:
            print(f"Warning: No active session for user {user_id}")
            return
        
        _, arm_id = self.current_session
        
        # Update global conversion tracking
        self.total_sessions += 1
        if conversion:
            self.total_conversions += 1
        
        # Track this individual conversion result
        self.conversion_history.append(1 if conversion else 0)
        
        # Update rolling average
        self._update_rolling_average()
        
        # Update the arm that was pulled
        arm = self.arms[arm_id]
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
    
    def get_best_orders(self, top_n: int = 10) -> List[Tuple[int, float, List[int], int]]:
        """Get the top N performing question orders."""
        # Only orders with minimum pulls
        min_pulls = 10
        valid_arms = [(arm.order_id, arm.get_conversion_rate(), arm.question_order, arm.pulls) 
                      for arm in self.arms.values() if arm.pulls >= min_pulls]
        
        # Sort by conversion rate (descending)
        valid_arms.sort(key=lambda x: x[1], reverse=True)
        
        return valid_arms[:top_n]
    
    def plot_conversion_over_time(self, save_path: str = "order_conversion_over_time.png") -> None:
        """Plot conversion rate over time."""
        if not self.conversion_history:
            print("No conversion history to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot the raw conversion history (1 for conversion, 0 for no conversion)
        plt.plot(self.conversion_history, 'k.', alpha=0.2, markersize=2)
        
        # Calculate and plot the rolling average
        window_size = min(100, len(self.conversion_history))
        rolling_avg = []
        for i in range(len(self.conversion_history) - window_size + 1):
            rolling_avg.append(sum(self.conversion_history[i:i+window_size]) / window_size)
        
        plt.plot(range(window_size-1, len(self.conversion_history)), rolling_avg, 'b-', linewidth=2,
                label=f'Rolling Average (window={window_size})')
        
        # Overall average line
        overall_rate = self.total_conversions / max(1, self.total_sessions)
        plt.axhline(y=overall_rate, color='r', linestyle='--', 
                   label=f'Overall Average: {overall_rate:.3f}')
        
        # Get the best performing orders
        best_orders = self.get_best_orders(3)
        
        # Add lines for the best orders
        for i, (arm_id, conv_rate, _, pulls) in enumerate(best_orders):
            if pulls >= 10:  # Only show if we have enough data
                plt.axhline(y=conv_rate, color=f'C{i+1}', linestyle='-.', 
                           label=f'Order #{arm_id}: {conv_rate:.3f} ({pulls} pulls)')
        
        plt.xlabel('Sessions')
        plt.ylabel('Conversion Rate')
        plt.title('Conversion Rate Over Time for Different Question Orders')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Dynamic y-axis range
        max_y = max(max(rolling_avg) if rolling_avg else 0, overall_rate)
        plt.ylim(-0.05, max_y * 1.1)  # Add some padding
        
        # Save the plot
        if save_path:
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.join(OUTPUT_DIR, os.path.basename(save_path))), exist_ok=True)
            full_path = os.path.join(OUTPUT_DIR, os.path.basename(save_path))
            plt.tight_layout()
            plt.savefig(full_path)
            print(f"Plot saved to {full_path}")
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

# Test case scenarios for order optimization
def run_order_test_case(test_name: str, conversion_rule: Callable, num_sessions: int = 1000,
                       top_questions: List[int] = None) -> None:
    """Run a test case with the given conversion rule."""
    print(f"\n========== Running Test Case: {test_name} ==========\n")
    
    # Create a test-specific data file
    data_file = f"order_bandit_data_{test_name}.json"
    
    # Remove existing data file to start fresh
    full_path = os.path.join(OUTPUT_DIR, data_file)
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Load questions
    questions = load_questions()
    
    # Create the bandit
    bandit = OrderBandit(
        questions=questions,
        top_questions=top_questions,
        max_arms=100,
        exploration_factor=0.2,
        data_file=data_file
    )
    
    # Simulate user sessions
    for i in range(1, num_sessions + 1):
        user_id = f"user_{i}"
        
        # Get questions for this user in a specific order
        test_questions = bandit.get_test_questions(user_id)
        
        # Apply the conversion rule for this test case
        conversion = conversion_rule(test_questions)
        
        # Update the bandit with the result
        bandit.update_with_result(user_id, conversion)
    
    # Plot conversion over time
    bandit.plot_conversion_over_time(f"order_conversion_{test_name}.png")
    
    # Итоги тестирования
    print(f"\n===== ORDER TEST RESULTS: '{test_name}' =====")
    print(f"Total sessions: {bandit.total_sessions}")
    print(f"Overall conversion rate: {bandit.total_conversions / bandit.total_sessions:.3f}")
    
    # Вывод лучших порядков вопросов
    best_orders = bandit.get_best_orders(top_n=5)
    print("\nTOP PERFORMING QUESTION ORDERS:")
    for i, (order_id, conv_rate, question_order, pulls) in enumerate(best_orders, 1):
        print(f"  {i}. Order #{order_id}: {conv_rate:.3f} conversion rate (used {pulls} times)")
        print(f"     Questions in this order: {question_order[:10]}... (showing first 10)")

    print(f"\nConversion rate plot saved to order_conversion_{test_name}.png")

# Функции конверсии для сценариев с оптимизацией порядка вопросов

def first_cluster_order_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Если первый кластер связанных вопросов (1-10) идёт в правильном порядке,
    то конверсия выше. Конверсия максимальна если вопросы 1-10 идут в порядке возрастания.
    """
    ordered_ids = [q.id for q in questions]
    
    # Находим позиции вопросов из первого кластера
    first_cluster_positions = {qid: idx for idx, qid in enumerate(ordered_ids) if 1 <= qid <= 10}
    
    # Если не все вопросы из кластера присутствуют, используем базовую вероятность
    if len(first_cluster_positions) < 10:
        base_prob = 0.1
        return random.random() < base_prob
    
    # Проверяем относительный порядок первого кластера
    # Вычисляем количество пар, которые находятся в правильном порядке (возрастания)
    correct_order_pairs = 0
    total_pairs = 0
    
    for i in range(1, 10):
        for j in range(i+1, 11):
            total_pairs += 1
            if first_cluster_positions[i] < first_cluster_positions[j]:
                correct_order_pairs += 1
    
    # Вычисляем процент правильно упорядоченных пар
    correct_order_ratio = correct_order_pairs / total_pairs
    
    # Преобразуем это в вероятность конверсии:
    # - При правильном порядке (ratio = 1.0) -> вероятность 0.25
    # - При случайном порядке (ratio ~ 0.5) -> вероятность 0.1
    conversion_prob = 0.1 + (0.15 * correct_order_ratio)
    
    return random.random() < conversion_prob

def second_cluster_order_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Если вопросы идут в порядке уменьшения сложности (от сложных к простым),
    то конверсия выше. Сложность вопроса условно соответствует его ID (чем больше ID, тем сложнее).
    """
    ordered_ids = [q.id for q in questions]
    
    # Проверяем общую тенденцию порядка относительно ID вопросов
    # (насколько хорошо соблюдается убывающий порядок)
    decreasing_pairs = 0
    total_pairs = 0
    
    for i in range(len(ordered_ids) - 1):
        for j in range(i + 1, len(ordered_ids)):
            total_pairs += 1
            # Если вопрос с большим ID (более сложный) идет раньше, это правильный порядок
            if ordered_ids[i] > ordered_ids[j]:
                decreasing_pairs += 1
    
    # Вычисляем процент правильно упорядоченных пар (по убыванию)
    decreasing_ratio = decreasing_pairs / total_pairs if total_pairs > 0 else 0.5
    
    # Преобразуем это в вероятность конверсии:
    # - При идеальном убывающем порядке (ratio = 1.0) -> вероятность 0.35
    # - При случайном порядке (ratio ~ 0.5) -> вероятность 0.2
    conversion_prob = 0.2 + (0.15 * (decreasing_ratio - 0.5) * 2)
    
    return random.random() < conversion_prob

def third_cluster_order_rule(questions: List[Question]) -> bool:
    """
    Тест-кейс: Вопросы делятся на три группы, и оптимальный порядок - сначала средние по сложности (21-30),
    затем простые (1-20), и в конце сложные (31-50).
    """
    ordered_ids = [q.id for q in questions]
    
    # Классифицируем вопросы на три группы
    easy_ids = [qid for qid in ordered_ids if 1 <= qid <= 20]
    medium_ids = [qid for qid in ordered_ids if 21 <= qid <= 30]
    hard_ids = [qid for qid in ordered_ids if 31 <= qid <= 50]
    
    # Находим позицию последнего среднего вопроса
    last_medium_pos = max([ordered_ids.index(qid) for qid in medium_ids]) if medium_ids else -1
    
    # Находим позицию первого сложного вопроса
    first_hard_pos = min([ordered_ids.index(qid) for qid in hard_ids]) if hard_ids else len(ordered_ids)
    
    # Находим позицию первого легкого вопроса
    first_easy_pos = min([ordered_ids.index(qid) for qid in easy_ids]) if easy_ids else -1
    
    # Проверяем соответствие оптимальному порядку:
    # 1. Средние вопросы должны быть в начале
    medium_at_beginning = first_easy_pos > 0 and (all(ordered_ids.index(mid) < first_easy_pos for mid in medium_ids))
    
    # 2. Легкие вопросы должны быть в середине
    easy_in_middle = (first_easy_pos > last_medium_pos or last_medium_pos == -1) and (all(ordered_ids.index(easy) < first_hard_pos for easy in easy_ids))
    
    # 3. Сложные вопросы должны быть в конце
    hard_at_end = (first_hard_pos > last_medium_pos or last_medium_pos == -1) and (all(ordered_ids.index(hard) > first_easy_pos for hard in hard_ids))
    
    # Рассчитываем базовый балл на основе соответствия правилам (каждое правило дает 0.05 бонуса)
    base_score = 0.15  # Базовая вероятность
    if medium_at_beginning:
        base_score += 0.05
    if easy_in_middle:
        base_score += 0.05
    if hard_at_end:
        base_score += 0.05
    
    # Добавляем бонус за общую согласованность в каждой группе
    # (средние вопросы идут первыми, легкие вторыми, сложные последними)
    ideal_positions = {}
    pos = 0
    
    # Идеальный порядок: сначала средние вопросы
    for qid in range(21, 31):
        if qid in [q.id for q in questions]:
            ideal_positions[qid] = pos
            pos += 1
    
    # Затем легкие вопросы
    for qid in range(1, 21):
        if qid in [q.id for q in questions]:
            ideal_positions[qid] = pos
            pos += 1
    
    # Затем сложные вопросы
    for qid in range(31, 51):
        if qid in [q.id for q in questions]:
            ideal_positions[qid] = pos
            pos += 1
    
    # Вычисляем коэффициент корреляции Спирмена между идеальными и реальными позициями
    real_positions = {qid: idx for idx, qid in enumerate(ordered_ids)}
    
    common_ids = set(ideal_positions.keys()).intersection(set(real_positions.keys()))
    if common_ids:
        ideal_ranks = [ideal_positions[qid] for qid in common_ids]
        real_ranks = [real_positions[qid] for qid in common_ids]
        
        # Простая корреляция для определения степени совпадения порядков
        n = len(common_ids)
        sum_d_squared = sum((ideal_ranks[i] - real_ranks[i])**2 for i in range(n))
        
        # Вычисляем ρ (ро) формулой для коэффициента корреляции Спирмена
        # ρ = 1 - (6 * Σd²) / (n * (n² - 1))
        spearman_coef = 1 - (6 * sum_d_squared) / (n * (n**2 - 1)) if n > 1 else 0
        
        # Масштабируем коэффициент корреляции в диапазон 0-0.1 для дополнительного бонуса
        order_bonus = 0.1 * (spearman_coef + 1) / 2
        
        conversion_prob = base_score + order_bonus
    else:
        conversion_prob = base_score
    
    return random.random() < conversion_prob

# Main function to run order test cases
def run_order_bandit_test_cases():
    """Run test cases for the order bandit."""
    print(f"Saving results to directory: {OUTPUT_DIR}")
    
    # Define a set of top questions to optimize (e.g., first 30 questions)
    top_questions = list(range(1, 31))
    
    # Тест-кейс 1: Оптимизация порядка первого кластера вопросов (1-10)
    run_order_test_case("first_cluster", first_cluster_order_rule, 
                       num_sessions=1000, top_questions=top_questions)
    
    # Тест-кейс 2: Оптимизация порядка по сложности (от сложных к простым)
    run_order_test_case("second_cluster", second_cluster_order_rule, 
                       num_sessions=1000, top_questions=top_questions)
    
    # Тест-кейс 3: Оптимизация порядка по группам сложности (средние, простые, сложные)
    run_order_test_case("third_cluster", third_cluster_order_rule, 
                       num_sessions=1000, top_questions=top_questions)

if __name__ == "__main__":
    run_order_bandit_test_cases()  # Run test cases for order optimization 