import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
import random
import json
import os
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

# Создаем директорию для выходных файлов
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "position_bandit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Question:
    """Represents a single test question with unique ID and content."""
    def __init__(self, id: int, content: str):
        self.id = id
        self.content = content

class PositionArm:
    """Represents an arm for a specific question at a specific position."""
    def __init__(self, question_id: int):
        self.question_id = question_id
        # Beta distribution parameters for Thompson Sampling
        self.alpha = 1  # successes + 1 (1 is the prior)
        self.beta = 1   # failures + 1 (1 is the prior)
        self.pulls = 0  # number of times this arm has been pulled
        self.conversions = 0  # number of successful conversions (moved to next question)
    
    def get_conversion_rate(self) -> float:
        """Get the current estimated conversion rate."""
        if self.pulls == 0:
            return 0
        return self.conversions / self.pulls
    
    def sample_value(self) -> float:
        """Sample a value from the Beta distribution for this arm."""
        return np.random.beta(self.alpha, self.beta)

class PositionBandit:
    """
    Multi-armed bandit for optimizing questions at each position sequentially.
    Focuses on next-question conversion rather than overall conversion.
    """
    def __init__(self, 
                 questions: List[Question], 
                 candidate_questions: List[int] = None,
                 test_size: int = 30,
                 exploration_factor: float = 0.2,
                 min_trials_per_position: int = 100,
                 data_file: str = "position_bandit_data.json"):
        
        self.questions = questions
        self.question_dict = {q.id: q for q in questions}
        
        # Если список кандидатов не предоставлен, используем все вопросы
        self.candidate_questions = candidate_questions or [q.id for q in questions]
        
        self.test_size = min(test_size, len(questions))  # Общее количество вопросов в тесте
        self.exploration_factor = exploration_factor
        self.min_trials_per_position = min_trials_per_position  # Минимальное количество тестов на позицию
        # Сохраняем файл данных в поддиректорию
        self.data_file = os.path.join(OUTPUT_DIR, data_file)
        
        # Текущая позиция, для которой мы оптимизируем
        self.current_position = 0
        
        # Уже выбранные вопросы для предыдущих позиций
        self.fixed_questions: List[int] = []
        
        # Банки "рук" для каждой позиции
        # position -> question_id -> PositionArm
        self.arms: Dict[int, Dict[int, PositionArm]] = {}
        
        # Текущая сессия и текущий вопрос на тестируемой позиции
        self.current_session: Optional[Tuple[str, int, int]] = None  # (user_id, position, question_id)
        
        # Метрики по каждой позиции
        self.position_metrics: Dict[int, Dict] = {}
        
        # Загружаем данные или инициализируем с нуля
        self._load_data()
    
    def _initialize_position(self, position: int) -> None:
        """Инициализация новой позиции для оптимизации."""
        if position not in self.arms:
            self.arms[position] = {}
        
        # Определяем кандидатов для этой позиции (исключая уже выбранные вопросы)
        available_candidates = [q_id for q_id in self.candidate_questions 
                              if q_id not in self.fixed_questions]
        
        # Создаем "руки" для каждого доступного вопроса на этой позиции
        for question_id in available_candidates:
            if question_id not in self.arms[position]:
                self.arms[position][question_id] = PositionArm(question_id)
        
        # Инициализируем метрики для этой позиции
        if position not in self.position_metrics:
            self.position_metrics[position] = {
                'total_trials': 0,          # общее количество испытаний
                'total_conversions': 0,     # общее количество конверсий
                'conversion_history': [],   # история конверсий
                'best_question_id': None,   # ID лучшего вопроса
                'best_conversion_rate': 0,  # конверсия лучшего вопроса
                'is_finalized': False       # финализирована ли позиция
            }
    
    def _load_data(self) -> None:
        """Загрузка существующих данных или инициализация с нуля."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Загружаем текущую позицию и зафиксированные вопросы
                self.current_position = data.get('current_position', 0)
                self.fixed_questions = data.get('fixed_questions', [])
                
                # Загружаем данные о "руках" для каждой позиции
                for pos_str, arms_data in data.get('arms', {}).items():
                    position = int(pos_str)
                    if position not in self.arms:
                        self.arms[position] = {}
                    
                    for q_id_str, arm_data in arms_data.items():
                        question_id = int(q_id_str)
                        arm = PositionArm(question_id)
                        arm.alpha = arm_data.get('alpha', 1)
                        arm.beta = arm_data.get('beta', 1)
                        arm.pulls = arm_data.get('pulls', 0)
                        arm.conversions = arm_data.get('conversions', 0)
                        self.arms[position][question_id] = arm
                
                # Загружаем метрики по позициям
                self.position_metrics = data.get('position_metrics', {})
                # Преобразуем строковые ключи в int
                self.position_metrics = {int(k): v for k, v in self.position_metrics.items()}
                
                print(f"Загружены данные для {len(self.arms)} позиций с {sum(len(arms) for arms in self.arms.values())} рук")
                
                # Проверяем, нужно ли инициализировать текущую позицию
                self._ensure_current_position_initialized()
                
            except Exception as e:
                print(f"Ошибка загрузки данных: {e}")
                # Если ошибка, начинаем с первой позиции
                self.current_position = 0
                self.fixed_questions = []
                self._initialize_position(0)
        else:
            # Если файла нет, начинаем с первой позиции
            self._initialize_position(0)
    
    def _ensure_current_position_initialized(self) -> None:
        """Проверяет, что текущая позиция инициализирована."""
        if self.current_position not in self.arms:
            self._initialize_position(self.current_position)
    
    def _save_data(self) -> None:
        """Сохранение текущих данных бандита."""
        data = {
            'current_position': self.current_position,
            'fixed_questions': self.fixed_questions,
            'arms': {},
            'position_metrics': self.position_metrics
        }
        
        # Сохраняем данные о "руках" для каждой позиции
        for position, position_arms in self.arms.items():
            data['arms'][str(position)] = {}
            for question_id, arm in position_arms.items():
                data['arms'][str(position)][str(question_id)] = {
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'pulls': arm.pulls,
                    'conversions': arm.conversions
                }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения данных: {e}")
    
    def _check_position_finalization(self, position: int) -> bool:
        """
        Проверяет, можно ли финализировать позицию 
        (выбрать лучший вопрос и перейти к следующей).
        """
        metrics = self.position_metrics[position]
        
        # Если уже финализирована, просто возвращаем True
        if metrics['is_finalized']:
            return True
        
        # Проверяем, достаточно ли испытаний
        if metrics['total_trials'] < self.min_trials_per_position:
            return False
        
        # Находим вопросы с минимальным количеством испытаний
        position_arms = self.arms[position]
        min_pulls = min(arm.pulls for arm in position_arms.values() if arm.pulls > 0)
        
        # Если какие-то вопросы недостаточно протестированы, не финализируем
        if min_pulls < 10:
            return False
        
        # Находим лучший вопрос для этой позиции
        best_question_id = None
        best_conversion_rate = 0
        
        for qid, arm in position_arms.items():
            if arm.pulls >= 10:  # проверяем только вопросы с достаточной статистикой
                conv_rate = arm.get_conversion_rate()
                if conv_rate > best_conversion_rate:
                    best_conversion_rate = conv_rate
                    best_question_id = qid
        
        # Если не нашли подходящий вопрос, не финализируем
        if best_question_id is None:
            return False
        
        # Если мы дошли до этой точки, финализируем позицию
        metrics['is_finalized'] = True
        metrics['best_question_id'] = best_question_id
        metrics['best_conversion_rate'] = best_conversion_rate
        
        # Добавляем лучший вопрос в список фиксированных
        self.fixed_questions.append(best_question_id)
        
        # Переходим к следующей позиции
        self.current_position += 1
        
        # Инициализируем следующую позицию, если она в пределах test_size
        if self.current_position < self.test_size:
            self._initialize_position(self.current_position)
        
        # Сохраняем изменения
        self._save_data()
        
        return True
    
    def get_test_questions(self, user_id: str) -> List[Question]:
        """Получение списка вопросов для тестирования пользователем."""
        # Проверяем, инициализирована ли текущая позиция
        self._ensure_current_position_initialized()
        
        # Если все позиции финализированы, просто возвращаем финальный порядок
        if self.current_position >= self.test_size:
            return [self.question_dict[qid] for qid in self.fixed_questions]
        
        # Получаем вопросы для фиксированных позиций
        fixed_questions = [self.question_dict[qid] for qid in self.fixed_questions]
        
        # Проверяем финализацию текущей позиции
        if self._check_position_finalization(self.current_position):
            # Если позиция была финализирована, рекурсивно вызываем функцию для новой позиции
            return self.get_test_questions(user_id)
        
        # Выбираем вопрос для текущей позиции (exploration или exploitation)
        current_question_id = self._select_question_for_position(self.current_position)
        
        # Сохраняем информацию о текущей сессии (для последующего обновления)
        self.current_session = (user_id, self.current_position, current_question_id)
        
        # Помещаем выбранный вопрос на текущую позицию
        current_question = self.question_dict[current_question_id]
        
        # Формируем полный список вопросов для теста
        test_questions = fixed_questions + [current_question]
        
        # Добавляем случайные вопросы, чтобы заполнить test_size
        # (исключаем уже выбранные вопросы)
        used_question_ids = set(self.fixed_questions + [current_question_id])
        available_questions = [q for q in self.questions 
                             if q.id not in used_question_ids]
        
        # Если нужно добавить еще вопросы
        if len(test_questions) < self.test_size and available_questions:
            # Выбираем случайные вопросы из доступных
            additional_questions = random.sample(
                available_questions, 
                min(self.test_size - len(test_questions), len(available_questions))
            )
            test_questions.extend(additional_questions)
        
        return test_questions
    
    def _select_question_for_position(self, position: int) -> int:
        """Выбор вопроса для тестирования на указанной позиции."""
        position_arms = self.arms[position]
        
        # Исследование (exploration) с вероятностью exploration_factor
        if random.random() < self.exploration_factor:
            # Выбираем случайный вопрос из доступных для этой позиции
            available_question_ids = list(position_arms.keys())
            return random.choice(available_question_ids)
        
        # Эксплуатация (exploitation) - используем Thompson Sampling
        best_value = -1
        best_question_id = None
        
        for qid, arm in position_arms.items():
            value = arm.sample_value()
            if value > best_value:
                best_value = value
                best_question_id = qid
        
        return best_question_id
    
    def update_with_result(self, user_id: str, conversion: bool) -> None:
        """
        Обновление результатов тестирования.
        conversion = True означает, что пользователь перешел к следующему вопросу.
        """
        if self.current_session is None or self.current_session[0] != user_id:
            print(f"Предупреждение: Нет активной сессии для пользователя {user_id}")
            return
        
        user_id, position, question_id = self.current_session
        
        # Получаем "руку" для этой позиции и вопроса
        arm = self.arms[position][question_id]
        
        # Обновляем метрики
        metrics = self.position_metrics[position]
        metrics['total_trials'] += 1
        if conversion:
            metrics['total_conversions'] += 1
        metrics['conversion_history'].append(1 if conversion else 0)
        
        # Обновляем "руку"
        arm.pulls += 1
        if conversion:
            arm.conversions += 1
            arm.alpha += 1
        else:
            arm.beta += 1
        
        # Сбрасываем текущую сессию
        self.current_session = None
        
        # Проверяем, можно ли финализировать позицию
        self._check_position_finalization(position)
        
        # Сохраняем данные
        self._save_data()
    
    def get_best_questions_for_position(self, position: int, top_n: int = 5) -> List[Tuple[int, float, int]]:
        """Получение списка лучших вопросов для указанной позиции."""
        if position not in self.arms:
            return []
        
        position_arms = self.arms[position]
        
        # Формируем список вопросов с минимальным количеством испытаний
        min_pulls = 10
        valid_arms = [(qid, arm.get_conversion_rate(), arm.pulls) 
                     for qid, arm in position_arms.items() 
                     if arm.pulls >= min_pulls]
        
        # Сортируем по конверсии (по убыванию)
        valid_arms.sort(key=lambda x: x[1], reverse=True)
        
        return valid_arms[:top_n]
    
    def get_finalized_order(self) -> List[int]:
        """Получение финализированного порядка вопросов."""
        return self.fixed_questions
    
    def plot_position_conversion(self, position: int, save_path: str = None) -> None:
        """Построение графика конверсии для позиции."""
        if position not in self.position_metrics:
            print(f"Нет данных для позиции {position}")
            return
        
        metrics = self.position_metrics[position]
        history = metrics['conversion_history']
        
        if not history:
            print(f"Нет истории конверсий для позиции {position}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Строим скользящее среднее (на окне в 100 значений)
        window_size = min(100, len(history))
        avg_conversions = []
        for i in range(len(history) - window_size + 1):
            avg_conversions.append(sum(history[i:i+window_size]) / window_size)
        
        # Строим график
        plt.plot(range(window_size-1, len(history)), avg_conversions, 'b-', linewidth=2,
                label=f'Скользящая средняя (окно {window_size})')
        
        # Добавляем линию общей средней конверсии
        overall_rate = metrics['total_conversions'] / metrics['total_trials'] if metrics['total_trials'] > 0 else 0
        plt.axhline(y=overall_rate, color='r', linestyle='--', 
                   label=f'Общая средняя: {overall_rate:.2f}')
        
        # Если позиция финализирована, отмечаем лучший вопрос
        if metrics['is_finalized']:
            best_qid = metrics['best_question_id']
            best_rate = metrics['best_conversion_rate']
            plt.axhline(y=best_rate, color='g', linestyle='-.', 
                       label=f'Лучший вопрос #{best_qid}: {best_rate:.2f}')
        
        # Добавляем подписи
        plt.xlabel('Количество испытаний')
        plt.ylabel('Конверсия')
        plt.title(f'Конверсия на позиции {position}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Динамически устанавливаем диапазон оси Y
        plt.ylim(0, max(max(avg_conversions) if avg_conversions else 0, overall_rate) * 1.1)
        
        # Сохраняем график, если указан путь
        if save_path:
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"График сохранен в {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_positions(self, base_path: str = "position_conversion_") -> None:
        """Построение графиков для всех позиций."""
        for position in self.position_metrics.keys():
            save_path = os.path.join(OUTPUT_DIR, f"{base_path}{position}.png")
            self.plot_position_conversion(position, save_path)
    
    def print_status(self) -> None:
        """Вывод текущего статуса оптимизации."""
        print("\n===== СТАТУС ОПТИМИЗАЦИИ ПО ПОЗИЦИЯМ =====")
        print(f"Всего позиций: {self.test_size}")
        print(f"Текущая позиция: {self.current_position}")
        print(f"Финализированных позиций: {len(self.fixed_questions)}")
        
        if self.fixed_questions:
            print("\nФИНАЛИЗИРОВАННЫЙ ПОРЯДОК ВОПРОСОВ:")
            for pos, qid in enumerate(self.fixed_questions):
                metrics = self.position_metrics.get(pos, {})
                conv_rate = metrics.get('best_conversion_rate', 0)
                print(f"  Позиция {pos}: Вопрос #{qid} (конверсия: {conv_rate:.3f})")
        
        if self.current_position < self.test_size:
            print(f"\nТЕКУЩАЯ ОПТИМИЗИРУЕМАЯ ПОЗИЦИЯ: {self.current_position}")
            metrics = self.position_metrics.get(self.current_position, {})
            print(f"  Испытаний: {metrics.get('total_trials', 0)}")
            print(f"  Общая конверсия: {metrics.get('total_conversions', 0) / max(1, metrics.get('total_trials', 1)):.3f}")
            
            best_questions = self.get_best_questions_for_position(self.current_position)
            if best_questions:
                print("\n  ЛУЧШИЕ ВОПРОСЫ ДЛЯ ТЕКУЩЕЙ ПОЗИЦИИ:")
                for i, (qid, conv_rate, pulls) in enumerate(best_questions, 1):
                    print(f"    {i}. Вопрос #{qid}: {conv_rate:.3f} конверсия ({pulls} испытаний)")

# Загрузка вопросов
def load_questions() -> List[Question]:
    """Загрузка вопросов из файла или генерация тестовых."""
    questions = []
    for i in range(1, 51):
        questions.append(Question(i, f"Содержание вопроса #{i}"))
    return questions

# Тестовые сценарии для бандита по позициям
def run_position_test_case(test_name: str, conversion_rule: Callable, num_sessions: int = 1000,
                          candidate_questions: List[int] = None, test_size: int = 30) -> None:
    """Запуск тестового сценария для бандита по позициям."""
    print(f"\n========== Запуск теста для позиционного бандита: {test_name} ==========\n")
    
    # Создаем файл данных специфичный для этого теста
    data_file = f"position_bandit_data_{test_name}.json"
    
    # Удаляем существующий файл данных, чтобы начать с чистого листа
    full_path = os.path.join(OUTPUT_DIR, data_file)
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Загружаем вопросы
    questions = load_questions()
    
    # Создаем бандит
    bandit = PositionBandit(
        questions=questions,
        candidate_questions=candidate_questions,
        test_size=test_size,
        exploration_factor=0.2,
        min_trials_per_position=100,  # Минимум 100 испытаний на позицию
        data_file=data_file
    )
    
    # Симулируем пользовательские сессии
    for i in range(1, num_sessions + 1):
        user_id = f"user_{i}"
        
        # Получаем вопросы для пользователя
        test_questions = bandit.get_test_questions(user_id)
        
        # Если все позиции финализированы, прерываем цикл
        if bandit.current_position >= bandit.test_size:
            print(f"Все позиции финализированы после {i} сессий")
            break
        
        # Применяем правило конверсии для этого теста
        conversion = conversion_rule(test_questions, bandit.current_position)
        
        # Обновляем бандит с результатом
        bandit.update_with_result(user_id, conversion)
        
        # Периодически выводим статус
        if i % 100 == 0:
            print(f"Сессия {i}/{num_sessions}. Текущая позиция: {bandit.current_position}")
    
    # Строим графики для всех позиций
    bandit.plot_all_positions(f"position_conversion_{test_name}_")
    
    # Выводим финальный статус
    bandit.print_status()

# Функции конверсии для тестовых сценариев

def position_dependent_rule(questions: List[Question], current_position: int) -> bool:
    """
    Тест-кейс: Определенные вопросы имеют лучшую конверсию на определенных позициях.
    Например, вопросы с ID, кратным позиции + 1, имеют повышенную конверсию.
    """
    # Если текущая позиция выходит за пределы массива, используем базовую вероятность
    if current_position >= len(questions):
        return random.random() < 0.5
    
    # Получаем вопрос на текущей позиции
    question_id = questions[current_position].id
    
    # Базовая вероятность конверсии
    base_conversion = 0.5
    
    # Бонус за "правильный" вопрос на позиции
    position_bonus = 0.3
    
    # Вопросы, которые хорошо работают на каждой позиции
    # (ID вопроса кратно номеру позиции + 1)
    if question_id % (current_position + 1) == 0:
        conversion_prob = base_conversion + position_bonus
    else:
        conversion_prob = base_conversion
    
    return random.random() < conversion_prob

def exact_position_match_rule(questions: List[Question], current_position: int) -> bool:
    """
    Тест-кейс: Для каждой позиции есть только один идеальный вопрос (N-й вопрос для N-й позиции).
    Этот вопрос имеет высокую конверсию (50%), все остальные - очень низкую (5%).
    """
    # Если текущая позиция выходит за пределы массива, используем базовую вероятность
    if current_position >= len(questions):
        return random.random() < 0.5
    
    # Получаем вопрос на текущей позиции
    question_id = questions[current_position].id
    
    # Если номер вопроса соответствует позиции + 1 (т.к. позиции с 0, а вопросы с 1)
    if question_id == current_position + 1:
        # Высокая конверсия для идеального вопроса (50%)
        conversion_prob = 0.5
    else:
        # Низкая конверсия для всех остальных вопросов (5%)
        conversion_prob = 0.05
    
    return random.random() < conversion_prob

def question_sequence_rule(questions: List[Question], current_position: int) -> bool:
    """
    Тест-кейс: Некоторые последовательности вопросов имеют лучшую конверсию.
    Например, вопросы с возрастающими ID имеют лучшую конверсию, если идут по порядку.
    """
    # Если текущая позиция выходит за пределы массива, используем базовую вероятность
    if current_position >= len(questions):
        return random.random() < 0.5
    
    # Получаем вопрос на текущей позиции
    current_question = questions[current_position]
    
    # Базовая вероятность конверсии
    base_conversion = 0.5
    
    # Проверяем, сколько предыдущих вопросов образуют возрастающую последовательность
    sequence_length = 1
    for i in range(current_position - 1, -1, -1):
        if questions[i].id < current_question.id:
            sequence_length += 1
        else:
            break
    
    # Бонус за длину последовательности (максимум 0.3 за 5+ последовательных вопросов)
    sequence_bonus = min(0.3, 0.05 * sequence_length)
    
    # Итоговая вероятность
    conversion_prob = base_conversion + sequence_bonus
    
    return random.random() < conversion_prob

def question_group_rule(questions: List[Question], current_position: int) -> bool:
    """
    Тест-кейс: Вопросы делятся на группы, и определенные группы лучше работают в начале,
    середине или конце теста.
    """
    # Если текущая позиция выходит за пределы массива, используем базовую вероятность
    if current_position >= len(questions):
        return random.random() < 0.5
    
    # Получаем вопрос на текущей позиции
    question_id = questions[current_position].id
    
    # Базовая вероятность конверсии
    base_conversion = 0.5
    
    # Определяем группу вопроса
    if 1 <= question_id <= 10:
        group = "A"  # Легкие вопросы
    elif 11 <= question_id <= 30:
        group = "B"  # Средние вопросы
    else:
        group = "C"  # Сложные вопросы
    
    # Определяем, на каком этапе теста мы находимся
    test_size = len(questions)
    if current_position < test_size * 0.3:
        stage = "начало"
    elif current_position < test_size * 0.7:
        stage = "середина"
    else:
        stage = "конец"
    
    # Бонусы для разных групп в зависимости от этапа
    bonus = 0.0
    
    if group == "A" and stage == "начало":
        bonus = 0.25  # Легкие вопросы лучше в начале
    elif group == "B" and stage == "середина":
        bonus = 0.25  # Средние вопросы лучше в середине
    elif group == "C" and stage == "конец":
        bonus = 0.25  # Сложные вопросы лучше в конце
    
    # Итоговая вероятность
    conversion_prob = base_conversion + bonus
    
    return random.random() < conversion_prob

# Основная функция для запуска тестов
def run_position_bandit_test_cases():
    """Запуск тестовых сценариев для бандита по позициям."""
    print(f"Сохранение результатов в директорию: {OUTPUT_DIR}")
    
    # Определяем список кандидатов (например, первые 30 вопросов)
    candidate_questions = list(range(1, 31))
    
    # Тест-кейс 1: Определенные вопросы лучше на определенных позициях
    run_position_test_case("position_dependent", position_dependent_rule, 
                          num_sessions=3000, candidate_questions=candidate_questions, 
                          test_size=10)  # Используем меньший размер для ускорения тестов
    
    # Тест-кейс 2: Конверсия зависит от последовательности вопросов
    run_position_test_case("question_sequence", question_sequence_rule, 
                          num_sessions=3000, candidate_questions=candidate_questions, 
                          test_size=10)
    
    # Тест-кейс 3: Разные группы вопросов лучше в разных частях теста
    run_position_test_case("question_group", question_group_rule, 
                          num_sessions=3000, candidate_questions=candidate_questions, 
                          test_size=10)
    
    # Тест-кейс 4: Для каждой позиции есть только один идеальный вопрос (N-й вопрос для N-й позиции)
    run_position_test_case("exact_position_match", exact_position_match_rule, 
                          num_sessions=3000, candidate_questions=candidate_questions, 
                          test_size=10)

if __name__ == "__main__":
    run_position_bandit_test_cases()  # Запуск тестов для позиционного бандита 