# 🎶 Система рекомендаций на основе LightGBM учитывая музыкальные предпочтения и взаимодействия пользователя 🎧

Этот проект реализует систему рекомендаций товаров (в данном случае музыкальных товаров) на основе данных о пользователях, песнях и их оценках. Для реализации используется алгоритм **LightGBM** с целью оптимизации релевантности рекомендаций.

## 🔍 Описание проекта

Проект решает задачу рекомендаций для пользователей, предлагая товары, которые могут их заинтересовать, на основе их предыдущих оценок и взаимодействий с песнями и товарами. 

Алгоритм использует модель **LambdaRank** для оптимизации позиций в списках рекомендаций, что идеально подходит для задач с ранжированием элементов.

### Основные этапы работы:

1. **Подготовка данных**:
   - Приведение данных к нужным типам.
   - Вычисление временных признаков (разница во времени между событиями).
  
2. **Вычисление статистических характеристик**:
   - Средние оценки товаров и песен.
   - Дополнительные статистики, такие как стандартное отклонение и минимум/максимум.

3. **Обучение модели**:
   - Построение и обучение модели с использованием **LightGBM**.
  
4. **Генерация рекомендаций**:
   - Система формирует 10 наиболее релевантных товаров для каждого пользователя на основе предсказаний модели.

5. **Экспорт рекомендаций**:
   - Рекомендации сохраняются в формате **Parquet**.

## 🚀 Как запустить

### 1. Клонируйте репозиторий:

```bash
git clone https://github.com/ваш_пользователь/рекомендации_музыка.git
cd рекомендации_музыка
```

### 2. Установите зависимости:

Для установки всех необходимых зависимостей используйте файл `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Запустите основной скрипт:

```bash
python main.py
```

После выполнения скрипта, результаты будут сохранены в формате **Parquet** в папке `/output`.

## 📝 Структура проекта

```
/рекомендации_музыка
│
├── main.py            # Основной скрипт для подготовки данных, обучения и генерации рекомендаций
├── functions.py       # Вспомогательные функции
├── requirements.txt   # Список зависимостей
└── output/            # Папка для сохранения результатов
```

### Файл `main.py`:
Этот файл содержит основную логику работы системы: подготовку данных, обучение модели и создание рекомендаций.

### Файл `functions.py`:
Содержит функции для предобработки данных, расчета статистик и других вспомогательных операций.

### Файл `requirements.txt`:
Содержит список зависимостей, которые необходимы для работы проекта.

### Папка `output/`:
Сюда сохраняются результаты работы системы, включая файл с рекомендациями для пользователей.

## ⚙️ Зависимости

Для корректной работы проекта потребуются следующие библиотеки:

- `pandas` — для обработки данных.
- `numpy` — для математических операций.
- `lightgbm` — для обучения модели.
- `pyarrow` — для сохранения результатов в формате Parquet.

Вы можете установить все зависимости с помощью pip:

```bash
pip install -r requirements.txt
```

## 💡 Описание алгоритма

Проект использует модель **LightGBM** с параметрами для оптимизации задач ранжирования (метрика NDCG), что позволяет эффективно генерировать релевантные рекомендации для пользователей.

## 📄 Результаты

Рекомендации для каждого пользователя будут сохранены в файл в формате **Parquet** с колонками:

- **index** — индекс рекомендации.
- **user_id** — ID пользователя.
- **item_ids** — 10 рекомендованных товаров.

## 📚 Лицензия

Проект выпущен под лицензией [MIT](LICENSE).
