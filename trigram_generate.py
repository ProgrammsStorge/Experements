def search(query: str, data: str,text) -> list[tuple[str, int]]:
    gram_count=5
    if not query or len(query) < gram_count:
        q_trigrams = {query} if query else set()
    else:
        q_trigrams = set(query[i:i + gram_count] for i in range(len(query) - 2))
    results = []

    if len(data) < gram_count:
        t_trigrams = {data} if data else set()
    else:
        t_trigrams = set(data[i:i + gram_count] for i in range(len(data) - 2))
    score = len(q_trigrams & t_trigrams)
    results.append((text, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_trigrams(s: str) -> set:
    if len(s) < 3:
        return {s} if s else set()
    return set(s[i:i + 3] for i in range(len(s) - 2))


import re

# Расширенный стоп-лист (существительные-паразиты, местаимения, общие слова)
STOP_WORDS = frozenset([
    'это', 'так', 'как', 'что', 'все', 'она', 'его', 'но', 'да', 'же', 'бы', 'по', 'уже', 'или', 'ни',
    'быть', 'был', 'него', 'до', 'вас', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
    'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
    'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'тогда', 'кто', 'этот',
    'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
    'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
    'наконец', 'два', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас',
    'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо',
    'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более',
    'всегда', 'конечно', 'всю', 'между', 'просто', 'типа', 'короче', 'ну', 'ага', 'угу', 'нет',
    'день', 'год', 'час', 'минута', 'вещь', 'место', 'сторона', 'вид', 'случай', 'жизнь', 'рука',
    'работа', 'слово', 'лицо', 'друг', 'глаз', 'вопрос', 'дом', 'голос', 'страна', 'мир', 'конец',
    'head', 'get', 'make', 'know', 'think', 'take', 'come', 'want', 'look', 'use', 'find', 'give',
    'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep',
    'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start', 'show', 'hear', 'play', 'run', 'move',
    'like', 'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
    'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch',
    'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk',
    'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send',
    'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain'
])


def extract_keywords(text, top_n=5):
    """
    Вычленяет ключевые слова (существительные/объекты) из текста.
    Фильтрует мусор, stopwords и короткие слова.
    """
    if not text:
        return []

    # 1. Токенизация: только буквы (рус/англ), длина > 4
    tokens = [w for w in re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', text.lower()) if len(w) > 4]

    # 2. Фильтрация стоп-слов
    filtered = [w for w in tokens if w not in STOP_WORDS]

    # 3. Убираем дубликаты, сохраняя порядок (или просто берем уникальные)
    unique_words = list(dict.fromkeys(filtered))

    # 4. Возвращаем топ-n (или все, если их меньше)
    return unique_words[:top_n]

def generate(text_user):
    text = text_user.lower() + ""
    good_words = extract_keywords(text)
    #print(good_words)
    #print(text, end=" ")
    for i in range(10):
        scan_words = []
        for data in database:
            try:
                data = " ".join(data.split()[data.split().index(text.split()[-1]):])
                scan_words.append(data.split()[1])
            except Exception as e:
                # print(e)
                continue
        word_win = ["", 0]
        scan_words = words
        # print(scan_words)
        for word in scan_words:
            for data in database:
                try:
                    data = " ".join(data.split()[:data.split().index(word)])
                    # print(data)
                    res = list(search(text, data,text))
                    res[0] = list(res[0])
                    # print(res)
                    if word in good_words:
                        res[0][1] += 8
                    if word_win[1] < res[0][1]:
                        word_win = [word, res[0][1]]
                        # print(word,res[0][1])
                except ValueError as e:
                    # print(e)
                    continue
        text += f" {word_win[0]}"
        # print(text)
        print(word_win[0], end=" ")
    print()



if __name__ == "__main__":
    with open("input.txt", "r",encoding="utf-8") as f:
        readed=re.sub(r'([^\w\s])', r' \1 ', f.read()).replace("<"," <").replace(">","> ")
        database =readed.lower().splitlines()
        words = sorted(list(set(readed.lower().split()))+[])

    print(words)
    # database = [
    #     "Привет мир",
    #     "Как дела",
    #     "Приветствие",
    #     "Мироздание",
    #     "Случайный текст"
    # ]
    while True: generate(input(">> "))


