import random


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
    if not text:
        return []
    tokens = [w for w in re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', text.lower()) if len(w) > 4]
    filtered = [w for w in tokens if w not in STOP_WORDS]
    unique_words = list(dict.fromkeys(filtered))
    return unique_words[:top_n]

def generate(text_user):
    temperature = 0
    text = text_user.lower() + ' бот:'
    good_words = extract_keywords(text)
    dominate_data_block=0
    for i in range(15):
        scan_words = []
        for data in database:
            try:
                data = " ".join(data.split()[data.split().index(text.split()[-1]):])
                scan_words.append(data.split()[1])
            except Exception as e:
                continue
        word_win = ["", 0]
        last_winers=["","",""]
        scan_words=words
        for word in scan_words:
            for data in database:
                try:
                    data = " ".join(data.split()[:data.split().index(word)])
                    res = list(search(text, data,text))
                    res[0] = list(res[0])
                    res[0][1]+=random.randint(-temperature,temperature)
                    if word in good_words:
                        res[0][1] *= 1
                    if data == dominate_data_block:
                        res[0][1]/=1
                    dominate_data_block=data
                    if word_win[1] < res[0][1]:
                        word_win = [word, res[0][1]]
                        last_winers.insert(0,word_win[0])
                        last_winers.pop()
                except ValueError as e:
                    continue

        text += f" {word_win[0]}"
        # print(text)
        if "пользователь:" in word_win[0]: return text
        print(word_win[0], end=" ")# + str(last_winers)
    print()
    return text




if __name__ == "__main__":
    with open("input.txt", "r",encoding="utf-8") as f:
        readed=f.read().replace("<"," <").replace(">","> ").replace('"',"").replace(":",": ")[0:].lower()

        #database = list(set(['\n'.join(readed.split("\n")[i:i + 2]) for i in range(0, len(readed.split("\n")), 2)]))
        database =list(set(readed.lower().split("\n---\n")))

        database += ["\n".join(i.split("\n")[::-1])for i in database]
        words = sorted(list(set(readed.lower().split()))+["\n"])
    print(database)
    print(words)

    while True:
        text = ""
        text += generate(text+input("\n>> "))

