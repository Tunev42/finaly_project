import random


def check_text(text):
    text_lower = text.lower()
    if 'земля круглая' in text_lower:
        return 'Истина. Научно доказано', ['nasa.gov', 'ras.ru']
    if '10% мозга' in text_lower:
        return 'Миф. Люди используют весь мозг', ['mednet.ru', 'journalofneuroscience.com']
    if 'солнце вращается вокруг земли' in text_lower:
        return 'Ложь. Это геоцентрическая модель', ['astronet.ru']
    if 'коричневый сахар полезнее белого' in text_lower:
        return 'Частично. Разница минимальна', ['roscontrol.com']
    r = random.randint(1, 100)
    if r < 40:
        return 'Скорее правда', ['tass.ru', 'ria.ru']
    if r < 80:
        return 'Скорее ложь', ['factcheck.ru']
    return 'Недостаточно данных', []