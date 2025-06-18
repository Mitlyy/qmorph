from typing import List, Tuple

import pymorphy2


class Lemmatizer:
    """Класс для лемматизации и морфологического анализа токенов."""

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def lemmatize_token(self, token: str) -> Tuple[str, str]:
        """
        Лемматизация одного токена с получением леммы и морфологического тега.
        :param token: токен для анализа
        :return: кортеж (лемма, морфологический_тег)
        """
        parses = self.morph.parse(token)
        best = parses[0]
        lemma = best.normalized.word
        morph_tag = str(best.tag)
        return lemma, morph_tag

    def lemmatize(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Лемматизация списка токенов.
        :param tokens: список токенов
        :return: список кортежей (лемма, морфологический_тег)
        """
        return [self.lemmatize_token(token) for token in tokens]
