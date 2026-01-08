import numpy
from collections import Counter
import os

def dystans_euklidesowy(punkt1, punkt2):
    wynik = numpy.sqrt(numpy.sum((punkt1 - punkt2) ** 2))
    return wynik

sciezka = os.path.join(os.path.dirname(__file__), "iris.csv")
class KlasyfikatorKNN():
    def __init__(self, k):
        self.k = k
    def dopasuj(self, X_trening, y_trening):
        self.X_trening = X_trening
        self.y_trening = y_trening
    def przewiduj_zbior(self, X_test):
        przewidywania = [self._przewiduj(punkt) for punkt in X_test.to_numpy()]
        return przewidywania
    def _przewiduj(self, punkt):
        odleglosci = [dystans_euklidesowy(punkt, punkt_treningowy) for punkt_treningowy in self.X_trening]
        indeksy_najblizszych = numpy.argsort(odleglosci)[:self.k]
        najblizsze_etykiety = [self.y_trening[i] for i in indeksy_najblizszych]
        najczestsza_etykieta, liczba_glosow = Counter(najblizsze_etykiety).most_common(1)[0]
        srednia_odleglosc = numpy.mean([odleglosci[i] for i in indeksy_najblizszych])/100
        return najczestsza_etykieta, srednia_odleglosc