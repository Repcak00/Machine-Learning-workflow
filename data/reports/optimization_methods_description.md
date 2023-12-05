# Opis metod optymalizacji oraz wnioski
## RandomSearch

Losowo przeszukuje przestrzeń hiperparametrów, wybierając losowo zestawy hiperparametrów z zadanego zakresu i sprawdzając ich skuteczność w celu wybrania najlepszego zestawu. RandomSearch jest szybszy niż GridSearch, ale może nie znaleźć najlepszych hiperparametrów, gdy przestrzeń hiperparametrów jest duża i trudna do przeszukania.

## GridSearch

Przeszukuje wszystkie możliwe kombinacje hiperparametrów z określonej przestrzeni, a następnie porównuje wyniki dla każdej kombinacji w celu wybrania najlepszego zestawu. GridSearch jest bardzo dokładny, ale może być bardzo czasochłonny, zwłaszcza dla dużych przestrzeni hiperparametrów. 
Liczba parametrów to wszystkie możliwe kombinacje, dlatego trwa bardzo długo, przez co nie jest najlepszym wyborem.

## HalvingGridSearch

Przeszukuje przestrzeń hiperparametrów za pomocą dzielenia i podbierania, gdzie w każdej iteracji przeszukuje mniejszą liczbę hiperparametrów. W każdej iteracji odrzuca słabe modele i skupia się na tym, które mają najlepszą wydajność. Ta metoda jest szybsza niż tradycyjny GridSearch, ale może nie znaleźć najlepszych hiperparametrów, gdy przestrzeń hiperparametrów jest duża i trudna do przeszukania.

## HalvingRandomSearch

Podobnie jak HalvingGridSearch, przeszukuje przestrzeń hiperparametrów za pomocą dzielenia i podbierania, ale wybiera losowe zestawy hiperparametrów do przeszukania w każdej iteracji. Ta metoda jest szybsza niż RandomSearch i GridSearch, ale może nie znaleźć najlepszych hiperparametrów, gdy przestrzeń hiperparametrów jest duża i trudna do przeszukania.

# Dodatkowe wnioski
Dobieranie hiperparametrów metodą GridSearch okazało się bardzo uciążliwe. Przez to, że tworzy on każdą możliwą kombinację z dostępnych parametrów, a następnie ją uczy z wykorzystanie Cross Validation cała procedura trwa bardzo długo. Dlatego też od razu odrzuciłem ją, jako potencjalny kandydat do wykorzystania późniejszego. Może się sprawdzić, gdy do wyboru mamy mniejszą liczbę parametrów, jednak należy zauważyć, że jest on bardzo dokładny. 
Metody z Random w nazwie są dużo szybsze niż Grid Search, ale dobierają modele losowo, co jest pewnym uproszczeniem. Lepszą z dwóch zaproponowanych jest według mnie Halving Random State, bo zmniejsza ona liczbę kandydatów. 
Wybraną metodą do doboru parametrów do SVM (poprzednie 4 były na RF) okazał się Halving Grid Search, który także zmniejsza sukcesywnie liczbę kandydatów, co usprawnia proces, zachowując jednak wysoką dokładność.

Wykres zależności czasu dostępny jest w pliku data/reports/hiperparameters_tuning_fig.png