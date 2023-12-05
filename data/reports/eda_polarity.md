# Porównanie zbioru polarity do poprzedniego

Zbiór ten jest dużo mnie złożony. Przekazany został w formacie tsv, co ciekawe zakodowany z użyciem "Windows-1252". 
Składa się z około 10 tysięcy rekordów, które zawierają zdania ocenione przez użytkowników. Klasy są idealnie równoliczne,
zagadnienie jest binarne. Ponadto dane są wyczyszczone, zostały sprowadzone do lowercase. Po wczytaniu przechowywane są 
w dwóch kolumnach - reviewText oraz overall. Można zauważyć, że są dużo mniej złożone i skomplikowane w porównaniu do 
zbioru Amazon. Tam problematyczny okazał się atrybut stylu, z którym należało sobie poradzić, a także sprostać problemowi 
brakujących wartości. Jest ich także znacząco mniej, prawie dwukrotnie.