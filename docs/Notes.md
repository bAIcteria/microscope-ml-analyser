# Notes

## Table of Contents
- [Introduction](#introduction)
- [Done work](#done-work)
- [Currently working on](#currently-working-on)
- [Future works](#future-works)
- [Data](#data)
- [UI must-have's](#ui-must-haves)
- [02.09.2025 team meeting 17:30-20:30](#02092025-team-meeting-1730-2030)
- [Literature](#literature)

---

## Introduction
Celem projektu jest stworzenie narzędzia będącego ekwiwalentem programu służącego do wykrywania bakterii na zdjęciu, ich charakteryzacji (w szczegolności parametrów takich jak długość, szerokość oraz pole powierzchni) oraz klasyfikacji wraz z generacją raportu w excelu.

## Done work
 - Tworzenie repozytorium na githubie
 - stworzenie narzędzie do konwersji raportów w formacie .csr na pliki .csv (jeden raport dzielony 10 10 plików csv - o nazwach powiązanych z id)
 - stworzenie narzedzia do nakładania na zdjęcia elips w miejscach, w których wykryte zostały bakterie
 - Ustrukturyzowanie danych - powiązanie zdjęć z dotyczącymi ich raportami (początkowo numery zdjęć w folderach oraz raportach nie zgadzały się)
 - konwersja raportów (informacje o położeniu obiektów) na pliki labelowe w formacie yolo
 - stworzenie setu treningowego oraz testowego
 - wytrenowanie modelu na Google Codelab
 - Walidacja modelu (82% dokładności dla zbioru od Moniki, Ponad 90% po dołożeniu zbioru Agaty)
 - Stworzenie algorytmu do wyciągania ze zdjęć wykrytych fragmentów, w których znajdują sie bakterie
 - Stworzenie algorytmu do analizy zdjęć bakterii oraz zwracania ich w formie raportu w formacie .csv. (błąd kilkudziesięcioprocentowy)
 - Odpalenie makr w nowym excelu
 - Manualne testy poprawności wygenerowanych raportów
 - Analiza działania makr, wydobycie z nich algorytmów
 - Sprawdzenie, które kolumny są potrzebne do wygenerowania ostatecznego raportu (długość, szerokość, pole powierzchnii bakterii)
 - Ulepszenie algorytmu do analizy zdjęć bakterii (błąd kilku/kilkunastoprocentowy).
 - przygotowanie komponentów umozliwiających generacje pełnego raportu bez uzycia makr w excelu.

## Currently working on
 - Uzupełnienie dokumentacji
 - Rozkodowanie magic numbers

## Future works
 - Porzadna dokumencja
 - Testy
 - Generacja ostatecznego raportu w excelu
 - Stworzenie interfejsu

## Data
 - Niektórym obrazom brakuje opisów (raportów)
 - Na niektórych obrazach widoczne jest dużo fałszywych wykryć
 - Stary algorytm nie wykrywa wszystkich bakterii
 - Stary algorytm czasami traktuje skupiska bakterii jako jeden obiekt
 - Dane są pomieszane przez co wymaga to od nas ręcznego filtrowania danych.

## UI must-have's
- analiza pojedynczego zdjęcia
- analiza wielu zdjęć (bulk)
- konwersja raportu .csr na pełny raport
- generowanie raportu na podstawie oznaczonych bakterii
- możliwość oznaczania i odznaczania (opcjonalnie sprecyzowanie typu) baterii na obrazie wyjściowym
- możliwość zmiany pewności modelu (0-100%) w postaci suwaka
- możliwość ustawienia parametrów dotyczących przeskalowania rozmiaru zdjęcia na mm lub um
- przedziały wielkościowe bakterii jako parametr do ustawienia

## 02.09.2025 team meeting 17:30-20:30
 - ustalone zostały początkowe założenia UI ([patrz sekcja UI must-have's](#ui-must-haves))
 - zostało wystosowane zapytanie o dostęp do superkomputera "Kraken" do trenowania modelu
    - wniosek ma zostać rozpatrzony w kolejnym tygodniu (8-12.09)
 - zdefiniowano cele i kryteria akceptacji narzędzia
    - narzędzie nie powinno wykrywać więcej (lub mniej) bakterii niż liczba wykrywana przez stary program +- 20/30%
 - zaprezentowano prototyp narzędzia, co spotkało się z entuzjazmem oraz pozytywnym odbiorem
    - na obeznym etapie prototyp wykrywa znacznie dokładniej bakterie lecz wymaga jeszcze trochę pracy, aktualne wyniki zostały ocenione jako obiecujące
 - omówiono proces wykonywania badań, zdjęć oraz wpływu poszczególnych czynników na obraz np. kontrast
 - zdefiniowano ograniczenia dotyczące liczenia bakterii (stare i młode bakterie)
    - stare bakterie, jak i również te, których dna "ucieka" nie powinny być wliczane. Natomiast bakterie młode powinny zostać zaklasyfikowane. Wici należy wykluczyć.
 - zweryfikowano wyniki prototypu
    - obecny model klasyfikuje gromady bakterii jako jedna, niektórych w ogóle nie wykrywa. Natomiast nie znaleziono tzw False Positive'ów.
 - zarekomendowano i poproszono o kolejny zestaw danych
    - następny zestaw zawierać będzie 500 próbek o dużej różnorodności. Każda z nich będzie zweryfikowana i zawierać będzie oznaczenia precyzyjnie wskazujące położenie bakterii na zdjęciach.
 - zaplanowano kolejne spotkanie, na którym skalibrowany zostanie aparat względem miarki położonej pod mikroskopem. Ponadto zostaniemy przeprowadzeni na żywo przez cały proces badania jednej próbki. Od pobrania materiału do zrobienia zdjęcia i jego analizy w starym programie.

## 04.09.2025 team meeting 20:30 - 22:30
 - zbadano pochodzenie wzorów przepisanych ze starego makra
 - rozwinięto założenia UI
 - należy stosować treshold adaptacyjny dla każdej wykrytej bakterii
 - sformułowano kolejne pytania, które zostaną skierowane do zespołu mikrobiologów


## 08.09.2025 - czyszczenie folderów
 - metoda słożąca na trenowaniu modelu do przewidywania powiarzchni bakterii nie działa (oparta na sieciach CNN)

## Literature
- https://www.mvi-inc.com/wp-content/uploads/80i-Instruction-Manual.pdf
- https://www.nikonusa.com/fileuploads/pdfs/Digital%20Sight%20Series%20062005.pdf?srsltid=AfmBOopwVtTNL-nvItahE82CrKjrcwGQqZday2kDNpGfKavrtE8yu7Dq
- http://insilico.ehu.eus/counting_chamber/thoma.php
- https://chemometec.com/how-to-count-cells-with-a-hemocytometer/
- https://www.researchgate.net/publication/318318415_Relationship_of_Epilithic_Diatom_Communities_to_Environmental_Variables_in_Yedikir_Dam_Lake_Amasya_Turkey/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ
- https://entnemdept.ufl.edu/hodges/protectus/lp_webfolder/9_12_grade/student_handout_1a.pdf
- https://pdfs.semanticscholar.org/7d10/bfb93cf6b7dbce5e00f36ea85b0026e6aa76.pdf