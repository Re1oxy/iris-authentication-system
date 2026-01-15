Iris-Based Biometric Authentication System
1. Opis projektu

Celem projektu było zaprojektowanie i implementacja systemu biometrycznego do uwierzytelniania użytkownika na podstawie cech oka (iris). System został zaimplementowany w języku Python z wykorzystaniem metod uczenia maszynowego oraz zasad programowania obiektowego (OOP).

Projekt realizowany był z myślą o realnym zastosowaniu na platformie PC oraz spełnieniu wymagań bezpieczeństwa, w tym odporności na ataki typu spoofing.

2. Wykorzystane technologie

Python 3.x

Pandas, NumPy – analiza i przetwarzanie danych

Scikit-learn – modele uczenia maszynowego

Matplotlib / Seaborn – analiza danych (EDA)

Git + GitHub – system kontroli wersji

Jupyter Notebook – eksploracyjna analiza danych

3. Analiza danych (EDA)

Przeprowadzono wstępną analizę danych obejmującą:

analizę statystyczną cech (średnia, odchylenie standardowe),

sprawdzenie rozkładów danych,

analizę korelacji pomiędzy cechami,

identyfikację potencjalnych anomalii.

Dane zostały podzielone na zbiór treningowy oraz testowy na podstawie dostarczonych plików CSV.

4. Przetwarzanie danych

W ramach przetwarzania danych wykonano:

normalizację cech numerycznych (StandardScaler),

wektoryzację danych wejściowych,

konwersję etykiet użytkowników do postaci klas dyskretnych.

Zapewniło to poprawne działanie algorytmów klasyfikacji.

5. Trenowanie modelu

Do klasyfikacji zastosowano kilka algorytmów:

K-Nearest Neighbors (KNN),

Support Vector Machine (SVM),

Random Forest.

Przeprowadzono strojenie hiperparametrów (fine-tuning), a ostatecznie wybrano model SVM z estymacją prawdopodobieństwa jako najlepiej spełniający założenia projektu.

6. Mechanizmy anty-spoofingowe

W celu zwiększenia bezpieczeństwa systemu zaimplementowano mechanizm wykrywania anomalii (Isolation Forest), który pozwala odrzucać nietypowe próbki mogące pochodzić z ataków typu spoofing.

Dodatkowo zastosowano próg ufności (confidence threshold), który decyduje o przyznaniu dostępu.

7. Ocena wyników

Model osiągnął następujące wyniki:

Accuracy: 95.98%

Niski współczynnik False Acceptance Rate (FAR)

Akceptowalny poziom False Rejection Rate (FRR)

Macierz pomyłek (confusion matrix) potwierdziła poprawne działanie systemu w warunkach testowych.

8. Wnioski

Zaprojektowany system spełnia wymagania funkcjonalne oraz jakościowe. Osiągnięte wyniki wskazują, że model może być wykorzystany jako podstawa rzeczywistego systemu biometrycznego. Projekt potwierdza skuteczność metod uczenia maszynowego w zadaniach uwierzytelniania biometrycznego.

9. Kontrola wersji

Cały proces tworzenia projektu był wersjonowany przy użyciu systemu Git. Repozytorium GitHub zawiera historię commitów dokumentującą kolejne etapy rozwoju aplikacji.
