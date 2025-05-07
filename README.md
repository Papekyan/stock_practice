# Sentiment-Zeitreihenanalyse für Aktien

Dieses Projekt ermöglicht die Analyse des Sentiments (Stimmung) verschiedener Aktien im Zeitverlauf basierend auf Twitter-Daten.

## Überblick

Die Sentiment-Zeitreihenanalyse liest Sentiment-Daten zu Aktien und erstellt aussagekräftige Visualisierungen, die die Entwicklung des Sentiments über Monate hinweg darstellen. Sie zeigt sowohl den Sentiment-Score als auch die Verteilung positiver, neutraler und negativer Tweets.

## Funktionen

- **Zeitreihenanalyse**: Verfolgt das Sentiment von Aktien im Zeitverlauf (monatlich)
- **Sentiment-Score**: Berechnet einen Score basierend auf positiven und negativen Tweets
- **Visualisierung**: Erstellt detaillierte Diagramme für jede Aktie und ein Gesamtdiagramm
- **Demo-Daten**: Generiert realistische Demo-Daten, wenn keine ausreichenden Daten verfügbar sind

## Dateistruktur

- `time_series_sentiment.py`: Hauptskript für die Zeitreihenanalyse
- `sentiment_analysis_results.json`: Ergebnisse der Sentiment-Analyse im JSON-Format
- `monthly_sentiment_data.csv`: Monatliche Sentiment-Daten als CSV-Datei
- `sentiment_plots/`: Verzeichnis für die generierten Diagramme
  - `all_stocks_sentiment.png`: Überblick über alle Aktien
  - `[AKTIE]_sentiment_time_series.png`: Detaillierte Analyse pro Aktie

## Verwendung

1. Stellen Sie sicher, dass die erforderlichen Pakete installiert sind:
   ```
   pip install pandas matplotlib seaborn numpy
   ```

2. Führen Sie die Zeitreihenanalyse aus:
   ```
   python time_series_sentiment.py
   ```

3. Prüfen Sie die erzeugten Diagramme im Verzeichnis `sentiment_plots/`

## Interpretation der Diagramme

Die erzeugten Diagramme zeigen für jede Aktie:

- **Oberer Graph**: Sentiment-Score im Zeitverlauf
  - Positiver Score: Überwiegend positive Tweets
  - Negativer Score: Überwiegend negative Tweets
  - Nulllinie: Ausgeglichenes Verhältnis

- **Unterer Graph**: Anzahl der Tweets nach Sentiment-Kategorie
  - Grün: Positive Tweets
  - Grau: Neutrale Tweets
  - Rot: Negative Tweets

## Anpassungen

Sie können folgende Parameter in der Datei `time_series_sentiment.py` anpassen:

- Quelldatei für Sentiment-Daten (`file_path` in `load_sentiment_data()`)
- Ausgabeverzeichnis für Diagramme (`output_dir` in `plot_sentiment_time_series()`)
- Name der CSV-Ausgabedatei (`output_file` in `export_monthly_data()`)

## Beispielergebnis

Die Datei `monthly_sentiment_data.csv` enthält unter anderem folgende Informationen:
- `stock`: Aktien-Symbol
- `yearmonth`: Jahr und Monat im Format YYYY-MM
- `positive`, `neutral`, `negative`: Anzahl der Tweets je Kategorie
- `total`: Gesamtzahl der Tweets
- `sentiment_score`: Berechneter Sentiment-Score für diesen Monat 