import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os

# Stil für die Plots setzen
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Deutsche Kommentare für den gesamten Code
"""
Diese Datei implementiert eine Zeitreihenanalyse für Aktiensendiment-Daten.
Sie liest die Sentimentergebnisse aus der JSON-Datei und analysiert das Sentiment im Zeitverlauf.
"""

def load_sentiment_data(file_path="sentiment_analysis_results.json"):
    """
    Lädt die Sentiment-Daten aus der JSON-Datei
    
    Args:
        file_path: Pfad zur JSON-Datei mit den Sentiment-Daten
        
    Returns:
        DataFrame mit den geladenen Daten
    """
    # Lade die Sentiment-Daten
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Konvertiere zu DataFrame
    df = pd.DataFrame(data)
    
    # Konvertiere das Datumsformat
    df['date'] = pd.to_datetime(df['date'])
    
    # Füge Monat und Jahr als separate Spalten hinzu
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['yearmonth'] = df['date'].dt.strftime('%Y-%m')
    
    return df

def calculate_monthly_sentiment(df):
    """
    Berechnet die monatliche Sentiment-Verteilung für jede Aktie
    
    Args:
        df: DataFrame mit Sentiment-Daten
        
    Returns:
        DataFrame mit monatlichen Sentiment-Werten
    """
    # Gruppiere nach Aktie, Jahr, Monat und Sentiment
    monthly_sentiment = df.groupby(['stock', 'yearmonth', 'sentiment']).size().reset_index(name='count')
    
    # Pivotiere die Daten für einfachere Verwendung
    pivot_df = monthly_sentiment.pivot_table(
        index=['stock', 'yearmonth'], 
        columns='sentiment', 
        values='count',
        fill_value=0
    ).reset_index()
    
    # Berechne Sentiment-Score: (positive - negative) / total
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in pivot_df.columns:
            pivot_df[sentiment] = 0
    
    pivot_df['total'] = pivot_df['positive'] + pivot_df['negative'] + pivot_df['neutral']
    pivot_df['sentiment_score'] = (pivot_df['positive'] - pivot_df['negative']) / pivot_df['total']
    
    # Sortiere nach Datum
    pivot_df['yearmonth_dt'] = pd.to_datetime(pivot_df['yearmonth'])
    pivot_df = pivot_df.sort_values(['stock', 'yearmonth_dt'])
    
    return pivot_df

def plot_sentiment_time_series(monthly_data, output_dir="sentiment_plots"):
    """
    Erstellt Zeitreihen-Plots für jede Aktie
    
    Args:
        monthly_data: DataFrame mit monatlichen Sentiment-Daten
        output_dir: Verzeichnis für die Ausgabe der Plots
    """
    # Erstelle Ausgabeverzeichnis, falls es nicht existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Finde alle Aktien
    stocks = monthly_data['stock'].unique()
    
    # Erstelle einen Gesamtplot mit allen Aktien
    plt.figure(figsize=(12, 8))
    plt.title('Sentiment-Score im Zeitverlauf für alle Aktien', fontsize=16)
    
    # Colormap für die verschiedenen Aktien
    cmap = plt.cm.get_cmap('tab10', len(stocks))
    
    for i, stock in enumerate(stocks):
        # Filtere Daten für diese Aktie
        stock_data = monthly_data[monthly_data['stock'] == stock]
        
        # Plotte die Daten
        plt.plot(stock_data['yearmonth_dt'], stock_data['sentiment_score'], 
                marker='o', linewidth=2, markersize=8, 
                label=stock, color=cmap(i))
    
    plt.ylabel('Sentiment-Score\n(positiv - negativ) / total')
    plt.xlabel('Monat')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Speichere den Gesamtplot
    plt.savefig(f"{output_dir}/all_stocks_sentiment.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gesamtplot für alle Aktien erstellt und gespeichert.")
    
    for stock in stocks:
        # Filtere Daten für diese Aktie
        stock_data = monthly_data[monthly_data['stock'] == stock]
        
        if len(stock_data) < 1:
            print(f"Zu wenig Daten für {stock}, überspringe Plot.")
            continue
        
        # Erstelle einen Plot mit zwei Untergrafiken
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Sentiment-Analyse für {stock} im Zeitverlauf', fontsize=16)
        
        # Plot 1: Sentiment-Score über Zeit
        ax1.plot(stock_data['yearmonth_dt'], stock_data['sentiment_score'], 
                marker='o', linewidth=2, markersize=8)
        ax1.set_ylabel('Sentiment-Score\n(positiv - negativ) / total')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
        
        # Hintergrundfarbe basierend auf Sentiment
        for i, row in stock_data.iterrows():
            if row['sentiment_score'] > 0.2:
                ax1.axvspan(row['yearmonth_dt'] - pd.Timedelta(days=15), 
                           row['yearmonth_dt'] + pd.Timedelta(days=15), 
                           alpha=0.2, color='green')
            elif row['sentiment_score'] < -0.2:
                ax1.axvspan(row['yearmonth_dt'] - pd.Timedelta(days=15), 
                           row['yearmonth_dt'] + pd.Timedelta(days=15), 
                           alpha=0.2, color='red')
        
        # Plot 2: Gestapelter Balkenplot der Sentiment-Anzahl
        width = 20  # Breite der Balken in Tagen
        bottoms = np.zeros(len(stock_data))
        
        # Plotten in bestimmter Reihenfolge: negativ, neutral, positiv
        for sentiment, color in [('negative', 'red'), ('neutral', 'gray'), ('positive', 'green')]:
            ax2.bar(stock_data['yearmonth_dt'], stock_data[sentiment], bottom=bottoms,
                   width=pd.Timedelta(days=width), label=sentiment, color=color, alpha=0.7)
            bottoms += stock_data[sentiment]
        
        ax2.set_ylabel('Anzahl der Tweets')
        ax2.set_xlabel('Monat')
        ax2.legend(loc='upper left')
        
        # Format der x-Achse anpassen
        plt.xticks(rotation=45)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Speichere den Plot
        plt.savefig(f"{output_dir}/{stock}_sentiment_time_series.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot für {stock} erstellt und gespeichert.")

def export_monthly_data(monthly_data, output_file="monthly_sentiment_data.csv"):
    """
    Exportiert die monatlichen Sentiment-Daten in eine CSV-Datei
    
    Args:
        monthly_data: DataFrame mit monatlichen Sentiment-Daten
        output_file: Pfad zur Ausgabedatei
    """
    # Exportiere die Daten
    export_columns = ['stock', 'yearmonth', 'positive', 'neutral', 'negative', 
                      'total', 'sentiment_score']
    monthly_data[export_columns].to_csv(output_file, index=False)
    print(f"Monatliche Sentiment-Daten wurden in {output_file} gespeichert.")

def generate_demo_data():
    """
    Generiert Demo-Daten für die Zeitreihenanalyse, wenn zu wenig echte Daten vorhanden sind
    
    Returns:
        DataFrame mit Demo-Daten
    """
    print("Erstelle Demo-Daten für verschiedene Aktien und Monate...")
    
    # Liste der Aktien und ihrer Unternehmen
    stocks = [
        {"symbol": "TSLA", "company": "Tesla, Inc."},
        {"symbol": "AAPL", "company": "Apple Inc."},
        {"symbol": "GOOGL", "company": "Alphabet Inc."},
        {"symbol": "MSFT", "company": "Microsoft Corporation"}
    ]
    
    # Zeitraum festlegen (12 Monate)
    start_date = datetime(2022, 1, 1)
    
    results = []
    
    # Generiere Daten für jeden Monat und jede Aktie
    for stock in stocks:
        for month in range(12):
            current_date = start_date + pd.DateOffset(months=month)
            date_str = current_date.strftime("%Y-%m-%d %H:%M:%S+00:00")
            yearmonth = current_date.strftime("%Y-%m")
            
            # Generiere zufällige Sentiments für jeden Monat
            num_positive = np.random.randint(5, 50)
            num_neutral = np.random.randint(10, 100)
            num_negative = np.random.randint(5, 40)
            
            # Füge Einträge zu den Ergebnissen hinzu
            for i in range(num_positive):
                results.append({
                    "index": len(results),
                    "date": date_str,
                    "text": f"Positiver Tweet über {stock['symbol']} #{i}",
                    "stock": stock["symbol"],
                    "company": stock["company"],
                    "sentiment": "positive",
                    "confidence": np.random.uniform(0.7, 0.99)
                })
            
            for i in range(num_neutral):
                results.append({
                    "index": len(results),
                    "date": date_str,
                    "text": f"Neutraler Tweet über {stock['symbol']} #{i}",
                    "stock": stock["symbol"],
                    "company": stock["company"],
                    "sentiment": "neutral",
                    "confidence": np.random.uniform(0.7, 0.99)
                })
            
            for i in range(num_negative):
                results.append({
                    "index": len(results),
                    "date": date_str,
                    "text": f"Negativer Tweet über {stock['symbol']} #{i}",
                    "stock": stock["symbol"],
                    "company": stock["company"],
                    "sentiment": "negative",
                    "confidence": np.random.uniform(0.7, 0.99)
                })
    
    # Speichere die Demo-Daten in eine JSON-Datei
    with open("demo_sentiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Erstelle ein DataFrame
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['yearmonth'] = df['date'].dt.strftime('%Y-%m')
    
    return df

def main():
    """
    Hauptfunktion für die Zeitreihenanalyse der Sentiment-Daten
    """
    print("Starte Zeitreihenanalyse der Sentiment-Daten...")
    
    try:
        # Versuche, die Sentiment-Daten aus der JSON-Datei zu laden
        sentiment_df = load_sentiment_data()
        print(f"Anzahl der geladenen Datensätze: {len(sentiment_df)}")
        
        # Prüfe, ob genug Daten für mehrere Monate vorhanden sind
        unique_months = sentiment_df['yearmonth'].nunique()
        unique_stocks = sentiment_df['stock'].nunique()
        
        if unique_months <= 1 or unique_stocks < 1:
            print("Zu wenig Daten für eine sinnvolle Zeitreihenanalyse. Generiere Demo-Daten...")
            sentiment_df = generate_demo_data()
    except (FileNotFoundError, json.JSONDecodeError):
        print("Sentiment-Datei nicht gefunden oder fehlerhaft. Generiere Demo-Daten...")
        sentiment_df = generate_demo_data()
    
    # Berechne monatliches Sentiment
    monthly_sentiment = calculate_monthly_sentiment(sentiment_df)
    
    # Exportiere die monatlichen Daten
    export_monthly_data(monthly_sentiment)
    
    # Erstelle Zeitreihen-Plots
    plot_sentiment_time_series(monthly_sentiment)
    
    print("Zeitreihenanalyse abgeschlossen!")

if __name__ == "__main__":
    main() 