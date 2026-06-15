"""
Hotel Booking Analytics System — Flask Backend
Covers: File Handling, NumPy, Pandas, ML (Linear Regression, clustering), Report Generation
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'hotel_bookings.csv')
REPORT_PATH  = os.path.join(BASE_DIR, 'hotel_report.txt')
REPORT_PATH  = os.path.join(os.path.dirname(__file__), 'hotel_report.txt')


# ──────────────────────────────────────────────
# Helper: load and preprocess data
# ──────────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("hotel_bookings.csv not found. Check the dataset path.")

    # ── Part 2 / Part 3 string ops ──────────────
    df['guest_name'] = df['guest_name'].str.strip()

    # ── Part 6 new columns ──────────────────────
    df['revenue']          = df['nights'] * df['price_per_night']
    df['discounted_price'] = df['price_per_night'] * 0.90   # 10 % discount

    # ── Intentional missing-value demo ──────────
    df.loc[df.sample(frac=0.05, random_state=42).index, 'rating'] = np.nan
    df['rating'] = df['rating'].fillna(df['rating'].mean())

    return df


# ──────────────────────────────────────────────
# Route 1 — All bookings  (Part 2, menu item 1)
# ──────────────────────────────────────────────
@app.route('/api/bookings', methods=['GET'])
def get_all_bookings():
    df = load_data()
    return jsonify(df.to_dict(orient='records'))


# ──────────────────────────────────────────────
# Route 2 — Search by guest name  (Part 3)
# ──────────────────────────────────────────────
@app.route('/api/search', methods=['GET'])
def search_booking():
    df = load_data()
    query = request.args.get('name', '').lower().strip()
    if not query:
        return jsonify({'error': 'Provide ?name=<guest>'}), 400

    # lower(), strip(), replace() equivalents on the column
    mask    = df['guest_name'].str.lower().str.replace(' ', '').str.contains(query.replace(' ', ''))
    results = df[mask]
    return jsonify(results.to_dict(orient='records'))


# ──────────────────────────────────────────────
# Route 3 — City report  (Part 4 + Part 6 GroupBy)
# ──────────────────────────────────────────────
@app.route('/api/city-report', methods=['GET'])
def city_report():
    df = load_data()

    city_stats = df.groupby('city').agg(
        total_bookings=('booking_id', 'count'),
        total_revenue=('revenue', 'sum'),
        avg_rating=('rating', 'mean'),
        avg_nights=('nights', 'mean')
    ).reset_index().round(2)

    # Part 4 — dict of bookings per city
    bookings_per_city = df['city'].value_counts().to_dict()

    # Most profitable city
    most_profitable = city_stats.loc[city_stats['total_revenue'].idxmax(), 'city']

    return jsonify({
        'city_stats':       city_stats.to_dict(orient='records'),
        'bookings_per_city': bookings_per_city,
        'most_profitable_city': most_profitable
    })


# ──────────────────────────────────────────────
# Route 4 — Revenue report  (Part 6 + Part 7)
# ──────────────────────────────────────────────
@app.route('/api/revenue-report', methods=['GET'])
def revenue_report():
    df = load_data()

    # Part 6 GroupBy
    revenue_by_city      = df.groupby('city')['revenue'].sum().to_dict()
    revenue_by_room_type = df.groupby('room_type')['revenue'].sum().to_dict()
    avg_rating_by_city   = df.groupby('city')['rating'].mean().round(2).to_dict()

    # Part 7 — top 5 guests by total spend
    top_guests = df.groupby('guest_name')['revenue'].sum().nlargest(5).reset_index()
    top_guests.columns = ['guest_name', 'total_spend']

    # Part 7 — most popular room type
    popular_room = df['room_type'].value_counts().idxmax()

    # Part 7 — occupancy report
    occupancy = df['room_type'].value_counts().reset_index()
    occupancy.columns = ['room_type', 'bookings']

    return jsonify({
        'revenue_by_city':       revenue_by_city,
        'revenue_by_room_type':  revenue_by_room_type,
        'avg_rating_by_city':    avg_rating_by_city,
        'top_guests':            top_guests.to_dict(orient='records'),
        'most_popular_room_type': popular_room,
        'occupancy':             occupancy.to_dict(orient='records'),
        'total_revenue':         int(df['revenue'].sum()),
        'total_bookings':        int(len(df))
    })


# ──────────────────────────────────────────────
# Route 5 — NumPy stats  (Part 5)
# ──────────────────────────────────────────────
@app.route('/api/numpy-stats', methods=['GET'])
def numpy_stats():
    df = load_data()

    nights    = np.array(df['nights'])
    prices    = np.array(df['price_per_night'])
    ratings   = np.array(df['rating'])

    # Normalise prices
    normalized = (prices - prices.min()) / (prices.max() - prices.min())

    # Above-average price
    above_avg_count = int(np.sum(prices > prices.mean()))

    return jsonify({
        'avg_nights':           round(float(np.mean(nights)), 2),
        'highest_price':        int(np.max(prices)),
        'lowest_price':         int(np.min(prices)),
        'std_rating':           round(float(np.std(ratings)), 4),
        'avg_rating':           round(float(np.mean(ratings)), 2),
        'normalized_prices':    [round(float(v), 4) for v in normalized.tolist()],
        'above_avg_price_count': above_avg_count
    })


# ──────────────────────────────────────────────
# Route 6 — Pandas filter + sort  (Part 6)
# ──────────────────────────────────────────────
@app.route('/api/filter', methods=['GET'])
def filter_data():
    df   = load_data()
    mode = request.args.get('mode', 'suites')

    if mode == 'suites':
        result = df[df['room_type'] == 'Suite']
    elif mode == 'chennai':
        result = df[df['city'] == 'Chennai']
    elif mode == 'high_rating':
        result = df[df['rating'] > 4.5]
    elif mode == 'price_asc':
        result = df.sort_values('price_per_night')
    elif mode == 'price_desc':
        result = df.sort_values('price_per_night', ascending=False)
    elif mode == 'rating_desc':
        result = df.sort_values('rating', ascending=False)
    else:
        result = df

    return jsonify(result.head(20).to_dict(orient='records'))


# ──────────────────────────────────────────────
# Route 7 — ML: Revenue Prediction  (Linear Regression)
# ──────────────────────────────────────────────
@app.route('/api/ml/predict', methods=['POST'])
def predict_revenue():
    """
    Given nights + price_per_night, predict revenue using a trained LR model.
    Also returns model R² score.
    """
    df = load_data()

    X = df[['nights', 'price_per_night']].values
    y = df['revenue'].values

    model = LinearRegression()
    model.fit(X, y)

    data = request.get_json()
    nights_in      = float(data.get('nights', 3))
    price_in       = float(data.get('price_per_night', 2500))
    prediction     = model.predict([[nights_in, price_in]])[0]

    r2 = model.score(X, y)

    return jsonify({
        'predicted_revenue': round(float(prediction), 2),
        'r2_score':          round(float(r2), 4),
        'intercept':         round(float(model.intercept_), 2),
        'coefficients':      {
            'nights':            round(float(model.coef_[0]), 2),
            'price_per_night':   round(float(model.coef_[1]), 2)
        }
    })


# ──────────────────────────────────────────────
# Route 8 — ML: Guest Segmentation  (KMeans clustering)
# ──────────────────────────────────────────────
@app.route('/api/ml/clusters', methods=['GET'])
def guest_clusters():
    """
    Cluster guests into 3 segments based on total spending + avg rating.
    Returns each guest's segment label.
    """
    df = load_data()

    guest_profile = df.groupby('guest_name').agg(
        total_spend=('revenue', 'sum'),
        avg_rating=('rating', 'mean'),
        total_nights=('nights', 'sum')
    ).reset_index()

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(guest_profile[['total_spend', 'avg_rating', 'total_nights']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    guest_profile['segment'] = kmeans.fit_predict(X_scaled)

    # Label segments meaningfully
    segment_labels = {0: 'Budget Traveller', 1: 'Business Guest', 2: 'Premium Guest'}
    guest_profile['segment_label'] = guest_profile['segment'].map(segment_labels)

    # Segment summary
    segment_summary = guest_profile.groupby('segment_label').agg(
        count=('guest_name', 'count'),
        avg_spend=('total_spend', 'mean'),
        avg_rating=('avg_rating', 'mean')
    ).round(2).reset_index()

    return jsonify({
        'guests':          guest_profile[['guest_name', 'total_spend', 'avg_rating', 'segment_label']].to_dict(orient='records'),
        'segment_summary': segment_summary.to_dict(orient='records')
    })


# ──────────────────────────────────────────────
# Route 9 — Generate text report  (Part 8)
# ──────────────────────────────────────────────
@app.route('/api/report/generate', methods=['GET'])
def generate_report():
    df = load_data()

    top_city      = df.groupby('city')['revenue'].sum().idxmax()
    top_room      = df['room_type'].value_counts().idxmax()
    avg_rating    = round(float(df['rating'].mean()), 2)
    total_rev     = int(df['revenue'].sum())
    total_book    = len(df)

    top5 = df.groupby('guest_name')['revenue'].sum().nlargest(5)

    report = f"""
╔══════════════════════════════════════════════╗
║        HOTEL ANALYTICS REPORT               ║
║        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}          ║
╚══════════════════════════════════════════════╝

SUMMARY
───────────────────────────────────────────────
Total Bookings  : {total_book}
Total Revenue   : ₹{total_rev:,}
Top City        : {top_city}
Top Room Type   : {top_room}
Average Rating  : {avg_rating}

REVENUE BY CITY
───────────────────────────────────────────────
"""
    for city, rev in df.groupby('city')['revenue'].sum().sort_values(ascending=False).items():
        report += f"  {city:<15} ₹{int(rev):>10,}\n"

    report += "\nREVENUE BY ROOM TYPE\n───────────────────────────────────────────────\n"
    for rtype, rev in df.groupby('room_type')['revenue'].sum().sort_values(ascending=False).items():
        report += f"  {rtype:<15} ₹{int(rev):>10,}\n"

    report += "\nTOP 5 GUESTS BY SPENDING\n───────────────────────────────────────────────\n"
    for guest, spend in top5.items():
        report += f"  {guest:<20} ₹{int(spend):>8,}\n"

    report += "\nOCCUPANCY REPORT\n───────────────────────────────────────────────\n"
    for rtype, cnt in df['room_type'].value_counts().items():
        report += f"  {rtype:<15} {cnt} bookings\n"

    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    return jsonify({'report': report, 'saved_to': 'hotel_report.txt'})


# ──────────────────────────────────────────────
# Route 10 — Export filtered data as CSV  (Part 1)
# ──────────────────────────────────────────────
@app.route('/api/export', methods=['GET'])
def export_csv():
    df       = load_data()
    city     = request.args.get('city', '')
    room     = request.args.get('room_type', '')

    if city:
        df = df[df['city'].str.lower() == city.lower()]
    if room:
        df = df[df['room_type'].str.lower() == room.lower()]

    export_path = os.path.join(os.path.dirname(__file__), 'exported_bookings.csv')
    df.to_csv(export_path, index=False)
    return send_file(export_path, as_attachment=True, download_name='exported_bookings.csv')


# ──────────────────────────────────────────────
# Route 11 — Overview stats for dashboard cards
# ──────────────────────────────────────────────
@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    df = load_data()
    return jsonify({
        'total_bookings':   int(len(df)),
        'total_revenue':    int(df['revenue'].sum()),
        'avg_rating':       round(float(df['rating'].mean()), 2),
        'avg_nights':       round(float(df['nights'].mean()), 2),
        'top_city':         df.groupby('city')['revenue'].sum().idxmax(),
        'top_room':         df['room_type'].value_counts().idxmax(),
        'cities':           sorted(df['city'].unique().tolist()),
        'room_types':       sorted(df['room_type'].unique().tolist()),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
