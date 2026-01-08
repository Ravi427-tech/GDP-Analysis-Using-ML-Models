
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # For server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import io
import base64
from datetime import datetime
import json

app = Flask(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# GDP Data
gdp_data = pd.DataFrame({
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'GDP_Trillion_USD': [1.68, 1.82, 1.83, 1.86, 2.04, 2.10, 2.29, 2.65, 2.71, 2.87, 2.67, 3.18, 3.39, 3.73, 3.94],
    'Growth_Rate': [10.3, 6.6, 5.5, 6.4, 7.4, 8.0, 8.3, 6.8, 6.5, 3.9, -5.8, 9.1, 7.0, 7.8, 6.8],
    'Agriculture': [18.2, 18.0, 17.5, 17.8, 17.0, 16.5, 16.3, 15.4, 14.9, 14.6, 16.8, 16.2, 15.8, 15.4, 15.0],
    'Manufacturing': [15.3, 15.1, 14.9, 14.7, 15.2, 15.8, 15.5, 15.3, 15.0, 14.8, 13.2, 13.8, 14.2, 14.5, 14.8],
    'Services': [55.4, 55.8, 56.5, 56.8, 57.2, 57.8, 58.4, 59.2, 60.1, 60.5, 60.0, 60.3, 60.5, 60.8, 61.2],
})

# Train models on startup
X = gdp_data[['Year']].values
y = gdp_data['GDP_Trillion_USD'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)


def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/overview')
def get_overview():
    """Get overview statistics"""
    latest = gdp_data.iloc[-1]
    return jsonify({
        'current_gdp': float(latest['GDP_Trillion_USD']),
        'growth_rate': float(latest['Growth_Rate']),
        'agriculture': float(latest['Agriculture']),
        'manufacturing': float(latest['Manufacturing']),
        'services': float(latest['Services']),
        'avg_growth': float(gdp_data['Growth_Rate'].mean()),
        'highest_growth': float(gdp_data['Growth_Rate'].max()),
        'lowest_growth': float(gdp_data['Growth_Rate'].min())
    })


@app.route('/api/data')
def get_data():
    """Get all GDP data"""
    return jsonify(gdp_data.to_dict(orient='records'))


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict GDP for a given year"""
    data = request.get_json()
    year = int(data.get('year', 2025))

    X_future = np.array([[year]])

    lr_pred = float(lr_model.predict(X_future)[0])
    rf_pred = float(rf_model.predict(X_future)[0])
    gb_pred = float(gb_model.predict(X_future)[0])

    ensemble_pred = (lr_pred + rf_pred + gb_pred) / 3

    base_gdp = float(gdp_data.iloc[-1]['GDP_Trillion_USD'])
    growth = ((ensemble_pred - base_gdp) / base_gdp) * 100

    return jsonify({
        'year': year,
        'linear_regression': round(lr_pred, 2),
        'random_forest': round(rf_pred, 2),
        'gradient_boosting': round(gb_pred, 2),
        'ensemble': round(ensemble_pred, 2),
        'growth_from_2024': round(growth, 2)
    })


@app.route('/api/plot/gdp_trend')
def plot_gdp_trend():
    """Generate GDP trend plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gdp_data['Year'], gdp_data['GDP_Trillion_USD'],
            marker='o', linewidth=3, markersize=8, color='#3b82f6', label='GDP')
    ax.fill_between(gdp_data['Year'], gdp_data['GDP_Trillion_USD'], alpha=0.3, color='#3b82f6')
    ax.set_title('India GDP Trend (2010-2024)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('GDP (Trillion USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    img = create_plot_base64(fig)
    return jsonify({'image': img})


@app.route('/api/plot/growth_rate')
def plot_growth_rate():
    """Generate growth rate plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in gdp_data['Growth_Rate']]
    ax.bar(gdp_data['Year'], gdp_data['Growth_Rate'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Annual GDP Growth Rate (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Growth Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3)

    img = create_plot_base64(fig)
    return jsonify({'image': img})


@app.route('/api/plot/sectors')
def plot_sectors():
    """Generate sector trends plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gdp_data['Year'], gdp_data['Agriculture'], marker='o', linewidth=2, label='Agriculture', markersize=6)
    ax.plot(gdp_data['Year'], gdp_data['Manufacturing'], marker='s', linewidth=2, label='Manufacturing', markersize=6)
    ax.plot(gdp_data['Year'], gdp_data['Services'], marker='^', linewidth=2, label='Services', markersize=6)
    ax.set_title('Sectoral Contribution to GDP (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    img = create_plot_base64(fig)
    return jsonify({'image': img})


@app.route('/api/model_performance')
def get_model_performance():
    """Get model performance metrics"""
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)

    return jsonify({
        'linear_regression': {
            'r2_score': float(r2_score(y_test, lr_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, lr_pred)))
        },
        'random_forest': {
            'r2_score': float(r2_score(y_test, rf_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred)))
        },
        'gradient_boosting': {
            'r2_score': float(r2_score(y_test, gb_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, gb_pred)))
        }
    })


@app.route('/download/data')
def download_data():
    """Download GDP data as CSV"""
    csv_data = gdp_data.to_csv(index=False)
    buffer = io.BytesIO()
    buffer.write(csv_data.encode())
    buffer.seek(0)
    return send_file(buffer, mimetype='text/csv', as_attachment=True,
                     download_name=f'india_gdp_data_{datetime.now().strftime("%Y%m%d")}.csv')


if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)

    # For production, use:
    # gunicorn app:app --bind 0.0.0.0:5000