"""
India GDP Analysis with Machine Learning
Complete Python implementation for PyCharm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class IndiaGDPAnalysis:
    def __init__(self):
        """Initialize the GDP Analysis with historical data"""
        self.gdp_data = pd.DataFrame({
            'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            'GDP_Trillion_USD': [1.68, 1.82, 1.83, 1.86, 2.04, 2.10, 2.29, 2.65, 2.71, 2.87, 2.67, 3.18, 3.39, 3.73,
                                 3.94],
            'Growth_Rate': [10.3, 6.6, 5.5, 6.4, 7.4, 8.0, 8.3, 6.8, 6.5, 3.9, -5.8, 9.1, 7.0, 7.8, 6.8],
            'Agriculture': [18.2, 18.0, 17.5, 17.8, 17.0, 16.5, 16.3, 15.4, 14.9, 14.6, 16.8, 16.2, 15.8, 15.4, 15.0],
            'Manufacturing': [15.3, 15.1, 14.9, 14.7, 15.2, 15.8, 15.5, 15.3, 15.0, 14.8, 13.2, 13.8, 14.2, 14.5, 14.8],
            'Services': [55.4, 55.8, 56.5, 56.8, 57.2, 57.8, 58.4, 59.2, 60.1, 60.5, 60.0, 60.3, 60.5, 60.8, 61.2]
        })
0        self.policy_data = pd.DataFrame({
            'Policy': ['GST Reform', 'Make in India', 'Digital India', 'Demonetization', 'COVID Relief', 'PLI Scheme'],
            'Year': [2017, 2014, 2015, 2016, 2020, 2021],
            'Impact': [1.2, 2.5, 1.8, -1.5, 3.2, 2.0]
        })

        self.challenges = pd.DataFrame({
            'Challenge': ['Unemployment Rate', 'Income Inequality (Gini)', 'Poverty Rate', 'Inflation Rate'],
            'Value': [7.8, 35.7, 21.2, 5.4]
        })

    def display_overview(self):
        """Display key metrics and overview"""
        print("\n" + "=" * 80)
        print("INDIA GDP ANALYSIS - OVERVIEW".center(80))
        print("=" * 80)

        latest = self.gdp_data.iloc[-1]
        print(f"\n📊 Current GDP (2024): ${latest['GDP_Trillion_USD']}T")
        print(f"📈 Growth Rate (2024): {latest['Growth_Rate']}%")
        print(f"🏭 Top Sector: Services ({latest['Services']}%)")
        print(f"👥 Unemployment Rate: 7.8%")

        print(f"\n📉 Historical Summary:")
        print(f"   • Highest Growth: {self.gdp_data['Growth_Rate'].max()}% (2010)")
        print(f"   • Lowest Growth: {self.gdp_data['Growth_Rate'].min()}% (2020 - COVID)")
        print(f"   • Average Growth (2010-2024): {self.gdp_data['Growth_Rate'].mean():.2f}%")
        print(
            f"   • GDP Increased from ${self.gdp_data['GDP_Trillion_USD'].iloc[0]}T to ${self.gdp_data['GDP_Trillion_USD'].iloc[-1]}T")

    def plot_gdp_trends(self):
        """Plot GDP and growth rate trends"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # GDP Trend
        axes[0].plot(self.gdp_data['Year'], self.gdp_data['GDP_Trillion_USD'],
                     marker='o', linewidth=2, markersize=8, color='#3b82f6')
        axes[0].fill_between(self.gdp_data['Year'], self.gdp_data['GDP_Trillion_USD'],
                             alpha=0.3, color='#3b82f6')
        axes[0].set_title('India GDP Trend (2010-2024)', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('GDP (Trillion USD)', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Growth Rate
        colors = ['green' if x > 0 else 'red' for x in self.gdp_data['Growth_Rate']]
        axes[1].bar(self.gdp_data['Year'], self.gdp_data['Growth_Rate'], color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Annual GDP Growth Rate (%)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Growth Rate (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gdp_trends.png', dpi=300, bbox_inches='tight')
        print("\n✅ GDP trends plot saved as 'gdp_trends.png'")
        plt.show()

    def analyze_sectors(self):
        """Analyze sectoral contributions"""
        print("\n" + "=" * 80)
        print("SECTORAL ANALYSIS".center(80))
        print("=" * 80)

        latest = self.gdp_data.iloc[-1]
        print(f"\n🌾 Agriculture Sector: {latest['Agriculture']}%")
        print(f"   • Declining trend (18.2% → 15.0%)")
        print(f"   • Still employs ~40% of workforce")
        print(f"   • Need for modernization and productivity")

        print(f"\n🏭 Manufacturing Sector: {latest['Manufacturing']}%")
        print(f"   • Moderate growth (15.3% → 14.8%)")
        print(f"   • Boosted by Make in India & PLI schemes")
        print(f"   • Focus: Electronics, automobiles, textiles")

        print(f"\n💼 Services Sector: {latest['Services']}%")
        print(f"   • Dominant sector (55.4% → 61.2%)")
        print(f"   • Includes IT, finance, telecom")
        print(f"   • Major contributor to GDP growth")

        # Plot sectoral trends
        plt.figure(figsize=(14, 7))
        plt.plot(self.gdp_data['Year'], self.gdp_data['Agriculture'],
                 marker='o', linewidth=2, label='Agriculture', markersize=6)
        plt.plot(self.gdp_data['Year'], self.gdp_data['Manufacturing'],
                 marker='s', linewidth=2, label='Manufacturing', markersize=6)
        plt.plot(self.gdp_data['Year'], self.gdp_data['Services'],
                 marker='^', linewidth=2, label='Services', markersize=6)

        plt.title('Sectoral Contribution to GDP (%) Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Contribution (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sector_trends.png', dpi=300, bbox_inches='tight')
        print("\n✅ Sector trends plot saved as 'sector_trends.png'")
        plt.show()

    def analyze_policies(self):
        """Analyze government policy impacts"""
        print("\n" + "=" * 80)
        print("GOVERNMENT POLICY IMPACT ANALYSIS".center(80))
        print("=" * 80)

        print("\n✅ POSITIVE REFORMS:")
        positive = self.policy_data[self.policy_data['Impact'] > 0]
        for _, row in positive.iterrows():
            print(f"   • {row['Policy']} ({row['Year']}): +{row['Impact']}% impact")

        print("\n❌ CHALLENGING PERIODS:")
        negative = self.policy_data[self.policy_data['Impact'] < 0]
        for _, row in negative.iterrows():
            print(f"   • {row['Policy']} ({row['Year']}): {row['Impact']}% impact")

        # Plot policy impacts
        plt.figure(figsize=(12, 6))
        colors = ['green' if x > 0 else 'red' for x in self.policy_data['Impact']]
        bars = plt.bar(self.policy_data['Policy'], self.policy_data['Impact'], color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.title('Impact of Government Policies on GDP Growth', fontsize=16, fontweight='bold')
        plt.xlabel('Policy', fontsize=12)
        plt.ylabel('Impact on Growth (%)', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('policy_impact.png', dpi=300, bbox_inches='tight')
        print("\n✅ Policy impact plot saved as 'policy_impact.png'")
        plt.show()

    def analyze_challenges(self):
        """Analyze economic challenges"""
        print("\n" + "=" * 80)
        print("KEY ECONOMIC CHALLENGES".center(80))
        print("=" * 80)

        print("\n⚠️  UNEMPLOYMENT CRISIS")
        print(f"   • Current Rate: {self.challenges.iloc[0]['Value']}%")
        print(f"   • Youth unemployment: ~15%")
        print(f"   • Issues: Skills mismatch, insufficient job creation")

        print("\n⚠️  INCOME INEQUALITY")
        print(f"   • Gini Coefficient: {self.challenges.iloc[1]['Value']}")
        print(f"   • Top 10% hold 57% of national income")
        print(f"   • Rural-urban divide widening")

        print("\n⚠️  POVERTY & DEVELOPMENT")
        print(f"   • Below poverty line: {self.challenges.iloc[2]['Value']}%")
        print(f"   • Regional disparities persist")
        print(f"   • Infrastructure gaps in rural areas")

        print("\n⚠️  INFLATION PRESSURE")
        print(f"   • Current Inflation: {self.challenges.iloc[3]['Value']}%")
        print(f"   • Food price volatility")
        print(f"   • Energy import dependency")

        # Plot challenges
        plt.figure(figsize=(10, 6))
        colors_map = ['#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6']
        plt.barh(self.challenges['Challenge'], self.challenges['Value'], color=colors_map, alpha=0.7)
        plt.title('Key Economic Challenges', fontsize=16, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('challenges.png', dpi=300, bbox_inches='tight')
        print("\n✅ Challenges plot saved as 'challenges.png'")
        plt.show()

    def train_ml_models(self):
        """Train machine learning models for GDP prediction"""
        print("\n" + "=" * 80)
        print("MACHINE LEARNING MODEL TRAINING".center(80))
        print("=" * 80)

        # Prepare data
        X = self.gdp_data[['Year']].values
        y = self.gdp_data['GDP_Trillion_USD'].values

        # Split data (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        print("\n🤖 Training Linear Regression Model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

        print(f"   ✓ R² Score: {lr_r2:.4f}")
        print(f"   ✓ RMSE: ${lr_rmse:.4f}T")

        # Random Forest
        print("\n🌲 Training Random Forest Model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        print(f"   ✓ R² Score: {rf_r2:.4f}")
        print(f"   ✓ RMSE: ${rf_rmse:.4f}T")

        # Store models
        self.lr_model = lr_model
        self.rf_model = rf_model

        # Visualize predictions
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test, lr_pred, alpha=0.6, s=100)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Linear Regression (R²={lr_r2:.4f})', fontsize=14, fontweight='bold')
        plt.xlabel('Actual GDP (Trillion USD)', fontsize=11)
        plt.ylabel('Predicted GDP (Trillion USD)', fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, rf_pred, alpha=0.6, s=100, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Random Forest (R²={rf_r2:.4f})', fontsize=14, fontweight='bold')
        plt.xlabel('Actual GDP (Trillion USD)', fontsize=11)
        plt.ylabel('Predicted GDP (Trillion USD)', fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("\n✅ Model performance plot saved as 'model_performance.png'")
        plt.show()

    def predict_future_gdp(self, year):
        """Predict GDP for future years"""
        if not hasattr(self, 'lr_model'):
            print("⚠️  Please train models first using train_ml_models()")
            return

        print(f"\n{'=' * 80}")
        print(f"GDP PREDICTION FOR {year}".center(80))
        print(f"{'=' * 80}")

        future_year = np.array([[year]])

        # Linear Regression Prediction
        lr_prediction = self.lr_model.predict(future_year)[0]

        # Random Forest Prediction
        rf_prediction = self.rf_model.predict(future_year)[0]

        # Calculate growth rate (from 2024)
        base_gdp = self.gdp_data.iloc[-1]['GDP_Trillion_USD']
        lr_growth = ((lr_prediction - base_gdp) / base_gdp) * 100
        rf_growth = ((rf_prediction - base_gdp) / base_gdp) * 100

        print(f"\n📊 LINEAR REGRESSION PREDICTION:")
        print(f"   • Predicted GDP: ${lr_prediction:.2f}T")
        print(f"   • Growth from 2024: {lr_growth:.2f}%")

        print(f"\n🌲 RANDOM FOREST PREDICTION:")
        print(f"   • Predicted GDP: ${rf_prediction:.2f}T")
        print(f"   • Growth from 2024: {rf_growth:.2f}%")

        print(f"\n📈 AVERAGE PREDICTION:")
        avg_prediction = (lr_prediction + rf_prediction) / 2
        avg_growth = (lr_growth + rf_growth) / 2
        print(f"   • Predicted GDP: ${avg_prediction:.2f}T")
        print(f"   • Growth from 2024: {avg_growth:.2f}%")

        print("\n⚠️  NOTE: Predictions are based on historical trends.")
        print("   Actual GDP may vary due to policy changes, global events, and economic reforms.")

    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE GDP ANALYSIS REPORT".center(80))
        print("=" * 80)

        self.display_overview()
        self.plot_gdp_trends()
        self.analyze_sectors()
        self.analyze_policies()
        self.analyze_challenges()
        self.train_ml_models()

        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE".center(80))
        print("=" * 80)
        print("\n📁 Generated Files:")
        print("   • gdp_trends.png")
        print("   • sector_trends.png")
        print("   • policy_impact.png")
        print("   • challenges.png")
        print("   • model_performance.png")
        print("\n✅ All visualizations saved successfully!")


def main():
    """Main function to run the analysis"""
    print("\n" + "🇮🇳 " * 40)
    print("INDIA GDP ANALYSIS WITH MACHINE LEARNING")
    print("🇮🇳 " * 40)

    # Initialize analyzer
    analyzer = IndiaGDPAnalysis()

    # Generate full report
    analyzer.generate_full_report()

    # Future predictions
    print("\n" + "=" * 80)
    print("FUTURE GDP PREDICTIONS".center(80))
    print("=" * 80)

    for year in [2025, 2027, 2030]:
        analyzer.predict_future_gdp(year)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - Thank you!".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()