import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global visual style for clean, publication-ready figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"

# Create output folder for visual assets if it doesn't exist
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Dataset Initialization (Indian E-Commerce Context)
# ---------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
data = {
    "Date": dates,
    "Sales_INR": np.random.normal(loc=50000, scale=15000, size=100).cumsum(),
    "Orders": np.random.randint(100, 500, size=100),
    "Category": np.random.choice(
        ["Electronics", "Apparel", "Home & Kitchen", "Beauty"], size=100
    ),
    "Region": np.random.choice(
        ["South", "North", "West", "East"], size=100, p=[0.4, 0.3, 0.2, 0.1]
    ),
    "Discount_Pct": np.random.uniform(5, 30, size=100),
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Line Plot: Revenue Trend Over Time (Matplotlib + Seaborn)
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Date", y="Sales_INR", color="#1f77b4", linewidth=2)
plt.title("Cumulative Daily Sales Trend (INR)", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=11)
plt.ylabel("Sales (INR)", fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sales_trend.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Distribution & Outlier Analysis (Histplot + Boxplot)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Distribution of Order Quantities
sns.histplot(
    df["Orders"], kde=True, ax=axes[0], color="#2ca02c", bins=15
)
axes[0].set_title("Distribution of Daily Orders", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Order Count")

# Subplot 2: Order Volume Across Product Categories
sns.boxplot(
    data=df, x="Category", y="Orders", ax=axes[1], palette="Set2"
)
axes[1].set_title("Order Distribution by Category", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Product Category")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "orders_distribution.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# 4. Correlation & Relationship Analysis
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Scatter plot with regression line (Discount vs. Orders)
sns.regplot(
    data=df,
    x="Discount_Pct",
    y="Orders",
    ax=axes[0],
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "red"},
)
axes[0].set_title("Discount % vs Order Volume", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Discount Percentage (%)")

# Subplot 2: Correlation Heatmap
numeric_df = df[["Sales_INR", "Orders", "Discount_Pct"]]
sns.heatmap(
    numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1], cbar=True
)
axes[1].set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "relationships_correlation.png"), dpi=300)
plt.close()

print("Visualization pipeline completed. Plots saved to 'visualizations/' directory.")