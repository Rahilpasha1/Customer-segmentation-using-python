import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", None)


df = pd.read_csv("online_retail.csv")
df.head()


print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Top 10 Description plot
plt.figure(figsize=(12,6))
top_desc = df["Description"].value_counts().head(10)

sns.barplot(x=top_desc.values, y=top_desc.index, palette="Set2")
plt.title("Top 10 Most Frequent Product Descriptions", fontsize=16)
plt.xlabel("Count")
plt.ylabel("Description")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df["Quantity"], bins=20, kde=True, color="skyblue")
plt.title("Quantity Distribution", fontsize=16)
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()


num_cols = df.select_dtypes(include=["float64", "int64"]).columns
sample_df = df[num_cols].sample(500, random_state=42)

sns.pairplot(sample_df, diag_kind="kde", corner=True)
plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = df.dropna()
# Convert InvoiceDate column to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceMonth"] = df["InvoiceDate"].dt.month
df["InvoiceHour"] = df["InvoiceDate"].dt.hour
X = df[["UnitPrice", "Quantity", "InvoiceMonth", "InvoiceHour"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=1000, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
df.head()

from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


df_sorted = df.sort_values("InvoiceDate")

-
top_codes = df["StockCode"].value_counts().head(5).index
df_filtered = df_sorted[df_sorted["StockCode"].isin(top_codes)]

def _plot_series(series, series_name, series_index=0):
    palette = list(sns.palettes.mpl_palette('Dark2'))
    xs = series['InvoiceDate']
    ys = series['UnitPrice']
    plt.plot(xs, ys, label=series_name,
             color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

for i, (series_name, series) in enumerate(df_filtered.groupby('StockCode')):
    _plot_series(series, series_name, i)

fig.legend(title='StockCode', bbox_to_anchor=(1, 1), loc='upper left')
sns.despine(fig=fig, ax=ax)
plt.xlabel('InvoiceDate')
plt.ylabel('UnitPrice')
plt.title

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(
    x="PCA1", y="PCA2",
    hue="Cluster",
    palette="Set2",
    data=df,
    s=80
)
plt.title("Customer Segments Visualization (PCA)")
plt.show()

feature_cols = ["UnitPrice", "Quantity", "InvoiceMonth", "InvoiceHour"]
feature_cols = ["UnitPrice", "Quantity"]
cluster_summary = df.groupby("months")[feature_cols].mean().reset_index()
cluster_summary

pip install dash
pip install dash==2.16.1 jupyter-dash==0.4.2 pyngrok plotly scikit-learn

from pyngrok import ngrok
ngrok.set_auth_token("35ytSTpxWaf4wavXM87thfruO7q_6kkavvXDENerWTbZSZSmn")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from pyngrok import ngrok

# Load Dataset
df = pd.read_csv("online_retail.csv")

# Preprocessing
df = df.dropna().drop_duplicates()

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceHour'] = df['InvoiceDate'].dt.hour

# Features for Clustering
features = ["UnitPrice", "Quantity", "InvoiceMonth", "InvoiceHour"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_components[:, 0], pca_components[:, 1]

# Dash App
app = Dash(__name__)

app.layout = html.Div([

    html.H1("Customer Segmentation Dashboard",
            style={"textAlign": "center", "color": "#003366"}),

    html.Label("Select Cluster:", style={"fontSize": "20px"}),
    dcc.Dropdown(
        id="cluster-filter",
        options=[{"label": f"Cluster {c}", "value": c} for c in df["Cluster"].unique()],
        placeholder="Select a cluster",
        style={"width": "50%"}
    ),

    dcc.Graph(id="cluster-distribution"),
    dcc.Graph(id="pca-plot"),
    dcc.Graph(id="unitprice-vs-quantity"),
    dcc.Graph(id="month-vs-hour"),
])


@app.callback(
    [
        Output("cluster-distribution", "figure"),
        Output("pca-plot", "figure"),
        Output("unitprice-vs-quantity", "figure"),
        Output("month-vs-hour", "figure"),
    ],
    [Input("cluster-filter", "value")]
)
def update_dashboard(selected_cluster):

    if selected_cluster is None:
        data = df
    else:
        data = df[df["Cluster"] == selected_cluster]

    # FIXED: Now shows correct filtered count
    fig1 = px.histogram(
        data, 
        x="Cluster",
        color="Cluster",
        title="Cluster Distribution (Filtered)"
    )

    fig2 = px.scatter(
        data, x="PCA1", y="PCA2", color="Cluster",
        title="PCA Cluster Visualization"
    )

    fig3 = px.scatter(
        data, x="UnitPrice", y="Quantity", color="Cluster",
        title="UnitPrice vs Quantity"
    )

    fig4 = px.scatter(
        data, x="InvoiceMonth", y="InvoiceHour", color="Cluster",
        title="InvoiceMonth vs InvoiceHour"
    )

    return fig1, fig2, fig3, fig4


# Start Ngrok Tunnel
public_url = ngrok.connect(8050)
print("🔗 Dash App Public URL:", public_url)

# Run Server
app.run_server(host="0.0.0.0", port=8050)
