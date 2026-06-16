"""
dashboard_app.py

Dash dashboard to visualize students at risk from MySQL 'dropout_master' table.

How to run:
    python dashboard_app.py
Then open: http://127.0.0.1:8050

Config: edit DB_* constants below or set environment variables if you prefer.
"""

import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from sqlalchemy import create_engine
import urllib.parse

# ---------------- CONFIG ----------------
DB_USER = os.environ.get("DB_USER", "root")
DB_PASS = os.environ.get("DB_PASS", "baha1528")   # change or export as env var
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "MiniProj")
TABLE_NAME = "dropout_master"

SHAP_CSV = "shap_feature_importances_fixed.csv"  # optional; used for top feature bar chart

# Dash app settings
PAGE_SIZE = 25
# ----------------------------------------

def connect_engine():
    # Using sqlalchemy engine with mysql+mysqlconnector
    password_escaped = urllib.parse.quote_plus(DB_PASS)
    uri = f"mysql+mysqlconnector://{DB_USER}:{password_escaped}@{DB_HOST}/{DB_NAME}"
    return create_engine(uri, pool_pre_ping=True)

def fetch_dropouts_from_db():
    engine = connect_engine()
    query = f"SELECT * FROM {TABLE_NAME}"
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        # Provide a helpful error message
        raise RuntimeError(f"Failed to read from DB. Check DB credentials/connection. Details: {e}")
    return df

def load_shap_importances():
    if os.path.exists(SHAP_CSV):
        try:
            s = pd.read_csv(SHAP_CSV, index_col=0, header=None, squeeze=False)
            # shap csv might be a single column series. Coerce to Series for plotting.
            try:
                # If file has one column of feature importance values but no header
                shap_df = pd.read_csv(SHAP_CSV, header=None, index_col=0)
                shap_df.columns = ["importance"]
            except Exception:
                shap_df = pd.read_csv(SHAP_CSV)
            # Normalize to DataFrame(feature, importance)
            if shap_df.shape[1] == 1:
                shap_df = shap_df.reset_index().rename(columns={'index': 'feature', shap_df.columns[0]: 'importance'})
            elif 'feature' in shap_df.columns and 'importance' in shap_df.columns:
                pass
            else:
                shap_df = shap_df.reset_index().rename(columns={shap_df.columns[0]:'feature', shap_df.columns[1]:'importance'})
            shap_df = shap_df.sort_values("importance", ascending=False)
            return shap_df
        except Exception:
            return None
    return None

# Initialize app
app = Dash(__name__)
server = app.server
app.title = "Student Dropout Risk Dashboard"

# Layout
app.layout = html.Div([
    html.Div([
        html.H2("🎓 Student Dropout Risk Dashboard", style={'margin':'6px 0'}),
        html.Div("Source: MySQL table 'dropout_master' — shows students predicted at-risk", style={'fontSize':12, 'color':'#666'}),
    ], style={'padding':'8px 16px', 'borderBottom':'1px solid #ddd'}),

    html.Div(id='error-box', style={'color':'red', 'padding':'8px 16px'}),

    html.Div([
        html.Button("Refresh data", id="refresh-btn", n_clicks=0),
        html.Button("Download filtered CSV", id="download-csv-btn", n_clicks=0, style={'marginLeft':'8px'}),
        dcc.Download(id="download-dataframe-csv"),
        html.Span(id='last-refresh', style={'marginLeft':'16px', 'color':'#666'})
    ], style={'padding':'8px 16px'}),

    html.Div([
        # Left column: filters
        html.Div([
            html.H4("Filters"),
            html.Label("Attendance rate ≤ (%)"),
            dcc.Slider(id='att-slider', min=0, max=100, step=1, value=100,
                       marks={0:"0",50:"50",100:"100"}),
            html.Br(),
            html.Label("Avg grade ≤"),
            dcc.Slider(id='grade-slider', min=0, max=10, step=0.1, value=10,
                       marks={0:"0",5:"5",10:"10"}),
            html.Br(),
            html.Label("Failed courses ≥"),
            dcc.Input(id='failed-min', type='number', value=0, min=0, style={'width':'100%'}),
            html.Br(), html.Br(),
            html.Label("Search by name or student_id"),
            dcc.Input(id='search-box', placeholder='type name or id...', style={'width':'100%'}),
            html.Br(), html.Br(),
            html.Label("Top N by risk score"),
            dcc.Input(id='top-n', type='number', value=100, min=1, style={'width':'100%'}),
        ], style={'flex':'0 0 300px', 'padding':'8px 16px', 'borderRight':'1px solid #eee'}),

        # Right column: charts + table
        html.Div([
            html.Div([
                html.Div(id='metrics-row', style={'display':'flex', 'gap':'24px', 'marginBottom':'12px'}),
            ]),
            html.Div([
                dcc.Tabs(id='tabs', value='tab-table', children=[
                    dcc.Tab(label='Table', value='tab-table', children=[
                        html.Div(id='table-container')
                    ]),
                    dcc.Tab(label='Charts', value='tab-charts', children=[
                        html.Div([
                            dcc.Graph(id='scatter-risk', style={'height':'400px'}),
                            dcc.Graph(id='hist-risk', style={'height':'300px'}),
                            dcc.Graph(id='shap-bar', style={'height':'300px'})
                        ])
                    ])
                ])
            ])
        ], style={'flex':'1 1 auto', 'padding':'8px 16px'}),
    ], style={'display':'flex', 'minHeight':'60vh'}),

    # Hidden store for data
    dcc.Store(id='df-store'),

    html.Div(style={'padding':'12px 16px', 'color':'#666', 'fontSize':12}, children=[
        "Note: This dashboard reads the live `dropout_master` table. Use the Refresh button after running model training.",
        html.Br(),
        html.Strong("Scope for improvements: "),
        html.Ul([
            html.Li("Add authentication + role-based views (faculty vs admin)."),
            html.Li("Add per-student SHAP explanation popups (requires per-student SHAP saved)."),
            html.Li("Allow threshold tuning from UI and re-run predictions."),
            html.Li("Add export to Excel and scheduled run integration."),
        ])
    ])
])

# ---------------- Callbacks ----------------

@app.callback(
    Output('df-store', 'data'),
    Output('error-box', 'children'),
    Output('last-refresh', 'children'),
    Input('refresh-btn', 'n_clicks'),
)
def refresh_data(n_clicks):
    """Read from DB when refresh pressed (or on initial load)."""
    try:
        df = fetch_dropouts_from_db()
        # Ensure types and sensible defaults
        if 'dropout_prob' in df.columns:
            df['dropout_prob'] = pd.to_numeric(df['dropout_prob'], errors='coerce').fillna(0.0)
        else:
            df['dropout_prob'] = 0.0
        # keep numeric columns present
        for col in ['attendance_rate','avg_grade','failed_courses','attendance_count']:
            if col not in df.columns:
                df[col] = 0
        # convert student_id to string to avoid JSON issues
        if 'student_id' in df.columns:
            df['student_id'] = df['student_id'].astype(str)
        payload = df.to_dict(orient='records')
        return payload, "", f"Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return None, f"Error fetching data: {e}", ""

@app.callback(
    Output('metrics-row', 'children'),
    Input('df-store', 'data'),
)
def update_metrics(data):
    if not data:
        return html.Div("No data loaded.")
    df = pd.DataFrame(data)
    total = len(df)
    at_risk = df['predicted_label'].astype(int).sum()
    avg_risk = df['dropout_prob'].mean() if 'dropout_prob' in df.columns else 0.0
    return [
        html.Div([
            html.H3(f"{at_risk}", style={'margin':'0'}),
            html.Div("Current at-risk students")
        ], style={'padding':'8px','border':'1px solid #eee','width':'220px','borderRadius':'6px'}),
        html.Div([
            html.H3(f"{total}", style={'margin':'0'}),
            html.Div("Total rows in table")
        ], style={'padding':'8px','border':'1px solid #eee','width':'220px','borderRadius':'6px'}),
        html.Div([
            html.H3(f"{avg_risk:.3f}", style={'margin':'0'}),
            html.Div("Average risk score")
        ], style={'padding':'8px','border':'1px solid #eee','width':'220px','borderRadius':'6px'}),
    ]

@app.callback(
    Output('table-container', 'children'),
    Input('df-store', 'data'),
    Input('att-slider', 'value'),
    Input('grade-slider', 'value'),
    Input('failed-min', 'value'),
    Input('search-box', 'value'),
    Input('top-n', 'value'),
)
def update_table(data, att_val, grade_val, failed_min, search_text, top_n):
    if not data:
        return html.Div("No data loaded.")
    df = pd.DataFrame(data)
    # apply filters
    df = df[pd.to_numeric(df.get('attendance_rate', 0)) <= att_val]
    df = df[pd.to_numeric(df.get('avg_grade', 0)) <= grade_val]
    df = df[pd.to_numeric(df.get('failed_courses', 0)) >= (failed_min or 0)]
    if search_text:
        s = str(search_text).lower()
        df = df[df.apply(lambda r: s in str(r.get('student_name','')).lower() or s in str(r.get('student_id','')), axis=1)]
    # focus on at-risk only (the table is meant to show at-risk students)
    df = df[df['predicted_label'].astype(int) == 1]
    # order by risk score desc and limit top_n
    df = df.sort_values('dropout_prob', ascending=False)
    if top_n:
        df = df.head(int(top_n))
    # columns to show
    show_cols = ['student_id','student_name','failed_courses','attendance_count','attendance_rate','avg_grade','dropout_prob']
    available_cols = [c for c in show_cols if c in df.columns]
    # Dash DataTable
    table = dash_table.DataTable(
        id='risk-table',
        columns=[{"name": c.replace('_',' ').title(), "id": c} for c in available_cols],
        data=df[available_cols].to_dict('records'),
        page_size=PAGE_SIZE,
        style_table={'overflowX':'auto'},
        sort_action='native',
        filter_action='native',
        row_selectable='single'
    )
    return table

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State('df-store', 'data'),
    State('att-slider', 'value'),
    State('grade-slider', 'value'),
    State('failed-min', 'value'),
    State('search-box', 'value'),
    State('top-n', 'value'),
    prevent_initial_call=True,
)
def export_filtered(n_clicks, data, att_val, grade_val, failed_min, search_text, top_n):
    if not data:
        return dcc.send_data_frame(pd.DataFrame().to_csv, "no_data.csv")
    df = pd.DataFrame(data)
    df = df[pd.to_numeric(df.get('attendance_rate', 0)) <= att_val]
    df = df[pd.to_numeric(df.get('avg_grade', 0)) <= grade_val]
    df = df[pd.to_numeric(df.get('failed_courses', 0)) >= (failed_min or 0)]
    if search_text:
        s = str(search_text).lower()
        df = df[df.apply(lambda r: s in str(r.get('student_name','')).lower() or s in str(r.get('student_id','')), axis=1)]
    df = df[df['predicted_label'].astype(int) == 1]
    df = df.sort_values('dropout_prob', ascending=False)
    if top_n:
        df = df.head(int(top_n))
    return dcc.send_data_frame(df.to_csv, "dropout_master_filtered.csv", index=False)

@app.callback(
    Output('scatter-risk', 'figure'),
    Output('hist-risk', 'figure'),
    Output('shap-bar', 'figure'),
    Input('df-store', 'data'),
)
def update_charts(data):
    if not data:
        empty_fig = px.scatter()
        return empty_fig, empty_fig, empty_fig
    df = pd.DataFrame(data)
    # Scatter: attendance_rate vs avg_grade colored by dropout_prob
    fig_scatter = px.scatter(df, x='attendance_rate', y='avg_grade', color='dropout_prob',
                             hover_data=['student_id','student_name','failed_courses'],
                             title='Attendance vs Average Grade (color = risk score)')
    # Histogram of risk scores
    fig_hist = px.histogram(df, x='dropout_prob', nbins=30, title='Risk Score Distribution (all rows)')
    # SHAP bar chart if available
    shap_df = load_shap_importances()
    if shap_df is not None:
        topk = shap_df.head(15)
        fig_shap = px.bar(topk, x='feature', y='importance', title='Top SHAP feature importances')
    else:
        fig_shap = px.bar(x=[], y=[], title='SHAP importances not available')
    return fig_scatter, fig_hist, fig_shap

# Run app
if __name__ == "__main__":
    # On app start, do an initial refresh (simulate button press)
    try:
        # This will trigger DB read on first HTTP load when user clicks refresh
        print("Starting dashboard. Use the Refresh button in the UI to load DB records.")
    except Exception:
        pass
    app.run(debug=True)

