import csv
from flask import Flask, render_template, request, redirect, url_for, session, flash
from collections import defaultdict
import logging
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from pandas import read_csv, to_datetime
from prophet import Prophet  # Use fbprophet instead of prophet
from matplotlib import pyplot as plt
import io
import base64


# Configuration
CSV_FILE = 'ames_liquor_sales_2021-24.csv'
USER_CREDENTIALS = {'admin': '123', 'user1': 'password'}  # Example credentials

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Logging configuration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read and process sales data
def read_sales(year=None):
    sales = defaultdict(
        lambda: {"brand": "", "retail_price": 0.0, "volume": "", "quantity_sold": 0, "total_sales": 0.0})

    try:
        with open(CSV_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                record_year = int(row['Date'].split('/')[-1])
                if year and record_year != year:
                    continue

                name = row['item_name']
                volume = row['volume']
                brand = row['brand']
                retail_price = float(row['retail_price'])
                quantity_sold = int(row['quantity_sold'])
                total_sales = retail_price * quantity_sold

                # Unique key based on name and volume
                key = (name, volume)

                # If the key already exists, aggregate and keep the highest price
                if key in sales:
                    sales[key]["quantity_sold"] += quantity_sold
                    sales[key]["total_sales"] += total_sales
                    # Update the retail price if the new price is higher
                    if retail_price > sales[key]["retail_price"]:
                        sales[key]["retail_price"] = retail_price
                        sales[key]["brand"] = brand  # Update the brand if price is updated
                else:
                    # Add a new entry for the name and volume
                    sales[key] = {
                        "brand": brand,
                        "retail_price": retail_price,
                        "volume": volume,
                        "quantity_sold": quantity_sold,
                        "total_sales": total_sales,
                    }
    except FileNotFoundError:
        pass  # If the file doesn't exist, return an empty list

    # Convert the dictionary to a sorted list by total_sales in descending order
    aggregated_sales = [
        {
            "name": key[0],
            "volume": key[1],
            **data,
            "total_sales": round(data["total_sales"], 2)  # Round total_sales to 2 decimal places
        }
        for key, data in sales.items()
    ]
    aggregated_sales = sorted(aggregated_sales, key=lambda x: x['total_sales'], reverse=True)

    return aggregated_sales
# Flask routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Authenticate user
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('homepage'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login_form.html')

@app.route('/logout')
def logout():
    # Clear session and redirect to login
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def homepage():
    if 'username' in session:
        return render_template('homepage.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/data', methods=['GET', 'POST'])
def data():
    selected_year = request.form.get('year')
    if selected_year:
        selected_year = int(selected_year)

    # Read and aggregate sales data for the selected year (or all years if not selected)
    alcohol_list = read_sales(year=selected_year)
    return render_template('display_sales.html', alcohol_list=alcohol_list, selected_year=selected_year)


# Load and preprocess data
df = pd.read_csv(CSV_FILE)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Dash layout
dash_app.layout = html.Div([
    html.H1("Ames Liquor Sales Dashboard"),
    dcc.Dropdown(
        id='store-dropdown',
        options=[{'label': store, 'value': store} for store in df['Store Name'].unique()] + [
            {'label': 'Total Ames Stores', 'value': 'Total Ames Stores'}],
        value='Total Ames Stores',
        clearable=False
    ),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.RadioItems(
        id='y-axis',
        options=[
            {'label': 'Sales ($)', 'value': 'sale'},
            {'label': 'Bottles Sold', 'value': 'quantity_sold'}
        ],
        value='sale',
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Dropdown(
        id='liquor-type-dropdown',
        options=[{'label': liquor_type, 'value': liquor_type} for liquor_type in df['Category Name'].unique()] + [
            {'label': 'All Liquors', 'value': 'All Liquors'}],
        value='All Liquors',
        clearable=False
    ),
    dcc.Graph(id='sales-graph'),
    dcc.RadioItems(
        id='time-period',
        options=[
            {'label': 'Weekly', 'value': 'W'},
            {'label': 'Monthly', 'value': 'M'},
            {'label': 'Quarterly', 'value': 'Q'},
            {'label': 'Annually', 'value': 'Y'}
        ],
        value='M',
        labelStyle={'display': 'inline-block'}
    )
])

# Dash callback
@dash_app.callback(
    Output('sales-graph', 'figure'),
    [Input('store-dropdown', 'value'),
     Input('time-period', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('y-axis', 'value'),
     Input('liquor-type-dropdown', 'value')]
)
def update_graph(selected_store, selected_period, start_date, end_date, y_axis, liquor_type):
    filtered_df = df.copy() if selected_store == 'Total Ames Stores' else df[df['Store Name'] == selected_store]

    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    if liquor_type != 'All Liquors':
        filtered_df = filtered_df[filtered_df['Category Name'] == liquor_type]

    if filtered_df.empty:
        return px.line(title='No data available for the selected criteria')

    resampled_df = filtered_df.resample(selected_period, on='Date').sum().reset_index()
    fig = px.line(resampled_df, x='Date', y=y_axis, title=f'{y_axis} for {selected_store} - {liquor_type}')
    fig.update_yaxes(rangemode="tozero")

    return fig


def load_data():
    # Load data
    CSV_FILE = 'ames_liquor_sales_2024.csv'
    df = read_csv(CSV_FILE, header=0)
    df.rename(columns={'Date': 'ds', 'sale': 'y', 'Store Name': 'Store Name'}, inplace=True)
    df['ds'] = to_datetime(df['ds'])
    return df

@app.route('/forecast_page')
def forecast_page():
    stores = df['Store Name'].unique()
    return render_template('index.html', stores=stores)

@app.route('/forecast', methods=['POST'])
def forecast():
    df = load_data()
    selected_store = request.form['store']
    df_filtered = df[df['Store Name'] == selected_store]

    # Group sales data by week and sum up the sales
    df_grouped = df_filtered.groupby(df_filtered['ds'].dt.to_period('W')).agg({'y': 'sum'}).reset_index()
    df_grouped['ds'] = df_grouped['ds'].dt.to_timestamp()

    # Define and fit the model
    model_grouped = Prophet()
    model_grouped.fit(df_grouped)

    # Define the period for which we want a prediction
    future_grouped = model_grouped.make_future_dataframe(periods=12, freq='M')
    forecast_grouped = model_grouped.predict(future_grouped)
    forecast_grouped = forecast_grouped[forecast_grouped['ds'] > df_grouped['ds'].max()]

    # Plot the forecast
    fig_grouped, ax_grouped = plt.subplots()
    model_grouped.plot(forecast_grouped, ax=ax_grouped)
    plt.title(f'Future Sales Forecast (Grouped by Week) for {selected_store}')

    # Save the plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('forecast.html', plot_url=plot_url, store=selected_store)

if __name__ == '__main__':
    app.run(debug=True)