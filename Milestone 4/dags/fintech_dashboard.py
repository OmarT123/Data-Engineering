import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px


def create_dashboard(fileName):
    df = pd.read_csv(fileName)

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Fintech Loan Analysis Dashboard", style={"textAlign": "center"}),
        html.P("Omar Tamer Abdelaty - 52-11870", style={"textAlign": "center", "fontSize": "14px"}),

        html.H2("1. What is the distribution of loan amounts across different grades?"),
        dcc.Graph(id="loan_distribution_graph"),

        html.H2("2. How does the loan amount relate to annual income across states ?"),
        dcc.Dropdown(
            id="state_filter",
            options=[
                {"label": "All States", "value": "all"}
            ] + [{"label": state, "value": state} for state in sorted(df["addr_state"].unique())],
            value="all",
            placeholder="Select a state",
        ),
        dcc.Graph(id="loan_income_scatter"),
        
        html.H2("3. What is the trend of loan issuance over the months (number of loans per month), filtered by year?"),
        dcc.Dropdown(
            id="year_filter",
            options=[{"label": year, "value": year} for year in sorted(df["issue_date"].dt.year.unique())],
            value=sorted(df["issue_date"].dt.year.unique())[0],
            placeholder="Select a year",
        ),
        dcc.Graph(id="loan_trend_graph"),

        html.H2("4. Which states have the highest average loan amount?"),
        dcc.Graph(id="state_average_loan_graph"),

        html.H2("5. What is the percentage distribution of loan grades in the dataset?"),
        dcc.Graph(id="loan_grade_distribution"),
    ])

    # Q1
    @app.callback(
        Output("loan_distribution_graph", "figure"),
        Input("loan_distribution_graph", "id")
    )
    def update_loan_distribution(_):
        fig = px.box(
            df,
            x="letter_grade",
            y="loan_amount",
            color="letter_grade",
            title="Distribution of Loan Amounts by Grades",
            labels={"grade": "Loan Grade", "loan_amount": "Loan Amount"}
        )
        fig.update_layout(showlegend=False)
        return fig

    # Q2
    @app.callback(
        Output("loan_income_scatter", "figure"),
        Input("state_filter", "value")
    )
    def update_loan_income_scatter(selected_state):
        filtered_data = df if selected_state == "all" else df[df["addr_state"] == selected_state]

        fig = px.scatter(
            filtered_data,
            x="annual_inc",
            y="loan_amount",
            color="loan_status",
            title="Loan Amount vs. Annual Income by State",
            labels={"annual_inc": "Annual Income", "loan_amount": "Loan Amount", "loan_status": "Loan Status"},
            hover_data=["addr_state"]
        )
        return fig

    # Q3
    @app.callback(
        Output("loan_trend_graph", "figure"),
        Input("year_filter", "value")
    )
    def update_loan_trend(selected_year):
        filtered_data = df[df["issue_date"].dt.year == selected_year]
        monthly_data = filtered_data.groupby(filtered_data["issue_date"].dt.month).size().reset_index(name="count")
        fig = px.line(
            monthly_data,
            x="issue_date",
            y="count",
            title=f"Number of Loans Issued per Month in {selected_year}",
            labels={"issue_date": "Month", "count": "Number of Loans"}
        )
        return fig

    # Q4
    @app.callback(
        Output("state_average_loan_graph", "figure"),
        Input("state_average_loan_graph", "id")
    )
    def update_state_average_loan(_):
        state_data = df.groupby("addr_state")["loan_amount"].mean().reset_index()
        fig = px.bar(
            state_data,
            x="addr_state",
            y="loan_amount",
            title="Average Loan Amount by State",
            labels={"addr_state": "State", "loan_amount": "Average Loan Amount"},
            color="loan_amount"
        )
        return fig

    # Q5
    @app.callback(
        Output("loan_grade_distribution", "figure"),
        Input("loan_grade_distribution", "id")
    )
    def update_loan_grade_distribution(_):
        grade_data = (df["grade"].value_counts(normalize=True) * 100).reset_index()
        grade_data.columns = ["grade", "percentage"]
        fig = px.histogram(
            grade_data,
            x="grade",
            y="percentage",
            title="Percentage Distribution of Loan Grades",
            labels={"grade": "Loan Grade", "percentage": "Percentage"},
        )
        return fig

    app.run(
        debug=False,
        host='0.0.0.0',
        port=8050,
        use_reloader=False
    )

