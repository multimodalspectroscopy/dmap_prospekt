import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from UCLN_implementation import calculate_concentrations
from SRS_implementation import compute_SRS
import numpy as np 
# Set Matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Concentration Calculation GUI"),
    
    # Calculate concentrations section
    html.Div([
        html.Label("Calculate Concentrations", style={'font-weight': 'bold'}),
        html.Hr(style={'margin': '5px 0'}),
        html.Label("Select Algorithm:"),
        dcc.Dropdown(
            id='calculation-method',
            options=[
                {'label': 'UCLN', 'value': 'ucln'},
                {'label': 'SRS', 'value': 'srs'}
            ],
            value='ucln'
        ),
    ], style={'border': '1px solid black', 'padding': '10px', 'margin-bottom': '20px'}),
    
    # Select algorithm section
    html.Div([
        html.Label("Select Algorithm", style={'font-weight': 'bold'}),
        html.Hr(style={'margin': '5px 0'}),
        # Inputs for UCLN calculation
        html.Div(id='ucln-inputs', children=[
            html.Label("Spectra File Path:"),
            dcc.Input(id='spectra-file', type='text'),
            
            html.Label("Wavelengths File Path:"),
            dcc.Input(id='wavelengths-file', type='text'),
            
            html.Label("Optode Distance:"),
            dcc.Input(id='optode-dist', type='number'),
            
            html.Button('Calculate', id='calculate-ucln'),
            html.Div(id='ucln-output'),
            html.Div(id='ucln-plot')
        ], style={'display': 'none'}),
        
        # Inputs for SRS calculation
        html.Div(id='srs-inputs', children=[
            html.Label("Subject Path:"),
            dcc.Input(id='subject-path', type='text'),
            
            html.Label("Chrom:"),
            dcc.Dropdown(
                id='chrom',
                options=[
                    {'label': 'HHb', 'value': 'HHb'},
                    {'label': 'HbO2', 'value': 'HbO2'},
                    {'label': 'water', 'value': 'water'},
                    {'label': 'fat', 'value': 'fat'}
                ],
                value=['HHb', 'HbO2', 'water'],  # Default value is ["HHb", "HbO2", "water"]
                multi=True  # Allow multiple selections
            ),
            
            html.Label("Start Wavelength:"),
            dcc.Input(id='start-wavelength', type='number'),
            
            html.Label("End Wavelength:"),
            dcc.Input(id='end-wavelength', type='number'),
            
            html.Button('Calculate', id='calculate-srs'),
            html.Div(id='srs-output'),
            html.Div(id='srs-plot')
        ], style={'display': 'none'}),
    ], style={'border': '1px solid black', 'padding': '10px', 'margin-bottom': '20px'}),

    # Enter inputs section
    html.Div([
        html.Label("Enter Inputs", style={'font-weight': 'bold'}),
        html.Hr(style={'margin': '5px 0'}),
        # Display input data
        html.Div(id='input-data-output', style={'margin-top': '10px'})
    ], style={'border': '1px solid black', 'padding': '10px'}),
])

# Callback to show relevant inputs based on selected calculation method
@app.callback(
    [Output('ucln-inputs', 'style'),
     Output('srs-inputs', 'style')],
    [Input('calculation-method', 'value')]
)
def update_inputs(calculation_method):
    if calculation_method == 'ucln':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback to perform UCLN calculation and plot UCLN
@app.callback(
    [Output('ucln-output', 'children'),
     Output('ucln-plot', 'children')],
    [Input('calculate-ucln', 'n_clicks')],
    [dash.dependencies.State('spectra-file', 'value'),
     dash.dependencies.State('wavelengths-file', 'value'),
     dash.dependencies.State('optode-dist', 'value')]
)
def update_ucln_output(n_clicks, spectra_file, wavelengths_file, optode_dist):
    if n_clicks is None:
        return "", None
    else:
        # Perform UCLN calculation with constant dpf and plot_data=True
        dpf = 4.9900
        plot_data = True
        result = calculate_concentrations(spectra_file, wavelengths_file, optode_dist, dpf=dpf, plot_data=plot_data)
        time_values = list(range(1, len(result) + 1))  # Convert range to a list

        # Create Plotly traces for each concentration
        traces = []
        concentrations = result.T * 1000  # Transpose and convert to uMol
        labels = ['HbO2', 'HHb', 'CCO']  # Labels for concentrations
        for i, concentration in enumerate(concentrations):
            trace = go.Scatter(x=time_values, y=concentration, mode='lines', name=labels[i])
            traces.append(trace)
        
        # Create Plotly layout
        layout = go.Layout(title='Concentration changes over time',
                           xaxis=dict(title='Time (sec)'),
                           yaxis=dict(title='Concentration (uMol)'),
                           legend=dict(orientation='h'))
        
        # Combine traces and layout into a Plotly figure
        figure = go.Figure(data=traces, layout=layout)
        
        return f"UCLN result: {result}", dcc.Graph(figure=figure)

# Callback to perform SRS calculation and plot k_mua
@app.callback(
    [Output('srs-output', 'children'),
     Output('srs-plot', 'children')],
    [Input('calculate-srs', 'n_clicks')],
    [dash.dependencies.State('subject-path', 'value'),
     dash.dependencies.State('chrom', 'value'),
     dash.dependencies.State('start-wavelength', 'value'),
     dash.dependencies.State('end-wavelength', 'value')]
)
def update_srs_output(n_clicks, subject_path, chrom, start_wavelength, end_wavelength):
    if n_clicks is None:
        return "", None
    else:
        # Perform SRS calculation with plot_data=True
        chrom = np.array(chrom)
        # Convert list to numpy array
        C, StO2, SD, k_mua = compute_SRS(subject_path, chrom, start_wavelength, end_wavelength, plot_data=True)
        
        # Create Plotly traces for k_mua
        traces = []
        wavelengths = np.linspace(start_wavelength, end_wavelength, len(k_mua[0]))
        for i, k_mua_array in enumerate(k_mua):
            trace = go.Scatter(x=wavelengths, y=k_mua_array, mode='lines', name=f'Array {i+1}')
            traces.append(trace)
        
        # Create Plotly layout
        layout = go.Layout(title='Absorption coefficient (k_mua) vs Wavelength',
                           xaxis=dict(title='Wavelength (nm)'),
                           yaxis=dict(title='Absorption coefficient (1/cm)'),
                           legend=dict(orientation='h'))
        
        # Combine traces and layout into a Plotly figure
        figure = go.Figure(data=traces, layout=layout)
        
        # Format the SRS results for display
        srs_result = html.Div([
            html.Label('C:'),
            html.Pre(str(C)),
            html.Label('StO2:'),
            html.Pre(str(StO2)),
            html.Label('SD:'),
            html.Pre(str(SD)),
            html.Label('k_mua:'),
            html.Pre(str(k_mua))
        ])

        return srs_result, dcc.Graph(figure=figure)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)