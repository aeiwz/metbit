# Load your NMR spectra data
df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
spectra = df.iloc[:,1:]
ppm = spectra.columns.astype(float).to_list()

# Create instance of the class with spectra and ppm data
stocsy_app = STOCSY_app(spectra, ppm)

# Get the app instance
app = stocsy_app.run_ui()

# Run the app
app.run_server(debug=True, port=8051)