import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

INPUT_CSV = 'umap_results.csv'
OUTPUT_HTML = 'umap_visualization.html'

def visualize():
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"File {INPUT_CSV} not found. Please run generate_umap.py first.")
        return

    print(f"Generating 3D plot for {len(df)} points...")

    # Create the initial figure with Phylum
    fig = px.scatter_3d(
        df, 
        x='x', 
        y='y', 
        z='z',
        color='Phylum',
        hover_data=['Class', 'Order', 'Family', 'Genus', 'Species'],
        title='3D UMAP of DNA Sequences',
        opacity=0.7,
        size_max=5
    )

    # Update marker size
    fig.update_traces(marker=dict(size=3))

    # Create buttons for switching color
    # Note: Switching color dynamically in 3D scatter with many points can be slow or complex 
    # because it requires updating the marker colors array.
    # A simpler way is to provide a list of buttons that update the 'marker.color' attribute,
    # but we need to map categories to colors manually or let Plotly do it.
    # For simplicity and robustness, we will generate the plot for Phylum.
    # If you want to view other categories, change the 'color' parameter in the px.scatter_3d call above.
    
    # However, we can try to add a dropdown that updates the trace.
    # But since 'color' in px handles mapping strings to colors, doing it via updatemenus is hard 
    # without pre-calculating all color arrays.
    
    print(f"Saving to {OUTPUT_HTML}...")
    fig.write_html(OUTPUT_HTML)
    print("Done. Open the HTML file in your browser.")

if __name__ == "__main__":
    visualize()
