"""
Test file for exploring table display options in Streamlit.
"""
import pandas as pd
import streamlit as st
import random

# Set up the page
st.set_page_config(layout="wide", page_title="Table Display Test")

# Function to generate sample data
def generate_sample_data(rows=20):
    """Generate sample Twitter-like account data."""
    data = {
        "Rank": list(range(1, rows+1)),
        "Username": [f"@user_{i}" for i in range(1, rows+1)],
        "Connection": [],
        "CloutRank": [round(random.random() * 0.01, 5) for _ in range(rows)],
        "In-Degree": [random.randint(1, 30) for _ in range(rows)],
        "Followers": [random.randint(100, 1000000) for _ in range(rows)],
        "Following": [random.randint(10, 5000) for _ in range(rows)],
        "Description": [],
        "Tweet Summary": []
    }
    
    # Add connection types
    connection_types = ["Original", "1st Degree", "2nd Degree", "Other"]
    connection_weights = [0.05, 0.25, 0.5, 0.2]  # Probability weights
    data["Connection"] = random.choices(connection_types, weights=connection_weights, k=rows)
    
    # Make first row the Original account
    data["Connection"][0] = "Original"
    
    # Generate realistic descriptions and tweet summaries
    descriptions = [
        "Data scientist and AI researcher. Working on NLP and graph networks. #MachineLearning #DataScience",
        "Software engineer at BigTech. Passionate about distributed systems and cloud architecture.",
        "Entrepreneur and startup founder. Building the future of social media.",
        "Digital marketing specialist with 10+ years of experience. Speaker, consultant, coffee lover.",
        "Passionate about climate change and sustainability. Working to make a difference in the world.",
        "Product manager by day, YouTuber by night. I talk about tech, innovation, and productivity.",
        "Professor of Computer Science at University of Technology. Researching machine learning and AI ethics.",
        "Mother of 2, tech enthusiast, and avid reader. I share my thoughts on parenthood and technology.",
        "Content creator and digital nomad. Currently traveling around the world while working remotely.",
        "Investor focused on early-stage tech startups. Looking for the next big thing.",
        "Political analyst and journalist. Writing about the intersection of technology and politics.",
        "UX/UI designer with a passion for creating beautiful and functional interfaces.",
        "Full-stack developer specializing in web applications. JavaScript enthusiast.",
        "Professional photographer capturing the beauty of urban landscapes.",
        "Health and wellness coach helping people live their best lives.",
    ]
    
    tweet_summaries = [
        "Frequently discusses machine learning advancements and shares academic papers. Often engages with AI research community and posts about recent conference talks.",
        "Shares tech industry news and commentary. Focuses on software engineering trends, coding tips, and development practices. Occasionally posts about tech conferences.",
        "Discusses startup funding rounds and entrepreneurship advice. Frequently shares insights about business growth and venture capital. Promotes own company regularly.",
        "Posts about digital marketing strategies and industry trends. Shares case studies and success stories. Frequently promotes speaking engagements and workshops.",
        "Shares environmental news and climate change research. Often posts about sustainable practices and advocates for policy changes. Engages with climate activist community.",
        "Creates threads about product development and tech reviews. Shares productivity tips and tool recommendations. Frequently promotes YouTube channel and recent videos.",
        "Discusses academic research and teaching experiences. Shares thoughts on AI ethics and responsible technology. Posts about student projects and academic conferences.",
        "Balances personal content about family life with professional tech commentary. Shares book recommendations and thoughts on work-life balance in tech industry.",
        "Documents travel experiences while working remotely. Shares tips about digital nomad lifestyle and location-independent work. Posts spectacular views from various countries.",
        "Analyzes investment trends and market opportunities. Shares thoughts on emerging technologies and startup potential. Occasionally posts about portfolio companies.",
        "Writes detailed threads analyzing political events and tech policy. Frequently shares articles and publications. Engages in discussions about technology regulation.",
        "Showcases design work and UX processes. Shares inspiration and design resources. Discusses trends in interface design and user experience.",
        "Shares coding tips and project updates. Discusses web development frameworks and best practices. Often posts code snippets and tutorials.",
        "Mostly posts original photography with minimal text. Shares behind-the-scenes of photo shoots and editing techniques. Occasionally promotes photography services.",
        "Creates content about health strategies and wellness practices. Shares workout routines and nutrition advice. Frequently posts motivational content.",
    ]
    
    # Randomly assign descriptions and tweet summaries
    data["Description"] = [random.choice(descriptions) for _ in range(rows)]
    data["Tweet Summary"] = [random.choice(tweet_summaries) for _ in range(rows)]
    
    # Make some descriptions and summaries exceptionally long
    for i in range(3):
        idx = random.randint(0, rows-1)
        data["Description"][idx] = " ".join([data["Description"][idx]] * 3)
        data["Tweet Summary"][idx] = " ".join([data["Tweet Summary"][idx]] * 4)
    
    # Format follower counts
    data["Followers"] = [f"{count:,}" for count in data["Followers"]]
    data["Following"] = [f"{count:,}" for count in data["Following"]]
        
    return pd.DataFrame(data)

# Create sample dataframe
df = generate_sample_data(20)

# Page title
st.title("Table Display Options Testing")
st.markdown("Testing different approaches for optimal table display in Streamlit")

# Sidebar controls
st.sidebar.header("Display Options")
display_method = st.sidebar.radio(
    "Select Display Method",
    ["st.table", "st.dataframe", "HTML Table", "HTML with Custom CSS", "Streamlit Container with CSS"]
)

show_all_columns = st.sidebar.checkbox("Show All Columns", True)
if not show_all_columns:
    columns_to_show = st.sidebar.multiselect(
        "Select Columns to Display",
        df.columns.tolist(),
        default=["Rank", "Username", "Connection", "CloutRank", "Description"]
    )
    df_display = df[columns_to_show]
else:
    df_display = df

# Main content area
st.header(f"Method 1: {display_method}")

# Apply different display methods
if display_method == "st.table":
    st.subheader("Standard st.table")
    st.markdown("**Pros:** Simple, built-in wrapping. **Cons:** No scrolling, limited customization.")
    st.table(df_display)

elif display_method == "st.dataframe":
    st.subheader("Standard st.dataframe")
    st.markdown("**Pros:** Interactive, sortable, scrollable. **Cons:** Text can be cut off.")
    
    column_config = {
        "Rank": st.column_config.NumberColumn(width="small"),
        "Username": st.column_config.TextColumn(width="medium"),
        "Connection": st.column_config.TextColumn(width="medium"),
        "CloutRank": st.column_config.NumberColumn(format="%.5f", width="small"),
        "In-Degree": st.column_config.NumberColumn(width="small"),
        "Followers": st.column_config.TextColumn(width="small"),
        "Following": st.column_config.TextColumn(width="small"),
        "Description": st.column_config.TextColumn(width="large"),
        "Tweet Summary": st.column_config.TextColumn(width="large")
    }
    
    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=400
    )

elif display_method == "HTML Table":
    st.subheader("Custom HTML Table")
    st.markdown("**Pros:** Full HTML customization. **Cons:** Requires HTML knowledge.")
    
    # Convert DataFrame to HTML table
    html_table = df_display.to_html(index=False, escape=False)
    
    # Display the HTML table
    st.markdown(html_table, unsafe_allow_html=True)

elif display_method == "HTML with Custom CSS":
    st.subheader("HTML Table with Custom CSS")
    st.markdown("**Pros:** Full control over styling and behavior. **Cons:** More complex.")
    
    # Define custom CSS styles
    css = """
    <style>
        .custom-table-container {
            width: 100%;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 1000px;
            font-size: 14px;
        }
        
        .custom-table th {
            background-color: rgba(108, 166, 205, 0.3);
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        
        .custom-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            word-wrap: break-word;
            max-width: 300px;
        }
        
        .custom-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .custom-table tr:hover {
            background-color: rgba(108, 166, 205, 0.2);
        }
        
        /* Connection column styling */
        .connection-original {
            color: #ff9d00;
            font-weight: bold;
        }
        
        .connection-first {
            color: #00c3ff;
            font-weight: bold;
        }
        
        .connection-second {
            color: #8bc34a;
            font-weight: bold;
        }
        
        .username-column {
            font-weight: bold;
            white-space: nowrap;
        }
    </style>
    """
    
    # Create a function to format cells based on column
    def format_cell(col_name, value):
        if col_name == "Connection":
            if value == "Original":
                return f'<span class="connection-original">{value}</span>'
            elif value == "1st Degree":
                return f'<span class="connection-first">{value}</span>'
            elif value == "2nd Degree":
                return f'<span class="connection-second">{value}</span>'
            else:
                return value
        elif col_name == "Username":
            return f'<span class="username-column">{value}</span>'
        elif col_name in ["Description", "Tweet Summary"]:
            # Replace newlines with <br> and wrap in a div with max height
            formatted = value.replace('\n', '<br>')
            return f'<div style="max-height: 100px;">{formatted}</div>'
        else:
            return value
    
    # Generate HTML table with custom formatting
    html_rows = []
    
    # Add header row
    header_cells = [f"<th>{col}</th>" for col in df_display.columns]
    html_rows.append(f"<tr>{''.join(header_cells)}</tr>")
    
    # Add data rows
    for _, row in df_display.iterrows():
        cells = []
        for col in df_display.columns:
            formatted_value = format_cell(col, str(row[col]))
            cells.append(f"<td>{formatted_value}</td>")
        html_rows.append(f"<tr>{''.join(cells)}</tr>")
    
    html_table = f"""
    {css}
    <div class="custom-table-container">
        <table class="custom-table">
            {''.join(html_rows)}
        </table>
    </div>
    """
    
    # Display the custom HTML table
    st.markdown(html_table, unsafe_allow_html=True)

elif display_method == "Streamlit Container with CSS":
    st.subheader("Streamlit Container with CSS")
    st.markdown("**Pros:** Combines native Streamlit with custom styling. **Cons:** Limited to container capabilities.")
    
    css = """
    <style>
        /* Style the st-container */
        div[data-testid="stVerticalBlock"] > div:nth-child(2) {
            overflow-x: auto !important;
            max-height: 500px;
            overflow-y: auto;
        }
        
        /* Style dataframe elements */
        .dataframe-wrapper table {
            width: 100%;
            min-width: 1000px; 
            border-collapse: collapse;
        }
        
        .dataframe-wrapper th {
            background-color: rgba(108, 166, 205, 0.3);
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        
        .dataframe-wrapper td {
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            max-width: 300px;
            word-wrap: break-word;
        }
        
        .dataframe-wrapper tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .dataframe-wrapper tr:hover {
            background-color: rgba(108, 166, 205, 0.2);
        }
        
        /* Keep usernames on one line */
        .dataframe-wrapper td:nth-child(2) {
            white-space: nowrap;
            font-weight: bold;
        }
        
        /* Apply colors to connection types */
        .dataframe-wrapper td:nth-child(3) {
            font-weight: bold;
        }
        
        .connection-original {
            color: #ff9d00 !important;
        }
        
        .connection-first {
            color: #00c3ff !important;
        }
        
        .connection-second {
            color: #8bc34a !important;
        }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)
    
    # Function to format cells with HTML
    def format_df_for_display(df):
        df_html = df.copy()
        
        # Format connection column with colors
        def color_connection(val):
            if val == "Original":
                return f'<span class="connection-original">{val}</span>'
            elif val == "1st Degree":
                return f'<span class="connection-first">{val}</span>'
            elif val == "2nd Degree":
                return f'<span class="connection-second">{val}</span>'
            return val
        
        df_html["Connection"] = df_html["Connection"].apply(color_connection)
        
        # Ensure descriptions have proper formatting
        if "Description" in df_html.columns:
            df_html["Description"] = df_html["Description"].apply(
                lambda x: '<div style="white-space: normal;">' + x.replace('\n', '<br>') + '</div>'
            )
        
        # Ensure tweet summaries have proper formatting
        if "Tweet Summary" in df_html.columns:
            df_html["Tweet Summary"] = df_html["Tweet Summary"].apply(
                lambda x: '<div style="white-space: normal;">' + x.replace('\n', '<br>') + '</div>'
            )
            
        return df_html
    
    formatted_df = format_df_for_display(df_display)
    
    # Create a container and display the dataframe
    with st.container():
        st.markdown('<div class="dataframe-wrapper">', unsafe_allow_html=True)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Additional options section
st.header("Compare Two Methods Side by Side")
col1, col2 = st.columns(2)

with col1:
    st.subheader("st.table (Native)")
    st.table(df_display.head(5))

with col2:
    st.subheader("HTML with CSS (Custom)")
    
    custom_css = """
    <style>
        .mini-table-container {
            width: 100%;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .mini-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        .mini-table th {
            background-color: rgba(108, 166, 205, 0.3);
            color: white;
            font-weight: bold;
            padding: 8px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        
        .mini-table td {
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            word-wrap: break-word;
        }
        
        .mini-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .mini-table tr:hover {
            background-color: rgba(108, 166, 205, 0.2);
        }
        
        /* Username should not wrap */
        .mini-table td:nth-child(2) {
            white-space: nowrap;
            font-weight: bold;
        }
    </style>
    """
    
    # Generate a simple HTML table with the custom css
    html_mini_table = f"""
    {custom_css}
    <div class="mini-table-container">
        <table class="mini-table">
            <tr>
                {''.join([f"<th>{col}</th>" for col in df_display.columns])}
            </tr>
            {''.join([
                f"<tr>{''.join([f'<td>{value}</td>' for value in row.values])}</tr>" 
                for _, row in df_display.head(5).iterrows()
            ])}
        </table>
    </div>
    """
    
    st.markdown(html_mini_table, unsafe_allow_html=True)

# Notes section
st.header("Implementation Notes")

st.markdown("""
### Key Issues and Solutions:

1. **Username Wrapping**: Usernames should not wrap to two lines
   - Solution: Use `white-space: nowrap` for username columns

2. **Horizontal Scrolling**: Tables with many columns need horizontal scrolling
   - Solution: Wrap table in container with `overflow-x: auto`

3. **Vertical Scrolling**: Long tables should have a fixed height with vertical scrolling
   - Solution: Set `max-height` and `overflow-y: auto` on container

4. **Long Text Fields**: Description and Tweet Summary fields need appropriate wrapping
   - Solution: Use `white-space: normal` and `word-wrap: break-word` with max-width constraints

5. **Styling Consistency**: Tables should match the overall dark theme
   - Solution: Use custom CSS with appropriate background colors and hover effects

6. **Performance**: Large tables should load and render efficiently
   - Solution: Limit initial data load, virtualize if needed
""")

# Code snippets
st.header("Recommended Implementation Code")

st.markdown("Based on testing, here's the recommended code for your tables:")

recommended_code = """
# HTML with Custom CSS approach
def display_table(df, container_height="500px", min_width="1000px"):
    css = '''
    <style>
        .table-container {
            width: 100%;
            overflow-x: auto;
            max-height: CONTAINER_HEIGHT;
            overflow-y: auto;
        }
        
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            min-width: MIN_WIDTH;
            font-size: 14px;
        }
        
        .custom-table th {
            background-color: rgba(108, 166, 205, 0.3);
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
            /* Add a solid background color with higher opacity for better visibility */
            background-color: rgb(38, 39, 48);
            border-bottom: 2px solid rgba(108, 166, 205, 0.7);
        }
        
        .custom-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            word-wrap: break-word;
            max-width: 300px;
        }
        
        /* Keep usernames on one line */
        .custom-table td.username {
            white-space: nowrap;
            font-weight: bold;
        }
        
        /* Format connection types */
        .custom-table td.connection {
            font-weight: bold;
            white-space: nowrap;
        }
        
        .connection-original { color: #ff9d00; }
        .connection-first { color: #00c3ff; }
        .connection-second { color: #8bc34a; }
        
        .custom-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .custom-table tr:hover {
            background-color: rgba(108, 166, 205, 0.2);
        }
    </style>
    '''.replace('CONTAINER_HEIGHT', container_height).replace('MIN_WIDTH', min_width)
    
    # Format each row with appropriate styling
    rows = []
    
    # Header row
    header = "".join([f"<th>{col}</th>" for col in df.columns])
    rows.append(f"<tr>{header}</tr>")
    
    # Data rows
    for _, row in df.iterrows():
        cells = []
        for i, (col, val) in enumerate(row.items()):
            # Format value based on column type
            if col == "Username":
                cells.append(f'<td class="username">{val}</td>')
            elif col == "Connection":
                css_class = ""
                if val == "Original":
                    css_class = "connection-original"
                elif val == "1st Degree":
                    css_class = "connection-first"
                elif val == "2nd Degree":
                    css_class = "connection-second"
                cells.append(f'<td class="connection {css_class}">{val}</td>')
            elif col in ["Description", "Tweet Summary"]:
                # Format long text fields for readability
                formatted_text = str(val).replace("\\n", "<br>")
                cells.append(f'<td>{formatted_text}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    # Combine everything
    html = f'''
    {css}
    <div class="table-container">
        <table class="custom-table">
            {"".join(rows)}
        </table>
    </div>
    '''
    
    return st.markdown(html, unsafe_allow_html=True)

# Usage
display_table(your_dataframe)
"""

st.code(recommended_code, language="python")

# Try the recommended code on our test data
st.header("Try the Recommended Implementation")

def display_table(df, container_height="500px", min_width="1000px"):
    css = '''
    <style>
        .table-container {
            width: 100%;
            overflow-x: auto;
            max-height: CONTAINER_HEIGHT;
            overflow-y: auto;
        }
        
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            min-width: MIN_WIDTH;
            font-size: 14px;
        }
        
        .custom-table th {
            background-color: rgba(108, 166, 205, 0.3);
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
            /* Add a solid background color with higher opacity for better visibility */
            background-color: rgb(38, 39, 48);
            border-bottom: 2px solid rgba(108, 166, 205, 0.7);
        }
        
        .custom-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            white-space: normal;
            word-wrap: break-word;
            max-width: 300px;
        }
        
        /* Keep usernames on one line */
        .custom-table td.username {
            white-space: nowrap;
            font-weight: bold;
        }
        
        /* Format connection types */
        .custom-table td.connection {
            font-weight: bold;
            white-space: nowrap;
        }
        
        .connection-original { color: #ff9d00; }
        .connection-first { color: #00c3ff; }
        .connection-second { color: #8bc34a; }
        
        .custom-table tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .custom-table tr:hover {
            background-color: rgba(108, 166, 205, 0.2);
        }
    </style>
    '''.replace('CONTAINER_HEIGHT', container_height).replace('MIN_WIDTH', min_width)
    
    # Format each row with appropriate styling
    rows = []
    
    # Header row
    header = "".join([f"<th>{col}</th>" for col in df.columns])
    rows.append(f"<tr>{header}</tr>")
    
    # Data rows
    for _, row in df.iterrows():
        cells = []
        for i, (col, val) in enumerate(row.items()):
            # Format value based on column type
            if col == "Username":
                cells.append(f'<td class="username">{val}</td>')
            elif col == "Connection":
                css_class = ""
                if val == "Original":
                    css_class = "connection-original"
                elif val == "1st Degree":
                    css_class = "connection-first"
                elif val == "2nd Degree":
                    css_class = "connection-second"
                cells.append(f'<td class="connection {css_class}">{val}</td>')
            elif col in ["Description", "Tweet Summary"]:
                # Format long text fields for readability
                formatted_text = str(val).replace("\n", "<br>")
                cells.append(f'<td>{formatted_text}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    # Combine everything
    html = f'''
    {css}
    <div class="table-container">
        <table class="custom-table">
            {"".join(rows)}
        </table>
    </div>
    '''
    
    return st.markdown(html, unsafe_allow_html=True)

# Try the recommended implementation
display_table(df)
