import streamlit as st
import seaborn as sns  
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sklearn
from openai import OpenAI

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


data = {
    'Advertising Spend': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'Price': [20, 25, 30, 35, 40, 45, 50, 55, 60],
    'Discount': [5, 10, 15, 20, 25, 30, 35, 40, 45],
    'Sales': [172, 219, 256, 310, 335, 368, 412, 460, 490]
}
example_df = pd.DataFrame(data)


# Function to plot individual scatter plots for each feature against output_var
def plot_sample_charts(df, output_var):
    features = [col for col in df.columns if col != output_var]
    color_palettes = sns.color_palette("tab10", len(features))  # Generate different colors for each feature
    
    col_objs = []  # List to store the column objects
    for i, feature in enumerate(features):
        if i % 2 == 0:  # Create new columns for every two features
            col1, col2 = st.columns(2)
            col_objs.extend([col1, col2])
        
        with col_objs[i]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=feature, y=output_var, data=df, color=color_palettes[i])
            plt.title(f'{feature} vs {output_var}')
            st.pyplot(plt)
            plt.clf()  # Clear the current figure


def generate_gpt_summary(model_type, features, target, mse, df):
    try:
        mean_values = df[features].mean().to_dict()
        target_positive_ratio = (df[target] > 0).mean() * 100
        prompt = f'''
        We trained a {model_type} model to predict {target} based on historical data. The model had a mean squared error of {mse}.
        Here are some key statistics from the historical data:
        - Average values: {mean_values}
        - Percentage of positive cases in target variable: {target_positive_ratio:.2f}%
        
        Given this context, what are the business implications of this model? How should you interpret and act on these predictions and historical trends? Give a concise but insightful answer on why this matters to the business/me. Sound human and not like chatGPT (tone: spartan, insightful, no corporate jargon)
        '''
        temperature = 0.4
        max_tokens = 500
        response = client.completions.create(model="text-davinci-003",  # Replace with the model you intend to use
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens)
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Function to run ML models and display results
def run_ml_models(df, output_var):
    X = df.drop(columns=output_var)
    y = df[output_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[model_name] = mse
    
    # Displaying model results
    st.write("### Model Results")
    st.write("""
    We ran three powerful machine learning models to extract meaningful insights from your data:
    1. **Linear Regression**: Unveils straightforward cause-and-effect relationships, such as understanding how changes in advertising spending directly impact sales.
    2. **Random Forest**: Excels in complex scenarios, providing precise recommendations. For instance, when dealing with a multitude of marketing variables affecting customer behavior, Random Forest can navigate through these variables to recommend the most effective marketing strategy.
    3. **Decision Tree**: Simplifies intricate situations, offering a clear view of strategic options. Imagine launching a new product with multiple pricing and marketing choices; Decision Tree breaks down the pros and cons of each choice, giving you a clear strategy.
    """)
    st.table(pd.DataFrame(list(results.items()), columns=['Model', 'Mean Squared Error']).set_index('Model'))
    best_model_name = min(results, key=results.get)
    st.write(f"The best model is {best_model_name} with a Mean Squared Error of {results[best_model_name]:.2f}.")
    
    # GPT-4 Explanation
      # Button to generate GPT-4 insights
    if st.button('What this means'):
        st.write("### Data Analysis and Insights")
        st.write("Powered by GPT-4")
        gpt_response = generate_gpt_summary(best_model_name, X.columns.tolist(), output_var, results[best_model_name], df)
        st.write(gpt_response)

# Process Uploaded File
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            output_var = st.selectbox('Select the output variable:', df.columns.tolist())
            st.dataframe(df)
            plot_sample_charts(df, output_var)
            run_ml_models(df, output_var)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Welcome Screen
def welcome_screen():
    logo_path = "/Users/samriddhiagnihotri/Desktop/eai.png"
    st.image(logo_path, use_column_width=True)
    st.markdown("""
    <h2 style='text-align: center; font-weight: bold;'> <i> Elevating Data into Intelligent Strategy </i> </h2>
 
    """, unsafe_allow_html=True)
    
    option = st.selectbox("What would you like to do?", ["About Us", "How the App Works", "File Upload", "Example Output"])
    
    if option == "About Us":
        st.markdown("""
        <h3 style='text-align: left;'>Welcome to a world where <i>Your Data</i> works for <i>You</i>.</h2>
        <p style='text-align: left;'>In an era where information is abundant, the difference between success and stagnation is the ability to leverage the right data at the right time.</p>
        <p style='text-align: left;'>At ExcaliburAI, we make the complex simple, giving you the tools to understand your data and make smarter business decisions.</p>
      
        """, unsafe_allow_html=True)



         # Adding a styled header
        st.markdown("<h3 font-weight: bold; text-decoration: underline;'>Why Choose ExcaliburAI?</h3>", unsafe_allow_html=True)

        # Creating sub-headers and their respective content
        sub_headers = [
            ("We Make Data Science Easy", "Harness the power of Machine Learning and Generative AI. We render complex data into easy-to-understand, actionable insights, making the sophisticated simple and accessible."),
            ("Find Opportunities You Didn’t Know Existed", "Uncover hidden gems in your data and find new ways to grow and succeed. With us, you see what others don’t."),
            ("Turn Your Information into Your Strategy", "Use your data to stay ahead of the competition. We help you turn your insights into actions, making your business stronger and smarter."),
        ]

        for header, content in sub_headers:
            # Using markdown with inline HTML styling for smaller, bolded subheadings
            st.markdown(f"<p style='font-weight: bold;'>{header}</p>", unsafe_allow_html=True)
            st.write(content)  # Displaying the content for each sub-header

        st.markdown("""
        <h3 style='text-align: left;'> <i> Innovation, Simplified. </i> </h2>
        <p style='text-align: left;'>In our commitment to excellence, we strive to deliver services and solutions characterized by simplicity, efficacy, and reliability. Just like how a sword cuts through ambiguity, ExcaliburAI slices through the noise, offering you the pure, undiluted essence of your data.</p>      
        """, unsafe_allow_html=True)



    elif option == "Example Output":
        st.write("""
        Below is an example of the kind of data this app can process and visualize. 
        This dataset represents sample advertising data from a medium-sized company. 
        It includes 'Advertising Spend', 'Price', 'Discount', and 'Sales' columns.
        
        In this scenario, a user might be interested in understanding how different factors 
        like advertising spend, pricing strategy, and discount levels impact the sales of the company. 
        The output variable for this dataset is 'Sales', and the user is possibly looking to analyze 
        and draw insights on how to optimize the advertising spend and pricing to maximize sales.
        """)
        st.dataframe(example_df)
        plot_sample_charts(example_df, 'Sales')
        run_ml_models(example_df, 'Sales')
    
    elif option == "How the App Works":
        st.write("""
        ### **How to Use This App**
        1. **Upload Your Data**: Use the "File Upload" option to upload your own data in Excel or CSV format.
        2. **Select Output Variable**: Choose the variable you would like to predict or analyze.
        3. **Run ML Models**: The app will run three ML models and display the results with insightful charts.
        4. **Analyze Results**: Understand the implications and insights drawn from the model results.
        5. **Download Report**: You have the option to download the analysis report for future reference.
        ### **Example Use Cases**
        - **Sales Prediction**: Analyze the impact of advertising spend, pricing, and discounts on sales.
        - **Customer Churn Analysis**: Understand the factors influencing customer retention and churn.
        - **Supply Chain Optimization**: Optimize inventory levels and logistics based on demand forecasts.
        ### **Business Utility**
        This app empowers businesses to leverage machine learning insights for informed decision-making. By understanding the correlations and patterns in your data, you can optimize your strategies, improve operations, and drive business growth.
        - **Informed Decision-Making**: Make data-driven decisions to optimize business outcomes.
        - **Strategic Insights**: Understand the impact of various factors on your target variable.
        - **Enhanced Productivity**: Quickly analyze and visualize data without the need for extensive data science knowledge.
        - **Business Growth**: Optimize business strategies and operations for enhanced profitability and growth.
        ### **How It Works**
        The app accepts datasets in Excel or CSV formats, allows users to select an output variable, and runs multiple machine learning models to analyze the impact of input variables on the selected output variable. It utilizes advanced machine learning algorithms and AI (GPT-4) to provide comprehensive insights and explanations, enabling users to understand and leverage the patterns and correlations in their data.
        Whether you are a business analyst, manager, or data enthusiast, this app provides a user-friendly interface to extract valuable insights from your data, helping you to drive your business forward.
        """)
    
    elif option == "File Upload":
        st.write("### **Upload Your File**\nPlease upload your data file in Excel (.xlsx) or CSV (.csv) format. After uploading, you'll need to select the output variable you want to analyze.")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        process_uploaded_file(uploaded_file)

    # Add the copyright notice at the end
    st.markdown("<p style='text-align: center; font-weight: bold;'>© 2023 Sam Agnihotri (sagniho LLC)</p>", unsafe_allow_html=True)
    #st.image('https://source.unsplash.com/featured/?artificialintelligence', use_column_width=True)

def main():
    welcome_screen()

def main():
    welcome_screen()

if __name__ == "__main__":
    main()
