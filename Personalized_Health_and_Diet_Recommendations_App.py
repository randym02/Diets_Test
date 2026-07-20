import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# Setup OpenAI API key
openai.api_key = "__"

# Load the necessary CSV files
foundation_food_df = pd.read_csv("foundation_food.csv")  # Contains food details
food_df = pd.read_csv("food.csv")  # Contains food names and descriptions
food_nutrient_df = pd.read_csv("food_nutrient.csv")  # Contains nutrient data for each food
nutrient_df = pd.read_csv("nutrient.csv")  # Contains nutrient names and units

# Merge foundation_food with food_df to get the description column
food_with_description_df = foundation_food_df.merge(food_df[['fdc_id', 'description']], on='fdc_id', how='left')

# Merge food_nutrient with nutrient (to get nutrient names)
food_nutrient_with_names_df = food_nutrient_df.merge(nutrient_df[['id', 'name']], left_on="nutrient_id", right_on="id", how="left", suffixes=('', '_nutrient'))

# Filter for essential nutrients by checking if the nutrient name is in the list of essential nutrients
essential_nutrients = [
    "Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference"
]

food_nutrient_with_names_df = food_nutrient_with_names_df[food_nutrient_with_names_df["name"].isin(essential_nutrients)]

# Aggregate nutrients for each food item (taking mean if there are multiple records for one food)
food_nutrient_aggregated_df = food_nutrient_with_names_df.groupby(['fdc_id', 'name'], as_index=False).agg({
    'amount': 'mean'
})

# Pivot the nutrient data to create a wide format (each nutrient as a column)
pivoted_nutrient_df = food_nutrient_aggregated_df.pivot(index="fdc_id", columns="name", values="amount").reset_index()

# Merge the food data with nutrient data
food_with_nutrients_df = food_with_description_df.merge(pivoted_nutrient_df, on="fdc_id", how="left")

# Clean the data: Remove any rows with missing essential nutrients or duplicate food items
food_with_nutrients_df = food_with_nutrients_df.dropna(subset=essential_nutrients).drop_duplicates(subset='description')

# Handle missing values by filling with 0 for simplicity
food_with_nutrients_df = food_with_nutrients_df.fillna(0)

# Define a healthiness score based on the nutrients
def assign_healthiness_score(row):
    score = 0
    if row['Total lipid (fat)'] > 20:  # High fat
        score -= 1
    if row['Carbohydrate, by difference'] > 50:  # High carbs
        score -= 1
    if row['Energy'] > 500:  # High calorie
        score -= 1
    if row['Protein'] < 5:  # Low protein
        score -= 1
    return score

# Apply the score to the data
food_with_nutrients_df['healthiness_score'] = food_with_nutrients_df.apply(assign_healthiness_score, axis=1)

# Prepare features and labels for the decision tree classifier
X = food_with_nutrients_df[essential_nutrients]  # Nutrient columns as features
y = food_with_nutrients_df['healthiness_score']  # Healthiness score as the label

# Check class distribution to handle imbalance
print("Class distribution in the training set:\n", y.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Train the Decision Tree Classifier with balanced class weights to handle class imbalance
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Measure the effectiveness of the model using accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Function to suggest foods based on BMI and healthiness score
def suggest_foods(bmi):
    # Determine food suggestions based on BMI and healthiness score
    if bmi > 30:
        food_options = food_with_nutrients_df[food_with_nutrients_df['healthiness_score'] >= -1]  # Slightly unhealthy foods allowed
    else:
        food_options = food_with_nutrients_df[food_with_nutrients_df['healthiness_score'] >= 0]  # Healthy foods only

    if not food_options.empty:
        sample_size = min(3, len(food_options))
        return food_options.sample(sample_size)
    else:
        return pd.DataFrame()

# Function to get diet advice from OpenAI
def get_diet_advice(bmi, recommended_foods):
    food_list = ", ".join(recommended_foods['description'].tolist())
    prompt = (
        f"My BMI is {bmi:.1f}. Based on the following foods: {food_list}, provide a "
        "diet recommendation that explains why these foods are good for my health. If an ingredient is listed, provide foods that can be made from it. Take into account the user's age, health goal, activity level, and gender."
        "For a high BMI person, avoid suggesting foods that are high in fat, sugar, or empty calories."
        "For a low BMI person, avoid suggesting foods that are lower in fat, sugar, or empty calories."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            timeout=70
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.Timeout as e:
        return f"Request timed out: {e}"
    except openai.error.OpenAIError as e:
        return f"Error with OpenAI API: {e}"

# BMI Calculation function
def calculate_bmi(weight, height):
    return (weight / (height ** 2)) * 703

# Streamlit UI
st.title("Personalized Health and Diet Recommendations")
st.write("Enter your age, weight, and height to get your BMI and personalized food suggestions.")

# User inputs (sliders for age, weight, height)
age = st.slider("Age", min_value=1, max_value=120, value=25)
weight = st.slider("Weight (lbs)", min_value=1, max_value=1000, value=244)

# Height input (feet and inches side by side)
col1, col2 = st.columns(2)

with col1:
    feet = st.number_input("Height (feet)", min_value=1, max_value=8, value=5)

with col2:
    inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=9)

# Calculate total height in inches
total_height_in_inches = feet * 12 + inches

# Inputs for health goal, activity level, and gender
health_goal = st.radio("Health Goal", options=["Weight Loss", "Maintenance", "Weight Gain"])
activity_level = st.selectbox("Activity Level", options=["Inactive", "Moderately Active", "Active"])
gender = st.selectbox("Gender", options=["Male", "Female"])

# Button to calculate BMI and receive personalized diet advice
if st.button("Calculate BMI and Receive Personalized Diet Advice"):
    bmi = calculate_bmi(weight, total_height_in_inches)
    st.write(f"Your BMI is: {bmi:.2f}")
    
    # Suggest food based on BMI
    recommended_foods = suggest_foods(bmi)
    st.write("Based on your BMI, here are some food suggestions:")
    
    if recommended_foods.empty:
        st.write("No food recommendations available for your BMI.")
    else:
        st.dataframe(recommended_foods[['description', 'Energy', 'Protein', 'Total lipid (fat)', 'Carbohydrate, by difference']])
        
        # Get diet advice from OpenAI
        advice = get_diet_advice(bmi, recommended_foods)
        st.write("Personalized Diet Advice:")
        
        st.markdown(f'<div style="max-height: 400px; overflow-y: auto;">{advice}</div>', unsafe_allow_html=True)

# Display model accuracy and performance metrics
st.write(f"Model accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")
