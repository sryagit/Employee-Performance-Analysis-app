# Import necessary libraries
import streamlit as st #for creating the app
import pandas as pd #for loading the dataset and manipulating
import numpy as np #for numerical operations
import matplotlib.pyplot as plt     #for plotting and visualization
import seaborn as sns #for visualization
from sklearn.model_selection import train_test_split, GridSearchCV #for splitting the dataset into training and testing 
from sklearn.ensemble import RandomForestClassifier #for creating a random forest classifier
from sklearn.metrics import classification_report, accuracy_score #for evaluating the model's performance
from sklearn.preprocessing import StandardScaler, LabelEncoder #for preprocessing the data

# App title 
st.title("Employee Performance Analysis app")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Homepage", "Data Information", "Visualization", "Machine Learning Model"])

# Load the dataset
df = pd.read_excel("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls")



# Homepage
if option == "Homepage":
    st.title('HOMEPAGE')
    st.write('''Welcome to employee hiring app homepage
             
At INX Future Inc., CEO Mr. Brain is deeply concerned about declining employee performance, which has negatively impacted customer satisfaction and led to a rise in customer escalations.
To address this issue, Mr. Brain emphasizes the need for a thorough analysis of the factors contributing to poor employee performance. 
This analysis will consider various attributes such as environmental satisfaction, salary increases over the past year, overtime, and work-life balance. By understanding these factors,
Mr. Brain can identify areas where the company aims to improve its performance index and enhance customer satisfaction.

To tackle this challenge, predictive analytics will be employed to forecast employee performance and address issues before hiring. 
The project will leverage machine learning, specifically Random Forest, to analyze historical data and make accurate predictions about future employee performance. 
This approach will provide valuable insights for Mr. Brain to make informed decisions and take appropriate actions to boost employee performance and maintain the company’s competitive edge in the market.
''')
   

# Data Information
elif option == "Data Information":
    st.write("### Data Information")
    
    # Display the first 5 rows
    st.write('The first five rows of the dataset:', df.head())
    
    # User input: number of rows to display
    num_rows = st.slider("Select the number of rows", min_value=1, max_value=len(df), value=5)
    st.write("Here are the rows you have selected from the dataset:")
    st.write(df.head(num_rows))
    
    # Display the summary statistics of the dataset
    st.write('Summary statistics of the dataset:')
    st.write(df.describe())
    
    # Check for duplicates
    if st.checkbox("Check for duplicates"):
        st.write(df[df.duplicated()])
    
    # Total number of duplicates
    if st.checkbox("Check for total number of duplicates"):
        st.write(df.duplicated().sum())
    
    #display the number of rows and columns
    df.shape
    st.write('number of rows:',df.shape[0])
    st.write('number of columns:',df.shape[1])

    # Check for missing values
    if st.checkbox("Check for missing values"):
        st.write(df.isnull().sum())

    # Total number of missing values
    if st.checkbox("Check for total number of missing values"):
        st.write(df.isnull().sum().sum())

# Visualization
elif option == "Visualization":
    st.write("### Data Visualization")

    # Plot histogram for age distribution
    st.write('Age distribution')
    plt.figure(figsize=(6,4))
    sns.histplot(data=df, x='Age', color='orange',bins=10)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())
    st.write('''
    From the figure above of age distribution,more workers age ranges between 30-40yrs.
    The 50-60 years age group belong around 40-80 workers
    Around 50-150 wrokers belong to the age group 20-29 years''')

    # 2. DISTRIBUTION OF PERFOMANCE RATING
    st.write('Distribution of Performance Rating')
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='PerformanceRating',color='orange')
    plt.title('Distribution of Performance Ratings')
    plt.xlabel('Performance Rating')
    plt.ylabel('Count')
    st.pyplot(plt.gcf()) #plt.show()
    st.write('''
    More workers (3) excellent perfomance led both Good (2) and Outstanding (4) in terms of performance rating.
    There were less than 200 exceptional performers(4), and there were about 200 good performers(2).''')
    
    # 3. Calculate departmental average performance
    st.write('Departmental Average Performance Rating')
    department_avg_performance = df.groupby('EmpDepartment')['PerformanceRating'].mean()
    # Create a pie chart
    plt.figure(figsize=(6,6))
    plt.pie(department_avg_performance, labels=department_avg_performance.index, autopct='%1.1f%%')
    plt.title('Departmental Average Performance Rating')
    st.pyplot(plt.gcf())
    st.write('''
    FROM THE FIG ABOVE;
    The department of development is in charge of all other departments at 17.5%, with the data science department coming in second at 17.3%. 
    However there is a a slight difference of 0.20% between Development and data science department
    Human Resource and Reserch and Development share the same percentage at 16.6% in perfomance
    The department with the lowest average perfomance was finance at 15.8%
    This figure shows that there might be differences in hiring personel,allocations of resources and employee rewards in different departments that might affect there perfomance. 
    The business must maintain equity and strike a balance.''')

  # 4. Calculate departmental average performance
    st.write('Departmental Average Performance Rating')
    department_avg_performance = df.groupby('EmpDepartment')['PerformanceRating'].mean()

   # Create a bar chart with slanted x-labels
    plt.figure(figsize=(6,4))
    department_avg_performance.plot(kind='bar', color='skyblue')
    plt.title('Departmental Average Performance Rating')
    plt.xlabel('Department')
    plt.ylabel('Average Rating (Integer)')
    plt.xticks(rotation=45, ha='right')  # Slant x-axis labels
    plt.grid(axis='y')
    st.pyplot(plt.gcf())
    st.write('''
    FROM THE FIGURE,WE UNDERSTAND THAT;
    Development perfomed better and finace the least. To confim the pie chart above''')

    # 5. perfomance rating distribution per department
    st.write('Performance Rating Distribution per Department')
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='EmpDepartment', hue='PerformanceRating', palette='viridis')
    plt.title('Performance Rating Distribution per Department')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.legend(title='Performance Rating', loc='upper right')
    plt.xticks(rotation=45)  # Rotate department names for better readability
    st.pyplot(plt.gcf())
    st.write('''
    1.** SALES DEPARTMENT ** -In sales department,the perfomance rating was 3,excellent at 250,followed by good at 2 with a perfomance rating of 90 and the 4 outstanding was the last one at less than 50
             
     2. *Human Resource *
    The perfomance rating was close at each other,with outstanding 4,was leading followed by 2 good and then 3 excellent was the last.However these perfomances were below 50.
    3. Development
    Department did well and ser passsed all other departments with excellent perfomance rating at 300,in that same department,outstanding was at 50 with Good perfomance was last at less than 40.
    4. Data science
    In data science department 3 excellent perfomance was leading with 10. however the other rating was less.
    5. *Research and development * Excellent perfomance was topping at 250,followed by good at 70 and outstanding was at 40.
    6. FINANCE Outstanding perfomance was at 30,followed by good and excellent perfomance.
    In the figure above, it can be conlcuded that all perfomance rating was recoded at 3,excellent leading with development leading the other 5. However,the departments should aim at being at 4 which is outstanding perfomance.
    There should be increased personell,resources and employees willingness to work.
             ''')
    
    st.write('Gender frequency')

   # Set the figure size before plotting
    plt.figure(figsize=(6, 4))

# Plotting the frequency of 'Gender' using countplot
    sns.countplot(x='Gender', data=df)

# Adding labels and title
    plt.xlabel("Gender")
    plt.ylabel("Frequency")
    plt.title("Frequency of Gender")

# Display the plot
    st.pyplot(plt.gcf())
    st.write('''
    More workers were male above 700 compaired to female workers around 400. 
    The comapny should have strategies to improve on balancing gender and try not to be gender biased.''')
   
    # 6. Calculate the average Performance Rating per Gender in each Department
    st.write('Average Performance Rating per Gender in Each Department')
    avg_perf_rating = df.groupby(['EmpDepartment', 'Gender'])['PerformanceRating'].mean().reset_index()

    # Plotting the results
    plt.figure(figsize=(6,4))
    sns.barplot(data=avg_perf_rating, x='EmpDepartment', y='PerformanceRating', hue='Gender', palette='tab10')
    plt.title('Average Performance Rating per Gender in Each Department')
    plt.xlabel('Department')
    plt.ylabel('Average Performance Rating')
    plt.legend(title='Gender', loc='upper right')
    plt.xticks(rotation=45)  # Rotate department names for better readability
    st.pyplot(plt.gcf())
    st.write('''
    In the fig above,
    The pattern indicates that both genders' performance is generally balanced across the departments. It should be highlighted, nonetheless, 
    that only two departments—Data Science and Development—had average performance ratings of 3 (excellent) or above for both genders.
    Therefore, the business should concentrate on providing performance enhancement trainings to both men and women and look into any potential gender-based hiring practices that may be occurring covertly in the departments.
       ''')
    # Plotting the frequency of 'Attrition' using countplot
    st.write('Atrrition frequency')

   # Set the figure size before plotting
    plt.figure(figsize=(6, 4))

# Plotting the frequency of 'Gender' using countplot
    sns.countplot(x='Attrition', data=df)
# Adding labels and title
    plt.xlabel("Attrition")
    plt.ylabel("Frequency")
    plt.title("Frequency of Attrition")
# Display the plot
    st.pyplot(plt.gcf())
    st.write('''- Many workers did not leave the company compared to those who did, with less than 200 employees leaving.''')
# Plotting the frequency of 'OverTime' using countplot
    st.write('OverTime frequency')
# Set the figure size before plotting
    plt.figure(figsize=(6, 4))
# Plotting the frequency of 'Gender' using countplot
    sns.countplot(x='OverTime', data=df)
#Adding labels and title
    plt.xlabel("overTime")
    plt.ylabel("Frequency")
    plt.title("Frequency of overTime")
# Display the plot
    st.pyplot(plt.gcf())
    st.write('''From the figure above, More people did not work overtime over 800, compaired to yes at 300''')

    #7.  Calculate last year training times vs average performance
    st.write('Last year Traning Times vs Average Performance Rating')
    lastyr_avg_traning_times= df.groupby('TrainingTimesLastYear')['PerformanceRating'].mean()
    # Create a pie chart
    plt.figure(figsize=(6,6))
    plt.pie(lastyr_avg_traning_times, labels=lastyr_avg_traning_times.index, autopct='%1.1f%%')
    plt.title('Last year Traning Times vs Average Performance Rating')
    st.pyplot(plt.gcf())
    st.write('''
    INTERPRETANTION
    As shown in the figure, those who had three training sessions last year performed better at 14.5%
    Workers who received no training during the previous year did the worst at 14.0%
    It should be mentioned, nevertheless, that workers who had one or two training sessions performed the same at 14.4% with those who got traning 5 times. The business should thus take note of this and refrain from spending so much money educating staff members up to six times at 14.0%
    The money need to go toward providing employees with zero(0) training time with instruction so they can perform better.
             ''')
    
    # 8. Employee relationship satisfaction with perfomance rating
    st.write('Relationship Satisfaction vs Performance Rating')
     #Mapping for Relationship Satisfaction
    satisfaction_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
    # Plotting
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x='EmpRelationshipSatisfaction', hue='PerformanceRating', palette='viridis')
    # Setting the x-ticks to the mapped values
    plt.xticks(ticks=[0, 1, 2, 3], labels=[satisfaction_mapping[i] for i in range(1, 5)])
    # Adding titles and labels
    plt.title('Relationship Satisfaction vs Performance Rating')
    plt.xlabel('Relationship Satisfaction')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())# Show the plot
    st.write('''
    INTERPRETATION *RELATIONSHIP SATISFACTION *
    People with high relationship satisfaction perfomed excellent compared to other levels.
    With very high satistifaction there satisfaction rating was outstanding compared to the levels
    The workers should moderete there relationship in order to perform better.
              ''')
        

     #10# List of columns to analyze
    st.write('List of columns to analyze')
    columns_analyze = ['Age', 'DistanceFromHome', 'EmpHourlyRate', 'EmpLastSalaryHikePercent',
                   'TotalWorkExperienceInYears', 'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
     # Set up the plotting area for subplots
    plt.figure(figsize=(14, 16))
      # Loop through the columns to analyze and create a bar plot for each
    for i, column in enumerate(columns_analyze, 1):
       plt.subplot(4, 2, i)
      # Create a barplot showing average performance rating per department for each selected column
       avg_performance = df.groupby(['EmpDepartment', column])['PerformanceRating'].mean().reset_index()
        # Plot
       sns.barplot(x='EmpDepartment', y='PerformanceRating', hue=column, data=avg_performance, palette='viridis')
       plt.title(f'Average Performance Rating by {column}')
       plt.xlabel('EmpDepartment')
       plt.ylabel('Average Performance Rating')
       plt.xticks(rotation=45)
       plt.legend(title=column)
       plt.tight_layout()
       st.pyplot(plt.gcf())#plt.show()
       st.write('''
       INTERPRETATION 
       1. *DEPARTMENTWISE AVERAGE PERFOMANCE RATING BY AGE*
       Different ages performed differently in different departments. For example the 24years age group performed outstandingly in the Data Science compared to all other departments. The age between 15-24 performed well in Development. The age 55+ performed poorly in Finance. The company needs to evaluate the age group in finance department and should have all age groups catered in all departments.
       2. *DEPARTMENTAL AVERAGE PERFOMACE WITH DISTANCE FROM HOME. Additionally, performance is impacted differentially by distance. This can be a result of the type of work that the staff members conduct.However,in development,reseache and development and sales departments perfomance was not affected much by the distance travelled by workers.
       3. Deparmentwise average perfomace by employee hourlyrent
       Different departments are impacted differently by the hourly rate as well. There appears to be a more equitable distribution of the hourly rate of 60-74 among all departments. However, develpoment,reserch and development sales did not affected by hourly rate. In other departments eg data science; the perfomance was high when the hourly rate was at 75.
       As a result, the business must pay its workers properly and faily for the same work done in order to encourage a positive outlook and higher performance.
       4. DEPARTMENTALWISE PERFOMANCE BY EMPLOYEE LAST SALARY HIKE
       It is evident that employee performance in every department is impacted by the percentage of wage increases. Workers who received a pay increase of twenty percent or more outperformed all other workers who received a smaller raise. This company's undercompensation problem has to be looked at. As a result, the business must establish a uniform wage increase percentage that is applied to every employee.
       5. DEPARTMENTWISE PERFOMANCE BY TOTAL WORK EXPERIRNCE IN YEARS
       Depending on their level of experience, employees in different departments are impacted in different ways for example,in sales people with work experience of more than 40years perfomed better. As a result, it is advised that every department hires workers with the kind of work experience that results in exceptional performance.
       6. DEPARTMENTALWISE PERFOMANCE BY EXPERIENCE IN YEARS IN THE CURRENT ROLE.
       Employees with zero to nine years of work experience in this organization demonstrate balanced performance. Workers with between 20 and 29 years of experience performed poorly in every department of this organization. However,in sales department,more people with 18yrs perfomed poorly. In data science,people with 0-3 yrs perfomed better. In finance,6-9yrs also perfomed better.
       As a result, the organization ought to handle the problem of employee overstay in several departments.
       7.** DEPARMENTALWISE PERFOMANCE BY YEARS SINCE LAST PROMOTION**
       The number of years since the last promotion obscures the performance grade. In every department, it is always changing. In development,perfomance was excellent despite the years since last promoted. However, in human resource and develpment people who got promoted 10-15yrs ago perfomed poorly. In data science people who got promoted 5 yrs a go perfomed well.
       As a result, the business must set up the promotion schedule by department.
       8. DEPARTMENTALWISE AVERAGE PERFOMANCE BY THE CURRENT MANAGER
                
       The Data Science and Finance departments unequivocally show that having the same boss for an extended period of time had a negative impact on their performance,they perfomance was excellent and outsanding when workers spent 6-9yrs with the manager.The business ought to think about having a rotating manager in these divisions.
       In development there was no major change. However, in reserch and development and sales, people who spent 12-15yr with the current manager perfomed excellent. Manager rotation should therefore be kept to at minimum and should be changed as per department.
               ''')

     #11 correlation matrix

    st.write('Correlation Matrix')
    corr_data=df[['EmpWorkLifeBalance','EmpLastSalaryHikePercent','PerformanceRating','DistanceFromHome','EmpEducationLevel','EmpEnvironmentSatisfaction','EmpHourlyRate','TotalWorkExperienceInYears',]]
    #calculating correlation matrix
    corr_matrix =corr_data.corr() 
    #heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt.gcf())
    st.write('''
    *INTERPRETATION *
    Correlation matrix is a visual representation of the correlation between variables in a dataset.
    There is a weak positive correlation of 0.12 between employee work balance and perfomance rating
    There is a positive correlation of 0.33 between salary hike and perfomance rating
    When it comes to distance from home there is a weak negative correlation of -0.05 with perfomance rating.
    There is a Weak positive correlation between employee educutaion level and perfomance rating at 0.02
    There is a psoitive correlation between employee job satisfaction at 0.40 with perfomance rating
    There is negative correlation of -0.04 between employee hourly rate and perfomance rating
    There is negative correation of -0.07 between total work experirience in years and perfomance rating
    ''')

# Machine Learning Model
elif option == "Machine Learning Model":
     st.title('Performance Rating Prediction')

    # Drop unnecessary columns and specified features
     df.drop(columns=['EmpNumber', 'Attrition', 'NumCompaniesWorked', 'Gender', 'MaritalStatus',
                     'EducationBackground', 'EmpJobLevel', 'EmpRelationshipSatisfaction'], inplace=True)

    # Filter target variable to include only specific Performance Rating values
     df = df[df['PerformanceRating'].isin([2, 3, 4])]

    # Identify categorical columns to be encoded
     encoded_columns = ['EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime']

    # Encode categorical columns
     le_dict = {col: LabelEncoder().fit(df[col]) for col in encoded_columns}
     for col in encoded_columns:
         df[col] = le_dict[col].transform(df[col])

    # Identify numerical columns
     numerical_columns = ['Age', 'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
                         'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobSatisfaction', 'EmpLastSalaryHikePercent',
                         'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'EmpWorkLifeBalance',
                         'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
                         'YearsWithCurrManager']

    # Separate features and target variable
     X = df.drop(columns=['PerformanceRating'])
     y = df['PerformanceRating']

    # Feature scaling of numerical columns
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier
     model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
     model.fit(X_train, y_train)

    # Make predictions on the test set
     y_pred = model.predict(X_test)

    # User input for new data
     st.write("## Enter new data for prediction")

    # Collecting user input for each feature using the unique values from the dataset
     EmpDepartment = st.selectbox("Employee Department", le_dict['EmpDepartment'].classes_)
     EmpJobRole = st.selectbox("Employee Job Role", le_dict['EmpJobRole'].classes_)
     BusinessTravelFrequency = st.selectbox("Business Travel Frequency", le_dict['BusinessTravelFrequency'].classes_)
     Overtime = st.selectbox("Overtime", le_dict['OverTime'].classes_)
     Age = st.number_input("Age")
     DistanceFromHome = st.number_input("Distance from Home")
     EmpEducationLevel = st.number_input("Employee Education Level")
     EmpEnvironmentSatisfaction = st.number_input("Employee Environment Satisfaction")
     EmpHourlyRate = st.number_input("Employee Hourly Rate")
     EmpJobInvolvement = st.number_input("Employee Job Involvement")
     EmpJobSatisfaction = st.number_input("Employee Job Satisfaction")
     EmpLastSalaryHikePercent = st.number_input("Employee Last Salary Hike Percent")
     TrainingTimesLastYear = st.number_input("Training Times Last Year")
     EmpWorkLifeBalance = st.number_input("Employee Work Life Balance")
     YearsSinceLastPromotion = st.number_input("Years Since Last Promotion")
     YearsWithCurrManager = st.number_input("Years with Current Manager")
     TotalWorkExperienceInYears = st.number_input("Total Work Experience in Years")
     ExperienceYearsAtThisCompany = st.number_input("Experience Years at This Company")
     ExperienceYearsInCurrentRole = st.number_input("Experience Years in Current Role")

    # Encode user input
     encoded_input = {
        'EmpDepartment': le_dict['EmpDepartment'].transform([EmpDepartment])[0],
        'EmpJobRole': le_dict['EmpJobRole'].transform([EmpJobRole])[0],
        'BusinessTravelFrequency': le_dict['BusinessTravelFrequency'].transform([BusinessTravelFrequency])[0],
        'OverTime': le_dict['OverTime'].transform([Overtime])[0],
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole
    }

    # Ensure that the order of columns in input_df matches the order of columns in X
     input_df = pd.DataFrame([encoded_input], columns=X.columns)

    # Now scale the input data
     input_df_scaled = scaler.transform(input_df)


     # Add a button for making predictions
     if st.button('Make Prediction'):
    # Predict using the model
       prediction = model.predict(input_df_scaled)

    # Display the predicted performance rating
       st.write("Predicted Performance Rating: ", prediction[0])

    # Display performance rating based on prediction
       if prediction[0] == 2:
        st.write("Performance rating: Good")
       elif prediction[0] == 3:
        st.write("Performance rating: Excellent")
       else:
        st.write("Performance rating: Outstanding")
